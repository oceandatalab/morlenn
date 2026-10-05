#!/usr/bin/env python
"""build_neighbor_pool.py — pools voisins Argo pour l'ENTRAINEMENT assimilé (KNN symétrique).

Reprend la logique du builder d'origine (KNN causal->symétrique) mais PARAMETRE par la config :
le vecteur voisin = [MLD | static_features] dans l'espace normalisé du précompute (stats=None).
Avec le set aligné (11 statics) -> feat_dim = 1 + 11 = 12.

Voisinage : KD-tree sur (position 3D scalée /LS_km, jour /LT_jours), K plus proches en excluant
soi-même (symétrique = passé+futur, leave-one-out). Sort neighbor_{train,val}.npz {feat[N,K,F], mask[N,K]}.

  build_neighbor_pool.py --config POOLCFG \
     --precompute-train ... --precompute-val ... --coords-train ... --coords-val ... \
     --out-train ... --out-val ... [--k 5 --ls 100 --lt 10]
"""
import argparse
import numpy as np
from sklearn.neighbors import KDTree

from morlenn.config import load_config
from morlenn.data import PrecomputedTensorDataset


def static_mld(precompute_dir, cfg):
    ds = PrecomputedTensorDataset(
        precompute_dir,
        dynamic_features=cfg.dataset.dynamic_features,
        static_features=cfg.dataset.static_features,
        stats=None,  # espace natif du précompute == espace d'entrée du modèle
        target_names=cfg.dataset.targets,
        mld_target_transform=cfg.dataset.mld_target_transform,
    )
    P = ds.single_payload
    return P["static"].numpy().astype(np.float32), P["target"][:, 0].numpy().astype(np.float32)


def load_coords(path):
    z = np.load(path)
    return z["lat"].astype("f8"), z["lon"].astype("f8"), z["day"].astype("f8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    # Mode global (un seul set) :
    ap.add_argument("--precompute", default=None, help="Global: precompute unique (data/pool/precompute/all).")
    ap.add_argument("--coords", default=None, help="Global: coords uniques (coords_all.npz).")
    ap.add_argument("--out", default=None, help="Global: sortie unique (neighbor_all_aligned.npz).")
    # Mode legacy (train/val séparés) :
    ap.add_argument("--precompute-train", default=None)
    ap.add_argument("--precompute-val", default=None)
    ap.add_argument("--coords-train", default=None)
    ap.add_argument("--coords-val", default=None)
    ap.add_argument("--out-train", default=None)
    ap.add_argument("--out-val", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ls", type=float, default=100.0, help="échelle spatiale (km)")
    ap.add_argument("--lt", type=float, default=10.0, help="échelle temporelle (jours)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    R, K, LS, LT = 6371.0, a.k, a.ls, a.lt

    if a.precompute is not None:
        # ---- Mode GLOBAL : KDTree sur l'ensemble, une seule sortie alignée au precompute global.
        if a.coords is None or a.out is None:
            raise SystemExit("Mode global : --precompute, --coords et --out requis ensemble.")
        pool_s, pool_m = static_mld(a.precompute, cfg)
        pool_feat = np.concatenate([pool_m[:, None], pool_s], 1).astype(np.float32)
        print(f"pool_feat {pool_feat.shape} (feat_dim={pool_feat.shape[1]})", flush=True)
        lat, lon, day = load_coords(a.coords)
        assert len(lat) == len(pool_s), f"coords {len(lat)} != precompute {len(pool_s)}"
        la, lo = np.deg2rad(lat), np.deg2rad(lon)
        gx, gy, gz = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
        pt = np.stack([R * gx / LS, R * gy / LS, R * gz / LS, day / LT], 1)
        tree = KDTree(pt)
        F = pool_feat.shape[1]
        N = len(lat)
        feat = np.zeros((N, K, F), np.float32)
        mask = np.zeros((N, K), np.float32)
        ind = tree.query(pt, k=60, return_distance=False)
        for r in range(N):
            keep = []
            for j in ind[r]:
                if j == r:
                    continue  # symétrique : passé+futur, leave-one-out
                keep.append(j)
                if len(keep) == K:
                    break
            for c, j in enumerate(keep):
                feat[r, c] = pool_feat[j]; mask[r, c] = 1.0
            if r and r % 50000 == 0:
                print(f"  all {r}/{N}", flush=True)
        np.savez(a.out, feat=feat, mask=mask)
        print(f"all: {feat.shape} full{K}={(mask.sum(1) == K).mean() * 100:.1f}% -> {a.out}", flush=True)
        print("neighbor pool GLOBAL DONE", flush=True)
        return

    for req in ("precompute_train", "precompute_val", "coords_train", "coords_val", "out_train", "out_val"):
        if getattr(a, req) is None:
            raise SystemExit(f"Mode legacy : --{req.replace('_','-')} requis (ou utilise le mode global --precompute/--coords/--out).")
    tr_s, tr_m = static_mld(a.precompute_train, cfg)
    va_s, va_m = static_mld(a.precompute_val, cfg)
    Ntr = len(tr_s)
    pool_s = np.concatenate([tr_s, va_s]); pool_m = np.concatenate([tr_m, va_m])
    pool_feat = np.concatenate([pool_m[:, None], pool_s], 1).astype(np.float32)  # [Npool, 1+static]
    print(f"pool_feat {pool_feat.shape} (feat_dim={pool_feat.shape[1]})", flush=True)

    tl, to, td = load_coords(a.coords_train); vl, vo, vd = load_coords(a.coords_val)
    # garde-fou d'alignement : coords <-> précompute doivent avoir le MEME nombre d'échantillons
    assert len(tl) == len(tr_s), f"coords train {len(tl)} != precompute train {len(tr_s)}"
    assert len(vl) == len(va_s), f"coords val {len(vl)} != precompute val {len(va_s)}"

    lat = np.concatenate([tl, vl]); lon = np.concatenate([to, vo]); day = np.concatenate([td, vd])
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    gx, gy, gz = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
    pt = np.stack([R * gx / LS, R * gy / LS, R * gz / LS, day / LT], 1)
    tree = KDTree(pt)

    def build(name, idxs, out):
        F = pool_feat.shape[1]
        feat = np.zeros((len(idxs), K, F), np.float32)
        mask = np.zeros((len(idxs), K), np.float32)
        ind = tree.query(pt[idxs], k=60, return_distance=False)
        for r, t in enumerate(idxs):
            keep = []
            for j in ind[r]:
                if j == t:
                    continue  # symétrique : passé+futur, leave-one-out
                keep.append(j)
                if len(keep) == K:
                    break
            for c, j in enumerate(keep):
                feat[r, c] = pool_feat[j]; mask[r, c] = 1.0
            if r and r % 50000 == 0:
                print(f"  {name} {r}/{len(idxs)}", flush=True)
        np.savez(out, feat=feat, mask=mask)
        print(f"{name}: {feat.shape} full{K}={(mask.sum(1) == K).mean() * 100:.1f}% -> {out}", flush=True)

    build("train", np.arange(Ntr), a.out_train)
    build("val", np.arange(Ntr, len(lat)), a.out_val)
    print("neighbor pool DONE", flush=True)


if __name__ == "__main__":
    main()
