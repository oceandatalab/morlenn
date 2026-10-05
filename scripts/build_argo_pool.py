#!/usr/bin/env python
"""
build_argo_pool.py:
    Build neighbooring pool of ARGO data when running assimlated inference
    format grid: feat, xyz, day
    with feat=[MLD, 19 statics from surface observations and model]
    Data should be in the same normalised space as model

USAGE:
     python build_argo_pool.py --precompute <PRECOMP/val> \
                               --coords <coords_val_split2023.npz> \
                               --config <config_19statics.toml> \
                               --out <data/neighbor/grid_pool_argo_2023.npz>
ARGUMENTS: 
    --precompute: add folder that contains precomputed fields
    --coords: Add numpy (npz) with spatio-temporal coordinates
    --config: toml with configuration (statitics fields)
    --year: Filter to keep only one year
    
RETURNS:
    numpy object with neighboring pool
"""
import argparse, datetime
import numpy as np
from morlenn.config import load_config
from morlenn.data import PrecomputedTensorDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precompute", required=True,
                    help="Folder that contains precomputed val (tensors+metadata)")
    ap.add_argument("--coords", required=True,
                    help="coords_val_*.npz (lat, lon, day)")
    ap.add_argument("--config", required=True, help="config TOML with statics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=None,
                    help="Optional, filter profiles for a specific year")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ds = PrecomputedTensorDataset(
        a.precompute,
        dynamic_features=cfg.dataset.dynamic_features,
        static_features=cfg.dataset.static_features,
        stats=None,  # precompute-native == original model space 
        target_names=cfg.dataset.targets, mld_target_transform="log1p",
    )
    P = ds.single_payload
    s = P["static"].numpy().astype("f4")
    m = P["target"][:, 0].numpy().astype("f4")
    feat = np.concatenate([m[:, None], s], 1).astype("f4")  # [N, 1+19=20]

    z = np.load(a.coords)
    lat = z["lat"].astype("f8")
    lon = z["lon"].astype("f8")
    day = z["day"].astype("f8")
    if a.year is not None:
        keep = np.array([datetime.date.fromordinal(int(d)).year == a.year for d in day])
        feat, lat, lon, day = feat[keep], lat[keep], lon[keep], day[keep]
        print(f"filtering year {a.year}: {keep.sum()}/{keep.size} profils kept")
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1).astype("f4")

    np.savez(a.out, feat=feat, xyz=xyz, day=day)
    d0, d1 = datetime.date.fromordinal(int(day.min())), datetime.date.fromordinal(int(day.max()))
    print(f"OK -> {a.out}  | {feat.shape[0]} profils  feat{feat.shape}  days {d0}..{d1}")


if __name__ == "__main__":
    main()
