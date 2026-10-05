from __future__ import annotations

"""Inference UpperDyn AUX POINTS Argo (train / val / all).

Pendant "points" de infer_output_daily_maps_stream.py : meme modele, memes stats
(snapshot du checkpoint), meme espace de normalisation, memes voisins assimiles —
mais les entrees sont lues DIRECTEMENT aux stations Argo (`data/mld_split/{train,val,all}`,
fichiers `mld_argo_natl_era5_*` + companions ocean/eddies) au lieu des cartes journalieres.
C'est exactement la chaine d'entree de l'entrainement (MonthlyNetCDFDataset), donc la
prediction aux points est directement comparable a la verite Argo du meme fichier.

Sortie : UN NetCDF de dimension `nstation` contenant, par profil :
  <target>_pred / <target>_true / <target>_std  (mld en metres, N2 dans son unite native)
  lat, lon, year, month, day, time, WMO, cycle, month_key, station_index
  n_neighbors, n1_dist_km, n1_dt_days (assimilation)

Voisins (profils assim) : KNN 4D (position/LS, jour/LT) sur un pool Argo au format
`grid_pool_argo_<annee>.npz` (feat=[MLD|statics] dans l'espace natif du precompute),
avec **leave-one-out** — le profil courant, s'il est dans le pool, est exclu de ses
propres voisins, exactement comme build_neighbor_pool.py a l'entrainement.

Exemples
--------
# val 2023, modele assim (voisins), GPU 0
CUDA_VISIBLE_DEVICES=0 "$PY" prod/code/scripts/infer_output_points.py \
    --config prod/config/profiles/assim.toml --checkpoint prod/train/canonical/assim/assim.pt \
    --input-dir data/mld_split/val --neighbor-pool "data/pool/grid_pool_argo_*.npz" \
    --output prod/infer_points/assim__val.nc

# points d'entrainement (2010-2022), meme modele
CUDA_VISIBLE_DEVICES=0 "$PY" prod/code/scripts/infer_output_points.py \
    --config prod/config/profiles/assim.toml --checkpoint prod/train/canonical/assim/assim.pt \
    --input-dir data/mld_split/train --neighbor-pool "data/pool/grid_pool_argo_*.npz" \
    --output prod/infer_points/assim__train.nc

# baseline (sans assimilation) : aucun pool requis
CUDA_VISIBLE_DEVICES=0 "$PY" prod/code/scripts/infer_output_points.py \
    --config prod/config/profiles/baseline.toml --checkpoint prod/train/canonical/baseline/baseline.pt \
    --input-dir data/mld_split/val --no-neighbors --output prod/infer_points/baseline__val.nc
"""

import argparse
import glob as _glob
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]          # prod/code
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_output_daily_maps_stream import build_model  # meme construction de modele que les cartes
from morlenn.config import load_config
from morlenn.data import (
    MonthlyNetCDFDataset,
    _companion_paths,
    _to_numpy,
    _variable_name,
    denormalize_targets,
    list_netcdf_files,
    load_stats,
)
from morlenn.model import split_prediction_params
from morlenn.notebook_analysis import _apply_checkpoint_snapshot
from morlenn.train import resolve_device, set_seed

EARTH_R_KM = 6371.0
META_VARS = ("lat", "lon", "year", "month", "day", "time", "WMO", "cycle", "station_mode")


# --------------------------------------------------------------------------------------
# metadonnees stations (ordre EXACT du dataset : fichier par fichier, stations valides)
# --------------------------------------------------------------------------------------
def read_station_metadata(dataset: MonthlyNetCDFDataset, targets: list[str]) -> dict[str, np.ndarray]:
    """lat/lon/date/WMO/... + verite Argo, dans l'ordre des echantillons du dataset.

    Lecture vectorisee par fichier (les indices de stations valides sont deja calcules
    par MonthlyNetCDFDataset), donc O(nb_fichiers) ouvertures et pas O(nb_profils).
    """
    wanted = list(META_VARS) + list(targets)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    month_keys: list[np.ndarray] = []
    station_index: list[np.ndarray] = []
    file_index: list[np.ndarray] = []

    for file_idx, primary_path in enumerate(dataset.files):
        idx = np.asarray(dataset._valid_station_indices[file_idx], dtype=np.int64)
        sources = {name: netCDF4.Dataset(path, "r") for name, path in _companion_paths(Path(primary_path)).items()}
        try:
            for name in wanted:
                values = None
                for source_name in ("primary", "ocean", "eddy"):
                    src = sources.get(source_name)
                    if src is None:
                        continue
                    var_name = _variable_name(src, name)
                    if var_name is None:
                        continue
                    array = _to_numpy(src.variables[var_name][:])
                    if array.ndim > 1:                 # variable temporelle -> derniere heure connue
                        array = array[:, -1]
                    values = array.reshape(-1)[idx]
                    break
                columns[name].append(
                    np.full(idx.size, np.nan, dtype=np.float32) if values is None else values.astype(np.float32)
                )
        finally:
            for src in sources.values():
                src.close()
        key = "".join(ch for ch in Path(primary_path).stem if ch.isdigit())[-6:]
        month_keys.append(np.full(idx.size, int(key) if key else -1, dtype=np.int32))
        station_index.append(idx.astype(np.int32))
        file_index.append(np.full(idx.size, file_idx, dtype=np.int32))

    meta = {name: np.concatenate(parts) for name, parts in columns.items()}
    meta["month_key"] = np.concatenate(month_keys)
    meta["station_index"] = np.concatenate(station_index)
    meta["file_index"] = np.concatenate(file_index)
    meta["day_ordinal"] = np.array(
        [_ordinal(y, m, d) for y, m, d in zip(meta["year"], meta["month"], meta["day"])], dtype=np.float64
    )
    if meta["lat"].size != len(dataset):
        raise ValueError(f"metadata {meta['lat'].size} != dataset {len(dataset)} (desalignement).")
    return meta


def _ordinal(year: float, month: float, day: float) -> float:
    try:
        return float(date(int(year), int(month), int(day)).toordinal())
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------------------
# voisins Argo (KNN 4D leave-one-out, memes echelles que build_neighbor_pool.py)
# --------------------------------------------------------------------------------------
def load_neighbor_pool(patterns: list[str]) -> dict[str, np.ndarray]:
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(_glob.glob(pattern))
        paths += matched if matched else [pattern]
    paths = list(dict.fromkeys(paths))
    feats, xyzs, days = [], [], []
    for path in paths:
        z = np.load(path)
        feats.append(z["feat"].astype(np.float32))
        xyzs.append(z["xyz"].astype(np.float32))
        days.append(z["day"].astype(np.float64))
    if not feats:
        raise SystemExit("Aucun pool voisin trouve (--neighbor-pool).")
    dims = {f.shape[1] for f in feats}
    if len(dims) != 1:
        raise SystemExit(f"Pools voisins de dimensions incompatibles: {dims}")
    return {
        "feat": np.concatenate(feats, 0),
        "xyz": np.concatenate(xyzs, 0),
        "day": np.concatenate(days, 0),
        "paths": paths,
    }


def build_neighbors(
    *, pool, lat, lon, day_ordinal, k, ls_km, lt_days, causal, query_k,
    gate_km=None, gate_days=10.0, self_km=0.05,
):
    """[N,k,F] feats + [N,k] masque + distance/ecart temporel du 1er voisin.

    Metrique identique a l'entrainement : KD-tree sur (R*xyz/LS, jour/LT), les k plus
    proches. Le profil lui-meme est retire UNE fois (leave-one-out, comme
    build_neighbor_pool.py) : est considere "soi" le premier candidat a moins de
    `self_km` km ET le meme jour — le test est physique (et pas une egalite a 1e-6)
    car les xyz du pool sont stockes en float32. Sans cette exclusion la MLD vraie du
    profil entrerait dans ses propres features voisines (fuite du label).
    `causal` ne garde que les voisins strictement anterieurs.
    """
    n = lat.size
    feat_dim = pool["feat"].shape[1]
    neighbors = np.zeros((n, k, feat_dim), dtype=np.float32)
    mask = np.zeros((n, k), dtype=np.float32)
    n1_km = np.full(n, np.nan, dtype=np.float32)
    n1_dt = np.full(n, np.nan, dtype=np.float32)
    n_self_dropped = 0

    pool_xyz = pool["xyz"].astype(np.float64)
    pool_day = pool["day"].astype(np.float64)
    pool_pt = np.concatenate([EARTH_R_KM * pool_xyz / ls_km, (pool_day / lt_days)[:, None]], axis=1)
    tree = KDTree(pool_pt)

    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    query_xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
    query_pt = np.concatenate(
        [EARTH_R_KM * query_xyz / ls_km, (day_ordinal / lt_days)[:, None]], axis=1
    )

    query_k = min(int(query_k), pool_pt.shape[0])
    ind = tree.query(query_pt, k=query_k, return_distance=False)   # deja tries par distance 4D
    for row in range(n):
        candidates = ind[row]
        chord = np.linalg.norm(query_xyz[row][None, :] - pool_xyz[candidates], axis=1)
        km = 2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0)) * EARTH_R_KM
        dday = day_ordinal[row] - pool_day[candidates]
        usable = np.ones(candidates.size, dtype=bool)
        is_self = (km <= float(self_km)) & (np.abs(dday) < 0.5)
        if is_self.any():
            usable[np.flatnonzero(is_self)[0]] = False       # leave-one-out : une seule fois
            n_self_dropped += 1
        if causal:
            usable &= dday > 0.0
        order = np.flatnonzero(usable)[:k]
        if order.size == 0:
            continue
        kept = candidates[order]
        km, dday = km[order], dday[order]
        keep_flag = np.ones(kept.size, dtype=bool)
        if gate_km is not None:
            keep_flag = (km <= float(gate_km)) & (np.abs(dday) <= float(gate_days))
        neighbors[row, : kept.size] = pool["feat"][kept]
        mask[row, : kept.size] = keep_flag.astype(np.float32)
        n1_km[row] = km[0]
        n1_dt[row] = dday[0]
        if row and row % 50000 == 0:
            print(json.dumps({"neighbors": f"{row}/{n}"}), flush=True)
    print(json.dumps({"self_excluded": n_self_dropped, "profiles": int(n)}), flush=True)
    return neighbors, mask, n1_km, n1_dt


# --------------------------------------------------------------------------------------
# metriques
# --------------------------------------------------------------------------------------
def _pair_stats(t: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "bias": float(np.mean(p - t)),
        "mae": float(np.mean(np.abs(p - t))),
        "corr": float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else float("nan"),
        "std_ratio": float(p.std() / t.std()) if t.std() > 0 else float("nan"),
    }


def metrics(true, pred, tail_quantile=0.95, transform="identity", target_iqr=None) -> dict[str, float]:
    """Metriques en UNITES PHYSIQUES (+ espace transforme pour la MLD).

    `*_transformed` = espace log1p, celui dans lequel eval_checkpoint.py/train.py
    calculent mld_rmse (divise par l'IQR cible -> `rmse_normalized`), mld_corr et
    mld_stdR : c'est ce qu'il faut comparer aux tableaux d'eval, pas la corr en metres.
    La queue utilise le meme quantile que l'entrainement (config.training.tail_quantile).
    """
    finite = np.isfinite(true) & np.isfinite(pred)
    t, p = np.asarray(true)[finite].astype(np.float64), np.asarray(pred)[finite].astype(np.float64)
    if t.size < 2:
        return {"n": int(t.size)}
    out = {"n": int(t.size)}
    out.update(_pair_stats(t, p))
    threshold = float(np.quantile(t, tail_quantile))
    tail = t >= threshold
    if tail.sum() >= 2:
        out["tail_quantile"] = float(tail_quantile)
        out["tail_threshold"] = threshold
        out["tail_count"] = int(tail.sum())
        out["tail_rmse"] = float(np.sqrt(np.mean((p[tail] - t[tail]) ** 2)))
        out["tail_bias"] = float(np.mean(p[tail] - t[tail]))
    if transform == "log1p":
        tt, pt = np.log1p(np.clip(t, 0.0, None)), np.log1p(np.clip(p, 0.0, None))
        transformed = _pair_stats(tt, pt)
        out["rmse_transformed"] = transformed["rmse"]
        out["corr_transformed"] = transformed["corr"]
        out["std_ratio_transformed"] = transformed["std_ratio"]
        if target_iqr:
            out["rmse_normalized"] = transformed["rmse"] / float(target_iqr)
    return out


# --------------------------------------------------------------------------------------
# sortie NetCDF
# --------------------------------------------------------------------------------------
def write_output(path: Path, meta, preds, stds, targets, attrs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.nc")
    if tmp.exists():
        tmp.unlink()
    n = preds.shape[0]
    with netCDF4.Dataset(tmp, "w") as ds:
        ds.createDimension("nstation", n)

        def var(name, values, dtype="f4", **kwargs):
            v = ds.createVariable(name, dtype, ("nstation",), zlib=True, complevel=4)
            v[:] = values
            for key, value in kwargs.items():
                v.setncattr(key, value)
            return v

        var("lat", meta["lat"], units="degrees_north")
        var("lon", meta["lon"], units="degrees_east")
        var("year", meta["year"])
        var("month", meta["month"])
        var("day", meta["day"])
        var("time", meta["time"], long_name="temps du profil Argo (unite du fichier source)")
        var("day_ordinal", meta["day_ordinal"], dtype="f8", long_name="jour proleptic Gregorian ordinal")
        var("WMO", meta["WMO"], dtype="f8")
        var("cycle", meta["cycle"])
        var("station_mode", meta["station_mode"])
        var("month_key", meta["month_key"], dtype="i4", long_name="AAAAMM du fichier source")
        var("station_index", meta["station_index"], dtype="i4", long_name="index nstation dans le fichier source")
        for index, name in enumerate(targets):
            var(f"{name}_pred", preds[:, index], long_name=f"{name} predit par le modele")
            var(f"{name}_true", meta[name], long_name=f"{name} observe (Argo)")
            if stds is not None:
                var(f"{name}_std", stds[:, index], long_name=f"ecart-type predictif de {name} (unites physiques)")
        for name in ("n_neighbors", "n1_dist_km", "n1_dt_days"):
            if name in meta:
                var(name, meta[name])
        for key, value in attrs.items():
            ds.setncattr(key, value)
    tmp.replace(path)


# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Inference UpperDyn MLD/N2 aux points Argo (train/val), pendant points de infer_output_daily_maps_stream.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="profil TOML (architecture + features)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-dir", required=True, help="data/mld_split/{val,train,all}")
    p.add_argument("--output", required=True, help="NetCDF de sortie (dimension nstation)")
    p.add_argument("--years", default=None, help="ne garder que ces annees, ex '2023' ou '2010,2011'")
    p.add_argument("--exclude-years", default=None, help="exclure ces annees (ex '2023' pour ne garder que le train)")
    p.add_argument("--max-samples", type=int, default=None, help="tronque (debug)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", help="bf16 autocast sur le forward (leger ecart numerique)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-clamp-bathy", action="store_true", help="ne pas borner la MLD par la profondeur (defaut: borne, comme les cartes/eval)")
    p.add_argument("--tail-quantile", type=float, default=None,
                   help="quantile de la queue MLD pour les metriques (defaut: config.training.tail_quantile)")
    # --- assimilation ---
    p.add_argument("--no-neighbors", action="store_true", help="force le mode sans assimilation")
    p.add_argument("--neighbor-pool", nargs="+", default=None,
                   help="pool(s) Argo format grid (feat/xyz/day), globs OK: 'data/pool/grid_pool_argo_*.npz'")
    p.add_argument("--neighbor-aligned", default=None,
                   help="alternative: matrices voisins DEJA alignees ligne a ligne (neighbor_all_aligned.npz) — "
                        "valable seulement si --input-dir est le dossier ayant servi au precompute et sans filtre d'annee")
    p.add_argument("--neighbor-k", type=int, default=None, help="defaut: config.model.num_neighbors")
    p.add_argument("--neighbor-ls", type=float, default=100.0, help="echelle spatiale du KNN (km)")
    p.add_argument("--neighbor-lt", type=float, default=10.0, help="echelle temporelle du KNN (jours)")
    p.add_argument("--neighbor-mode", choices=["symmetric", "causal"], default="symmetric",
                   help="symmetric = passe+futur, leave-one-out (semantique d'entrainement)")
    p.add_argument("--neighbor-query-k", type=int, default=None, help="candidats interroges avant filtrage (defaut 60)")
    p.add_argument("--self-km", type=float, default=0.05,
                   help="rayon (km) sous lequel un candidat du MEME jour est considere comme le profil lui-meme "
                        "et exclu de ses propres voisins (leave-one-out). 0 desactive l'exclusion.")
    p.add_argument("--gate-neighbors-km", type=float, default=None,
                   help="si defini: masque=1 seulement si voisin <= ce rayon ET <= --gate-neighbors-days")
    p.add_argument("--gate-neighbors-days", type=float, default=10.0)
    args = p.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} existe deja (utilise --overwrite).")

    # ---- config / stats / modele : strictement la meme sequence que l'inference cartes ----
    config_path, checkpoint_path = Path(args.config), Path(args.checkpoint)
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = load_stats(config.paths.stats_path) if config.paths.stats_path.exists() else None
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    if stats is None:
        raise SystemExit("Stats introuvables (ni config.paths.stats_path ni snapshot du checkpoint).")
    set_seed(config.training.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = resolve_device(args.device or config.training.device)
    use_neighbors = bool(config.model.use_mld_neighbors) and not args.no_neighbors
    model = build_model(config, checkpoint, not use_neighbors, device)
    targets = list(config.dataset.targets)
    print(json.dumps({"device": str(device), "use_neighbors": use_neighbors, "targets": targets}), flush=True)

    # ---- dataset stations (entrees identiques a l'entrainement) ----
    files = list_netcdf_files(args.input_dir)
    dataset = MonthlyNetCDFDataset(
        files=files,
        dynamic_features=config.dataset.dynamic_features,
        static_features=config.dataset.static_features,
        targets=targets,
        mld_target_transform=config.dataset.mld_target_transform,
        stats=stats,
        replace_nan_with_zero=config.dataset.replace_nan_with_zero,
        clip_mld_max=config.dataset.clip_mld_max,
        dynamic_history_hours=config.dataset.dynamic_history_hours,
        strict_static_features=config.dataset.strict_static_features,
        eddy_max_distance_radius=config.dataset.eddy_max_distance_radius,
    )
    print(json.dumps({"input_dir": str(args.input_dir), "files": len(dataset.files), "samples": len(dataset)}), flush=True)

    t0 = time.time()
    meta = read_station_metadata(dataset, targets)
    print(json.dumps({"metadata_s": round(time.time() - t0, 1)}), flush=True)

    # ---- selection (annees / troncature) ----
    keep = np.ones(len(dataset), dtype=bool)
    years = meta["year"].astype(int)
    if args.years:
        wanted = {int(y) for y in args.years.split(",") if y.strip()}
        keep &= np.isin(years, sorted(wanted))
    if args.exclude_years:
        dropped = {int(y) for y in args.exclude_years.split(",") if y.strip()}
        keep &= ~np.isin(years, sorted(dropped))
    keep_idx = np.flatnonzero(keep)
    if args.max_samples is not None:
        keep_idx = keep_idx[: args.max_samples]
    if keep_idx.size == 0:
        raise SystemExit("Aucun profil selectionne (verifie --years/--exclude-years).")
    filtered = keep_idx.size != len(dataset)
    meta = {name: values[keep_idx] for name, values in meta.items()}
    subset = Subset(dataset, keep_idx.tolist())
    print(json.dumps({"selected": int(keep_idx.size),
                      "years": sorted({int(y) for y in meta["year"]})}), flush=True)

    # ---- voisins ----
    neighbors = neighbor_mask = None
    neighbor_source = "none"
    if use_neighbors:
        k = int(args.neighbor_k or config.model.num_neighbors)
        if args.neighbor_aligned:
            if filtered:
                raise SystemExit("--neighbor-aligned exige l'ensemble complet du dossier (pas de --years/--max-samples).")
            z = np.load(args.neighbor_aligned)
            neighbors = z["feat"].astype(np.float32)
            neighbor_mask = z["mask"].astype(np.float32)
            if neighbors.shape[0] != keep_idx.size:
                raise SystemExit(
                    f"{args.neighbor_aligned}: {neighbors.shape[0]} lignes != {keep_idx.size} profils "
                    "(les matrices alignees ne valent que pour le dossier du precompute)."
                )
            neighbor_source = str(args.neighbor_aligned)
            meta["n_neighbors"] = neighbor_mask.sum(axis=1).astype(np.float32)
        else:
            if not args.neighbor_pool:
                raise SystemExit("Modele assimile : passe --neighbor-pool (ou --neighbor-aligned, ou --no-neighbors).")
            pool = load_neighbor_pool(args.neighbor_pool)
            if pool["feat"].shape[1] != int(config.model.neighbor_feat_dim):
                raise SystemExit(
                    f"pool feat_dim={pool['feat'].shape[1]} != modele neighbor_feat_dim={config.model.neighbor_feat_dim}."
                )
            print(json.dumps({"neighbor_pool": pool["paths"], "pool_size": int(pool["feat"].shape[0]),
                              "feat_dim": int(pool["feat"].shape[1])}), flush=True)
            t0 = time.time()
            neighbors, neighbor_mask, n1_km, n1_dt = build_neighbors(
                pool=pool, lat=meta["lat"].astype(np.float64), lon=meta["lon"].astype(np.float64),
                day_ordinal=meta["day_ordinal"], k=k, ls_km=args.neighbor_ls, lt_days=args.neighbor_lt,
                causal=(args.neighbor_mode == "causal"), query_k=args.neighbor_query_k or 60,
                gate_km=args.gate_neighbors_km, gate_days=args.gate_neighbors_days,
                self_km=args.self_km,
            )
            neighbor_source = ",".join(pool["paths"])
            meta["n1_dist_km"] = n1_km
            meta["n1_dt_days"] = n1_dt
            meta["n_neighbors"] = neighbor_mask.sum(axis=1).astype(np.float32)
            print(json.dumps({"neighbors_s": round(time.time() - t0, 1),
                              "full_k_frac": round(float((neighbor_mask.sum(1) == k).mean()), 4),
                              "median_n1_km": round(float(np.nanmedian(n1_km)), 1)}), flush=True)

    # ---- forward ----
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    n_targets = len(targets)
    n_total = keep_idx.size
    pred_norm = np.full((n_total, n_targets), np.nan, dtype=np.float32)
    std_raw = np.full((n_total, n_targets), np.nan, dtype=np.float32)   # deja en unites physiques
    elevation_phys = np.full(n_total, np.nan, dtype=np.float32)
    static_features = list(config.dataset.static_features)
    elevation_index = static_features.index("elevation") if "elevation" in static_features else None
    has_std = False

    offset = 0
    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        dynamic = batch["dynamic"].to(device, non_blocking=True).float()
        static = batch["static"].to(device, non_blocking=True).float()
        size = dynamic.shape[0]
        stop = offset + size
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            if use_neighbors:
                nb = torch.from_numpy(neighbors[offset:stop]).to(device)
                nm = torch.from_numpy(neighbor_mask[offset:stop]).to(device)
                output = model(dynamic, static, neighbors=nb, neighbor_mask=nm)
            else:
                output = model(dynamic, static)
            mean_t, std_t = split_prediction_params(output, n_targets)
        pred_norm[offset:stop] = mean_t.float().cpu().numpy()
        if std_t is not None:
            has_std = True
            std_raw[offset:stop] = std_t.float().cpu().numpy()
        if elevation_index is not None:
            scale = stats.static_iqr[elevation_index]
            elevation_phys[offset:stop] = (
                batch["static"][:, elevation_index].numpy() * (1.0 if scale == 0.0 else scale)
                + stats.static_median[elevation_index]
            )
        offset = stop
        if batch_idx % 20 == 0:
            done = max(offset, 1)
            print(json.dumps({"batch": batch_idx, "done": offset, "total": n_total,
                              "eta_s": round((time.time() - t0) / done * (n_total - offset), 1)}), flush=True)
    if offset != n_total:
        raise RuntimeError(f"forward incomplet: {offset}/{n_total}")
    print(json.dumps({"forward_s": round(time.time() - t0, 1)}), flush=True)

    # ---- denormalisation (+ clamp bathymetrie, comme les cartes et eval_checkpoint) ----
    preds = denormalize_targets(pred_norm, stats=stats, target_names=targets,
                                mld_target_transform=config.dataset.mld_target_transform)
    # La tete std du modele predit deja dans les UNITES PHYSIQUES (la NLL gaussienne de
    # losses.py la compare a prediction_phys/target_phys) -> aucune denormalisation ici,
    # exactement comme mld_std/N2_std des cartes.
    stds = std_raw if has_std else None
    clamped = 0
    if not args.no_clamp_bathy and elevation_index is not None and "mld" in targets:
        mld_index = targets.index("mld")
        ocean = np.isfinite(elevation_phys) & (elevation_phys < 0.0)
        depth = np.where(ocean, -elevation_phys, np.nan)
        clamped = int(np.sum(ocean & (preds[:, mld_index] > depth)))
        preds[ocean, mld_index] = np.minimum(preds[ocean, mld_index], depth[ocean])
    print(json.dumps({"clamped_to_bathy": clamped}), flush=True)

    # ---- metriques vs Argo ----
    tail_quantile = args.tail_quantile if args.tail_quantile is not None else config.training.tail_quantile
    summary = {
        name: metrics(
            meta[name], preds[:, index], tail_quantile=tail_quantile,
            transform=(config.dataset.mld_target_transform if name == "mld" else "identity"),
            target_iqr=(stats.target_iqr[index] if name == "mld" else None),
        )
        for index, name in enumerate(targets)
    }
    print(json.dumps({"metrics": summary}, indent=2, default=float), flush=True)

    attrs = {
        "title": "UpperDyn — inference MLD/N2 aux points Argo",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "input_dir": str(args.input_dir),
        "num_profiles": int(n_total),
        "use_neighbors": int(use_neighbors),
        "neighbor_source": neighbor_source,
        "neighbor_mode": args.neighbor_mode if use_neighbors else "none",
        "neighbor_k": int(args.neighbor_k or config.model.num_neighbors) if use_neighbors else 0,
        "neighbor_length_scale_km": float(args.neighbor_ls),
        "neighbor_time_scale_days": float(args.neighbor_lt),
        "neighbor_gate_km": float(args.gate_neighbors_km) if args.gate_neighbors_km is not None else "none",
        "neighbor_self_exclusion_km": float(args.self_km),
        "clamp_mld_to_bathymetry": int(not args.no_clamp_bathy),
        "mld_target_transform": config.dataset.mld_target_transform,
        "dynamic_features": ",".join(config.dataset.dynamic_features),
        "static_features": ",".join(static_features),
        "targets": ",".join(targets),
        "metrics_json": json.dumps(summary, default=float),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_script": "prod/code/scripts/infer_output_points.py",
    }
    write_output(output_path, meta, preds, stds, targets, attrs)
    print(json.dumps({"output": str(output_path), "status": "done"}), flush=True)


if __name__ == "__main__":
    main()
