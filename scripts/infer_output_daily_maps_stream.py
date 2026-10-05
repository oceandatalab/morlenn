from __future__ import annotations

"""Streaming UpperDyn map inference.

Same physics/outputs as infer_output_daily_maps_assim.py, but the 720h dynamic
history is kept as a GPU-resident rolling ring buffer over ocean columns. Output
hours are produced in strict chronological order across the whole run: the buffer
is seeded once (720h), then each new output hour pushes ONLY the 1 new hour
(read O(1) from the slim cache) instead of re-assembling the full 720h window per
hour / per lat-tile. Forward runs on the full ocean grid in GPU batches, indexing
the buffer directly (no per-batch H2D of the history).

Validated against the per-day script: regenerate a day and compare with np.allclose.
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np
import torch
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import infer_output_daily_maps_assim as A  # reuse the battle-tested helpers
from morlenn.config import load_config
from morlenn.data import load_stats
from morlenn.model import split_prediction_params, UpperDynTCNAttentionModel
from morlenn.notebook_analysis import _apply_checkpoint_snapshot, _load_checkpoint_state
from morlenn.train import resolve_device, set_seed


def build_model(config, checkpoint, no_neighbors: bool, device):
    """Identical explicit build to the per-day script's main()."""
    M = config.model
    g = lambda n, d: getattr(M, n, d)
    model = UpperDynTCNAttentionModel(
        num_dynamic=len(config.dataset.dynamic_features), num_static=len(config.dataset.static_features),
        num_targets=len(config.dataset.targets),
        tcn_channels=M.tcn_channels, dilations=M.dilations, kernel_size=M.kernel_size, dropout=M.dropout,
        static_hidden_dim=M.static_hidden_dim, static_num_heads=M.static_num_heads,
        use_dynamic_context=M.use_dynamic_context, dynamic_context_mode=M.dynamic_context_mode,
        dynamic_context_hidden_dim=M.dynamic_context_hidden_dim, dynamic_summary_kernel_size=M.dynamic_summary_kernel_size,
        dynamic_summary_windows=M.dynamic_summary_windows, film_hidden_dim=M.film_hidden_dim,
        attention_hidden_dim=M.attention_hidden_dim, attention_pooling_mode=M.attention_pooling_mode,
        attention_tokens_per_segment=M.attention_tokens_per_segment, num_attention_experts=M.num_attention_experts,
        expert_kernel_sizes=M.expert_kernel_sizes, fusion_hidden_dim=M.fusion_hidden_dim, backbone_hidden_dim=M.backbone_hidden_dim,
        predictive_distribution=M.predictive_distribution, min_std=M.min_std,
        mld_branch_hidden_dim=M.mld_branch_hidden_dim, mld_branch_depth=M.mld_branch_depth,
        detach_n2_head_input=M.detach_n2_head_input,
        n2_separate_temporal_branch=M.n2_separate_temporal_branch, n2_separate_attention_branch=M.n2_separate_attention_branch,
        n2_attention_hidden_dim=M.n2_attention_hidden_dim, n2_num_attention_experts=M.n2_num_attention_experts,
        n2_expert_kernel_sizes=M.n2_expert_kernel_sizes, n2_fusion_hidden_dim=M.n2_fusion_hidden_dim,
        n2_backbone_hidden_dim=M.n2_backbone_hidden_dim, n2_backbone_grad_scale=M.n2_backbone_grad_scale,
        n2_branch_hidden_dim=M.n2_branch_hidden_dim, n2_branch_depth=M.n2_branch_depth,
        use_mld_neighbors=not no_neighbors,
        neighbor_feat_dim=g("neighbor_feat_dim", 20), neighbor_hidden_dim=g("neighbor_hidden_dim", 96),
        neighbor_modality_dropout=g("neighbor_modality_dropout", 0.5),
        neighbor_increment=bool(g("neighbor_increment", False)), neighbor_increment_full=bool(g("neighbor_increment_full", False)),
    )
    model.load_state_dict(_load_checkpoint_state(checkpoint), strict=False)
    model.to(device).eval()
    return model


def eddy_grid_from_dataset(dataset, hour, latitudes, longitudes, threshold):
    """Gridded eddy static features read DIRECTLY from the merged file at `hour`,
    with the exact training admissibility gate (data.py _eddy_match_is_admissible):
    a point keeps the eddy field value iff eddy_type==0 (value is 0 anyway) OR it is
    within `threshold` effective radii of the eddy center; otherwise 0. This replaces
    the 2010-atlas reconstruction so real 2023 eddies are used."""
    et = A.read_merged(dataset, "eddy_type", hour)                     # (H, W)
    clat = A.read_merged(dataset, "eddy_center_lat", hour)
    clon = A.read_merged(dataset, "eddy_center_lon", hour)
    rad = A.read_merged(dataset, "eddy_effective_radius", hour)
    if threshold is None:
        admissible = np.ones(et.shape, dtype=bool)
    else:
        lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
        lat1 = np.deg2rad(lat_grid); lon1 = np.deg2rad(lon_grid)
        lat2 = np.deg2rad(clat); lon2 = np.deg2rad(clon)
        dlat = lat2 - lat1; dlon = lon2 - lon1
        val = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        dist = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(val, 0.0, 1.0)))   # haversine km, R=6371
        radius_km = np.where(rad > 1000.0, rad / 1000.0, rad)              # meters -> km like training
        finite = np.isfinite(clat) & np.isfinite(clon) & np.isfinite(rad)
        with np.errstate(invalid="ignore", divide="ignore"):
            dor = dist / np.where(radius_km > 0.0, radius_km, np.inf)
        admissible = (et == 0.0) | (finite & (rad > 0.0) & (radius_km > 0.0) & (dor <= float(threshold)))
    out = {}
    for name in A.EDDY_STATIC_NAMES:
        v = np.nan_to_num(A.read_merged(dataset, name, hour), nan=0.0)  # eddy fields are NaN where no eddy -> 0
        out[name] = np.where(admissible, v, 0.0).astype(np.float32)
    return out


def read_hour(buf, dyn_files, features, dt, ocean_flat):
    """One output hour for all dynamic features + interp_count over ocean columns.

    Returns (dyn (n_dyn, ncol) float32, ic (ncol,) float32). Slices the per-day
    decompressed cache arrays at dt.hour -> O(1) vs re-reading the 720h window.
    """
    path = dyn_files[dt.date()]
    ncol = ocean_flat.size
    dyn = np.empty((len(features), ncol), dtype=np.float32)
    for fi, f in enumerate(features):
        arr = buf.get(path, f)  # (24, H, W), decompressed once per (day, feature)
        dyn[fi] = arr[dt.hour].reshape(-1)[ocean_flat]
    icarr = buf.get(path, "interp_count")
    ic = icarr[dt.hour].reshape(-1)[ocean_flat].astype(np.float32)
    return dyn, ic


def infer_stream(
    *, model, stats, config, files_by_date, dyn_files, buf, cache,
    bathy_path, output_dir, checkpoint_path, config_path, target_days, hours,
    history_hours, batch_size, device, sss_source, pool, assim_km, assim_days,
    eddy, eddy_threshold, use_neighbors, climato_lookup, neighbor_window_days,
    overwrite, eddy_from_merged=False, eddy_radius_gate=2.0, amp=False, symmetric=False,
    row_start=None, row_end=None, name_tag="", gate_km=None, gate_days=10.0,
):
    feats = config.dataset.dynamic_features
    eddy_static_idx = [i for i, nm in enumerate(config.dataset.static_features) if nm in A.EDDY_STATIC_NAMES]
    nd = len(feats)
    ns = len(config.dataset.static_features)
    ntarg = len(config.dataset.targets)
    T = history_hours

    # --- grid geometry + ocean columns (buffer covers exactly the predictable candidates) ---
    target_path0 = files_by_date[target_days[0]]
    with netCDF4.Dataset(target_path0, "r") as src, netCDF4.Dataset(bathy_path, "r") as bathy:
        latitudes = A.to_numpy(src.variables["latitude"][:])
        longitudes = A.to_numpy(src.variables["longitude"][:])
        elevation = A.to_numpy(bathy.variables["elevation"][:])
    nlat, nlon = len(latitudes), len(longitudes)
    base = np.isfinite(elevation) & (elevation <= 0.0)        # ocean candidate mask
    if row_start is not None or row_end is not None:
        _rs = row_start if row_start is not None else 0
        _re = row_end if row_end is not None else nlat
        _band = np.zeros_like(base); _band[_rs:_re, :] = True
        base = base & _band
        print(json.dumps({"band_rows": [int(_rs), int(_re)]}), flush=True)
    ocean_flat = np.flatnonzero(base.reshape(-1)).astype(np.int64)
    ncol = ocean_flat.size
    i_lat = (ocean_flat // nlon)
    i_lon = (ocean_flat % nlon)
    ocean_lat = latitudes[i_lat].astype(np.float64)
    ocean_lon = longitudes[i_lon].astype(np.float64)
    la = np.deg2rad(ocean_lat); lo = np.deg2rad(ocean_lon)
    ocean_xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1).astype(np.float32)
    elev_col = elevation.reshape(-1)[ocean_flat]
    bathy_depth_col = np.where(elev_col < 0.0, -elev_col, np.nan).astype(np.float32)  # nan where elevation==0 (no clamp), like the per-day script
    print(json.dumps({"grid": [nlat, nlon], "ocean_cols": int(ncol), "history_hours": T}), flush=True)

    # --- GPU ring buffer ---
    H = torch.empty((nd, T, ncol), device=device, dtype=torch.float32)
    IC = torch.empty((T, ncol), device=device, dtype=torch.float32)
    head = 0
    buf_last_dt = None  # newest hour currently in the buffer

    # stats as GPU tensors (same guard as normalize_dynamic/static)
    dyn_med = torch.tensor(stats.dynamic_median, device=device, dtype=torch.float32).view(nd, 1, 1)
    dyn_iqr = torch.tensor(np.where(stats.dynamic_iqr == 0.0, 1.0, stats.dynamic_iqr), device=device, dtype=torch.float32).view(nd, 1, 1)

    def push(dt):
        nonlocal head
        dyn, ic = read_hour(buf, dyn_files, feats, dt, ocean_flat)
        H[:, head, :] = torch.from_numpy(dyn).to(device)
        IC[head, :] = torch.from_numpy(ic).to(device)
        head = (head + 1) % T

    def seed(target_dt):
        nonlocal head, buf_last_dt
        head = 0
        start_dt = target_dt - timedelta(hours=T - 1)
        for k in range(T):
            push(start_dt + timedelta(hours=k))
        buf_last_dt = target_dt

    R = 6371.0
    for day in target_days:
        output_path = output_dir / (f"morlenn_mld_{day:%Y%m%d}__{name_tag}.nc" if name_tag else f"morlenn_mld_{day:%Y%m%d}.nc")
        if output_path.exists() and not overwrite:
            print(json.dumps({"date": day.isoformat(), "status": "skip_exists"}), flush=True)
            # keep the buffer coherent for the next day even when skipping
            buf_last_dt = None
            continue
        print(json.dumps({"date": day.isoformat(), "status": "start"}), flush=True)
        target_path = files_by_date[day]
        temp_path = output_path.with_name(f"{output_path.stem}.tmp.nc")
        if temp_path.exists():
            temp_path.unlink()
        output = A.create_output(temp_path, target_path, bathy_path, checkpoint_path, config_path, hours)

        # neighbor sub-tree for this day (same selection as the per-day script)
        day_grid = float(day.toordinal())
        if symmetric:
            sel = np.ones(pool["day"].shape, dtype=bool)
        elif neighbor_window_days is not None:
            sel = np.abs(pool["day"] - day_grid) <= float(neighbor_window_days)
        else:
            sel = pool["day"] < day_grid
        n_sel = int(sel.sum())
        if n_sel >= 1:
            sub_xyz = pool["xyz"][sel]; sub_day = pool["day"][sel]; sub_feat = pool["feat"][sel]
            sub_4d = np.concatenate([R * sub_xyz / 100.0, (sub_day / 10.0)[:, None]], axis=1).astype(np.float32)
            nb_tree = KDTree(sub_4d); nb_k = min(5, n_sel)
        else:
            sub_xyz = sub_day = sub_feat = None; nb_tree = None; nb_k = 0
        feat_dim = pool["feat"].shape[1]
        eddy_day = A.build_eddy_day(eddy, day) if eddy is not None else None

        completed = False
        day_t0 = time.time()
        try:
            for out_hour_index, hour in enumerate(hours):
                hour_t0 = time.time()
                target_dt = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
                _ts = time.time()
                if buf_last_dt is None or target_dt != buf_last_dt + timedelta(hours=1):
                    seed(target_dt)              # first hour ever, or a gap -> full 720h read
                else:
                    push(target_dt); buf_last_dt = target_dt
                _t_seed = time.time() - _ts; _ts = time.time()

                # --- statics for this output time over ocean columns (full grid build, then index) ---
                static = A.build_static_features(
                    cache=cache, target_path=target_path, target_day=day, hour=hour,
                    static_features=config.dataset.static_features, latitudes=latitudes, longitudes=longitudes,
                    elevation_tile=elevation, lat_slice=slice(0, nlat), sss_source=sss_source,
                    eddy_day=(None if eddy_from_merged else eddy_day), eddy_threshold=eddy_threshold,
                    climato_lookup=climato_lookup,
                )  # (ns, nlat, nlon)
                if eddy_from_merged and eddy_static_idx:
                    # overwrite atlas-derived eddy rows with the real eddy fields from the merged file
                    eg = eddy_grid_from_dataset(cache.get(target_path), hour, latitudes, longitudes, eddy_radius_gate)
                    for i in eddy_static_idx:
                        static[i] = eg[config.dataset.static_features[i]]
                static_cols = static.reshape(ns, -1)[:, ocean_flat]            # (ns, ncol)
                sfin = np.all(np.isfinite(static_cols), axis=0)                 # (ncol,)
                static_norm = A.normalize_static(static_cols, stats).astype(np.float32)
                static_g = torch.from_numpy(static_norm).to(device)            # (ns, ncol)

                # --- validity over the window (order-independent reductions on the buffer) ---
                _t_static = time.time() - _ts; _ts = time.time()
                order = (head + torch.arange(T, device=device)) % T
                dyn_finite = torch.isfinite(H).all(dim=0).all(dim=0).cpu().numpy()  # (ncol,)
                nan_count = IC.sum(dim=0).cpu().numpy().astype(np.float64)          # (ncol,)
                valid_pred = dyn_finite & sfin
                valid_input = valid_pred & (nan_count == 0)
                interp_frac = (nan_count / (nd * T)).astype(np.float32)
                pred_cols = np.flatnonzero(valid_pred).astype(np.int64)
                _t_valid = time.time() - _ts; _ts = time.time()

                # outputs over ocean columns (then scattered to grid)
                mld_c = np.full(ncol, np.nan, np.float32); n2_c = np.full(ncol, np.nan, np.float32)
                mlds_c = np.full(ncol, np.nan, np.float32); n2s_c = np.full(ncol, np.nan, np.float32)
                n1km_c = np.full(ncol, np.nan, np.float32); n1dt_c = np.full(ncol, np.nan, np.float32)
                nnb_c = np.zeros(ncol, np.float32)

                for s in range(0, pred_cols.size, batch_size):
                    bc = pred_cols[s:s + batch_size]
                    bc_t = torch.from_numpy(bc).to(device)
                    dyn = H.index_select(2, bc_t).index_select(1, order)        # (nd, T, B) ascending time
                    dyn = (dyn - dyn_med) / dyn_iqr
                    dyn_t = dyn.permute(2, 0, 1).contiguous()                   # (B, nd, T)
                    static_t = static_g.index_select(1, bc_t).t().contiguous()  # (B, ns)
                    # neighbors for these columns
                    nb_feat = np.zeros((bc.size, 5, feat_dim), dtype=np.float32)
                    nb_mask = np.zeros((bc.size, 5), dtype=np.float32)
                    if nb_tree is not None and nb_k > 0:
                        gxyz = ocean_xyz[bc]
                        q4d = np.concatenate([R * gxyz / 100.0, np.full((bc.size, 1), day_grid / 10.0, np.float32)], axis=1)
                        knn_idx = nb_tree.query(q4d, k=nb_k, return_distance=False)
                        nb_feat[:, :nb_k, :] = sub_feat[knn_idx]
                        nb_mask[:, :nb_k] = 1.0
                        # distance (km) et ecart temporel (jours) par voisin
                        nbr_xyz = sub_xyz[knn_idx]                                     # (B, nb_k, 3)
                        chord_all = np.linalg.norm(gxyz[:, None, :] - nbr_xyz, axis=2)
                        km_all = (2.0 * np.arcsin(np.minimum(chord_all / 2.0, 1.0)) * R).astype(np.float32)
                        dday_all = (day_grid - sub_day[knn_idx]).astype(np.float32)    # (B, nb_k)
                        if gate_km is not None:
                            # gate physique : n'active un voisin que s'il est reellement proche
                            # (fenetre d'assimilation). A l'entrainement le masque=1 correspondait
                            # toujours a un Argo proche (echantillons AUX positions Argo) ; sur la
                            # grille pleine, marquer 1 des voisins a >1000 km est hors-distribution
                            # et cree des patches. Ce gate replique la semantique d'entrainement.
                            keep = (km_all <= float(gate_km)) & (np.abs(dday_all) <= float(gate_days))
                            nb_mask[:, :nb_k] = keep.astype(np.float32)
                            nnb_c[bc] = keep.sum(axis=1).astype(np.float32)
                        else:
                            nnb_c[bc] = float(nb_k)
                        n1km_c[bc] = km_all[:, 0]
                        n1dt_c[bc] = dday_all[:, 0]
                    nb_feat_t = torch.from_numpy(nb_feat).to(device)
                    nb_mask_t = torch.from_numpy(nb_mask).to(device)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                        if use_neighbors:
                            out = model(dyn_t, static_t, neighbors=nb_feat_t, neighbor_mask=nb_mask_t)
                        else:
                            out = model(dyn_t, static_t)
                        mean_t, std_t = split_prediction_params(out, ntarg)
                    mean_t = mean_t.float()
                    std_t = std_t.float() if std_t is not None else None
                    mean_phys = A.denormalize_prediction(
                        mean_t.detach().cpu().numpy(), stats=stats,
                        target_names=config.dataset.targets, transform=config.dataset.mld_target_transform,
                    )
                    mld_c[bc] = mean_phys[:, 0]; n2_c[bc] = mean_phys[:, 1]
                    if std_t is not None:
                        sp = std_t.detach().cpu().numpy()
                        mlds_c[bc] = sp[:, 0]; n2s_c[bc] = sp[:, 1]

                # mld clamp to bathy depth where elevation<0 (matches per-day script)
                clamp = np.isfinite(bathy_depth_col)
                mld_c = np.where(clamp, np.minimum(mld_c, bathy_depth_col), mld_c)

                # scatter columns -> grid and write this hour
                def to_grid(col, fill):
                    g = np.full(nlat * nlon, fill, dtype=col.dtype)
                    g[ocean_flat] = col
                    return g.reshape(nlat, nlon)
                output.variables["mld"][out_hour_index] = to_grid(mld_c, np.nan)
                output.variables["N2"][out_hour_index] = to_grid(n2_c, np.nan)
                output.variables["mld_std"][out_hour_index] = to_grid(mlds_c, np.nan)
                output.variables["N2_std"][out_hour_index] = to_grid(n2s_c, np.nan)
                vi = np.zeros(ncol, np.int8); vi[valid_input] = 1
                output.variables["valid_input"][out_hour_index] = to_grid(vi, np.int8(0))
                output.variables["n1_dist_km"][out_hour_index] = to_grid(n1km_c, np.nan)
                output.variables["n1_dt_days"][out_hour_index] = to_grid(n1dt_c, np.nan)
                output.variables["n_neighbors"][out_hour_index] = to_grid(nnb_c.astype(np.int8), np.int8(0))
                assim = ((n1km_c <= assim_km) & (np.abs(n1dt_c) <= assim_days) & (nnb_c > 0)).astype(np.int8)
                output.variables["assim_mask"][out_hour_index] = to_grid(assim, np.int8(0))
                interp_out = np.where(valid_pred, interp_frac, np.nan).astype(np.float32)
                output.variables["interp_fraction"][out_hour_index] = to_grid(interp_out, np.nan)
                _t_fwd = time.time() - _ts
                print(json.dumps({"date": day.isoformat(), "hour": int(hour), "n_pred": int(pred_cols.size),
                                  "seed_s": round(_t_seed, 1), "static_s": round(_t_static, 1),
                                  "valid_s": round(_t_valid, 1), "fwd_s": round(_t_fwd, 1),
                                  "tot_s": round(time.time() - hour_t0, 1)}), flush=True)
            output.sync()
            completed = True
        finally:
            output.close()
        if completed:
            temp_path.replace(output_path)
        print(json.dumps({"date": day.isoformat(), "status": "done"}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Streaming UpperDyn MLD/N2 map inference (GPU rolling history buffer).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--bathy", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name-tag", default="", help="suffixe tracable insere dans le nom des fichiers de sortie")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--hours", default="all")
    p.add_argument("--history-hours", type=int, default=720)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-land", action="store_true")
    p.add_argument("--sss-source", choices=["sss", "sos"], default="sss")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--row-start", type=int, default=None, help="ocean mask restreint aux rangees de latitude [row-start,row-end) — decoupage spatial haute resolution")
    p.add_argument("--row-end", type=int, default=None)
    p.add_argument("--neighbor-pool", required=True)
    p.add_argument("--assim-km", type=float, default=50.0)
    p.add_argument("--assim-days", type=float, default=10.0)
    p.add_argument("--gate-neighbors-km", type=float, default=None, help="si defini: n'active un voisin dans le masque MODELE que si <= ce rayon (km) ET <= --gate-neighbors-days. Corrige les patches OOD loin de tout Argo (le masque etait mis a 1 sans coupure de distance).")
    p.add_argument("--gate-neighbors-days", type=float, default=10.0)
    p.add_argument("--neighbor-window-days", type=float, default=None)
    p.add_argument("--neighbor-mode", choices=["causal", "symmetric"], default="causal",
        help="symmetric: passe+futur, colocalise inclus (modele v3 entraine sym)")
    p.add_argument("--eddy-data", required=True)
    p.add_argument("--eddy-threshold", type=float, default=2.0)
    p.add_argument("--eddy-from-merged", action="store_true", help="Read real eddy_* static fields directly from the merged input file (with the training 2r admissibility gate) instead of the npz atlas. Use for years the atlas does not cover (e.g. 2023).")
    p.add_argument("--no-neighbors", action="store_true")
    p.add_argument("--climato-nc", default=None)
    p.add_argument("--cache-dir", required=True, help="Slim filled input cache (required for streaming).")
    p.add_argument("--mem-days", type=int, default=34)
    p.add_argument("--map-prefix", default="merged_", help="prefixe des cartes journalieres (glorys_merged_ pour GLORYS)")
    p.add_argument("--amp", action="store_true", help="bf16 autocast on the forward (~2x faster; tiny numeric shift vs fp32). Use for production, not bit-exact validation.")
    args = p.parse_args()

    config_path = Path(args.config); checkpoint_path = Path(args.checkpoint)
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = load_stats(config.paths.stats_path) if config.paths.stats_path.exists() else None
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    set_seed(config.training.seed)
    # set_seed forces cudnn.deterministic=True (slow deterministic conv kernels, ~100x slower forward).
    # Inference needs no determinism -> restore the fast autotuned kernels.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = resolve_device(args.device or config.training.device)
    model = build_model(config, checkpoint, args.no_neighbors, device)

    if args.no_neighbors:
        # baseline / sans assimilation : pas besoin du pool -> pool VIDE (0 ligne) -> nb_tree=None,
        # aucun voisin, et aucun crash si le fichier grid_pool est absent (build_argo_pool non lancé).
        _fdim = int(getattr(config.model, "neighbor_feat_dim", 12))
        pool = {"feat": np.zeros((0, _fdim), np.float32), "xyz": np.zeros((0, 3), np.float32), "day": np.zeros((0,), np.float64)}
        print(json.dumps({"neighbor_pool": "(ignore, --no-neighbors)"}), flush=True)
    else:
        pool_npz = np.load(args.neighbor_pool)
        pool = {"feat": pool_npz["feat"].astype(np.float32), "xyz": pool_npz["xyz"].astype(np.float32), "day": pool_npz["day"].astype(np.float64)}
        print(json.dumps({"neighbor_pool": str(args.neighbor_pool), "pool_size": int(pool["feat"].shape[0]), "feat_dim": int(pool["feat"].shape[1])}), flush=True)
    ez = np.load(args.eddy_data)
    eddy = {"day": ez["day"].astype(np.float64), "lat": ez["lat"].astype(np.float32), "lon": ez["lon"].astype(np.float32),
            "amp_m": ez["amp_m"].astype(np.float32), "eff_km": ez["eff_km"].astype(np.float32),
            "spdavg": ez["spdavg"].astype(np.float32), "sprad_km": ez["sprad_km"].astype(np.float32),
            "etype": ez["etype"].astype(np.float32), "ltype": ez["ltype"].astype(np.float32)}
    print(json.dumps({"eddy_data": str(args.eddy_data), "eddy_obs": int(eddy["day"].size)}), flush=True)

    climato_lookup = None
    if "mld_climato" in config.dataset.static_features:
        from climato_lookup import ClimatoLookup
        climato_lookup = ClimatoLookup(args.climato_nc)

    input_dir = Path(args.input_dir); bathy_path = Path(args.bathy); output_dir = Path(args.output_dir)
    start = A.parse_date(args.start_date) if args.start_date else None
    end = A.parse_date(args.end_date) if args.end_date else None
    hours = A.parse_hours(args.hours)
    files_by_date = A.discover_daily_files(input_dir, prefix=args.map_prefix)
    required_vars = A.required_variables(
        config.dataset.dynamic_features,
        [A.static_source_name(f, sss_source=args.sss_source) for f in config.dataset.static_features],
    )
    target_days = A.valid_target_dates(files_by_date=files_by_date, required_vars=required_vars,
                                       history_hours=args.history_hours, start=start, end=end)
    dyn_files = A.discover_cache_files(Path(args.cache_dir))
    n_dyn = len(config.dataset.dynamic_features)
    buf = A.DayArrayCache(max_entries=args.mem_days * (n_dyn + 1))

    def _history_in_cache(day):
        first_dt = datetime.combine(day, datetime.min.time())
        hstart = (first_dt - timedelta(hours=args.history_hours - 1)).date()
        return all((hstart + timedelta(days=o)) in dyn_files for o in range((day - hstart).days + 1))

    target_days = [d for d in target_days if d in files_by_date and _history_in_cache(d)]
    if args.max_days is not None:
        target_days = target_days[: args.max_days]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"output_dir": str(output_dir), "num_days": len(target_days), "device": str(device)}), flush=True)
    if not target_days:
        return

    cache = A.DatasetCache(max_open=32)
    try:
        infer_stream(
            model=model, stats=stats, config=config, files_by_date=files_by_date, dyn_files=dyn_files,
            buf=buf, cache=cache, bathy_path=bathy_path, output_dir=output_dir, checkpoint_path=checkpoint_path,
            config_path=config_path, target_days=target_days, hours=hours, history_hours=args.history_hours,
            batch_size=args.batch_size, device=device, sss_source=args.sss_source, pool=pool,
            assim_km=args.assim_km, assim_days=args.assim_days, eddy=eddy, eddy_threshold=args.eddy_threshold,
            use_neighbors=not args.no_neighbors, climato_lookup=climato_lookup,
            neighbor_window_days=args.neighbor_window_days, overwrite=args.overwrite,
            symmetric=(args.neighbor_mode == "symmetric"),
            eddy_from_merged=args.eddy_from_merged,
            eddy_radius_gate=getattr(config.dataset, "eddy_max_distance_radius", 2.0),
            amp=args.amp,
            row_start=args.row_start, row_end=args.row_end,
            name_tag=args.name_tag,
            gate_km=args.gate_neighbors_km, gate_days=args.gate_neighbors_days,
        )
    finally:
        cache.close()


if __name__ == "__main__":
    main()
