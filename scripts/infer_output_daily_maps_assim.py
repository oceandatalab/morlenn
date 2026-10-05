from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
UTC = timezone.utc
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np
import torch
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morlenn.config import load_config
from morlenn.data import RobustStats, load_stats
from morlenn.model import split_prediction_params, UpperDynTCNAttentionModel
from morlenn.notebook_analysis import (
    _apply_checkpoint_snapshot,
    _fit_model,
    _infer_model_overrides_from_checkpoint,
    _load_checkpoint_state,
)
from morlenn.train import resolve_device, set_seed


DYNAMIC_BASE_VARS = {
    "wind_speed": ("u10", "v10"),
    "wind_speed_neutral": ("u10n", "v10n"),
    "sea_slope": ("swh", "mwp"),
}

EDDY_R = 6371.0
EDDY_STATIC_NAMES = (
    "eddy_amplitude", "eddy_effective_radius", "eddy_speed_average",
    "eddy_speed_radius", "eddy_type", "eddy_lifetime_type",
)


class DatasetCache:
    def __init__(self, max_open: int = 40) -> None:
        self.max_open = max_open
        self._cache: OrderedDict[Path, netCDF4.Dataset] = OrderedDict()

    def get(self, path: Path) -> netCDF4.Dataset:
        dataset = self._cache.pop(path, None)
        if dataset is None:
            dataset = netCDF4.Dataset(path, "r")
        self._cache[path] = dataset
        while len(self._cache) > self.max_open:
            _, old = self._cache.popitem(last=False)
            old.close()
        return dataset

    def close(self) -> None:
        for dataset in self._cache.values():
            dataset.close()
        self._cache.clear()


class DayArrayCache:
    """LRU buffer of decompressed (24, H, W) feature arrays keyed by (cache_path, name).

    Used only in --cache-dir mode: a given (day, feature) is decompressed once and reused
    across every output hour / lat-tile / overlapping output day, instead of re-decompressing
    the same slim file 24*ntiles times. Memory ~ max_entries * (24*H*W*4 bytes).
    """

    def __init__(self, max_entries: int = 816) -> None:
        self.max_entries = max_entries
        self._cache: "OrderedDict[tuple[Path, str], np.ndarray]" = OrderedDict()

    def get(self, path: Path, name: str) -> np.ndarray:
        key = (path, name)
        arr = self._cache.pop(key, None)
        if arr is None:
            with netCDF4.Dataset(path, "r") as dataset:
                var = dataset.variables[name]
                var.set_auto_mask(False)  # float features keep stored NaN; interp_count keeps 0 (fill_value=0)
                arr = np.asarray(var[:], dtype=np.float32)  # (24, H, W)
        self._cache[key] = arr
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return arr


def discover_cache_files(cache_dir: Path) -> dict[date, Path]:
    files: dict[date, Path] = {}
    for path in sorted(cache_dir.glob("cache_*.nc")):
        stem = path.stem.removeprefix("cache_")
        if len(stem) != 8 or not stem.isdigit():
            continue
        files[datetime.strptime(stem, "%Y%m%d").date()] = path
    return files


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_hours(value: str) -> list[int]:
    if value == "all":
        return list(range(24))
    hours: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            hours.extend(range(int(start), int(end) + 1))
        else:
            hours.append(int(part))
    unique = sorted(set(hours))
    if any(hour < 0 or hour > 23 for hour in unique):
        raise ValueError("--hours must be 'all' or a comma-separated list/ranges in [0, 23].")
    return unique


def to_numpy(values: np.ndarray) -> np.ndarray:
    if isinstance(values, np.ma.MaskedArray):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float32)


def read_merged(dataset: netCDF4.Dataset, name: str, index=slice(None)) -> np.ndarray:
    """Lit une variable d'un fichier merged en IGNORANT valid_min / valid_max.

    Les fichiers merged v2 sont repackes en int16 avec un scale_factor/add_offset
    PROPRE au fichier, mais conservent les attributs valid_min/valid_max herites du
    produit amont (packing 0.01 K / 273.15 K pour analysed_sst). netCDF4 applique ces
    bornes aux entiers BRUTS : avec le nouveau packing elles ne couvrent plus qu'une
    fenetre de ~2 K, et ~96% de l'ocean est masque a tort (l'inference sort alors du
    NaN partout via la garde `sfin` sur les statiques). On decode donc a la main et on
    ne masque que sur _FillValue / missing_value.
    """
    var = dataset.variables[name]
    var.set_auto_maskandscale(False)
    try:
        raw = np.asarray(var[index])
    finally:
        # les Dataset sont mis en cache et partages (DatasetCache) -> on restaure le
        # defaut netCDF4 pour ne pas rendre des entiers bruts a un autre lecteur.
        var.set_auto_maskandscale(True)
    missing = np.zeros(raw.shape, dtype=bool)
    for attr in ("_FillValue", "missing_value"):
        if attr in var.ncattrs():
            for sentinel in np.atleast_1d(var.getncattr(attr)):
                missing |= raw == sentinel
    out = raw.astype(np.float32)
    if "scale_factor" in var.ncattrs():
        out = out * np.float32(var.getncattr("scale_factor"))
    if "add_offset" in var.ncattrs():
        out = out + np.float32(var.getncattr("add_offset"))
    out[missing] = np.nan
    return out


def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return num / np.where(np.abs(den) < eps, eps, den)


def geo_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat_deg).astype(np.float32)
    lon_rad = np.deg2rad(lon_deg).astype(np.float32)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def day_of_year(day: date) -> int:
    return day.timetuple().tm_yday


def discover_daily_files(input_dir: Path, prefix: str = "merged_") -> dict[date, Path]:
    # La date YYYYMMDD est lue sur les 8 CHIFFRES FINAUX du nom, ce qui tolere un eventuel
    # token de resolution entre le prefixe et la date : "merged_YYYYMMDD.nc" (output_daily_025)
    # ET "merged_025_YYYYMMDD.nc" (output_daily_025_v2) fonctionnent tous deux avec prefix="merged_",
    # "glorys_merged_YYYYMMDD.nc" avec prefix="glorys_merged_".
    files: dict[date, Path] = {}
    for path in sorted(input_dir.glob(f'{prefix}*.nc')):
        token = path.stem[-8:]
        if not token.isdigit():
            continue
        try:
            files[datetime.strptime(token, "%Y%m%d").date()] = path
        except ValueError:
            continue
    return files


def required_variables(dynamic_features: Iterable[str], static_features: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for feature in dynamic_features:
        names.update(DYNAMIC_BASE_VARS.get(feature, (feature,)))
    for feature in static_features:
        if feature in {"geo_x", "geo_y", "geo_z", "doy_sin", "doy_cos", "elevation", "mld_climato"}:
            continue
        if feature.startswith("eddy_"):  # computed from the eddy atlas, not read from daily files
            continue
        names.add(feature)
    return names


def static_source_name(feature: str, sss_source: str) -> str:
    if feature == "sss":
        return sss_source
    return feature


def has_required_variables(path: Path, names: set[str]) -> bool:
    try:
        with netCDF4.Dataset(path, "r") as dataset:
            return all(name in dataset.variables for name in names)
    except OSError:
        return False


def valid_target_dates(
    files_by_date: dict[date, Path],
    required_vars: set[str],
    history_hours: int,
    start: date | None,
    end: date | None,
) -> list[date]:
    available = sorted(files_by_date)
    if start is not None:
        available = [day for day in available if day >= start]
    if end is not None:
        available = [day for day in available if day <= end]
    # Only candidate targets and their history windows need a variable check; checking every
    # file (over a slow network mount) needlessly opens hundreds of unrelated daily files.
    needed: set[date] = set()
    for day in available:
        history_start = (datetime.combine(day, datetime.min.time()) - timedelta(hours=history_hours - 1)).date()
        for offset in range((day - history_start).days + 1):
            needed.add(history_start + timedelta(days=offset))
    good_file = {
        day: has_required_variables(path, required_vars)
        for day, path in files_by_date.items()
        if day in needed
    }
    valid: list[date] = []
    for day in available:
        if not good_file.get(day, False):
            continue
        first_dt = datetime.combine(day, datetime.min.time())
        history_start = first_dt - timedelta(hours=history_hours - 1)
        history_days = {
            (history_start + timedelta(days=offset)).date()
            for offset in range((day - history_start.date()).days + 1)
        }
        if all(history_day in files_by_date and good_file.get(history_day, False) for history_day in history_days):
            valid.append(day)
    return valid


def read_var(
    cache: DatasetCache,
    path: Path,
    name: str,
    hours: list[int],
    lat_slice: slice,
) -> np.ndarray:
    dataset = cache.get(path)
    return read_merged(dataset, name, np.index_exp[hours, lat_slice, :])


def grouped_hours(target_dt: datetime, history_hours: int) -> list[tuple[date, list[int]]]:
    by_day: dict[date, list[int]] = defaultdict(list)
    start_dt = target_dt - timedelta(hours=history_hours - 1)
    for offset in range(history_hours):
        current = start_dt + timedelta(hours=offset)
        by_day[current.date()].append(current.hour)
    return sorted(by_day.items())


def read_dynamic_feature(
    cache: DatasetCache,
    files_by_date: dict[date, Path],
    feature: str,
    target_dt: datetime,
    history_hours: int,
    lat_slice: slice,
    precomputed: bool = False,
    buf: "DayArrayCache | None" = None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for day, hours in grouped_hours(target_dt, history_hours):
        path = files_by_date[day]
        if precomputed:
            # slim filled cache: every feature (incl. grad_*, sss, interp_count) stored by name,
            # gap-free; decompressed once per (day, feature) via the buffer.
            arr = buf.get(path, feature)
            # slice latitude FIRST (cheap view of the tile band) then fancy-index the hours,
            # so we copy only (len(hours), tile_lat, W) instead of the whole (24, H, W) grid.
            values = np.asarray(arr[:, lat_slice, :][hours], dtype=np.float32)
            chunks.append(values)
            continue
        if feature == "wind_speed":
            u10 = read_var(cache, path, "u10", hours, lat_slice)
            v10 = read_var(cache, path, "v10", hours, lat_slice)
            values = np.sqrt(u10**2 + v10**2).astype(np.float32)
        elif feature == "wind_speed_neutral":
            u10n = read_var(cache, path, "u10n", hours, lat_slice)
            v10n = read_var(cache, path, "v10n", hours, lat_slice)
            values = np.sqrt(u10n**2 + v10n**2).astype(np.float32)
        elif feature == "sea_slope":
            swh = read_var(cache, path, "swh", hours, lat_slice)
            mwp = read_var(cache, path, "mwp", hours, lat_slice)
            values = safe_ratio(swh, mwp)
        else:
            values = read_var(cache, path, feature, hours, lat_slice)
        chunks.append(values)
    return np.concatenate(chunks, axis=0)


def gradient_magnitude_per_m(field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Horizontal gradient magnitude (units of field per metre) for a [T,H,W] field.

    grad_analysed_sst is empty in the maps merged files; the training feature is the
    SST gradient magnitude in K/m, so we recompute it from analysed_sst on the grid.
    """
    DEG2M = 111000.0
    lats = lats.astype(np.float64)
    lons = lons.astype(np.float64)
    dlat = np.gradient(lats) if lats.size >= 2 else np.ones_like(lats)
    dlon = np.gradient(lons) if lons.size >= 2 else np.ones_like(lons)
    coslat = np.cos(np.deg2rad(lats))
    out = np.empty_like(field, dtype=np.float32)
    for t in range(field.shape[0]):
        f2 = field[t].astype(np.float64)
        gy = np.gradient(f2, axis=0) if f2.shape[0] >= 2 else np.zeros_like(f2)
        gx = np.gradient(f2, axis=1) if f2.shape[1] >= 2 else np.zeros_like(f2)
        gy_m = gy / (dlat[:, None] * DEG2M)
        gx_m = gx / (dlon[None, :] * DEG2M * np.where(np.abs(coslat) < 1e-6, 1e-6, coslat)[:, None])
        out[t] = np.sqrt(gx_m**2 + gy_m**2).astype(np.float32)
    return out


def temporal_fill(values: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN along the time axis (axis 0) of a [T, H, W] field.

    Fills flicker / record-start gaps in the input history per grid point. A point whose
    whole history is NaN (e.g. permanently ice-masked) is left NaN (no spatial fabrication).
    """
    T = values.shape[0]
    a = values.reshape(T, -1).astype(np.float64)  # [T, P]
    nan = np.isnan(a)
    if not nan.any():
        return values
    P = a.shape[1]
    tcol = np.arange(T)[:, None]
    rows = np.arange(P)[None, :]
    fin = ~nan
    fwd = np.maximum.accumulate(np.where(fin, tcol, -1), axis=0)            # last finite idx <= t (-1 if none)
    bwd = np.minimum.accumulate(np.where(fin, tcol, T)[::-1], axis=0)[::-1]  # next finite idx >= t (T if none)
    v_fwd = a[np.clip(fwd, 0, T - 1), rows]
    v_bwd = a[np.clip(bwd, 0, T - 1), rows]
    denom = (bwd - fwd).astype(np.float64)
    denom[denom == 0] = 1.0
    w = (tcol - fwd) / denom
    interp = v_fwd * (1.0 - w) + v_bwd * w           # linear between bracketing finite points
    interp = np.where(fwd < 0, v_bwd, interp)        # leading gap -> backfill (farthest-in-time edge)
    interp = np.where(bwd >= T, v_fwd, interp)       # trailing gap -> forward-fill
    out = np.where(nan, interp, a)
    out[:, ~fin.any(0)] = np.nan                     # columns with no finite at all stay NaN (no spatial fill)
    return out.reshape(values.shape).astype(np.float32)


def build_eddy_day(eddy: dict, target_day: date):
    """Eddies active on target_day -> KDTree on unit-sphere centers + their fields."""
    eday = float(target_day.toordinal())
    m = eddy["day"] == eday
    if not bool(m.any()):
        return None
    la = np.deg2rad(eddy["lat"][m].astype(np.float64))
    lo = np.deg2rad(eddy["lon"][m].astype(np.float64))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1).astype(np.float32)
    fields = {k: eddy[k][m] for k in ("amp_m", "eff_km", "spdavg", "sprad_km", "etype", "ltype")}
    return {"tree": KDTree(xyz), "fields": fields, "n": int(m.sum())}


def compute_eddy_grid(lat_grid: np.ndarray, lon_grid: np.ndarray, eday, threshold: float) -> dict:
    """For each grid point assign the fields of the most-central eddy with dist/radius<=threshold (else 0)."""
    shape = lat_grid.shape
    out = {name: np.zeros(shape, np.float32) for name in EDDY_STATIC_NAMES}
    if eday is None:
        return out
    la = np.deg2rad(lat_grid.ravel().astype(np.float64))
    lo = np.deg2rad(lon_grid.ravel().astype(np.float64))
    g = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1).astype(np.float32)
    k = min(8, eday["n"])
    chord, idx = eday["tree"].query(g, k=k)
    if k == 1:
        chord = chord.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    gc_km = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * EDDY_R
    eff = eday["fields"]["eff_km"][idx]
    ratio = gc_km / np.where(eff > 0.0, eff, np.inf)
    ratio = np.where(ratio <= threshold, ratio, np.inf)
    best = np.argmin(ratio, axis=1)
    rows = np.arange(best.size)
    has = np.isfinite(ratio[rows, best])
    chosen = idx[rows, best]
    fld = eday["fields"]
    for name, key in zip(EDDY_STATIC_NAMES, ("amp_m", "eff_km", "spdavg", "sprad_km", "etype", "ltype")):
        v = np.zeros(g.shape[0], np.float32)
        v[has] = fld[key][chosen[has]].astype(np.float32)
        out[name] = v.reshape(shape)
    return out


def build_static_features(
    cache: DatasetCache,
    target_path: Path,
    target_day: date,
    hour: int,
    static_features: list[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    elevation_tile: np.ndarray,
    lat_slice: slice,
    sss_source: str,
    eddy_day=None,
    eddy_threshold: float = 2.0,
    climato_lookup=None,
) -> np.ndarray:
    dataset = cache.get(target_path)
    lat_grid, lon_grid = np.meshgrid(latitudes[lat_slice], longitudes, indexing="ij")
    geo = geo_xyz(lat_grid, lon_grid)
    doy = day_of_year(target_day)
    eddy_fields = None
    _eddy_feats = [f for f in static_features if f.startswith("eddy_")]
    if _eddy_feats and not all(f in dataset.variables for f in _eddy_feats):
        eddy_fields = compute_eddy_grid(lat_grid, lon_grid, eddy_day, eddy_threshold)
    rows: list[np.ndarray] = []
    for feature in static_features:
        if feature.startswith("eddy_"):
            if feature in dataset.variables:          # eddy_* baked into daily files -> read directly
                rows.append(read_merged(dataset, feature, np.index_exp[hour, lat_slice, :]))
            else:
                rows.append(eddy_fields[feature])
        elif feature == "mld_climato":
            if climato_lookup is None:
                rows.append(np.zeros_like(elevation_tile))
            else:
                cl = climato_lookup(
                    lat_grid.ravel(), lon_grid.ravel(), np.full(lat_grid.size, float(doy))
                ).reshape(lat_grid.shape).astype(np.float32)
                rows.append(cl)
        elif feature == "geo_x":
            rows.append(geo[0])
        elif feature == "geo_y":
            rows.append(geo[1])
        elif feature == "geo_z":
            rows.append(geo[2])
        elif feature == "elevation":
            rows.append(elevation_tile)
        elif feature == "doy_sin":
            rows.append(np.full_like(elevation_tile, np.sin(2.0 * np.pi * doy / 365.0), dtype=np.float32))
        elif feature == "doy_cos":
            rows.append(np.full_like(elevation_tile, np.cos(2.0 * np.pi * doy / 365.0), dtype=np.float32))
        else:
            source_name = static_source_name(feature, sss_source=sss_source)
            rows.append(read_merged(dataset, source_name, np.index_exp[hour, lat_slice, :]))
    return np.stack(rows, axis=0).astype(np.float32)


def normalize_dynamic(dynamic: np.ndarray, stats: RobustStats) -> np.ndarray:
    return (dynamic - stats.dynamic_median[:, None]) / np.where(
        stats.dynamic_iqr[:, None] == 0.0,
        1.0,
        stats.dynamic_iqr[:, None],
    )


def normalize_static(static: np.ndarray, stats: RobustStats) -> np.ndarray:
    return (static - stats.static_median[:, None]) / np.where(
        stats.static_iqr[:, None] == 0.0,
        1.0,
        stats.static_iqr[:, None],
    )


def inverse_mld(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.astype(np.float32)
    if transform == "log1p":
        return np.expm1(values).astype(np.float32)
    raise ValueError(f"Unsupported MLD transform: {transform}")


def denormalize_prediction(prediction: np.ndarray, stats: RobustStats, target_names: list[str], transform: str) -> np.ndarray:
    restored = prediction * np.where(stats.target_iqr == 0.0, 1.0, stats.target_iqr) + stats.target_median
    output = restored.astype(np.float32, copy=True)
    for index, name in enumerate(target_names):
        if name == "mld":
            output[:, index] = inverse_mld(output[:, index], transform)
    return output


def create_output(
    path: Path,
    source_path: Path,
    bathy_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    hours: list[int],
) -> netCDF4.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(source_path, "r") as src, netCDF4.Dataset(bathy_path, "r") as bathy:
        dataset = netCDF4.Dataset(path, "w")
        dataset.createDimension("time", len(hours))
        dataset.createDimension("latitude", len(src.dimensions["latitude"]))
        dataset.createDimension("longitude", len(src.dimensions["longitude"]))
        time_var = dataset.createVariable("time", "f8", ("time",))
        time_var[:] = src.variables["time"][hours]
        for attr in src.variables["time"].ncattrs():
            if attr == "_FillValue": continue
            time_var.setncattr(attr, src.variables["time"].getncattr(attr))
        lat_var = dataset.createVariable("latitude", "f4", ("latitude",))
        lon_var = dataset.createVariable("longitude", "f4", ("longitude",))
        lat_var[:] = src.variables["latitude"][:]
        lon_var[:] = src.variables["longitude"][:]
        for name, var in [("latitude", lat_var), ("longitude", lon_var)]:
            src_var = src.variables[name]
            for attr in src_var.ncattrs():
                if attr == "_FillValue": continue
                var.setncattr(attr, src_var.getncattr(attr))
        elev = dataset.createVariable("elevation", "f4", ("latitude", "longitude"), zlib=True, complevel=4)
        elev[:] = bathy.variables["elevation"][:]
        elev.units = getattr(bathy.variables["elevation"], "units", "m")
        dataset.source_daily_file = str(source_path)
        dataset.source_bathymetry_file = str(bathy_path)
        dataset.checkpoint = str(checkpoint_path)
        dataset.config = str(config_path)
        dataset.mld_postprocess = "clamped to local bathymetry depth where elevation < 0"
        dataset.created_utc = datetime.now(UTC).isoformat(timespec="seconds")
        for name, units, long_name in [
            ("mld", "m", "UpperDyn inferred mixed layer depth"),
            ("N2", "s-2", "UpperDyn inferred N2"),
            ("mld_std", "m", "UpperDyn predictive standard deviation for MLD"),
            ("N2_std", "s-2", "UpperDyn predictive standard deviation for N2"),
        ]:
            var = dataset.createVariable(
                name,
                "f4",
                ("time", "latitude", "longitude"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
                chunksizes=(1, min(64, len(src.dimensions["latitude"])), len(src.dimensions["longitude"])),
            )
            var.units = units
            var.long_name = long_name
        valid = dataset.createVariable(
            "valid_input",
            "i1",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            fill_value=np.int8(0),
            chunksizes=(1, min(64, len(src.dimensions["latitude"])), len(src.dimensions["longitude"])),
        )
        valid.long_name = "1 where all required model inputs were finite before normalization"
        for name, dtype, units, long_name, fill in [
            ("n1_dist_km", "f4", "km", "Great-circle distance to the nearest causal Argo neighbor", np.float32(np.nan)),
            ("n1_dt_days", "f4", "days", "Age in days of the nearest causal Argo neighbor", np.float32(np.nan)),
            ("n_neighbors", "i1", "1", "Number of causal Argo neighbors fed to the model (0-5)", np.int8(0)),
            ("assim_mask", "i1", "1", "1 where a close recent neighbor is present (dist<=assim_km and age<=assim_days)", np.int8(0)),
            ("interp_fraction", "f4", "1", "Fraction of dynamic inputs (feature x hour over the 720h history) that were temporally interpolated (0=perfect, higher=imperfect)", np.float32(np.nan)),
        ]:
            v = dataset.createVariable(
                name, dtype, ("time", "latitude", "longitude"), zlib=True, complevel=4,
                fill_value=fill,
                chunksizes=(1, min(64, len(src.dimensions["latitude"])), len(src.dimensions["longitude"])),
            )
            v.units = units
            v.long_name = long_name
        return dataset


def infer_day(
    model: torch.nn.Module,
    stats: RobustStats,
    config,
    files_by_date: dict[date, Path],
    cache: DatasetCache,
    input_dir: Path,
    bathy_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    target_day: date,
    hours: list[int],
    history_hours: int,
    tile_lat: int,
    batch_size: int,
    device: torch.device,
    ocean_only: bool,
    sss_source: str,
    pool: dict,
    assim_km: float,
    assim_days: float,
    eddy: dict,
    eddy_threshold: float,
    use_neighbors: bool = True,
    symmetric: bool = False,
    climato_lookup=None,
    dyn_files: dict[date, Path] | None = None,
    precomputed: bool = False,
    buf: "DayArrayCache | None" = None,
) -> None:
    # dyn_files = source of the 720h dynamic history (slim filled cache when precomputed);
    # files_by_date stays the merged target used for the output template + lat/lon grid.
    if dyn_files is None:
        dyn_files = files_by_date
    target_path = files_by_date[target_day]
    with netCDF4.Dataset(target_path, "r") as src, netCDF4.Dataset(bathy_path, "r") as bathy:
        latitudes = to_numpy(src.variables["latitude"][:])
        longitudes = to_numpy(src.variables["longitude"][:])
        elevation = to_numpy(bathy.variables["elevation"][:])
        nlat = len(latitudes)
        nlon = len(longitudes)

    temp_path = output_path.with_name(f"{output_path.stem}.tmp.nc")
    if temp_path.exists():
        temp_path.unlink()
    output = create_output(temp_path, target_path, bathy_path, checkpoint_path, config_path, hours)

    # --- causal neighbor sub-tree for this day (pool profiles strictly before target_day) ---
    R = 6371.0
    day_grid = float(target_day.toordinal())
    sel = (np.ones(pool["day"].shape, dtype=bool) if symmetric else (pool["day"] < day_grid))
    n_sel = int(sel.sum())
    if n_sel >= 1:
        sub_xyz = pool["xyz"][sel]
        sub_day = pool["day"][sel]
        sub_feat = pool["feat"][sel]
        sub_4d = np.concatenate([R * sub_xyz / 100.0, (sub_day / 10.0)[:, None]], axis=1).astype(np.float32)
        nb_tree = KDTree(sub_4d)
        nb_k = min(5, n_sel)
    else:
        sub_xyz = sub_day = sub_feat = None
        nb_tree = None
        nb_k = 0
    feat_dim = pool["feat"].shape[1]

    # eddies active on this day, gridded on demand inside build_static_features
    eddy_day = build_eddy_day(eddy, target_day) if eddy is not None else None

    completed = False
    try:
        for out_hour_index, hour in enumerate(hours):
            target_dt = datetime.combine(target_day, datetime.min.time()) + timedelta(hours=hour)
            for lat_start in range(0, nlat, tile_lat):
                lat_stop = min(lat_start + tile_lat, nlat)
                lat_slice = slice(lat_start, lat_stop)
                elevation_tile = elevation[lat_slice, :]
                dynamic_features: list[np.ndarray] = []
                valid = np.isfinite(elevation_tile)
                if ocean_only:
                    valid &= elevation_tile <= 0.0
                valid_pred = valid.copy()
                nan_count = np.zeros(elevation_tile.shape, dtype=np.float64)
                if precomputed:
                    # slim filled cache: all 23 features stored by name, gap-free (temporal_fill baked).
                    # interp diagnostic comes from the precomputed interp_count, no per-day recompute.
                    for feature in config.dataset.dynamic_features:
                        values = read_dynamic_feature(
                            cache=cache, files_by_date=dyn_files, feature=feature,
                            target_dt=target_dt, history_hours=history_hours, lat_slice=lat_slice,
                            precomputed=True, buf=buf,
                        )
                        valid_pred &= np.isfinite(values).all(axis=0)  # only permanent-NaN (ice) excluded
                        dynamic_features.append(values)
                    ic = read_dynamic_feature(
                        cache=cache, files_by_date=dyn_files, feature="interp_count",
                        target_dt=target_dt, history_hours=history_hours, lat_slice=lat_slice,
                        precomputed=True, buf=buf,
                    )
                    nan_count = np.nansum(ic, axis=0).astype(np.float64)  # feature*hour interpolated in window
                    valid &= valid_pred & (nan_count == 0)               # strict: predictable AND no interp used
                else:
                    for feature in config.dataset.dynamic_features:
                        if feature == "grad_analysed_sst":
                            h0 = max(0, lat_start - 1)
                            h1 = min(nlat, lat_stop + 1)
                            sst = read_dynamic_feature(
                                cache=cache, files_by_date=files_by_date, feature="analysed_sst",
                                target_dt=target_dt, history_hours=history_hours, lat_slice=slice(h0, h1),
                            )
                            grad_full = gradient_magnitude_per_m(sst, latitudes[h0:h1], longitudes)
                            top = lat_start - h0
                            values = grad_full[:, top:top + (lat_stop - lat_start), :]
                        elif feature == "grad_sss":
                            # grad_sss flickers on the shelf; recompute from sos (psu/m) like grad_analysed_sst.
                            h0 = max(0, lat_start - 1)
                            h1 = min(nlat, lat_stop + 1)
                            sal = read_dynamic_feature(
                                cache=cache, files_by_date=files_by_date, feature="sos",
                                target_dt=target_dt, history_hours=history_hours, lat_slice=slice(h0, h1),
                            )
                            grad_full = gradient_magnitude_per_m(sal, latitudes[h0:h1], longitudes)
                            top = lat_start - h0
                            values = grad_full[:, top:top + (lat_stop - lat_start), :]
                        else:
                            read_feat = sss_source if feature == "sss" else feature  # apply sos everywhere
                            values = read_dynamic_feature(
                                cache=cache,
                                files_by_date=files_by_date,
                                feature=read_feat,
                                target_dt=target_dt,
                                history_hours=history_hours,
                                lat_slice=lat_slice,
                            )
                        nan_mask = np.isnan(values)
                        valid &= ~nan_mask.any(axis=0)              # strict: finite before interp
                        nan_count += nan_mask.sum(axis=0)
                        values = temporal_fill(values)               # temporal interpolation of input gaps
                        valid_pred &= np.isfinite(values).all(axis=0)  # finite after interp (predictable)
                        dynamic_features.append(values)
                static = build_static_features(
                    cache=cache,
                    target_path=target_path,
                    target_day=target_day,
                    hour=hour,
                    static_features=config.dataset.static_features,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    elevation_tile=elevation_tile,
                    lat_slice=lat_slice,
                    sss_source=sss_source,
                    eddy_day=eddy_day,
                    eddy_threshold=eddy_threshold,
                    climato_lookup=climato_lookup,
                )
                sfin = np.all(np.isfinite(static), axis=0)
                valid &= sfin           # strict (no interpolation) -> valid_input diagnostic
                valid_pred &= sfin      # predictable: dynamics finite after temporal interp + static finite
                dynamic = np.stack(dynamic_features, axis=0).reshape(
                    len(config.dataset.dynamic_features),
                    history_hours,
                    -1,
                )
                static_flat = static.reshape(len(config.dataset.static_features), -1)
                valid_flat = valid.reshape(-1)
                pred = np.full((valid_flat.size, len(config.dataset.targets)), np.nan, dtype=np.float32)
                pred_std = np.full_like(pred, np.nan)
                # predict where inputs are recoverable (dynamics temporally interpolated); no spatial fill.
                valid_indices = np.flatnonzero(valid_pred.reshape(-1))
                interp_frac = (nan_count / (len(config.dataset.dynamic_features) * history_hours)).reshape(-1).astype(np.float32)
                n1km = np.full(valid_flat.size, np.nan, dtype=np.float32)
                n1dt = np.full(valid_flat.size, np.nan, dtype=np.float32)
                nnb = np.zeros(valid_flat.size, dtype=np.float32)
                tile_lat_vals = latitudes[lat_slice].astype(np.float64)
                for start in range(0, valid_indices.size, batch_size):
                    indices = valid_indices[start : start + batch_size]
                    dyn_batch = dynamic[:, :, indices]
                    dyn_batch = normalize_dynamic(dyn_batch.reshape(len(config.dataset.dynamic_features), -1), stats)
                    dyn_batch = dyn_batch.reshape(len(config.dataset.dynamic_features), history_hours, indices.size)
                    # valid_indices already guarantees finite dynamics and statics -> nan_to_num was redundant
                    # (np.nan_to_num's isposinf/isneginf passes were ~20% of the per-day runtime).
                    dyn_tensor = torch.from_numpy(np.transpose(dyn_batch, (2, 0, 1))).to(device=device, dtype=torch.float32)
                    static_batch = normalize_static(static_flat[:, indices], stats)
                    static_tensor = torch.from_numpy(static_batch.T).to(device=device, dtype=torch.float32)
                    # --- causal neighbors for these grid points ---
                    i_lat_local = indices // nlon
                    i_lon = indices % nlon
                    lat_pts = tile_lat_vals[i_lat_local]
                    lon_pts = longitudes[i_lon].astype(np.float64)
                    la = np.deg2rad(lat_pts)
                    lo = np.deg2rad(lon_pts)
                    gxyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1).astype(np.float32)
                    nb_feat = np.zeros((indices.size, 5, feat_dim), dtype=np.float32)
                    nb_mask = np.zeros((indices.size, 5), dtype=np.float32)
                    if nb_tree is not None and nb_k > 0:
                        q4d = np.concatenate(
                            [R * gxyz / 100.0, np.full((indices.size, 1), day_grid / 10.0, dtype=np.float32)], axis=1
                        )
                        knn_idx = nb_tree.query(q4d, k=nb_k, return_distance=False)
                        nb_feat[:, :nb_k, :] = sub_feat[knn_idx]
                        nb_mask[:, :nb_k] = 1.0
                        j0 = knn_idx[:, 0]
                        chord = np.linalg.norm(gxyz - sub_xyz[j0], axis=1)
                        n1km[indices] = (2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0)) * R).astype(np.float32)
                        n1dt[indices] = (day_grid - sub_day[j0]).astype(np.float32)
                        nnb[indices] = float(nb_k)
                    nb_feat_t = torch.from_numpy(nb_feat).to(device=device, dtype=torch.float32)
                    nb_mask_t = torch.from_numpy(nb_mask).to(device=device, dtype=torch.float32)
                    with torch.no_grad():
                        if use_neighbors:
                            output_tensor = model(dyn_tensor, static_tensor, neighbors=nb_feat_t, neighbor_mask=nb_mask_t)
                        else:
                            output_tensor = model(dyn_tensor, static_tensor)
                        mean_tensor, std_tensor = split_prediction_params(output_tensor, len(config.dataset.targets))
                    mean_norm = mean_tensor.detach().cpu().numpy()
                    pred[indices] = denormalize_prediction(
                        mean_norm,
                        stats=stats,
                        target_names=config.dataset.targets,
                        transform=config.dataset.mld_target_transform,
                    )
                    if std_tensor is not None:
                        # The Gaussian heads are trained directly in physical units.
                        pred_std[indices] = std_tensor.detach().cpu().numpy()
                shape = (lat_stop - lat_start, nlon)
                mld_tile = pred[:, 0].reshape(shape)
                bathy_depth_tile = np.where(elevation_tile < 0.0, -elevation_tile, np.nan).astype(np.float32)
                clamp_mask = np.isfinite(bathy_depth_tile)
                mld_tile = np.where(clamp_mask, np.minimum(mld_tile, bathy_depth_tile), mld_tile)
                output.variables["N2"][out_hour_index, lat_slice, :] = pred[:, 1].reshape(shape)
                output.variables["mld"][out_hour_index, lat_slice, :] = mld_tile
                output.variables["mld_std"][out_hour_index, lat_slice, :] = pred_std[:, 0].reshape(shape)
                output.variables["N2_std"][out_hour_index, lat_slice, :] = pred_std[:, 1].reshape(shape)
                output.variables["valid_input"][out_hour_index, lat_slice, :] = valid.astype(np.int8)
                output.variables["n1_dist_km"][out_hour_index, lat_slice, :] = n1km.reshape(shape)
                output.variables["n1_dt_days"][out_hour_index, lat_slice, :] = n1dt.reshape(shape)
                output.variables["n_neighbors"][out_hour_index, lat_slice, :] = nnb.reshape(shape).astype(np.int8)
                assim = ((n1km <= assim_km) & (np.abs(n1dt) <= assim_days) & (nnb > 0)).astype(np.int8)
                output.variables["assim_mask"][out_hour_index, lat_slice, :] = assim.reshape(shape)
                interp_out = np.where(valid_pred.reshape(-1), interp_frac, np.nan).astype(np.float32)
                output.variables["interp_fraction"][out_hour_index, lat_slice, :] = interp_out.reshape(shape)
            output.sync()
        completed = True
    finally:
        output.close()
    if completed:
        temp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer UpperDyn MLD/N2 maps from output_daily merged files.")
    parser.add_argument("--config", default=str(ROOT / "configs/20260421-150203.toml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts/checkpoints/20260421-150203/best_mld_rmse.pt"))
    parser.add_argument("--input-dir", default=str(ROOT / "output_daily"))
    parser.add_argument("--bathy", default=str(ROOT / "output_daily/inputs_bathymetry.nc"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/map_inference/20260421-150203"))
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD. Defaults to first date with full history.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD. Defaults to last valid date.")
    parser.add_argument("--hours", default="all", help="'all', comma list, or ranges, e.g. '0,6,12,18' or '0-23'.")
    parser.add_argument("--history-hours", type=int, default=720)
    parser.add_argument("--tile-lat", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-land", action="store_true")
    parser.add_argument("--sss-source", choices=["sss", "sos"], default="sss")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--neighbor-pool", required=True, help="Path to grid_neighbor_pool_*.npz (feat,xyz,day).")
    parser.add_argument("--assim-km", type=float, default=50.0, help="assim_mask: max distance (km) to nearest neighbor.")
    parser.add_argument("--assim-days", type=float, default=10.0, help="assim_mask: max age (days) of nearest neighbor.")
    parser.add_argument("--neighbor-mode", choices=["causal", "symmetric"], default="causal",
        help="causal: voisins strictement passes (forecast). symmetric: passe+futur, metrique [xyz/100,jour/10], colocalise inclus (modele v3 entraine sym).")
    parser.add_argument("--eddy-data", required=True, help="Path to eddy_meta32_*_natl.npz (gridded eddy source).")
    parser.add_argument("--eddy-threshold", type=float, default=2.0, help="Match eddy if dist/effective_radius <= this.")
    parser.add_argument("--no-neighbors", action="store_true", help="Baseline mode: build model without the neighbor branch and predict without assimilation (diagnostic mask fields are still written).")
    parser.add_argument("--climato-nc", default=None, help="Path to climato200km.nc; required if static_features contains mld_climato.")
    parser.add_argument("--cache-dir", default=None, help="Slim filled input cache dir (build_input_cache + fill_input_cache). When set, the 720h dynamic history is read gap-free from cache_*.nc instead of deriving from merged_*.nc (I/O + temporal_fill baked once).")
    parser.add_argument("--mem-days", type=int, default=34, help="Days of (feature) arrays to keep decompressed in RAM in --cache-dir mode (~mem_days*24*21MB).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    input_dir = Path(args.input_dir)
    bathy_path = Path(args.bathy)
    output_dir = Path(args.output_dir)
    start = parse_date(args.start_date) if args.start_date else None
    end = parse_date(args.end_date) if args.end_date else None
    hours = parse_hours(args.hours)

    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = load_stats(config.paths.stats_path) if config.paths.stats_path.exists() else None
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    set_seed(config.training.seed)
    # set_seed() forces cudnn.deterministic=True (+benchmark=False), which makes every
    # conv forward ~100x slower. Inference needs no determinism — re-enable autotuned
    # kernels so an hourly month is feasible (~17 min/day vs ~13 h/day).
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = resolve_device(args.device or config.training.device)
    # Build the model explicitly from the (checkpoint-restored) config so the full
    # architecture is reproduced: double N2 backbone + neighbor assimilation branch.
    M = config.model
    ND_ = len(config.dataset.dynamic_features)
    NS_ = len(config.dataset.static_features)
    model = UpperDynTCNAttentionModel(
        num_dynamic=ND_, num_static=NS_, num_targets=len(config.dataset.targets),
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
        use_mld_neighbors=not args.no_neighbors,
        neighbor_feat_dim=getattr(M, "neighbor_feat_dim", 20), neighbor_hidden_dim=getattr(M, "neighbor_hidden_dim", 96),
        neighbor_modality_dropout=getattr(M, "neighbor_modality_dropout", 0.5),
        neighbor_increment=bool(getattr(M, "neighbor_increment", False)),
        neighbor_increment_full=bool(getattr(M, "neighbor_increment_full", False)),
    )
    model.load_state_dict(_load_checkpoint_state(checkpoint), strict=False)
    model.to(device)
    model.eval()
    overrides = {"build": "explicit_from_config", "neighbor_increment": bool(getattr(M, "neighbor_increment", False))}

    pool_npz = np.load(args.neighbor_pool)
    pool = {"feat": pool_npz["feat"].astype(np.float32), "xyz": pool_npz["xyz"].astype(np.float32), "day": pool_npz["day"].astype(np.float64)}
    print(json.dumps({"neighbor_pool": str(args.neighbor_pool), "pool_size": int(pool["feat"].shape[0]), "feat_dim": int(pool["feat"].shape[1])}), flush=True)

    ez = np.load(args.eddy_data)
    eddy = {"day": ez["day"].astype(np.float64), "lat": ez["lat"].astype(np.float32), "lon": ez["lon"].astype(np.float32),
            "amp_m": ez["amp_m"].astype(np.float32), "eff_km": ez["eff_km"].astype(np.float32),
            "spdavg": ez["spdavg"].astype(np.float32), "sprad_km": ez["sprad_km"].astype(np.float32),
            "etype": ez["etype"].astype(np.float32), "ltype": ez["ltype"].astype(np.float32)}
    print(json.dumps({"eddy_data": str(args.eddy_data), "eddy_obs": int(eddy["day"].size), "eddy_days": int(np.unique(eddy["day"]).size)}), flush=True)

    climato_lookup = None
    if "mld_climato" in config.dataset.static_features:
        if args.climato_nc is None:
            raise ValueError("static_features contains mld_climato but --climato-nc was not provided.")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from climato_lookup import ClimatoLookup
        climato_lookup = ClimatoLookup(args.climato_nc)
        print(json.dumps({"climato_nc": str(args.climato_nc)}), flush=True)

    files_by_date = discover_daily_files(input_dir)
    required_vars = required_variables(
        config.dataset.dynamic_features,
        [
            static_source_name(feature, sss_source=args.sss_source)
            for feature in config.dataset.static_features
        ],
    )
    target_days = valid_target_dates(
        files_by_date=files_by_date,
        required_vars=required_vars,
        history_hours=args.history_hours,
        start=start,
        end=end,
        )
    # --- slim filled input cache (opt-in): gap-free dynamic history, I/O + temporal_fill baked once ---
    precomputed = args.cache_dir is not None
    dyn_files = files_by_date
    buf = None
    if precomputed:
        dyn_files = discover_cache_files(Path(args.cache_dir))
        n_dyn = len(config.dataset.dynamic_features)
        buf = DayArrayCache(max_entries=args.mem_days * (n_dyn + 1))  # +1 for interp_count

        def _history_in_cache(day: date) -> bool:
            first_dt = datetime.combine(day, datetime.min.time())
            history_start = (first_dt - timedelta(hours=args.history_hours - 1)).date()
            span = range((day - history_start).days + 1)
            return all((history_start + timedelta(days=o)) in dyn_files for o in span)

        kept = [d for d in target_days if d in files_by_date and _history_in_cache(d)]
        dropped = [d for d in target_days if d not in kept]
        print(json.dumps({"cache_dir": str(args.cache_dir), "cache_days": len(dyn_files),
                          "target_days_with_cache_history": len(kept),
                          "dropped_missing_cache": [d.isoformat() for d in dropped[:10]]}), flush=True)
        target_days = kept
    if args.max_days is not None:
        target_days = target_days[: args.max_days]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "input_dir": str(input_dir),
        "bathy": str(bathy_path),
        "history_hours": args.history_hours,
        "hours": hours,
        "dynamic_features": config.dataset.dynamic_features,
        "static_features": config.dataset.static_features,
        "static_sources": {
            feature: static_source_name(feature, sss_source=args.sss_source)
            for feature in config.dataset.static_features
        },
        "target_days": [day.isoformat() for day in target_days],
        "device": str(device),
        "checkpoint_overrides": overrides,
        "neighbor_pool": str(args.neighbor_pool),
        "assim_km": args.assim_km,
        "assim_days": args.assim_days,
        "eddy_data": str(args.eddy_data),
        "eddy_threshold": args.eddy_threshold,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_days": len(target_days), "device": str(device)}), flush=True)

    cache = DatasetCache(max_open=32)
    try:
        for day in target_days:
            output_path = output_dir / f"morlenn_mld_{day:%Y%m%d}.nc"
            if output_path.exists() and not args.overwrite:
                print(json.dumps({"date": day.isoformat(), "status": "skip_exists", "path": str(output_path)}), flush=True)
                continue
            print(json.dumps({"date": day.isoformat(), "status": "start", "path": str(output_path)}), flush=True)
            infer_day(
                model=model,
                stats=stats,
                config=config,
                files_by_date=files_by_date,
                cache=cache,
                input_dir=input_dir,
                bathy_path=bathy_path,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                target_day=day,
                hours=hours,
                history_hours=args.history_hours,
                tile_lat=args.tile_lat,
                batch_size=args.batch_size,
                device=device,
                ocean_only=not args.include_land,
                sss_source=args.sss_source,
                pool=pool,
                assim_km=args.assim_km,
                symmetric=(args.neighbor_mode == "symmetric"),
                assim_days=args.assim_days,
                eddy=eddy,
                eddy_threshold=args.eddy_threshold,
                use_neighbors=not args.no_neighbors,
                climato_lookup=climato_lookup,
                dyn_files=dyn_files,
                precomputed=precomputed,
                buf=buf,
            )
            print(json.dumps({"date": day.isoformat(), "status": "done", "path": str(output_path)}), flush=True)
    finally:
        cache.close()


if __name__ == "__main__":
    main()
