from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from .config import load_config
from .data import MonthlyNetCDFDataset, list_netcdf_files


ERA5_BASE_VARS = ("swh", "mwp", "shww", "mpww", "sshf", "slhf", "u10n", "v10n")
OCEAN_CORE_VARS = ("analysed_sst", "sos", "adt", "sla", "grad_analysed_sst", "grad_sla")
OCEAN_ALL_VARS = (
    "analysed_sst",
    "sos",
    "sss",
    "adt",
    "sla",
    "ugos",
    "vgos",
    "grad_analysed_sst",
    "grad_sla",
    "grad_sss",
    "dos",
)
EDDY_VARS = (
    "eddy_type",
    "eddy_lifetime_type",
    "eddy_amplitude",
    "eddy_speed_average",
    "eddy_effective_radius",
    "eddy_speed_radius",
    "eddy_center_lon",
    "eddy_center_lat",
)
COMPUTED_STATIC_VARS = {"geo_x", "geo_y", "geo_z", "doy_sin", "doy_cos"}


def profile_counts(config_paths: Iterable[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        for split, directory in (("train", config.paths.train_dir), ("val", config.paths.val_dir)):
            files = list_netcdf_files(directory)
            dataset = MonthlyNetCDFDataset(
                files=files,
                dynamic_features=config.dataset.dynamic_features,
                static_features=config.dataset.static_features,
                targets=config.dataset.targets,
                mld_target_transform=config.dataset.mld_target_transform,
                stats=None,
                replace_nan_with_zero=config.dataset.replace_nan_with_zero,
                clip_mld_max=config.dataset.clip_mld_max,
                strict_static_features=config.dataset.strict_static_features,
            )
            rows.append(
                {
                    "config": Path(config_path).name,
                    "split": split,
                    "input_files": len(files),
                    "kept_files": len(dataset.files),
                    "profiles": len(dataset),
                    "first_file": dataset.files[0].name,
                    "last_file": dataset.files[-1].name,
                }
            )
    return pd.DataFrame(rows)


def _to_numpy(array: object) -> np.ndarray:
    if isinstance(array, np.ma.MaskedArray):
        array = array.filled(np.nan)
    return np.asarray(array, dtype=np.float32)


def _month_key(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def _companion_paths(primary_path: Path) -> dict[str, Path]:
    key = _month_key(primary_path)
    directory = primary_path.parent
    return {
        "primary": primary_path,
        "ocean": directory / f"mld_argo_natl_sst_ssh_sss_ssscnr_bat_{key}.nc",
        "eddy": directory / f"eddies_{key}.nc",
    }


def _open_sources(primary_path: Path) -> dict[str, netCDF4.Dataset]:
    paths = _companion_paths(primary_path)
    return {name: netCDF4.Dataset(path, "r") for name, path in paths.items() if path.exists()}


def _close_sources(sources: dict[str, netCDF4.Dataset]) -> None:
    for dataset in sources.values():
        dataset.close()


def _source_with_variable(sources: dict[str, netCDF4.Dataset], variable: str) -> tuple[str, netCDF4.Dataset] | None:
    for source_name in ("primary", "ocean", "eddy"):
        dataset = sources.get(source_name)
        if dataset is not None and variable in dataset.variables:
            return source_name, dataset
    return None


def _station_mask(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        finite = np.isfinite(values)
        return finite
    axes = tuple(range(1, values.ndim))
    if mode == "all":
        return np.all(np.isfinite(values), axis=axes)
    if mode == "any":
        return np.any(np.isfinite(values), axis=axes)
    if mode == "t0":
        return np.isfinite(values[:, 0])
    raise ValueError(f"Unsupported finite mode: {mode}")


def _nan_count_mask(values: np.ndarray, max_missing: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return np.isfinite(values)
    axes = tuple(range(1, values.ndim))
    return np.sum(~np.isfinite(values), axis=axes) <= max_missing


def _target_mask(sources: dict[str, netCDF4.Dataset], targets: Iterable[str], clip_mld_max: float | None) -> np.ndarray:
    nstation = len(sources["primary"].dimensions["nstation"])
    mask = np.ones(nstation, dtype=bool)
    for target in targets:
        source = _source_with_variable(sources, target)
        if source is None:
            return np.zeros(nstation, dtype=bool)
        values = _to_numpy(source[1].variables[target][:])
        if target == "mld":
            if clip_mld_max is not None:
                values = np.minimum(values, clip_mld_max)
            mask &= values > 0.0
        mask &= np.isfinite(values)
    return mask


def _mask_for_variable(
    sources: dict[str, netCDF4.Dataset],
    variable: str,
    mode: str,
    nstation: int,
) -> np.ndarray:
    source = _source_with_variable(sources, variable)
    if source is None:
        return np.zeros(nstation, dtype=bool)
    values = _to_numpy(source[1].variables[variable][:])
    return _station_mask(values, mode)


def _dynamic_required_variables(dynamic_features: Iterable[str]) -> list[str]:
    variables: list[str] = []
    for feature in dynamic_features:
        if feature == "wind_speed":
            variables.extend(["u10", "v10"])
            continue
        if feature == "wind_speed_neutral":
            variables.extend(["u10n", "v10n"])
            continue
        if feature == "sea_slope":
            variables.extend(["swh", "mwp"])
            continue
        variables.append(feature)
    return list(dict.fromkeys(variables))


def scan_new_dataset(
    config_path: str | Path,
    data_root: str | Path,
    missing_thresholds: Iterable[int] = (0, 1, 3, 6, 12, 24, 72),
) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    data_root = Path(data_root)
    files = sorted(data_root.glob("mld_argo_natl_era5_*.nc"))
    dynamic_variables = _dynamic_required_variables(config.dataset.dynamic_features)
    feature_variables = list(
        dict.fromkeys(
            [
                *config.dataset.targets,
                *dynamic_variables,
                *(feature for feature in config.dataset.static_features if feature not in COMPUTED_STATIC_VARS),
                *OCEAN_ALL_VARS,
                *EDDY_VARS,
            ]
        )
    )

    mask_counts: dict[str, int] = {
        "total": 0,
        "target_mld_n2": 0,
        "era5_strict": 0,
        "ocean_t0_core": 0,
        "ocean_any_core": 0,
        "ocean_t0_all": 0,
        "ocean_any_all": 0,
        "eddy_all": 0,
        "full_strict_t0_core_eddy": 0,
        "full_any_core_eddy": 0,
        "current_loader_filter": 0,
    }
    tolerance_counts = {int(threshold): 0 for threshold in missing_thresholds}
    feature_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []

    for primary_path in files:
        sources = _open_sources(primary_path)
        try:
            nstation = len(sources["primary"].dimensions["nstation"])
            month = _month_key(primary_path)
            mask_counts["total"] += nstation
            target = _target_mask(sources, config.dataset.targets, config.dataset.clip_mld_max)
            mask_counts["target_mld_n2"] += int(target.sum())

            era5 = target.copy()
            for variable in ERA5_BASE_VARS:
                era5 &= _mask_for_variable(sources, variable, "all", nstation)
            mask_counts["era5_strict"] += int(era5.sum())

            ocean_t0_core = target.copy()
            ocean_any_core = target.copy()
            for variable in OCEAN_CORE_VARS:
                ocean_t0_core &= _mask_for_variable(sources, variable, "t0", nstation)
                ocean_any_core &= _mask_for_variable(sources, variable, "any", nstation)
            mask_counts["ocean_t0_core"] += int(ocean_t0_core.sum())
            mask_counts["ocean_any_core"] += int(ocean_any_core.sum())

            ocean_t0_all = target.copy()
            ocean_any_all = target.copy()
            for variable in OCEAN_ALL_VARS:
                ocean_t0_all &= _mask_for_variable(sources, variable, "t0", nstation)
                ocean_any_all &= _mask_for_variable(sources, variable, "any", nstation)
            mask_counts["ocean_t0_all"] += int(ocean_t0_all.sum())
            mask_counts["ocean_any_all"] += int(ocean_any_all.sum())

            eddy = target.copy()
            for variable in EDDY_VARS:
                eddy &= _mask_for_variable(sources, variable, "all", nstation)
            mask_counts["eddy_all"] += int(eddy.sum())
            mask_counts["full_strict_t0_core_eddy"] += int((era5 & ocean_t0_core & eddy).sum())
            mask_counts["full_any_core_eddy"] += int((era5 & ocean_any_core & eddy).sum())

            current = target.copy()
            for variable in dynamic_variables:
                source = _source_with_variable(sources, variable)
                if source is None:
                    current &= False
                    continue
                values = _to_numpy(source[1].variables[variable][:])
                if values.ndim > 1 and values.shape[-1] in {10, 11}:
                    current &= _station_mask(values, "any")
                else:
                    current &= _station_mask(values, "all")
            mask_counts["current_loader_filter"] += int(current.sum())

            for threshold in tolerance_counts:
                tolerant = target.copy()
                for variable in dynamic_variables:
                    source = _source_with_variable(sources, variable)
                    if source is None:
                        tolerant &= False
                        continue
                    values = _to_numpy(source[1].variables[variable][:])
                    if values.ndim > 1 and values.shape[-1] in {10, 11}:
                        tolerant &= _station_mask(values, "any")
                    else:
                        tolerant &= _nan_count_mask(values, threshold)
                tolerance_counts[threshold] += int(tolerant.sum())

            month_rows.append(
                {
                    "month": month,
                    "nstation": nstation,
                    "target_valid": int(target.sum()),
                    "era5_strict": int(era5.sum()),
                    "ocean_any_core": int(ocean_any_core.sum()),
                    "eddy_all": int(eddy.sum()),
                    "current_loader_filter": int(current.sum()),
                }
            )

            for variable in feature_variables:
                source = _source_with_variable(sources, variable)
                if source is None:
                    missing_rows.append({"month": month, "variable": variable})
                    continue
                source_name, dataset = source
                values = _to_numpy(dataset.variables[variable][:])
                all_mask = _station_mask(values, "all")
                any_mask = _station_mask(values, "any")
                t0_mask = _station_mask(values, "t0")
                feature_rows.append(
                    {
                        "month": month,
                        "source": source_name,
                        "variable": variable,
                        "nstation": nstation,
                        "all_finite": int(all_mask.sum()),
                        "any_finite": int(any_mask.sum()),
                        "t0_finite": int(t0_mask.sum()),
                    }
                )
        finally:
            _close_sources(sources)

    counts = pd.DataFrame(
        [
            {
                "mask": name,
                "profiles": count,
                "percent_total": 100.0 * count / max(mask_counts["total"], 1),
                "percent_target_valid": 100.0 * count / max(mask_counts["target_mld_n2"], 1),
            }
            for name, count in mask_counts.items()
        ]
    )
    tolerance = pd.DataFrame(
        [
            {
                "max_missing_per_dynamic_series": threshold,
                "profiles": count,
                "gain_vs_strict": count - tolerance_counts.get(0, 0),
                "percent_total": 100.0 * count / max(mask_counts["total"], 1),
            }
            for threshold, count in tolerance_counts.items()
        ]
    )
    features = pd.DataFrame(feature_rows)
    if not features.empty:
        features["all_percent"] = 100.0 * features["all_finite"] / features["nstation"]
        features["any_percent"] = 100.0 * features["any_finite"] / features["nstation"]
        features["t0_percent"] = 100.0 * features["t0_finite"] / features["nstation"]

    return {
        "counts": counts,
        "tolerance": tolerance,
        "features": features,
        "missing": pd.DataFrame(missing_rows),
        "months": pd.DataFrame(month_rows),
    }
