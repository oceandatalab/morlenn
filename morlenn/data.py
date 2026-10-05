from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Any

import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset


STRICT_DYNAMIC_FEATURES = {
    "swh",
    "mwp",
    "shww",
    "mpww",
    "sshf",
    "slhf",
    "wind_speed",
    "wind_speed_neutral",
    "sea_slope",
}
STRICT_STATIC_FEATURES = {
    "geo_x",
    "geo_y",
    "geo_z",
    "elevation",
    "doy_sin",
    "doy_cos",
}
VARIABLE_ALIASES = {
    "N2": ("N2_rho0",),
    "N2_rho0": ("N2",),
    "u10n": ("u10",),
    "v10n": ("v10",),
    "wind_speed_neutral": ("wind_speed",),
}


def list_netcdf_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    era5_files = sorted(directory.glob("mld_argo_natl_era5_*.nc"))
    if era5_files:
        return era5_files
    return sorted(directory.glob("*.nc"))


@dataclass
class RobustStats:
    dynamic_median: np.ndarray
    dynamic_iqr: np.ndarray
    static_median: np.ndarray
    static_iqr: np.ndarray
    target_median: np.ndarray
    target_iqr: np.ndarray


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    return np.where(scale == 0.0, 1.0, scale).astype(np.float32)


def _safe_scale_tensor(scale: torch.Tensor) -> torch.Tensor:
    return torch.where(scale == 0.0, torch.ones_like(scale), scale)


def _slice_dynamic_history_tensor(dynamic: torch.Tensor, history_hours: int | None) -> torch.Tensor:
    if history_hours is None:
        return dynamic
    if history_hours <= 0:
        raise ValueError("dynamic_history_hours must be strictly positive.")
    steps = min(int(history_hours), int(dynamic.shape[-1]))
    return dynamic[..., -steps:]


def _slice_dynamic_history_array(dynamic: np.ndarray, history_hours: int | None) -> np.ndarray:
    if history_hours is None:
        return dynamic
    if history_hours <= 0:
        raise ValueError("dynamic_history_hours must be strictly positive.")
    steps = min(int(history_hours), int(dynamic.shape[-1]))
    return dynamic[..., -steps:]


def load_stats(path: str | Path) -> RobustStats:
    data = np.load(Path(path), allow_pickle=True)
    return RobustStats(
        dynamic_median=data["dynamic_median"].astype(np.float32),
        dynamic_iqr=data["dynamic_iqr"].astype(np.float32),
        static_median=data["static_median"].astype(np.float32),
        static_iqr=data["static_iqr"].astype(np.float32),
        target_median=data["target_median"].astype(np.float32),
        target_iqr=data["target_iqr"].astype(np.float32),
    )


def stats_to_dict(stats: RobustStats) -> dict[str, list[float]]:
    return {
        "dynamic_median": stats.dynamic_median.astype(np.float32).tolist(),
        "dynamic_iqr": stats.dynamic_iqr.astype(np.float32).tolist(),
        "static_median": stats.static_median.astype(np.float32).tolist(),
        "static_iqr": stats.static_iqr.astype(np.float32).tolist(),
        "target_median": stats.target_median.astype(np.float32).tolist(),
        "target_iqr": stats.target_iqr.astype(np.float32).tolist(),
    }


def stats_from_dict(data: dict[str, list[float]]) -> RobustStats:
    return RobustStats(
        dynamic_median=np.asarray(data["dynamic_median"], dtype=np.float32),
        dynamic_iqr=np.asarray(data["dynamic_iqr"], dtype=np.float32),
        static_median=np.asarray(data["static_median"], dtype=np.float32),
        static_iqr=np.asarray(data["static_iqr"], dtype=np.float32),
        target_median=np.asarray(data["target_median"], dtype=np.float32),
        target_iqr=np.asarray(data["target_iqr"], dtype=np.float32),
    )


def save_stats(path: str | Path, stats: RobustStats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dynamic_median=stats.dynamic_median,
        dynamic_iqr=stats.dynamic_iqr,
        static_median=stats.static_median,
        static_iqr=stats.static_iqr,
        target_median=stats.target_median,
        target_iqr=stats.target_iqr,
    )


def _apply_mld_target_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.astype(np.float32)
    if transform == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None)).astype(np.float32)
    raise ValueError(f"Unsupported MLD target transform: {transform}")


def _inverse_mld_target_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.astype(np.float32)
    if transform == "log1p":
        return np.expm1(np.clip(values, a_min=None, a_max=9.0)).astype(np.float32)
    raise ValueError(f"Unsupported MLD target transform: {transform}")


def _apply_mld_target_transform_tensor(values: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return values
    if transform == "log1p":
        return torch.log1p(values.clamp_min(0.0))
    raise ValueError(f"Unsupported MLD target transform: {transform}")


def _inverse_mld_target_transform_tensor(values: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return values
    if transform == "log1p":
        return torch.expm1(values.clamp_max(9.0))
    raise ValueError(f"Unsupported MLD target transform: {transform}")


def transform_targets(
    values: np.ndarray,
    target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
    mld_target_transform: str = "identity",
) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float32).copy()
    for index, name in enumerate(target_names):
        if name == "mld":
            transformed[..., index] = _apply_mld_target_transform(transformed[..., index], mld_target_transform)
    return transformed


def inverse_transform_targets(
    values: np.ndarray,
    target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
    mld_target_transform: str = "identity",
) -> np.ndarray:
    restored = np.asarray(values, dtype=np.float32).copy()
    for index, name in enumerate(target_names):
        if name == "mld":
            restored[..., index] = _inverse_mld_target_transform(restored[..., index], mld_target_transform)
    return restored


def denormalize_targets(
    values: np.ndarray,
    stats: RobustStats,
    target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
    mld_target_transform: str = "identity",
) -> np.ndarray:
    transformed = values * _safe_scale(stats.target_iqr) + stats.target_median
    return inverse_transform_targets(transformed, target_names=target_names, mld_target_transform=mld_target_transform)


def denormalize_predictive_std(
    values_std: np.ndarray,
    values_mean: np.ndarray,
    stats: RobustStats,
    target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
    mld_target_transform: str = "identity",
) -> np.ndarray:
    restored_std = np.asarray(values_std, dtype=np.float32) * _safe_scale(stats.target_iqr)
    restored_mean = np.asarray(values_mean, dtype=np.float32) * _safe_scale(stats.target_iqr) + stats.target_median
    outputs = restored_std.copy()
    for index, name in enumerate(target_names):
        if name != "mld":
            continue
        if mld_target_transform == "identity":
            outputs[..., index] = restored_std[..., index]
            continue
        if mld_target_transform == "log1p":
            sigma = np.clip(restored_std[..., index], a_min=0.0, a_max=3.0)
            mu = np.clip(restored_mean[..., index], a_min=None, a_max=9.0)
            exponent = np.clip(2.0 * mu + sigma**2, a_min=None, a_max=80.0)
            variance = np.expm1(sigma**2) * np.exp(exponent)
            outputs[..., index] = np.sqrt(np.clip(variance, a_min=0.0, a_max=None)).astype(np.float32)
            continue
        raise ValueError(f"Unsupported MLD target transform: {mld_target_transform}")
    return outputs


def normalize_targets(
    values: np.ndarray,
    stats: RobustStats,
    target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
    mld_target_transform: str = "identity",
) -> np.ndarray:
    transformed = transform_targets(values, target_names=target_names, mld_target_transform=mld_target_transform)
    return (transformed - stats.target_median) / _safe_scale(stats.target_iqr)


def denormalize_targets_tensor(
    values: torch.Tensor,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
    mld_target_transform: str = "identity",
) -> torch.Tensor:
    restored = values * _safe_scale_tensor(target_iqr).unsqueeze(0) + target_median.unsqueeze(0)
    columns: list[torch.Tensor] = []
    for index, name in enumerate(target_names):
        if name == "mld":
            columns.append(_inverse_mld_target_transform_tensor(restored[:, index], mld_target_transform))
        else:
            columns.append(restored[:, index])
    return torch.stack(columns, dim=-1)


def denormalize_predictive_std_tensor(
    values_std: torch.Tensor,
    values_mean: torch.Tensor,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
    mld_target_transform: str = "identity",
) -> torch.Tensor:
    restored_std = values_std * _safe_scale_tensor(target_iqr).unsqueeze(0)
    restored_mean = values_mean * _safe_scale_tensor(target_iqr).unsqueeze(0) + target_median.unsqueeze(0)
    columns: list[torch.Tensor] = []
    for index, name in enumerate(target_names):
        if mld_target_transform == "identity":
            columns.append(restored_std[:, index])
            continue
        if name == "mld" and mld_target_transform == "log1p":
            sigma = restored_std[:, index].clamp(min=0.0, max=3.0)
            mu = restored_mean[:, index].clamp_max(9.0)
            exponent = (2.0 * mu + sigma.square()).clamp_max(80.0)
            variance = torch.expm1(sigma.square()) * torch.exp(exponent)
            columns.append(torch.sqrt(variance.clamp_min(0.0)))
            continue
        if name != "mld":
            columns.append(restored_std[:, index])
            continue
        raise ValueError(f"Unsupported MLD target transform: {mld_target_transform}")
    return torch.stack(columns, dim=-1)


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ma.MaskedArray):
        array = array.filled(np.nan)
    return np.asarray(array, dtype=np.float32)


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return num / np.where(np.abs(den) < eps, eps, den)


def _day_of_year(year: int, month: int, day: int) -> int:
    return date(int(year), int(month), int(day)).timetuple().tm_yday


def _geo_xyz(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x = float(np.cos(lat_rad) * np.cos(lon_rad))
    y = float(np.cos(lat_rad) * np.sin(lon_rad))
    z = float(np.sin(lat_rad))
    return x, y, z


def _haversine_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    radius_km = 6371.0
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius_km * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0))))


def _is_finite_stationwise(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return np.isfinite(array)
    axes = tuple(range(1, array.ndim))
    return np.all(np.isfinite(array), axis=axes)


def _most_recent_finite_stationwise(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        return values
    flat = values.reshape(values.shape[0], -1)
    finite = np.isfinite(flat)
    reversed_index = np.argmax(finite[:, ::-1], axis=1)
    last_index = flat.shape[1] - 1 - reversed_index
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    has_finite = finite.any(axis=1)
    out[has_finite] = flat[has_finite, last_index[has_finite]]
    return out


def _most_recent_finite_value(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 0:
        return float(values.item())
    flat = values.reshape(-1)
    finite = np.isfinite(flat)
    if not finite.any():
        return float("nan")
    return float(flat[np.flatnonzero(finite)[-1]])


def _month_key(path: Path) -> str | None:
    match = re.search(r"(\d{6})", path.name)
    return match.group(1) if match else None


def _companion_paths(primary_path: Path) -> dict[str, Path]:
    resolved = primary_path.resolve()
    directory = resolved.parent
    key = _month_key(resolved)
    sources = {"primary": resolved}
    if key is None:
        return sources
    ocean = directory / f"mld_argo_natl_sst_ssh_sss_ssscnr_bat_{key}.nc"
    eddies = directory / f"eddies_{key}.nc"
    if ocean.exists():
        sources["ocean"] = ocean
    if eddies.exists():
        sources["eddy"] = eddies
    return sources


def _source_with_variable(sources: dict[str, netCDF4.Dataset], name: str) -> netCDF4.Dataset:
    for source_name in ("primary", "ocean", "eddy"):
        dataset = sources.get(source_name)
        if dataset is not None and _variable_name(dataset, name) is not None:
            return dataset
    available = sorted({variable for dataset in sources.values() for variable in dataset.variables})
    raise KeyError(f"Variable {name!r} not found in sources. Available variables: {available}")


def _variable_name(dataset: netCDF4.Dataset, name: str) -> str | None:
    if name in dataset.variables:
        return name
    for alias in VARIABLE_ALIASES.get(name, ()):
        if alias in dataset.variables:
            return alias
    return None


def _feature_index(features: list[str], name: str) -> int:
    if name in features:
        return features.index(name)
    for alias in VARIABLE_ALIASES.get(name, ()):
        if alias in features:
            return features.index(alias)
    raise ValueError(f"{name!r} is not in feature list {features!r}")


def _variable_values(dataset: netCDF4.Dataset, name: str, index: object = slice(None)) -> np.ndarray:
    variable_name = _variable_name(dataset, name)
    if variable_name is None:
        available = sorted(dataset.variables)
        raise KeyError(f"Variable {name!r} not found. Available variables: {available}")
    return _to_numpy(dataset.variables[variable_name][index])


def _interp_with_missing(x_new: np.ndarray, x_old: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(x_new.shape, np.nan, dtype=np.float32)
    if finite.sum() == 1:
        return np.full(x_new.shape, float(values[finite][0]), dtype=np.float32)
    return np.interp(
        x_new,
        x_old[finite],
        values[finite],
        left=values[finite][0],
        right=values[finite][-1],
    ).astype(np.float32)


def _resample_to_reference(values: np.ndarray, source_time: np.ndarray | None, reference_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        return np.full(reference_time.shape, float(values), dtype=np.float32)
    if values.shape[-1] == reference_time.shape[0]:
        return values.astype(np.float32)
    if values.shape[-1] in {10, 11} and reference_time.shape[0] >= 240:
        # Ocean-surface files store daily points from oldest to most recent.
        # Expand the available 10-day history to the recent end of the 720h
        # sequence; fill older unavailable hours with the oldest available value.
        if values.shape[-1] == 11:
            values = values[..., 1:]
        x_old = np.linspace(0.0, 239.0, num=values.shape[-1], dtype=np.float32)
        x_recent = np.arange(240, dtype=np.float32)
        if values.ndim == 1:
            recent = _interp_with_missing(x_recent, x_old, values)
            older_value = recent[0] if np.isfinite(recent[0]) else np.nan
            older = np.full(reference_time.shape[0] - 240, older_value, dtype=np.float32)
            return np.concatenate([older, recent], axis=0)
        flat = values.reshape(-1, values.shape[-1])
        recent = np.stack([_interp_with_missing(x_recent, x_old, row) for row in flat], axis=0).astype(np.float32)
        older = np.repeat(recent[:, :1], reference_time.shape[0] - 240, axis=1).astype(np.float32)
        return np.concatenate([older, recent], axis=1).reshape(*values.shape[:-1], reference_time.shape[0])
    if source_time is None:
        x_old = np.linspace(0.0, 1.0, num=values.shape[-1], dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=reference_time.shape[0], dtype=np.float32)
    else:
        x_old = np.asarray(source_time, dtype=np.float32)
        x_new = np.asarray(reference_time, dtype=np.float32)
    if values.ndim == 1:
        return _interp_with_missing(x_new, x_old, values)
    flat = values.reshape(-1, values.shape[-1])
    resampled = np.stack(
        [_interp_with_missing(x_new, x_old, row) for row in flat],
        axis=0,
    )
    return resampled.reshape(*values.shape[:-1], reference_time.shape[0]).astype(np.float32)


def compute_valid_station_indices(
    dataset: netCDF4.Dataset,
    dynamic_features: list[str],
    static_features: list[str],
    targets: list[str],
    clip_mld_max: float | None = None,
) -> np.ndarray:
    nstation = len(dataset.dimensions["nstation"])
    valid_mask = np.ones(nstation, dtype=bool)

    for target_name in targets:
        target_values = _variable_values(dataset, target_name)
        if target_name == "mld":
            if clip_mld_max is not None:
                target_values = np.minimum(target_values, clip_mld_max)
            valid_mask &= np.isfinite(target_values)
            valid_mask &= target_values > 0.0
        else:
            valid_mask &= np.isfinite(target_values)

    for feature_name in static_features:
        if feature_name in {"doy_sin", "doy_cos"}:
            year = _to_numpy(dataset.variables["year"][:])
            month = _to_numpy(dataset.variables["month"][:])
            day = _to_numpy(dataset.variables["day"][:])
            valid_mask &= np.isfinite(year)
            valid_mask &= np.isfinite(month)
            valid_mask &= np.isfinite(day)
            continue
        if feature_name in {"geo_x", "geo_y", "geo_z"}:
            lat = _variable_values(dataset, "lat")
            lon = _variable_values(dataset, "lon")
            valid_mask &= np.isfinite(lat)
            valid_mask &= np.isfinite(lon)
            continue
        feature_values = _variable_values(dataset, feature_name)
        if feature_values.ndim > 1:
            feature_values = _most_recent_finite_stationwise(feature_values)
        valid_mask &= np.isfinite(feature_values)

    for feature_name in dynamic_features:
        if feature_name in {"wind_speed", "wind_speed_neutral"}:
            u_name, v_name = ("u10n", "v10n") if feature_name == "wind_speed_neutral" else ("u10", "v10")
            valid_mask &= _is_finite_stationwise(_variable_values(dataset, u_name))
            valid_mask &= _is_finite_stationwise(_variable_values(dataset, v_name))
            continue
        if feature_name == "sea_slope":
            swh = _variable_values(dataset, "swh")
            mwp = _variable_values(dataset, "mwp")
            valid_mask &= _is_finite_stationwise(swh)
            valid_mask &= _is_finite_stationwise(mwp)
            continue
        feature_values = _variable_values(dataset, feature_name)
        valid_mask &= _is_finite_stationwise(feature_values)

    return np.flatnonzero(valid_mask).astype(np.int64)


class MonthlyNetCDFDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        files: list[Path],
        dynamic_features: list[str],
        static_features: list[str],
        targets: list[str],
        mld_target_transform: str = "identity",
        stats: RobustStats | None = None,
        replace_nan_with_zero: bool = True,
        clip_mld_max: float | None = None,
        dynamic_history_hours: int | None = None,
        strict_static_features: list[str] | tuple[str, ...] | None = None,
        eddy_max_distance_radius: float | None = None,
    ) -> None:
        if not files:
            raise ValueError("No NetCDF files found.")
        self.files: list[Path] = []
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        self.targets = targets
        self.mld_target_transform = mld_target_transform
        self.stats = stats
        self.replace_nan_with_zero = replace_nan_with_zero
        self.clip_mld_max = clip_mld_max
        self.dynamic_history_hours = dynamic_history_hours
        self.strict_static_features = set(strict_static_features or ())
        self.eddy_max_distance_radius = eddy_max_distance_radius
        self._datasets: dict[Path, netCDF4.Dataset] = {}
        self._file_sources: list[dict[str, Path]] = []
        self._valid_station_indices: list[np.ndarray] = []
        self._offsets: list[int] = []
        total = 0
        for path in [Path(path) for path in files]:
            sources = _companion_paths(path)
            datasets = {name: netCDF4.Dataset(source_path, "r") for name, source_path in sources.items()}
            try:
                try:
                    if len(datasets) == 1:
                        valid_station_indices = compute_valid_station_indices(
                            dataset=datasets["primary"],
                            dynamic_features=self.dynamic_features,
                            static_features=self.static_features,
                            targets=self.targets,
                            clip_mld_max=self.clip_mld_max,
                        )
                    else:
                        valid_station_indices = self._compute_valid_station_indices_for_sources(datasets)
                except KeyError as exc:
                    print(f"Skipping {path}: {exc}")
                    continue
            finally:
                for dataset in datasets.values():
                    dataset.close()
            if len(valid_station_indices) == 0:
                continue
            self.files.append(Path(path))
            self._file_sources.append(sources)
            self._valid_station_indices.append(valid_station_indices)
            total += len(valid_station_indices)
            self._offsets.append(total)
        if not self.files:
            raise ValueError("No valid NetCDF samples found after filtering NaNs and non-positive MLD.")

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, index: int) -> tuple[int, Path, int]:
        file_idx = bisect_right(self._offsets, index)
        previous = 0 if file_idx == 0 else self._offsets[file_idx - 1]
        station_offset = index - previous
        return file_idx, self.files[file_idx], int(self._valid_station_indices[file_idx][station_offset])

    def _dataset(self, path: Path) -> netCDF4.Dataset:
        path = path.resolve()
        dataset = self._datasets.get(path)
        if dataset is None:
            dataset = netCDF4.Dataset(path, "r")
            self._datasets[path] = dataset
        return dataset

    def _source_datasets(self, file_idx: int) -> dict[str, netCDF4.Dataset]:
        return {name: self._dataset(path) for name, path in self._file_sources[file_idx].items()}

    def __del__(self) -> None:
        for dataset in self._datasets.values():
            try:
                dataset.close()
            except Exception:
                pass

    def _reference_time(self, sources: dict[str, netCDF4.Dataset]) -> np.ndarray:
        primary = sources["primary"]
        if "hours_before" in primary.variables:
            return _to_numpy(primary.variables["hours_before"][:])
        first_dynamic = _source_with_variable(sources, self.dynamic_features[0])
        return np.arange(first_dynamic.variables[self.dynamic_features[0]].shape[-1], dtype=np.float32)

    def _compute_valid_station_indices_for_sources(self, sources: dict[str, netCDF4.Dataset]) -> np.ndarray:
        nstation = len(sources["primary"].dimensions["nstation"])
        valid_mask = np.ones(nstation, dtype=bool)

        for dataset in sources.values():
            if len(dataset.dimensions["nstation"]) != nstation:
                raise ValueError("Companion NetCDF files must share the same nstation dimension.")

        for target_name in self.targets:
            dataset = _source_with_variable(sources, target_name)
            target_values = _variable_values(dataset, target_name)
            if target_name == "mld":
                if self.clip_mld_max is not None:
                    target_values = np.minimum(target_values, self.clip_mld_max)
                valid_mask &= np.isfinite(target_values)
                valid_mask &= target_values > 0.0
            else:
                valid_mask &= np.isfinite(target_values)

        for feature_name in self.static_features:
            strict_static = feature_name in STRICT_STATIC_FEATURES or feature_name in self.strict_static_features
            if feature_name in {"doy_sin", "doy_cos"}:
                if not strict_static:
                    continue
                dataset = sources["primary"]
                year = _to_numpy(dataset.variables["year"][:])
                month = _to_numpy(dataset.variables["month"][:])
                day = _to_numpy(dataset.variables["day"][:])
                valid_mask &= np.isfinite(year)
                valid_mask &= np.isfinite(month)
                valid_mask &= np.isfinite(day)
                continue
            if feature_name in {"geo_x", "geo_y", "geo_z"}:
                if not strict_static:
                    continue
                dataset = sources["primary"]
                lat = _to_numpy(dataset.variables["lat"][:])
                lon = _to_numpy(dataset.variables["lon"][:])
                valid_mask &= np.isfinite(lat)
                valid_mask &= np.isfinite(lon)
                continue
            dataset = _source_with_variable(sources, feature_name)
            if not strict_static and self.replace_nan_with_zero:
                continue
            values = _variable_values(dataset, feature_name)
            if values.ndim > 1:
                values = _most_recent_finite_stationwise(values)
            valid_mask &= np.isfinite(values)

        for feature_name in self.dynamic_features:
            strict_dynamic = feature_name in STRICT_DYNAMIC_FEATURES
            if feature_name in {"wind_speed", "wind_speed_neutral"}:
                u_name, v_name = ("u10n", "v10n") if feature_name == "wind_speed_neutral" else ("u10", "v10")
                dataset = _source_with_variable(sources, u_name)
                valid_mask &= _is_finite_stationwise(_variable_values(dataset, u_name))
                valid_mask &= _is_finite_stationwise(_variable_values(dataset, v_name))
                continue
            if feature_name == "sea_slope":
                dataset = _source_with_variable(sources, "swh")
                valid_mask &= _is_finite_stationwise(_to_numpy(dataset.variables["swh"][:]))
                valid_mask &= _is_finite_stationwise(_to_numpy(dataset.variables["mwp"][:]))
                continue
            dataset = _source_with_variable(sources, feature_name)
            if not strict_dynamic and self.replace_nan_with_zero:
                continue
            values = _variable_values(dataset, feature_name)
            valid_mask &= _is_finite_stationwise(values)

        return np.flatnonzero(valid_mask).astype(np.int64)

    def _dynamic_feature(self, sources: dict[str, netCDF4.Dataset], station_idx: int, name: str) -> np.ndarray:
        reference_time = self._reference_time(sources)
        if name == "sea_slope":
            dataset = _source_with_variable(sources, "swh")
            swh = _to_numpy(dataset.variables["swh"][station_idx, :])
            mwp = _to_numpy(dataset.variables["mwp"][station_idx, :])
            return _safe_ratio(swh, mwp)
        if name in {"wind_speed", "wind_speed_neutral"}:
            u_name, v_name = ("u10n", "v10n") if name == "wind_speed_neutral" else ("u10", "v10")
            dataset = _source_with_variable(sources, u_name)
            u = _variable_values(dataset, u_name, (station_idx, slice(None)))
            v = _variable_values(dataset, v_name, (station_idx, slice(None)))
            return np.sqrt(u ** 2 + v ** 2).astype(np.float32)
        dataset = _source_with_variable(sources, name)
        values = _variable_values(dataset, name, station_idx)
        if values.ndim == 0:
            return np.full(reference_time.shape, float(values), dtype=np.float32)
        source_time = _to_numpy(dataset.variables["hours_before"][:]) if "hours_before" in dataset.variables else None
        return _resample_to_reference(values, source_time, reference_time)

    def _static_feature(self, sources: dict[str, netCDF4.Dataset], station_idx: int, name: str) -> float:
        if name == "doy_sin":
            dataset = sources["primary"]
            doy = _day_of_year(
                int(dataset.variables["year"][station_idx]),
                int(dataset.variables["month"][station_idx]),
                int(dataset.variables["day"][station_idx]),
            )
            return float(np.sin(2.0 * np.pi * doy / 365.0))
        if name == "doy_cos":
            dataset = sources["primary"]
            doy = _day_of_year(
                int(dataset.variables["year"][station_idx]),
                int(dataset.variables["month"][station_idx]),
                int(dataset.variables["day"][station_idx]),
            )
            return float(np.cos(2.0 * np.pi * doy / 365.0))
        if name in {"geo_x", "geo_y", "geo_z"}:
            dataset = sources["primary"]
            lat = float(_to_numpy(dataset.variables["lat"][station_idx]).item())
            lon = float(_to_numpy(dataset.variables["lon"][station_idx]).item())
            x, y, z = _geo_xyz(lat, lon)
            if name == "geo_x":
                return x
            if name == "geo_y":
                return y
            return z
        if name.startswith("eddy_") and not self._eddy_match_is_admissible(sources, station_idx):
            return 0.0
        dataset = _source_with_variable(sources, name)
        value = _variable_values(dataset, name, station_idx)
        return _most_recent_finite_value(value)

    def _eddy_match_is_admissible(self, sources: dict[str, netCDF4.Dataset], station_idx: int) -> bool:
        if self.eddy_max_distance_radius is None:
            return True
        eddy = sources.get("eddy")
        primary = sources.get("primary")
        if eddy is None or primary is None:
            return True
        try:
            eddy_type = float(_variable_values(eddy, "eddy_type", station_idx).item())
            if eddy_type == 0.0:
                return True
            profile_lat = float(_variable_values(primary, "lat", station_idx).item())
            profile_lon = float(_variable_values(primary, "lon", station_idx).item())
            center_lat = float(_variable_values(eddy, "eddy_center_lat", station_idx).item())
            center_lon = float(_variable_values(eddy, "eddy_center_lon", station_idx).item())
            radius = float(_variable_values(eddy, "eddy_effective_radius", station_idx).item())
        except Exception:
            return False
        if not all(np.isfinite(value) for value in [profile_lat, profile_lon, center_lat, center_lon, radius]):
            return False
        if radius <= 0.0:
            return False
        radius_km = radius / 1000.0 if radius > 1000.0 else radius
        if radius_km <= 0.0:
            return False
        distance_over_radius = _haversine_km(profile_lat, profile_lon, center_lat, center_lon) / radius_km
        return bool(distance_over_radius <= self.eddy_max_distance_radius)

    def _target(self, sources: dict[str, netCDF4.Dataset], station_idx: int, name: str) -> float:
        dataset = _source_with_variable(sources, name)
        value = float(_variable_values(dataset, name, station_idx).item())
        if name == "mld" and self.clip_mld_max is not None:
            value = min(value, self.clip_mld_max)
        return value

    def _sample_raw(self, file_idx: int, station_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sources = self._source_datasets(file_idx)
        dynamic = np.stack(
            [self._dynamic_feature(sources, station_idx, feature) for feature in self.dynamic_features],
            axis=0,
        ).astype(np.float32)
        static = np.asarray(
            [self._static_feature(sources, station_idx, feature) for feature in self.static_features],
            dtype=np.float32,
        )
        target = np.asarray(
            [self._target(sources, station_idx, feature) for feature in self.targets],
            dtype=np.float32,
        )
        return dynamic, static, target

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, _, station_idx = self._locate(index)
        dynamic, static, target = self._sample_raw(file_idx, station_idx)

        if self.stats is not None:
            dynamic = (dynamic - self.stats.dynamic_median[:, None]) / np.where(
                self.stats.dynamic_iqr[:, None] == 0.0,
                1.0,
                self.stats.dynamic_iqr[:, None],
            )
            static = (static - self.stats.static_median) / np.where(
                self.stats.static_iqr == 0.0,
                1.0,
                self.stats.static_iqr,
            )
            target = normalize_targets(
                target[None, :],
                stats=self.stats,
                target_names=self.targets,
                mld_target_transform=self.mld_target_transform,
            )
            target = target[0]

        if self.replace_nan_with_zero:
            dynamic = np.nan_to_num(dynamic, nan=0.0, posinf=0.0, neginf=0.0)
            static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        dynamic = _slice_dynamic_history_array(dynamic, self.dynamic_history_hours)

        return {
            "dynamic": torch.from_numpy(dynamic),
            "static": torch.from_numpy(static),
            "target": torch.from_numpy(target),
        }


class PrecomputedTensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        directory: str | Path,
        dynamic_features: list[str] | None = None,
        static_features: list[str] | None = None,
        stats: RobustStats | None = None,
        target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
        mld_target_transform: str = "identity",
        dynamic_history_hours: int | None = None,
        mld_subset_min: float | None = None,
        mld_subset_max: float | None = None,
        eddy_balance: bool = False,
        eddy_balance_seed: int = 42,
        neighbor_path: str | Path | None = None,
    ) -> None:
        directory = Path(directory)
        self.dynamic_indices: torch.Tensor | None = None
        self.static_indices: torch.Tensor | None = None
        self.dynamic_history_hours = dynamic_history_hours
        self.single_file = directory / "dataset.pt"
        self._single_payload: dict[str, torch.Tensor] | None = None
        if self.single_file.exists():
            metadata_path = directory / "metadata.pt"
            metadata = torch.load(metadata_path, map_location="cpu") if metadata_path.exists() else {}
            source_dynamic_features = list(metadata.get("dynamic_features", dynamic_features or []))
            source_static_features = list(metadata.get("static_features", static_features or []))
            source_target_names = list(metadata.get("targets", target_names))
            source_stats_path = Path(metadata["stats_path"]) if metadata.get("stats_path") else None
            source_stats = load_stats(source_stats_path) if source_stats_path is not None and source_stats_path.exists() else None
            if metadata_path.exists():
                if dynamic_features is not None and "dynamic_features" in metadata:
                    self.dynamic_indices = torch.as_tensor(
                        [_feature_index(source_dynamic_features, name) for name in dynamic_features],
                        dtype=torch.long,
                    )
                if static_features is not None and "static_features" in metadata:
                    self.static_indices = torch.as_tensor(
                        [_feature_index(source_static_features, name) for name in static_features],
                        dtype=torch.long,
                    )
            payload = torch.load(self.single_file, map_location="cpu")
            dynamic = payload["dynamic"]
            static = payload["static"]
            target = payload["target"]
            if self.dynamic_indices is not None:
                dynamic = dynamic.index_select(1, self.dynamic_indices)
            if self.static_indices is not None:
                static = static.index_select(1, self.static_indices)
            if stats is not None and source_stats is not None:
                dynamic_source_idx = self.dynamic_indices if self.dynamic_indices is not None else torch.arange(dynamic.shape[1])
                static_source_idx = self.static_indices if self.static_indices is not None else torch.arange(static.shape[1])
                source_dynamic_median = torch.as_tensor(source_stats.dynamic_median, dtype=torch.float32).index_select(0, dynamic_source_idx)
                source_dynamic_iqr = torch.as_tensor(_safe_scale(source_stats.dynamic_iqr), dtype=torch.float32).index_select(0, dynamic_source_idx)
                current_dynamic_median = torch.as_tensor(stats.dynamic_median, dtype=torch.float32)
                current_dynamic_iqr = torch.as_tensor(_safe_scale(stats.dynamic_iqr), dtype=torch.float32)
                dynamic_raw = dynamic * source_dynamic_iqr[None, :, None] + source_dynamic_median[None, :, None]
                dynamic = (dynamic_raw - current_dynamic_median[None, :, None]) / current_dynamic_iqr[None, :, None]

                source_static_median = torch.as_tensor(source_stats.static_median, dtype=torch.float32).index_select(0, static_source_idx)
                source_static_iqr = torch.as_tensor(_safe_scale(source_stats.static_iqr), dtype=torch.float32).index_select(0, static_source_idx)
                current_static_median = torch.as_tensor(stats.static_median, dtype=torch.float32)
                current_static_iqr = torch.as_tensor(_safe_scale(stats.static_iqr), dtype=torch.float32)
                static_raw = static * source_static_iqr[None, :] + source_static_median[None, :]
                static = (static_raw - current_static_median[None, :]) / current_static_iqr[None, :]

                source_target_median = torch.as_tensor(source_stats.target_median, dtype=torch.float32)
                source_target_iqr = torch.as_tensor(source_stats.target_iqr, dtype=torch.float32)
                target_raw = denormalize_targets_tensor(
                    target,
                    target_median=source_target_median,
                    target_iqr=source_target_iqr,
                    target_names=source_target_names,
                    mld_target_transform=str(metadata.get("mld_target_transform", mld_target_transform)),
                )
                target_transformed = target_raw.clone()
                for index, name in enumerate(target_names):
                    if name == "mld":
                        target_transformed[:, index] = _apply_mld_target_transform_tensor(
                            target_transformed[:, index],
                            mld_target_transform,
                        )
                current_target_median = torch.as_tensor(stats.target_median, dtype=torch.float32)
                current_target_iqr = torch.as_tensor(_safe_scale(stats.target_iqr), dtype=torch.float32)
                target = (target_transformed - current_target_median[None, :]) / current_target_iqr[None, :]
            dynamic = _slice_dynamic_history_tensor(dynamic, self.dynamic_history_hours)
            self._single_payload = {"dynamic": dynamic, "static": static, "target": target}
            if (mld_subset_min is not None or mld_subset_max is not None) and stats is not None:
                mld_index = list(target_names).index("mld")
                mld_phys = denormalize_targets_tensor(
                    self._single_payload["target"],
                    target_median=torch.as_tensor(stats.target_median, dtype=torch.float32),
                    target_iqr=torch.as_tensor(stats.target_iqr, dtype=torch.float32),
                    target_names=list(target_names),
                    mld_target_transform=mld_target_transform,
                )[:, mld_index]
                mask = torch.ones(mld_phys.shape[0], dtype=torch.bool)
                if mld_subset_min is not None:
                    mask &= mld_phys > float(mld_subset_min)
                if mld_subset_max is not None:
                    mask &= mld_phys <= float(mld_subset_max)
                keep = torch.nonzero(mask, as_tuple=False).squeeze(1)
                for key in ("dynamic", "static", "target"):
                    self._single_payload[key] = self._single_payload[key].index_select(0, keep)
                print(
                    f"MLD subset filter [{mld_subset_min}, {mld_subset_max}] m: "
                    f"kept {int(keep.numel())} / {int(mld_phys.shape[0])} samples",
                    flush=True,
                )
            if eddy_balance and stats is not None and static_features is not None and "eddy_type" in static_features:
                et_idx = list(static_features).index("eddy_type")
                et_iqr = float(_safe_scale(torch.as_tensor(stats.static_iqr, dtype=torch.float32))[et_idx])
                et_med = float(torch.as_tensor(stats.static_median, dtype=torch.float32)[et_idx])
                eddy_type_raw = self._single_payload["static"][:, et_idx] * et_iqr + et_med
                inside = eddy_type_raw.abs() > 0.5  # eddy_type != 0 => admissible eddy within radius 1
                inside_idx = torch.nonzero(inside, as_tuple=False).squeeze(1)
                outside_idx = torch.nonzero(~inside, as_tuple=False).squeeze(1)
                k = int(inside_idx.numel())
                if 0 < k <= int(outside_idx.numel()):
                    gen = torch.Generator().manual_seed(int(eddy_balance_seed))
                    sampled_outside = outside_idx[torch.randperm(outside_idx.numel(), generator=gen)[:k]]
                    keep = torch.cat([inside_idx, sampled_outside])
                    keep = keep[torch.randperm(keep.numel(), generator=gen)]
                    for key in ("dynamic", "static", "target"):
                        self._single_payload[key] = self._single_payload[key].index_select(0, keep)
                    print(
                        f"Eddy balance: inside={k} + outside(sampled)={k} -> "
                        f"{int(keep.numel())} / {int(inside.numel())} samples (50/50)",
                        flush=True,
                    )
                else:
                    print(f"Eddy balance skipped: inside={k}, outside={int(outside_idx.numel())}", flush=True)
            if neighbor_path is not None:
                npz = np.load(str(neighbor_path))
                feat = torch.as_tensor(npz["feat"], dtype=torch.float32)
                nb_mask = torch.as_tensor(npz["mask"], dtype=torch.float32)
                n_payload = int(self._single_payload["target"].shape[0])
                if feat.shape[0] != n_payload:
                    raise ValueError(
                        f"Neighbor file {neighbor_path} has {feat.shape[0]} rows but payload has {n_payload} "
                        "(neighbor matrices must be aligned to the unfiltered precompute order)."
                    )
                self._single_payload["neighbors"] = feat
                self._single_payload["neighbor_mask"] = nb_mask
                print(f"Loaded neighbors {tuple(feat.shape)} from {neighbor_path}", flush=True)
            self.files = []
            self._cache = {}
            self._offsets = [int(self._single_payload["target"].shape[0])]
            return
        if mld_subset_min is not None or mld_subset_max is not None or eddy_balance:
            raise NotImplementedError(
                "mld_subset_min/mld_subset_max/eddy_balance are only supported for single-file (dataset.pt) precomputed datasets."
            )
        self.files = sorted(directory.glob("shard_*.pt"))
        if not self.files:
            raise ValueError(f"No precomputed shards found in {directory}")
        self._cache: dict[int, dict[str, torch.Tensor]] = {}
        self._offsets: list[int] = []
        total = 0
        for path in self.files:
            payload = torch.load(path, map_location="cpu")
            count = int(payload["target"].shape[0])
            total += count
            self._offsets.append(total)

    def __len__(self) -> int:
        return self._offsets[-1]

    @property
    def single_payload(self) -> dict[str, torch.Tensor] | None:
        return self._single_payload

    def subset(self, indices) -> "PrecomputedTensorDataset":
        """Return a new single-file dataset restricted to `indices` (into the current order).

        Shares the feature-subset/normalization already applied; slices every payload
        tensor (dynamic/static/target and neighbors/neighbor_mask if present). Lets a single
        global precompute be split into train/val at load time while keeping the fast path.
        """
        if self._single_payload is None:
            raise NotImplementedError("subset() requires a single-file (dataset.pt) precomputed dataset.")
        idx = torch.as_tensor(indices, dtype=torch.long)
        new = PrecomputedTensorDataset.__new__(PrecomputedTensorDataset)
        new.dynamic_indices = self.dynamic_indices
        new.static_indices = self.static_indices
        new.dynamic_history_hours = self.dynamic_history_hours
        new.single_file = self.single_file
        new.files = []
        new._cache = {}
        new._single_payload = {key: value.index_select(0, idx) for key, value in self._single_payload.items()}
        new._offsets = [int(new._single_payload["target"].shape[0])]
        return new

    @property
    def shard_bounds(self) -> list[tuple[int, int]]:
        bounds: list[tuple[int, int]] = []
        previous = 0
        for offset in self._offsets:
            bounds.append((previous, offset))
            previous = offset
        return bounds

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        cached = self._cache.get(shard_idx)
        if cached is None:
            cached = torch.load(self.files[shard_idx], map_location="cpu")
            self._cache = {shard_idx: cached}
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._single_payload is not None:
            sample = {
                "dynamic": self._single_payload["dynamic"][index],
                "static": self._single_payload["static"][index],
                "target": self._single_payload["target"][index],
            }
            if "neighbors" in self._single_payload:
                sample["neighbors"] = self._single_payload["neighbors"][index]
                sample["neighbor_mask"] = self._single_payload["neighbor_mask"][index]
            return sample
        shard_idx = bisect_right(self._offsets, index)
        previous = 0 if shard_idx == 0 else self._offsets[shard_idx - 1]
        local_idx = index - previous
        shard = self._load_shard(shard_idx)
        dynamic = shard["dynamic"][local_idx]
        static = shard["static"][local_idx]
        if self.dynamic_indices is not None:
            dynamic = dynamic.index_select(0, self.dynamic_indices)
        if self.static_indices is not None:
            static = static.index_select(0, self.static_indices)
        dynamic = _slice_dynamic_history_tensor(dynamic, self.dynamic_history_hours)
        return {
            "dynamic": dynamic,
            "static": static,
            "target": shard["target"][local_idx],
        }
