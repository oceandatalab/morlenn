from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm.auto import tqdm

from .config import ExperimentConfig, load_config
from .data import (
    MonthlyNetCDFDataset,
    PrecomputedTensorDataset,
    denormalize_targets_tensor,
    list_netcdf_files,
    load_stats,
    stats_to_dict,
)
from .losses import WeightedUpperDynLoss
from .model import UpperDynTCNAttentionModel, split_prediction_params


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = deterministic


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return unwrap_model(model).state_dict()


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def main_process() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


def setup_distributed() -> tuple[bool, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.destroy_process_group()


def gather_distributed_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not distributed_is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def reduce_distributed_sum(value: float, device: torch.device) -> float:
    if not distributed_is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


class ShardedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: PrecomputedTensorDataset, batch_size: int, drop_last: bool = False) -> None:
        self.bounds = dataset.shard_bounds
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        shard_order = list(range(len(self.bounds)))
        random.shuffle(shard_order)
        for shard_idx in shard_order:
            start, end = self.bounds[shard_idx]
            indices = list(range(start, end))
            random.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for start, end in self.bounds:
            shard_len = end - start
            if self.drop_last:
                total += shard_len // self.batch_size
            else:
                total += (shard_len + self.batch_size - 1) // self.batch_size
        return total


class PrecomputedBatchLoader:
    def __init__(
        self,
        dataset: PrecomputedTensorDataset,
        batch_size: int,
        shuffle: bool,
        distributed: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        payload = dataset.single_payload
        if payload is None:
            raise ValueError("PrecomputedBatchLoader requires a single-file precomputed dataset.")
        self.payload = payload
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = int(payload["target"].shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _indices(self) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.num_samples, generator=generator)
        else:
            indices = torch.arange(self.num_samples)
        if not self.distributed:
            return indices
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        per_rank = self.num_samples // world_size if self.drop_last else (self.num_samples + world_size - 1) // world_size
        total_size = per_rank * world_size
        if total_size > self.num_samples:
            indices = torch.cat([indices, indices[: total_size - self.num_samples]], dim=0)
        else:
            indices = indices[:total_size]
        return indices[rank:total_size:world_size].contiguous()

    def __iter__(self):
        indices = self._indices()
        for offset in range(0, len(indices), self.batch_size):
            batch_indices = indices[offset : offset + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            yield {key: value.index_select(0, batch_indices) for key, value in self.payload.items()}

    def __len__(self) -> int:
        if self.distributed:
            n = self.num_samples // dist.get_world_size() if self.drop_last else (self.num_samples + dist.get_world_size() - 1) // dist.get_world_size()
        else:
            n = self.num_samples
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.detach()
    target = target.detach()
    error = prediction - target
    rmse = error.square().mean(dim=0).sqrt()
    bias = error.mean(dim=0)
    pred_std = prediction.std(dim=0, unbiased=False)
    target_std = target.std(dim=0, unbiased=False)
    pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    corr = (pred_centered * target_centered).mean(dim=0) / (
        pred_std * target_std + 1e-6
    )
    return {
        "rmse_mean": float(rmse.mean()),
        "bias_mean": float(bias.mean()),
        "corr_mean": float(corr.mean()),
        "std_ratio_mean": float((pred_std / (target_std + 1e-6)).mean()),
        "mld_rmse": float(rmse[0]),
        "n2_rmse": float(rmse[1]),
        "mld_bias": float(bias[0]),
        "n2_bias": float(bias[1]),
        "mld_corr": float(corr[0]),
        "n2_corr": float(corr[1]),
        "mld_std_ratio": float(pred_std[0] / (target_std[0] + 1e-6)),
        "n2_std_ratio": float(pred_std[1] / (target_std[1] + 1e-6)),
    }


def physical_regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    target_names: list[str],
    mld_target_transform: str,
) -> dict[str, float]:
    prediction = prediction.detach()
    target = target.detach()
    prediction_phys = denormalize_targets_tensor(
        prediction,
        target_median=target_median,
        target_iqr=target_iqr,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )
    target_phys = denormalize_targets_tensor(
        target,
        target_median=target_median,
        target_iqr=target_iqr,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )
    error = prediction_phys - target_phys
    rmse = error.square().mean(dim=0).sqrt()
    bias = error.mean(dim=0)
    return {
        "mld_rmse_phys": float(rmse[0]),
        "n2_rmse_phys": float(rmse[1]),
        "mld_bias_phys": float(bias[0]),
        "n2_bias_phys": float(bias[1]),
    }


def mld_tail_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    target_names: list[str],
    mld_target_transform: str,
    quantile: float,
) -> dict[str, float]:
    prediction_phys = denormalize_targets_tensor(
        prediction.detach(),
        target_median=target_median,
        target_iqr=target_iqr,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )[:, 0]
    target_phys = denormalize_targets_tensor(
        target.detach(),
        target_median=target_median,
        target_iqr=target_iqr,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )[:, 0]
    threshold = torch.quantile(target_phys, quantile)
    mask = target_phys >= threshold
    pred_tail = prediction_phys[mask]
    target_tail = target_phys[mask]
    error = pred_tail - target_tail
    pred_std = pred_tail.std(unbiased=False)
    target_std = target_tail.std(unbiased=False)
    return {
        "mld_tail_quantile": float(quantile),
        "mld_tail_threshold_phys": float(threshold),
        "mld_tail_count": int(mask.sum()),
        "mld_tail_rmse_phys": float(error.square().mean().sqrt()),
        "mld_tail_bias_phys": float(error.mean()),
        "mld_tail_std_ratio": float(pred_std / (target_std + 1e-6)),
    }


def predictive_std_metrics(
    prediction_mean: torch.Tensor,
    prediction_std: torch.Tensor,
    target: torch.Tensor,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    target_names: list[str],
    mld_target_transform: str,
) -> dict[str, float]:
    prediction_std_phys = prediction_std.detach().clamp_min(1e-6)
    prediction_mean_phys = denormalize_targets_tensor(
        prediction_mean.detach(),
        target_median=target_median,
        target_iqr=target_iqr,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )
    target_phys = denormalize_targets_tensor(
        target.detach(),
        target_median=target_median,
        target_iqr=target_iqr,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )
    abs_error_phys = (prediction_mean_phys - target_phys).abs()
    one_sigma_coverage = (abs_error_phys <= prediction_std_phys.clamp_min(1e-6)).float().mean(dim=0)
    return {
        "mld_pred_std_phys_mean": float(prediction_std_phys[:, 0].mean()),
        "n2_pred_std_phys_mean": float(prediction_std_phys[:, 1].mean()),
        "mld_one_sigma_coverage": float(one_sigma_coverage[0]),
        "n2_one_sigma_coverage": float(one_sigma_coverage[1]),
    }


def missing_prediction_metrics() -> dict[str, float]:
    value = float("nan")
    return {
        "rmse_mean": value,
        "bias_mean": value,
        "corr_mean": value,
        "std_ratio_mean": value,
        "mld_rmse": value,
        "n2_rmse": value,
        "mld_bias": value,
        "n2_bias": value,
        "mld_corr": value,
        "n2_corr": value,
        "mld_std_ratio": value,
        "n2_std_ratio": value,
        "mld_rmse_phys": value,
        "n2_rmse_phys": value,
        "mld_bias_phys": value,
        "n2_bias_phys": value,
        "mld_tail_quantile": value,
        "mld_tail_threshold_phys": value,
        "mld_tail_count": 0,
        "mld_tail_rmse_phys": value,
        "mld_tail_bias_phys": value,
        "mld_tail_std_ratio": value,
    }


def clamp_normalized_mld_to_bathymetry(
    prediction: torch.Tensor,
    static: torch.Tensor,
    static_median: torch.Tensor,
    static_iqr: torch.Tensor,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    static_features: list[str],
    target_names: list[str],
    mld_target_transform: str,
) -> torch.Tensor:
    if "elevation" not in static_features or "mld" not in target_names:
        return prediction
    static_median = static_median.to(device=static.device, dtype=static.dtype)
    static_iqr = static_iqr.to(device=static.device, dtype=static.dtype)
    target_median = target_median.to(device=prediction.device, dtype=prediction.dtype)
    target_iqr = target_iqr.to(device=prediction.device, dtype=prediction.dtype)
    elevation_index = static_features.index("elevation")
    mld_index = target_names.index("mld")
    elevation = static[:, elevation_index] * torch.where(
        static_iqr[elevation_index] == 0.0,
        torch.ones_like(static_iqr[elevation_index]),
        static_iqr[elevation_index],
    ) + static_median[elevation_index]
    ocean_mask = torch.isfinite(elevation) & (elevation < 0.0)
    if not bool(ocean_mask.any()):
        return prediction

    prediction_phys = denormalize_targets_tensor(
        prediction,
        target_median=target_median,
        target_iqr=target_iqr,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )
    bathy_depth = -elevation[ocean_mask]
    clamped_mld = torch.minimum(prediction_phys[ocean_mask, mld_index], bathy_depth)
    if mld_target_transform == "identity":
        clamped_mld_transformed = clamped_mld
    elif mld_target_transform == "log1p":
        clamped_mld_transformed = torch.log1p(clamped_mld.clamp_min(0.0))
    else:
        raise ValueError(f"Unsupported MLD transform: {mld_target_transform}")

    output = prediction.clone()
    scale = torch.where(target_iqr[mld_index] == 0.0, torch.ones_like(target_iqr[mld_index]), target_iqr[mld_index])
    output[ocean_mask, mld_index] = (clamped_mld_transformed - target_median[mld_index]) / scale
    return output


def clamp_prediction_output_to_bathymetry(
    prediction_output: torch.Tensor,
    static: torch.Tensor,
    static_median: torch.Tensor,
    static_iqr: torch.Tensor,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    static_features: list[str],
    target_names: list[str],
    mld_target_transform: str,
) -> torch.Tensor:
    prediction_mean, prediction_std = split_prediction_params(prediction_output, len(target_names))
    prediction_mean = clamp_normalized_mld_to_bathymetry(
        prediction=prediction_mean,
        static=static,
        static_median=static_median,
        static_iqr=static_iqr,
        target_median=target_median,
        target_iqr=target_iqr,
        static_features=static_features,
        target_names=target_names,
        mld_target_transform=mld_target_transform,
    )
    if prediction_std is None:
        return prediction_mean
    return torch.cat([prediction_mean, prediction_std], dim=-1)


def selection_score_from_metrics(
    metrics: dict[str, float],
    selection_metric: str,
    bias_weight: float,
    std_weight: float,
    tail_weight: float,
) -> tuple[str, float]:
    if selection_metric == "loss":
        return "val_loss", float(metrics["val_loss"])
    if selection_metric == "mld_rmse":
        return "val_mld_rmse_phys", float(metrics["val_mld_rmse_phys"])
    if selection_metric == "mld_calibrated":
        value = (
            float(metrics["val_mld_rmse_phys"])
            + bias_weight * abs(float(metrics["val_mld_bias_phys"]))
            + std_weight * abs(1.0 - float(metrics["val_mld_std_ratio"]))
        )
        return "val_mld_calibrated_score", value
    if selection_metric == "mld_tail":
        return "val_mld_tail_rmse_phys", float(metrics["val_mld_tail_rmse_phys"])
    if selection_metric == "n2_rmse":
        return "val_n2_rmse_phys", float(metrics["val_n2_rmse_phys"])
    if selection_metric == "mld_tail_calibrated":
        value = (
            float(metrics["val_mld_rmse_phys"])
            + bias_weight * abs(float(metrics["val_mld_bias_phys"]))
            + std_weight * abs(1.0 - float(metrics["val_mld_std_ratio"]))
            + tail_weight * float(metrics["val_mld_tail_rmse_phys"])
        )
        return "val_mld_tail_calibrated_score", value
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def run_epoch(
    model: UpperDynTCNAttentionModel,
    loader: DataLoader,
    criterion: WeightedUpperDynLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    target_median: torch.Tensor,
    target_iqr: torch.Tensor,
    static_median: torch.Tensor,
    static_iqr: torch.Tensor,
    target_names: list[str],
    static_features: list[str],
    mld_target_transform: str,
    tail_quantile: float,
    progress: bool = True,
    metric_batches: int | None = None,
    finite_check_interval: int = 1,
    clamp_mld_to_bathy: bool = False,
    step_callback: Callable[[int], bool] | None = None,
    max_steps: int | None = None,
    regime_split: bool = False,
) -> dict[str, float]:
    is_training = optimizer is not None
    phase = "train" if is_training else "val"
    model.train(is_training)
    all_predictions: list[torch.Tensor] = []
    all_prediction_stds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_statics: list[torch.Tensor] = []
    running_loss = 0.0
    running_batches = 0
    running_examples = 0
    running_loss_metrics: dict[str, float] = {}
    show_progress = progress and main_process()

    iterator = tqdm(
        loader,
        desc=phase,
        leave=False,
        dynamic_ncols=True,
        mininterval=2.0,
        smoothing=0.05,
        disable=not show_progress,
    )
    for batch_idx, batch in enumerate(iterator, start=1):
        dynamic = batch["dynamic"].to(device=device, dtype=torch.float32, non_blocking=True)
        static = batch["static"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        neighbors = batch.get("neighbors")
        neighbor_mask = batch.get("neighbor_mask")
        if neighbors is not None:
            neighbors = neighbors.to(device=device, dtype=torch.float32, non_blocking=True)
            neighbor_mask = neighbor_mask.to(device=device, dtype=torch.float32, non_blocking=True)
        if finite_check_interval > 0 and batch_idx % finite_check_interval == 0:
            for name, tensor in (("dynamic", dynamic), ("static", static), ("target", target)):
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(f"Non-finite {phase} {name} tensor at batch {batch_idx}.")

        with torch.set_grad_enabled(is_training):
            if regime_split:
                prediction_output, regime_logits = model(
                    dynamic, static, return_regime_logits=True, neighbors=neighbors, neighbor_mask=neighbor_mask
                )
            else:
                prediction_output = model(dynamic, static, neighbors=neighbors, neighbor_mask=neighbor_mask)
                regime_logits = None
            if finite_check_interval > 0 and batch_idx % finite_check_interval == 0 and not torch.isfinite(prediction_output).all():
                raise FloatingPointError(f"Non-finite {phase} prediction tensor at batch {batch_idx}.")
            if clamp_mld_to_bathy:
                prediction_output = clamp_prediction_output_to_bathymetry(
                    prediction_output=prediction_output,
                    static=static,
                    static_median=static_median,
                    static_iqr=static_iqr,
                    target_median=target_median,
                    target_iqr=target_iqr,
                    static_features=static_features,
                    target_names=target_names,
                    mld_target_transform=mld_target_transform,
                )
            loss, loss_metrics = criterion(prediction_output, target, regime_logits=regime_logits)
            if finite_check_interval > 0 and batch_idx % finite_check_interval == 0 and not torch.isfinite(loss):
                details = ", ".join(f"{key}={value}" for key, value in sorted(loss_metrics.items()))
                raise FloatingPointError(f"Non-finite {phase} loss at batch {batch_idx}: {details}")
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if step_callback is not None and step_callback(batch_idx):
                    model.train(True)
                    break

        batch_examples = int(target.shape[0])
        running_loss += float(loss.detach()) * batch_examples
        running_batches += 1
        running_examples += batch_examples
        for key, value in loss_metrics.items():
            running_loss_metrics[key] = running_loss_metrics.get(key, 0.0) + float(value) * batch_examples
        if metric_batches is None or batch_idx <= metric_batches:
            prediction_mean, prediction_std = split_prediction_params(prediction_output.detach(), target.shape[-1])
            all_predictions.append(prediction_mean.cpu())
            if prediction_std is not None:
                all_prediction_stds.append(prediction_std.cpu())
            all_targets.append(target.detach().cpu())
            if clamp_mld_to_bathy:
                all_statics.append(static.detach().cpu())
        if batch_idx % 20 == 0:
            iterator.set_postfix(loss=f"{running_loss / max(running_examples, 1):.4f}")
        if max_steps is not None and is_training and batch_idx >= max_steps:
            break

    if all_predictions:
        prediction_tensor = gather_distributed_tensor(torch.cat(all_predictions, dim=0).to(device)).cpu()
        target_tensor = gather_distributed_tensor(torch.cat(all_targets, dim=0).to(device)).cpu()
        if clamp_mld_to_bathy and all_statics:
            static_tensor = gather_distributed_tensor(torch.cat(all_statics, dim=0).to(device)).cpu()
            prediction_tensor = clamp_normalized_mld_to_bathymetry(
                prediction=prediction_tensor,
                static=static_tensor,
                static_median=static_median.cpu(),
                static_iqr=static_iqr.cpu(),
                target_median=target_median.cpu(),
                target_iqr=target_iqr.cpu(),
                static_features=static_features,
                target_names=target_names,
                mld_target_transform=mld_target_transform,
            )
        metrics = regression_metrics(prediction_tensor, target_tensor)
        metrics.update(
            physical_regression_metrics(
                prediction=prediction_tensor,
                target=target_tensor,
                target_median=target_median,
                target_iqr=target_iqr,
                target_names=target_names,
                mld_target_transform=mld_target_transform,
            )
        )
        metrics.update(
            mld_tail_metrics(
                prediction=prediction_tensor,
                target=target_tensor,
                target_median=target_median,
                target_iqr=target_iqr,
                target_names=target_names,
                mld_target_transform=mld_target_transform,
                quantile=tail_quantile,
            )
        )
        if all_prediction_stds:
            prediction_std_tensor = gather_distributed_tensor(torch.cat(all_prediction_stds, dim=0).to(device)).cpu()
            metrics.update(
                predictive_std_metrics(
                    prediction_mean=prediction_tensor,
                    prediction_std=prediction_std_tensor,
                    target=target_tensor,
                    target_median=target_median,
                    target_iqr=target_iqr,
                    target_names=target_names,
                    mld_target_transform=mld_target_transform,
                )
            )
    else:
        metrics = missing_prediction_metrics()
    total_loss = reduce_distributed_sum(running_loss, device)
    total_examples = reduce_distributed_sum(float(running_examples), device)
    metrics["loss"] = total_loss / max(total_examples, 1.0)
    reduced_loss_metrics = {
        key: reduce_distributed_sum(value, device)
        for key, value in running_loss_metrics.items()
    }
    metrics.update(
        {
            f"loss_{key}": value / max(total_examples, 1.0)
            for key, value in reduced_loss_metrics.items()
            if key != "loss"
        }
    )
    return metrics


def build_dataloaders(config: ExperimentConfig, distributed: bool = False):
    train_files = list_netcdf_files(config.paths.train_dir)
    val_files = list_netcdf_files(config.paths.val_dir)
    stats = load_stats(config.paths.stats_path) if config.paths.stats_path.exists() else None
    print(
        f"Building datasets: train_files={len(train_files)} val_files={len(val_files)} stats={config.paths.stats_path}",
        flush=True,
    )

    train_dataset = None
    val_dataset = None
    # Global mode: one precompute + one neighbor pool, split by year at load time.
    if config.dataset.precomputed_dir is not None and Path(config.dataset.precomputed_dir).exists():
        import numpy as _np
        if config.dataset.coords_path is None or not Path(config.dataset.coords_path).exists():
            raise ValueError("Global mode requires dataset.coords_path (coords_all.npz with 'year').")
        if (config.dataset.mld_subset_min is not None or config.dataset.mld_subset_max is not None
                or config.dataset.eddy_balance):
            raise NotImplementedError("mld_subset/eddy_balance are not supported with the global val_years split.")
        _coords = _np.load(str(config.dataset.coords_path))
        if "year" in _coords:
            _years = _coords["year"].astype(int)
        else:
            import datetime as _dt
            _years = _np.array([_dt.date.fromordinal(int(d)).year for d in _coords["day"]], dtype=int)
        _val_years = sorted({int(y) for y in config.dataset.val_years})
        _val_mask = _np.isin(_years, _val_years) if _val_years else _np.zeros(len(_years), dtype=bool)
        _val_idx = _np.nonzero(_val_mask)[0]
        _train_idx = _np.nonzero(~_val_mask)[0]
        print(
            f"Using GLOBAL precompute {config.dataset.precomputed_dir} | val_years={_val_years}: "
            f"train={len(_train_idx)} val={len(_val_idx)} total={len(_years)}",
            flush=True,
        )
        _full = PrecomputedTensorDataset(
            config.dataset.precomputed_dir,
            dynamic_features=config.dataset.dynamic_features,
            static_features=config.dataset.static_features,
            stats=stats,
            target_names=config.dataset.targets,
            mld_target_transform=config.dataset.mld_target_transform,
            dynamic_history_hours=config.dataset.dynamic_history_hours,
            neighbor_path=config.dataset.neighbor_path,
        )
        if len(_full) != len(_years):
            raise ValueError(f"Global precompute has {len(_full)} samples but coords has {len(_years)}.")
        train_dataset = _full.subset(_train_idx)
        val_dataset = _full.subset(_val_idx)
        del _full

    if train_dataset is None and config.dataset.train_precomputed_dir is not None and config.dataset.train_precomputed_dir.exists():
        print(f"Using precomputed train dataset: {config.dataset.train_precomputed_dir}", flush=True)
        train_dataset = PrecomputedTensorDataset(
            config.dataset.train_precomputed_dir,
            dynamic_features=config.dataset.dynamic_features,
            static_features=config.dataset.static_features,
            stats=stats,
            target_names=config.dataset.targets,
            mld_target_transform=config.dataset.mld_target_transform,
            dynamic_history_hours=config.dataset.dynamic_history_hours,
            mld_subset_min=config.dataset.mld_subset_min,
            mld_subset_max=config.dataset.mld_subset_max,
            eddy_balance=config.dataset.eddy_balance,
            eddy_balance_seed=config.training.seed,
            neighbor_path=config.dataset.neighbor_train_path,
        )
    elif train_dataset is None:
        train_dataset = MonthlyNetCDFDataset(
            files=train_files,
            dynamic_features=config.dataset.dynamic_features,
            static_features=config.dataset.static_features,
            targets=config.dataset.targets,
            mld_target_transform=config.dataset.mld_target_transform,
            stats=stats,
            replace_nan_with_zero=config.dataset.replace_nan_with_zero,
            clip_mld_max=config.dataset.clip_mld_max,
            dynamic_history_hours=config.dataset.dynamic_history_hours,
            strict_static_features=config.dataset.strict_static_features,
            eddy_max_distance_radius=config.dataset.eddy_max_distance_radius,
        )
    if val_dataset is None and config.dataset.val_precomputed_dir is not None and config.dataset.val_precomputed_dir.exists():
        print(f"Using precomputed val dataset: {config.dataset.val_precomputed_dir}", flush=True)
        val_dataset = PrecomputedTensorDataset(
            config.dataset.val_precomputed_dir,
            dynamic_features=config.dataset.dynamic_features,
            static_features=config.dataset.static_features,
            stats=stats,
            target_names=config.dataset.targets,
            mld_target_transform=config.dataset.mld_target_transform,
            dynamic_history_hours=config.dataset.dynamic_history_hours,
            mld_subset_min=config.dataset.mld_subset_min,
            mld_subset_max=config.dataset.mld_subset_max,
            neighbor_path=config.dataset.neighbor_val_path,
        )
    elif val_dataset is None:
        val_dataset = MonthlyNetCDFDataset(
            files=val_files,
            dynamic_features=config.dataset.dynamic_features,
            static_features=config.dataset.static_features,
            targets=config.dataset.targets,
            mld_target_transform=config.dataset.mld_target_transform,
            stats=stats,
            replace_nan_with_zero=config.dataset.replace_nan_with_zero,
            clip_mld_max=config.dataset.clip_mld_max,
            dynamic_history_hours=config.dataset.dynamic_history_hours,
            strict_static_features=config.dataset.strict_static_features,
            eddy_max_distance_radius=config.dataset.eddy_max_distance_radius,
        )
    print(
        f"Datasets ready: train_samples={len(train_dataset)} val_samples={len(val_dataset)}",
        flush=True,
    )

    if distributed:
        if isinstance(train_dataset, PrecomputedTensorDataset) and train_dataset.single_payload is not None:
            train_loader = PrecomputedBatchLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                distributed=True,
                seed=config.training.seed,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                sampler=DistributedSampler(train_dataset, shuffle=True),
                num_workers=config.training.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=config.training.num_workers > 0,
                prefetch_factor=4 if config.training.num_workers > 0 else None,
            )
    elif isinstance(train_dataset, PrecomputedTensorDataset):
        if train_dataset.single_payload is not None:
            train_loader = PrecomputedBatchLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                distributed=False,
                seed=config.training.seed,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=ShardedBatchSampler(train_dataset, config.training.batch_size),
                num_workers=config.training.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=config.training.num_workers > 0,
                prefetch_factor=4 if config.training.num_workers > 0 else None,
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=4 if config.training.num_workers > 0 else None,
        )
    if isinstance(val_dataset, PrecomputedTensorDataset) and val_dataset.single_payload is not None:
        val_loader = PrecomputedBatchLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            distributed=distributed,
            seed=config.training.seed,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=DistributedSampler(val_dataset, shuffle=False) if distributed else None,
            num_workers=config.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=4 if config.training.num_workers > 0 else None,
        )
    return train_loader, val_loader


def build_legacy_val_loader(
    config: ExperimentConfig,
    stats: object,
    distributed: bool = False,
) -> DataLoader | None:
    if config.evaluation.legacy_val_dir is None:
        return None
    files = list_netcdf_files(config.evaluation.legacy_val_dir)
    if not files:
        raise ValueError(f"No legacy validation NetCDF files found in {config.evaluation.legacy_val_dir}")
    dataset = MonthlyNetCDFDataset(
        files=files,
        dynamic_features=config.dataset.dynamic_features,
        static_features=config.dataset.static_features,
        targets=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
        stats=stats,
        replace_nan_with_zero=config.dataset.replace_nan_with_zero,
        clip_mld_max=config.dataset.clip_mld_max,
        dynamic_history_hours=config.dataset.dynamic_history_hours,
        strict_static_features=config.dataset.strict_static_features,
        eddy_max_distance_radius=config.dataset.eddy_max_distance_radius,
    )
    if main_process():
        print(
            f"Legacy validation dataset: dir={config.evaluation.legacy_val_dir} samples={len(dataset)}",
            flush=True,
        )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=False) if distributed else None,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
        prefetch_factor=4 if config.training.num_workers > 0 else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UpperDyn structured-static TCN model.")
    parser.add_argument("--config", required=True, help="Path to the TOML config file.")
    return parser.parse_args()


def save_checkpoint(payload: dict[str, object], run_checkpoint_dir: Path, alias_checkpoint_dir: Path, filename: str) -> None:
    if not main_process():
        return
    torch.save(payload, run_checkpoint_dir / filename)
    torch.save(payload, alias_checkpoint_dir / filename)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    distributed, local_rank = setup_distributed()
    config_text = Path(args.config).read_text(encoding="utf-8")
    set_seed(
        config.training.seed,
        deterministic=config.training.deterministic,
        cudnn_benchmark=config.training.cudnn_benchmark,
    )
    if main_process():
        config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.paths.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        config.paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging requires the 'tensorboard' package in the project virtual environment."
        ) from exc

    config_stem = Path(args.config).stem
    run_name_object = [
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{config_stem}" if main_process() else ""
    ]
    if distributed_is_initialized():
        dist.broadcast_object_list(run_name_object, src=0)
    run_name = str(run_name_object[0])
    run_checkpoint_dir = config.paths.checkpoint_dir / run_name
    run_metrics_dir = config.paths.metrics_path.parent / run_name
    run_metrics_path = run_metrics_dir / config.paths.metrics_path.name
    latest_run_path = config.paths.checkpoint_dir / "latest_run.json"
    if main_process():
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_metrics_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.paths.tensorboard_dir / run_name)) if main_process() else None

    if main_process():
        with latest_run_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_name": run_name,
                    "config": str(args.config),
                    "checkpoint_dir": str(run_checkpoint_dir),
                    "metrics_path": str(run_metrics_path),
                    "tensorboard_dir": str(config.paths.tensorboard_dir / run_name),
                },
                handle,
                indent=2,
            )
        (run_checkpoint_dir / "config.toml").write_text(config_text, encoding="utf-8")
        (run_metrics_dir / "config.toml").write_text(config_text, encoding="utf-8")

    train_loader, val_loader = build_dataloaders(config, distributed=distributed)
    device = torch.device(f"cuda:{local_rank}") if distributed else resolve_device(config.training.device)
    if main_process():
        print(f"Using device={device}", flush=True)

    model = UpperDynTCNAttentionModel(
        num_dynamic=len(config.dataset.dynamic_features),
        num_static=len(config.dataset.static_features),
        num_targets=len(config.dataset.targets),
        tcn_channels=config.model.tcn_channels,
        dilations=config.model.dilations,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        static_hidden_dim=config.model.static_hidden_dim,
        static_num_heads=config.model.static_num_heads,
        use_dynamic_context=config.model.use_dynamic_context,
        dynamic_context_mode=config.model.dynamic_context_mode,
        dynamic_context_hidden_dim=config.model.dynamic_context_hidden_dim,
        dynamic_summary_kernel_size=config.model.dynamic_summary_kernel_size,
        dynamic_summary_windows=config.model.dynamic_summary_windows,
        film_hidden_dim=config.model.film_hidden_dim,
        attention_hidden_dim=config.model.attention_hidden_dim,
        attention_pooling_mode=config.model.attention_pooling_mode,
        attention_tokens_per_segment=config.model.attention_tokens_per_segment,
        num_attention_experts=config.model.num_attention_experts,
        expert_kernel_sizes=config.model.expert_kernel_sizes,
        fusion_hidden_dim=config.model.fusion_hidden_dim,
        backbone_hidden_dim=config.model.backbone_hidden_dim,
        predictive_distribution=config.model.predictive_distribution,
        min_std=config.model.min_std,
        mld_branch_hidden_dim=config.model.mld_branch_hidden_dim,
        mld_branch_depth=config.model.mld_branch_depth,
        detach_n2_head_input=config.model.detach_n2_head_input,
        n2_separate_temporal_branch=config.model.n2_separate_temporal_branch,
        n2_separate_attention_branch=config.model.n2_separate_attention_branch,
        n2_attention_hidden_dim=config.model.n2_attention_hidden_dim,
        n2_num_attention_experts=config.model.n2_num_attention_experts,
        n2_expert_kernel_sizes=config.model.n2_expert_kernel_sizes,
        n2_fusion_hidden_dim=config.model.n2_fusion_hidden_dim,
        n2_backbone_hidden_dim=config.model.n2_backbone_hidden_dim,
        n2_backbone_grad_scale=config.model.n2_backbone_grad_scale,
        n2_branch_hidden_dim=config.model.n2_branch_hidden_dim,
        n2_branch_depth=config.model.n2_branch_depth,
        mld_regime_split=config.model.mld_regime_split,
        num_mld_regimes=config.model.num_mld_regimes,
        regime_gate_hidden_dim=config.model.regime_gate_hidden_dim,
        mld_regime_attention_hidden_dim=config.model.mld_regime_attention_hidden_dim,
        mld_regime_fusion_hidden_dim=config.model.mld_regime_fusion_hidden_dim,
        mld_regime_backbone_hidden_dim=config.model.mld_regime_backbone_hidden_dim,
        mld_regime_branch_hidden_dim=config.model.mld_regime_branch_hidden_dim,
        mld_regime_branch_depth=config.model.mld_regime_branch_depth,
        use_mld_neighbors=config.model.use_mld_neighbors,
        num_neighbors=config.model.num_neighbors,
        neighbor_feat_dim=config.model.neighbor_feat_dim,
        neighbor_hidden_dim=config.model.neighbor_hidden_dim,
        neighbor_modality_dropout=config.model.neighbor_modality_dropout,
        neighbor_increment=config.model.neighbor_increment,
        neighbor_increment_full=config.model.neighbor_increment_full,
    ).to(device)
    if config.training.init_from_checkpoint is not None:
        init_state = torch.load(config.training.init_from_checkpoint, map_location=device)["model_state_dict"]
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if main_process():
            print(
                f"Initialized from {config.training.init_from_checkpoint}: "
                f"{len(missing)} new params (neighbor branch), {len(unexpected)} unexpected",
                flush=True,
            )
    if config.training.freeze_backbone:
        n_frozen = 0
        n_trainable = 0
        for name, param in model.named_parameters():
            if "neighbor" in name:
                n_trainable += param.numel()
            else:
                param.requires_grad_(False)
                n_frozen += param.numel()
        if main_process():
            print(f"Froze backbone: {n_frozen} frozen / {n_trainable} trainable (neighbor branch only)", flush=True)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=config.model.use_mld_neighbors)
        if main_process():
            print(f"Using DistributedDataParallel on {dist.get_world_size()} GPUs", flush=True)
    elif device.type == "cuda" and torch.cuda.device_count() > 1 and main_process():
        print("Multiple GPUs are visible, but DataParallel is disabled. Use torchrun for parallel training.", flush=True)

    stats = load_stats(config.paths.stats_path) if config.paths.stats_path.exists() else None
    if stats is None:
        raise FileNotFoundError(f"Missing stats file: {config.paths.stats_path}")
    run_snapshot = {
        "config_path": str(args.config),
        "config_text": config_text,
        "stats": stats_to_dict(stats),
    }
    criterion = WeightedUpperDynLoss(
        target_median=stats.target_median.tolist(),
        target_iqr=stats.target_iqr.tolist(),
        mld_log_weight=config.loss.mld_log_weight,
        n2_task_weight=config.loss.n2_task_weight,
        mld_deep_thresholds=config.loss.mld_deep_thresholds,
        mld_deep_weights=config.loss.mld_deep_weights,
        mld_underprediction_weight=config.loss.mld_underprediction_weight,
        mld_physical_weight=config.loss.mld_physical_weight,
        mld_physical_delta=config.loss.mld_physical_delta,
        mld_rmse_weight=config.loss.mld_rmse_weight,
        mld_tail_rmse_weight=config.loss.mld_tail_rmse_weight,
        mld_spread_weight=config.loss.mld_spread_weight,
        mld_corr_weight=config.loss.mld_corr_weight,
        n2_rmse_weight=config.loss.n2_rmse_weight,
        n2_spread_weight=config.loss.n2_spread_weight,
        n2_corr_weight=config.loss.n2_corr_weight,
        n2_warmup_epochs=config.loss.n2_warmup_epochs,
        n2_warmup_start_factor=config.loss.n2_warmup_start_factor,
        mld_tail_loss_quantile=config.loss.mld_tail_loss_quantile,
        huber_delta=config.loss.huber_delta,
        uncertainty_weight=config.loss.uncertainty_weight,
        predictive_distribution=config.model.predictive_distribution,
        min_std=config.model.min_std,
        target_names=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
        mld_regime_thresholds=config.loss.mld_regime_thresholds,
        regime_gate_weight=config.loss.regime_gate_weight,
        regime_load_balance_weight=config.loss.regime_load_balance_weight,
    ).to(device)
    target_median_tensor = torch.as_tensor(stats.target_median, dtype=torch.float32)
    target_iqr_tensor = torch.as_tensor(stats.target_iqr, dtype=torch.float32)
    static_median_tensor = torch.as_tensor(stats.static_median, dtype=torch.float32)
    static_iqr_tensor = torch.as_tensor(stats.static_iqr, dtype=torch.float32)
    legacy_val_loader = build_legacy_val_loader(config, stats=stats, distributed=distributed)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.min_learning_rate,
    )

    best_scores = {
        "loss": float("inf"),
        "mld_rmse": float("inf"),
        "mld_calibrated": float("inf"),
        "mld_tail": float("inf"),
        "mld_tail_calibrated": float("inf"),
        "n2_rmse": float("inf"),
    }
    patience = 0
    history: list[dict[str, float]] = []
    step_state = {"global_step": 0, "stop": False}

    def validate_checkpoint_step(epoch: int, global_step: int) -> None:
        nonlocal patience
        # Mode PROD (val_years=[]) : val vide -> pas de validation. On sauvegarde juste les
        # checkpoints périodiques (latest + step_N), sans métriques val, sans best, sans patience
        # (donc pas d'early-stopping). On sélectionne ensuite un step_N.pt / latest.pt à la main.
        _val_n = getattr(val_loader, "num_samples", None)
        if _val_n is None:
            _ds = getattr(val_loader, "dataset", None)
            _val_n = len(_ds) if _ds is not None else 1  # inconnu -> ne pas skipper
        if _val_n == 0:
            step_metrics = {"epoch": epoch, "global_step": global_step, "step_eval": False, "note": "no_val (prod)"}
            history.append(step_metrics)
            if main_process():
                print(json.dumps(step_metrics), flush=True)
                with run_metrics_path.open("w", encoding="utf-8") as handle:
                    json.dump(history, handle, indent=2)
                with config.paths.metrics_path.open("w", encoding="utf-8") as handle:
                    json.dump(history, handle, indent=2)
            payload = {
                "model_state_dict": model_state_dict(model),
                "config": str(args.config),
                **run_snapshot,
                "run_name": run_name,
                "run_checkpoint_dir": str(run_checkpoint_dir),
                "run_metrics_path": str(run_metrics_path),
                "epoch": epoch,
                "global_step": global_step,
                "attention_pooling_mode": config.model.attention_pooling_mode,
                "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                "dynamic_context_mode": config.model.dynamic_context_mode,
            }
            save_checkpoint(payload, run_checkpoint_dir, config.paths.checkpoint_dir, "latest.pt")
            save_checkpoint(payload, run_checkpoint_dir, config.paths.checkpoint_dir, f"step_{global_step}.pt")
            return
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip=config.training.grad_clip,
            target_median=target_median_tensor,
            target_iqr=target_iqr_tensor,
            static_median=static_median_tensor,
            static_iqr=static_iqr_tensor,
            target_names=config.dataset.targets,
            static_features=config.dataset.static_features,
            mld_target_transform=config.dataset.mld_target_transform,
            tail_quantile=config.training.tail_quantile,
            progress=False,
            metric_batches=None,
            finite_check_interval=config.training.finite_check_interval,
            clamp_mld_to_bathy=True,
            regime_split=config.model.mld_regime_split,
        )
        step_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "step_eval": True,
            "val_loss": val_metrics["loss"],
            "val_mld_rmse": val_metrics["mld_rmse"],
            "val_n2_rmse": val_metrics["n2_rmse"],
            "val_mld_corr": val_metrics["mld_corr"],
            "val_n2_corr": val_metrics["n2_corr"],
            "val_mld_std_ratio": val_metrics["mld_std_ratio"],
            "val_n2_std_ratio": val_metrics["n2_std_ratio"],
            "val_mld_rmse_phys": val_metrics["mld_rmse_phys"],
            "val_n2_rmse_phys": val_metrics["n2_rmse_phys"],
            "val_mld_bias_phys": val_metrics["mld_bias_phys"],
            "val_n2_bias_phys": val_metrics["n2_bias_phys"],
            "val_mld_tail_rmse_phys": val_metrics["mld_tail_rmse_phys"],
            "val_mld_tail_bias_phys": val_metrics["mld_tail_bias_phys"],
            "val_mld_tail_std_ratio": val_metrics["mld_tail_std_ratio"],
        }
        _, calibrated_score = selection_score_from_metrics(
            metrics=step_metrics,
            selection_metric="mld_calibrated",
            bias_weight=config.training.checkpoint_bias_weight,
            std_weight=config.training.checkpoint_std_weight,
            tail_weight=config.training.checkpoint_tail_weight,
        )
        step_metrics["val_mld_calibrated_score"] = calibrated_score
        _, tail_calibrated_score = selection_score_from_metrics(
            metrics=step_metrics,
            selection_metric="mld_tail_calibrated",
            bias_weight=config.training.checkpoint_bias_weight,
            std_weight=config.training.checkpoint_std_weight,
            tail_weight=config.training.checkpoint_tail_weight,
        )
        step_metrics["val_mld_tail_calibrated_score"] = tail_calibrated_score
        score_name, score_value = selection_score_from_metrics(
            metrics=step_metrics,
            selection_metric=config.training.selection_metric,
            bias_weight=config.training.checkpoint_bias_weight,
            std_weight=config.training.checkpoint_std_weight,
            tail_weight=config.training.checkpoint_tail_weight,
        )
        step_metrics[score_name] = score_value
        history.append(step_metrics)
        if main_process():
            print(json.dumps(step_metrics), flush=True)
            assert writer is not None
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            for key, value in step_metrics.items():
                if key in {"epoch", "global_step", "step_eval"}:
                    continue
                writer.add_scalar(f"step/{key}", value, global_step)
            with run_metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            with config.paths.metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

        payload = {
            "model_state_dict": model_state_dict(model),
            "config": str(args.config),
            **run_snapshot,
            "run_name": run_name,
            "run_checkpoint_dir": str(run_checkpoint_dir),
            "run_metrics_path": str(run_metrics_path),
            "epoch": epoch,
            "global_step": global_step,
            "val_metrics": val_metrics,
            "epoch_metrics": step_metrics,
            "selection_metric": config.training.selection_metric,
            "selection_name": score_name,
            "selection_value": score_value,
            "attention_pooling_mode": config.model.attention_pooling_mode,
            "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
            "dynamic_context_mode": config.model.dynamic_context_mode,
        }
        save_checkpoint(payload, run_checkpoint_dir, config.paths.checkpoint_dir, "latest.pt")
        save_checkpoint(payload, run_checkpoint_dir, config.paths.checkpoint_dir, f"step_{global_step}.pt")

        if step_metrics["val_loss"] < best_scores["loss"]:
            best_scores["loss"] = step_metrics["val_loss"]
            save_checkpoint({**payload, "selection_metric": "loss", "selection_name": "val_loss", "selection_value": step_metrics["val_loss"]}, run_checkpoint_dir, config.paths.checkpoint_dir, "best_loss.pt")
        if step_metrics["val_mld_rmse_phys"] < best_scores["mld_rmse"]:
            best_scores["mld_rmse"] = step_metrics["val_mld_rmse_phys"]
            save_checkpoint({**payload, "selection_metric": "mld_rmse", "selection_name": "val_mld_rmse_phys", "selection_value": step_metrics["val_mld_rmse_phys"]}, run_checkpoint_dir, config.paths.checkpoint_dir, "best_mld_rmse.pt")
        if step_metrics["val_mld_calibrated_score"] < best_scores["mld_calibrated"]:
            best_scores["mld_calibrated"] = step_metrics["val_mld_calibrated_score"]
            save_checkpoint({**payload, "selection_metric": "mld_calibrated", "selection_name": "val_mld_calibrated_score", "selection_value": step_metrics["val_mld_calibrated_score"]}, run_checkpoint_dir, config.paths.checkpoint_dir, "best_mld_calibrated.pt")
        if step_metrics["val_mld_tail_rmse_phys"] < best_scores["mld_tail"]:
            best_scores["mld_tail"] = step_metrics["val_mld_tail_rmse_phys"]
            save_checkpoint({**payload, "selection_metric": "mld_tail", "selection_name": "val_mld_tail_rmse_phys", "selection_value": step_metrics["val_mld_tail_rmse_phys"]}, run_checkpoint_dir, config.paths.checkpoint_dir, "best_mld_tail.pt")
        if step_metrics["val_n2_rmse_phys"] < best_scores["n2_rmse"]:
            best_scores["n2_rmse"] = step_metrics["val_n2_rmse_phys"]
            save_checkpoint({**payload, "selection_metric": "n2_rmse", "selection_name": "val_n2_rmse_phys", "selection_value": step_metrics["val_n2_rmse_phys"]}, run_checkpoint_dir, config.paths.checkpoint_dir, "best_n2_rmse.pt")
        if step_metrics["val_mld_tail_calibrated_score"] < best_scores["mld_tail_calibrated"]:
            best_scores["mld_tail_calibrated"] = step_metrics["val_mld_tail_calibrated_score"]
            save_checkpoint({**payload, "selection_metric": "mld_tail_calibrated", "selection_name": "val_mld_tail_calibrated_score", "selection_value": step_metrics["val_mld_tail_calibrated_score"]}, run_checkpoint_dir, config.paths.checkpoint_dir, "best_mld_tail_calibrated.pt")
        if score_value < best_scores[config.training.selection_metric]:
            best_scores[config.training.selection_metric] = score_value
            patience = 0
            save_checkpoint(payload, run_checkpoint_dir, config.paths.checkpoint_dir, "best.pt")
        else:
            patience += 1
        scheduler.step(score_value)

    def on_train_step(epoch: int) -> Callable[[int], bool] | None:
        if config.training.val_interval_steps <= 0 and config.training.max_steps is None:
            return None

        def callback(_: int) -> bool:
            step_state["global_step"] += 1
            global_step = int(step_state["global_step"])
            if config.training.val_interval_steps > 0 and global_step % config.training.val_interval_steps == 0:
                validate_checkpoint_step(epoch=epoch, global_step=global_step)
                model.train(True)
            if config.training.max_steps is not None and global_step >= config.training.max_steps:
                step_state["stop"] = True
                return True
            if patience >= config.training.early_stopping_patience:
                step_state["stop"] = True
                return True
            return False

        return callback

    for epoch in range(1, config.training.epochs + 1):
        criterion.set_epoch(epoch)
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        elif hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        if main_process():
            print(f"Starting epoch {epoch}/{config.training.epochs}", flush=True)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=config.training.grad_clip,
            target_median=target_median_tensor,
            target_iqr=target_iqr_tensor,
            static_median=static_median_tensor,
            static_iqr=static_iqr_tensor,
            target_names=config.dataset.targets,
            static_features=config.dataset.static_features,
            mld_target_transform=config.dataset.mld_target_transform,
            tail_quantile=config.training.tail_quantile,
            metric_batches=config.training.train_metric_batches,
            finite_check_interval=config.training.finite_check_interval,
            clamp_mld_to_bathy=True,
            step_callback=on_train_step(epoch),
            regime_split=config.model.mld_regime_split,
        )
        if step_state["stop"]:
            break
        # PROD (val_years=[]) : val vide -> pas de validation de fin d'epoch. Les checkpoints
        # périodiques sont déjà sauvés par le callback step ; on saute tout le bloc
        # val/epoch_metrics/best/scheduler/early-stop (mêmes clés val_metrics absentes sinon KeyError).
        _epoch_val_n = getattr(val_loader, "num_samples", None)
        if _epoch_val_n is None:
            _epoch_val_ds = getattr(val_loader, "dataset", None)
            _epoch_val_n = len(_epoch_val_ds) if _epoch_val_ds is not None else 1
        if _epoch_val_n == 0:
            if main_process():
                print(json.dumps({"epoch": epoch, "note": "no_val (prod) — fin epoch, validation sautée"}), flush=True)
            continue
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip=config.training.grad_clip,
            target_median=target_median_tensor,
            target_iqr=target_iqr_tensor,
            static_median=static_median_tensor,
            static_iqr=static_iqr_tensor,
            target_names=config.dataset.targets,
            static_features=config.dataset.static_features,
            mld_target_transform=config.dataset.mld_target_transform,
            tail_quantile=config.training.tail_quantile,
            metric_batches=None,
            finite_check_interval=config.training.finite_check_interval,
            clamp_mld_to_bathy=True,
            regime_split=config.model.mld_regime_split,
        )
        legacy_val_metrics = None
        if legacy_val_loader is not None:
            legacy_val_metrics = run_epoch(
                model=model,
                loader=legacy_val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                grad_clip=config.training.grad_clip,
                target_median=target_median_tensor,
                target_iqr=target_iqr_tensor,
                static_median=static_median_tensor,
                static_iqr=static_iqr_tensor,
                target_names=config.dataset.targets,
                static_features=config.dataset.static_features,
                mld_target_transform=config.dataset.mld_target_transform,
                tail_quantile=config.training.tail_quantile,
                progress=False,
                metric_batches=None,
                finite_check_interval=config.training.finite_check_interval,
                clamp_mld_to_bathy=True,
                regime_split=config.model.mld_regime_split,
            )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mld_rmse": train_metrics["mld_rmse"],
            "train_n2_rmse": train_metrics["n2_rmse"],
            "train_mld_corr": train_metrics["mld_corr"],
            "train_n2_corr": train_metrics["n2_corr"],
            "train_mld_std_ratio": train_metrics["mld_std_ratio"],
            "train_n2_std_ratio": train_metrics["n2_std_ratio"],
            "train_mld_rmse_phys": train_metrics["mld_rmse_phys"],
            "train_n2_rmse_phys": train_metrics["n2_rmse_phys"],
            "train_mld_bias_phys": train_metrics["mld_bias_phys"],
            "train_n2_bias_phys": train_metrics["n2_bias_phys"],
            "train_mld_tail_quantile": train_metrics["mld_tail_quantile"],
            "train_mld_tail_threshold_phys": train_metrics["mld_tail_threshold_phys"],
            "train_mld_tail_count": train_metrics["mld_tail_count"],
            "train_mld_tail_rmse_phys": train_metrics["mld_tail_rmse_phys"],
            "train_mld_tail_bias_phys": train_metrics["mld_tail_bias_phys"],
            "train_mld_tail_std_ratio": train_metrics["mld_tail_std_ratio"],
            "val_loss": val_metrics["loss"],
            "val_mld_rmse": val_metrics["mld_rmse"],
            "val_n2_rmse": val_metrics["n2_rmse"],
            "val_mld_corr": val_metrics["mld_corr"],
            "val_n2_corr": val_metrics["n2_corr"],
            "val_mld_std_ratio": val_metrics["mld_std_ratio"],
            "val_n2_std_ratio": val_metrics["n2_std_ratio"],
            "val_mld_rmse_phys": val_metrics["mld_rmse_phys"],
            "val_n2_rmse_phys": val_metrics["n2_rmse_phys"],
            "val_mld_bias_phys": val_metrics["mld_bias_phys"],
            "val_n2_bias_phys": val_metrics["n2_bias_phys"],
            "val_mld_tail_quantile": val_metrics["mld_tail_quantile"],
            "val_mld_tail_threshold_phys": val_metrics["mld_tail_threshold_phys"],
            "val_mld_tail_count": val_metrics["mld_tail_count"],
            "val_mld_tail_rmse_phys": val_metrics["mld_tail_rmse_phys"],
            "val_mld_tail_bias_phys": val_metrics["mld_tail_bias_phys"],
            "val_mld_tail_std_ratio": val_metrics["mld_tail_std_ratio"],
            "train_mld_log_loss": train_metrics["loss_mld_log_loss"],
            "train_mld_physical_loss": train_metrics["loss_mld_physical_loss"],
            "train_mld_rmse_loss": train_metrics.get("loss_mld_rmse_loss", 0.0),
            "train_mld_tail_rmse_loss": train_metrics.get("loss_mld_tail_rmse_loss", 0.0),
            "train_mld_spread_loss": train_metrics.get("loss_mld_spread_loss", 0.0),
            "train_mld_spread_ratio": train_metrics.get("loss_mld_spread_ratio", 0.0),
            "train_mld_corr_loss": train_metrics.get("loss_mld_corr_loss", 0.0),
            "train_mld_corr_batch": train_metrics.get("loss_mld_corr_batch", 0.0),
            "train_n2_loss": train_metrics["loss_n2_loss"],
            "train_n2_rmse_loss": train_metrics.get("loss_n2_rmse_loss", 0.0),
            "train_n2_spread_loss": train_metrics.get("loss_n2_spread_loss", 0.0),
            "train_n2_spread_ratio": train_metrics.get("loss_n2_spread_ratio", 0.0),
            "train_n2_corr_loss": train_metrics.get("loss_n2_corr_loss", 0.0),
            "train_n2_corr_batch": train_metrics.get("loss_n2_corr_batch", 0.0),
            "train_n2_weight_factor": train_metrics.get("loss_n2_weight_factor", 1.0),
            "train_uncertainty_loss": train_metrics.get("loss_uncertainty_loss", 0.0),
            "train_mld_uncertainty_loss": train_metrics.get("loss_mld_uncertainty_loss", 0.0),
            "train_n2_uncertainty_loss": train_metrics.get("loss_n2_uncertainty_loss", 0.0),
            "val_mld_log_loss": val_metrics["loss_mld_log_loss"],
            "val_mld_physical_loss": val_metrics["loss_mld_physical_loss"],
            "val_mld_rmse_loss": val_metrics.get("loss_mld_rmse_loss", 0.0),
            "val_mld_tail_rmse_loss": val_metrics.get("loss_mld_tail_rmse_loss", 0.0),
            "val_mld_spread_loss": val_metrics.get("loss_mld_spread_loss", 0.0),
            "val_mld_spread_ratio": val_metrics.get("loss_mld_spread_ratio", 0.0),
            "val_mld_corr_loss": val_metrics.get("loss_mld_corr_loss", 0.0),
            "val_mld_corr_batch": val_metrics.get("loss_mld_corr_batch", 0.0),
            "val_n2_loss": val_metrics["loss_n2_loss"],
            "val_n2_rmse_loss": val_metrics.get("loss_n2_rmse_loss", 0.0),
            "val_n2_spread_loss": val_metrics.get("loss_n2_spread_loss", 0.0),
            "val_n2_spread_ratio": val_metrics.get("loss_n2_spread_ratio", 0.0),
            "val_n2_corr_loss": val_metrics.get("loss_n2_corr_loss", 0.0),
            "val_n2_corr_batch": val_metrics.get("loss_n2_corr_batch", 0.0),
            "val_n2_weight_factor": val_metrics.get("loss_n2_weight_factor", 1.0),
            "val_uncertainty_loss": val_metrics.get("loss_uncertainty_loss", 0.0),
            "val_mld_uncertainty_loss": val_metrics.get("loss_mld_uncertainty_loss", 0.0),
            "val_n2_uncertainty_loss": val_metrics.get("loss_n2_uncertainty_loss", 0.0),
        }
        if legacy_val_metrics is not None:
            epoch_metrics.update(
                {
                    "legacy_val_loss": legacy_val_metrics["loss"],
                    "legacy_val_mld_rmse_phys": legacy_val_metrics["mld_rmse_phys"],
                    "legacy_val_mld_bias_phys": legacy_val_metrics["mld_bias_phys"],
                    "legacy_val_mld_corr": legacy_val_metrics["mld_corr"],
                    "legacy_val_mld_tail_rmse_phys": legacy_val_metrics["mld_tail_rmse_phys"],
                    "legacy_val_n2_rmse_phys": legacy_val_metrics["n2_rmse_phys"],
                }
            )
        if "mld_pred_std_phys_mean" in train_metrics:
            epoch_metrics["train_mld_pred_std_phys_mean"] = train_metrics["mld_pred_std_phys_mean"]
            epoch_metrics["train_n2_pred_std_phys_mean"] = train_metrics["n2_pred_std_phys_mean"]
            epoch_metrics["train_mld_one_sigma_coverage"] = train_metrics["mld_one_sigma_coverage"]
            epoch_metrics["train_n2_one_sigma_coverage"] = train_metrics["n2_one_sigma_coverage"]
        if "mld_pred_std_phys_mean" in val_metrics:
            epoch_metrics["val_mld_pred_std_phys_mean"] = val_metrics["mld_pred_std_phys_mean"]
            epoch_metrics["val_n2_pred_std_phys_mean"] = val_metrics["n2_pred_std_phys_mean"]
            epoch_metrics["val_mld_one_sigma_coverage"] = val_metrics["mld_one_sigma_coverage"]
            epoch_metrics["val_n2_one_sigma_coverage"] = val_metrics["n2_one_sigma_coverage"]
        _, calibrated_score = selection_score_from_metrics(
            metrics=epoch_metrics,
            selection_metric="mld_calibrated",
            bias_weight=config.training.checkpoint_bias_weight,
            std_weight=config.training.checkpoint_std_weight,
            tail_weight=config.training.checkpoint_tail_weight,
        )
        epoch_metrics["val_mld_calibrated_score"] = calibrated_score
        _, tail_calibrated_score = selection_score_from_metrics(
            metrics=epoch_metrics,
            selection_metric="mld_tail_calibrated",
            bias_weight=config.training.checkpoint_bias_weight,
            std_weight=config.training.checkpoint_std_weight,
            tail_weight=config.training.checkpoint_tail_weight,
        )
        epoch_metrics["val_mld_tail_calibrated_score"] = tail_calibrated_score
        score_name, score_value = selection_score_from_metrics(
            metrics=epoch_metrics,
            selection_metric=config.training.selection_metric,
            bias_weight=config.training.checkpoint_bias_weight,
            std_weight=config.training.checkpoint_std_weight,
            tail_weight=config.training.checkpoint_tail_weight,
        )
        epoch_metrics[score_name] = score_value
        history.append(epoch_metrics)
        if main_process():
            print(json.dumps(epoch_metrics), flush=True)
            assert writer is not None
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            for key, value in epoch_metrics.items():
                if key == "epoch":
                    continue
                writer.add_scalar(key, value, epoch)

            with run_metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            with config.paths.metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

        latest_payload = {
            "model_state_dict": model_state_dict(model),
            "config": str(args.config),
            **run_snapshot,
            "run_name": run_name,
            "run_checkpoint_dir": str(run_checkpoint_dir),
            "run_metrics_path": str(run_metrics_path),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "epoch_metrics": epoch_metrics,
            "selection_metric": config.training.selection_metric,
            "selection_name": score_name,
            "selection_value": score_value,
            "attention_pooling_mode": config.model.attention_pooling_mode,
            "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
            "dynamic_context_mode": config.model.dynamic_context_mode,
        }
        save_checkpoint(
            latest_payload,
            run_checkpoint_dir=run_checkpoint_dir,
            alias_checkpoint_dir=config.paths.checkpoint_dir,
            filename="latest.pt",
        )

        if epoch_metrics["val_loss"] < best_scores["loss"]:
            best_scores["loss"] = epoch_metrics["val_loss"]
            save_checkpoint(
                {
                    "model_state_dict": model_state_dict(model),
                    "config": str(args.config),
                    **run_snapshot,
                    "run_name": run_name,
                    "run_checkpoint_dir": str(run_checkpoint_dir),
                    "run_metrics_path": str(run_metrics_path),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "epoch_metrics": epoch_metrics,
                    "selection_metric": "loss",
                    "selection_name": "val_loss",
                    "selection_value": epoch_metrics["val_loss"],
                    "attention_pooling_mode": config.model.attention_pooling_mode,
                    "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                    "dynamic_context_mode": config.model.dynamic_context_mode,
                },
                run_checkpoint_dir=run_checkpoint_dir,
                alias_checkpoint_dir=config.paths.checkpoint_dir,
                filename="best_loss.pt",
            )

        if epoch_metrics["val_mld_rmse_phys"] < best_scores["mld_rmse"]:
            best_scores["mld_rmse"] = epoch_metrics["val_mld_rmse_phys"]
            save_checkpoint(
                {
                    "model_state_dict": model_state_dict(model),
                    "config": str(args.config),
                    **run_snapshot,
                    "run_name": run_name,
                    "run_checkpoint_dir": str(run_checkpoint_dir),
                    "run_metrics_path": str(run_metrics_path),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "epoch_metrics": epoch_metrics,
                    "selection_metric": "mld_rmse",
                    "selection_name": "val_mld_rmse_phys",
                    "selection_value": epoch_metrics["val_mld_rmse_phys"],
                    "attention_pooling_mode": config.model.attention_pooling_mode,
                    "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                    "dynamic_context_mode": config.model.dynamic_context_mode,
                },
                run_checkpoint_dir=run_checkpoint_dir,
                alias_checkpoint_dir=config.paths.checkpoint_dir,
                filename="best_mld_rmse.pt",
            )

        if epoch_metrics["val_mld_calibrated_score"] < best_scores["mld_calibrated"]:
            best_scores["mld_calibrated"] = epoch_metrics["val_mld_calibrated_score"]
            save_checkpoint(
                {
                    "model_state_dict": model_state_dict(model),
                    "config": str(args.config),
                    **run_snapshot,
                    "run_name": run_name,
                    "run_checkpoint_dir": str(run_checkpoint_dir),
                    "run_metrics_path": str(run_metrics_path),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "epoch_metrics": epoch_metrics,
                    "selection_metric": "mld_calibrated",
                    "selection_name": "val_mld_calibrated_score",
                    "selection_value": epoch_metrics["val_mld_calibrated_score"],
                    "attention_pooling_mode": config.model.attention_pooling_mode,
                    "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                    "dynamic_context_mode": config.model.dynamic_context_mode,
                },
                run_checkpoint_dir=run_checkpoint_dir,
                alias_checkpoint_dir=config.paths.checkpoint_dir,
                filename="best_mld_calibrated.pt",
            )

        if epoch_metrics["val_mld_tail_rmse_phys"] < best_scores["mld_tail"]:
            best_scores["mld_tail"] = epoch_metrics["val_mld_tail_rmse_phys"]
            save_checkpoint(
                {
                    "model_state_dict": model_state_dict(model),
                    "config": str(args.config),
                    **run_snapshot,
                    "run_name": run_name,
                    "run_checkpoint_dir": str(run_checkpoint_dir),
                    "run_metrics_path": str(run_metrics_path),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "epoch_metrics": epoch_metrics,
                    "selection_metric": "mld_tail",
                    "selection_name": "val_mld_tail_rmse_phys",
                    "selection_value": epoch_metrics["val_mld_tail_rmse_phys"],
                    "attention_pooling_mode": config.model.attention_pooling_mode,
                    "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                    "dynamic_context_mode": config.model.dynamic_context_mode,
                },
                run_checkpoint_dir=run_checkpoint_dir,
                alias_checkpoint_dir=config.paths.checkpoint_dir,
                filename="best_mld_tail.pt",
            )

        if epoch_metrics["val_n2_rmse_phys"] < best_scores["n2_rmse"]:
            best_scores["n2_rmse"] = epoch_metrics["val_n2_rmse_phys"]
            save_checkpoint(
                {
                    "model_state_dict": model_state_dict(model),
                    "config": str(args.config),
                    **run_snapshot,
                    "run_name": run_name,
                    "run_checkpoint_dir": str(run_checkpoint_dir),
                    "run_metrics_path": str(run_metrics_path),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "epoch_metrics": epoch_metrics,
                    "selection_metric": "n2_rmse",
                    "selection_name": "val_n2_rmse_phys",
                    "selection_value": epoch_metrics["val_n2_rmse_phys"],
                    "attention_pooling_mode": config.model.attention_pooling_mode,
                    "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                    "dynamic_context_mode": config.model.dynamic_context_mode,
                },
                run_checkpoint_dir=run_checkpoint_dir,
                alias_checkpoint_dir=config.paths.checkpoint_dir,
                filename="best_n2_rmse.pt",
            )

        if epoch_metrics["val_mld_tail_calibrated_score"] < best_scores["mld_tail_calibrated"]:
            best_scores["mld_tail_calibrated"] = epoch_metrics["val_mld_tail_calibrated_score"]
            save_checkpoint(
                {
                    "model_state_dict": model_state_dict(model),
                    "config": str(args.config),
                    **run_snapshot,
                    "run_name": run_name,
                    "run_checkpoint_dir": str(run_checkpoint_dir),
                    "run_metrics_path": str(run_metrics_path),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "epoch_metrics": epoch_metrics,
                    "selection_metric": "mld_tail_calibrated",
                    "selection_name": "val_mld_tail_calibrated_score",
                    "selection_value": epoch_metrics["val_mld_tail_calibrated_score"],
                    "attention_pooling_mode": config.model.attention_pooling_mode,
                    "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                    "dynamic_context_mode": config.model.dynamic_context_mode,
                },
                run_checkpoint_dir=run_checkpoint_dir,
                alias_checkpoint_dir=config.paths.checkpoint_dir,
                filename="best_mld_tail_calibrated.pt",
            )

        if score_value < best_scores[config.training.selection_metric]:
            best_scores[config.training.selection_metric] = score_value
            patience = 0
            save_checkpoint(
                {
                    "model_state_dict": model_state_dict(model),
                    "config": str(args.config),
                    **run_snapshot,
                    "run_name": run_name,
                    "run_checkpoint_dir": str(run_checkpoint_dir),
                    "run_metrics_path": str(run_metrics_path),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "epoch_metrics": epoch_metrics,
                    "selection_metric": config.training.selection_metric,
                    "selection_name": score_name,
                    "selection_value": score_value,
                    "attention_pooling_mode": config.model.attention_pooling_mode,
                    "attention_tokens_per_segment": config.model.attention_tokens_per_segment,
                    "dynamic_context_mode": config.model.dynamic_context_mode,
                },
                run_checkpoint_dir=run_checkpoint_dir,
                alias_checkpoint_dir=config.paths.checkpoint_dir,
                filename="best.pt",
            )
        else:
            patience += 1

        scheduler.step(score_value)

        if patience >= config.training.early_stopping_patience:
            break

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
