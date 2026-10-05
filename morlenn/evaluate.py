from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, load_config_text
from .data import MonthlyNetCDFDataset, PrecomputedTensorDataset, denormalize_targets, list_netcdf_files, load_stats, stats_from_dict
from .model import UpperDynTCNAttentionModel, split_prediction_params
from .notebook_analysis import _apply_checkpoint_snapshot, _infer_model_overrides_from_checkpoint, _load_checkpoint_state
from .train import resolve_device, set_seed


def per_target_metrics(prediction: np.ndarray, target: np.ndarray, target_names: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(target_names):
        pred = prediction[:, index]
        ref = target[:, index]
        error = pred - ref
        pred_std = float(np.std(pred))
        ref_std = float(np.std(ref))
        corr = float(np.corrcoef(pred, ref)[0, 1]) if pred.size > 1 else float("nan")
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "corr": corr,
            "pred_std": pred_std,
            "target_std": ref_std,
            "std_ratio": pred_std / (ref_std + 1e-6),
        }
    return metrics


def mld_bin_metrics(prediction_mld: np.ndarray, target_mld: np.ndarray) -> dict[str, dict[str, float]]:
    bins = [
        ("0_50", 0.0, 50.0),
        ("50_150", 50.0, 150.0),
        ("150_300", 150.0, 300.0),
        ("300_plus", 300.0, np.inf),
    ]
    metrics: dict[str, dict[str, float]] = {}
    for name, lower, upper in bins:
        mask = (target_mld >= lower) & (target_mld < upper)
        if not np.any(mask):
            metrics[name] = {"count": 0, "rmse": float("nan"), "bias": float("nan")}
            continue
        pred = prediction_mld[mask]
        ref = target_mld[mask]
        error = pred - ref
        metrics[name] = {
            "count": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
        }
    return metrics


def top_tail_metrics(prediction_mld: np.ndarray, target_mld: np.ndarray, quantile: float = 0.95) -> dict[str, float]:
    threshold = float(np.quantile(target_mld, quantile))
    mask = target_mld >= threshold
    pred = prediction_mld[mask]
    ref = target_mld[mask]
    error = pred - ref
    return {
        "quantile": quantile,
        "threshold": threshold,
        "count": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "pred_std": float(np.std(pred)),
        "target_std": float(np.std(ref)),
        "std_ratio": float(np.std(pred) / (np.std(ref) + 1e-6)),
    }


def qq_summary(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray) -> dict[str, list[float]]:
    pred_q = np.quantile(prediction, quantiles)
    target_q = np.quantile(target, quantiles)
    return {
        "quantiles": [float(q) for q in quantiles],
        "prediction": [float(v) for v in pred_q],
        "target": [float(v) for v in target_q],
    }


def predictive_uncertainty_metrics(
    prediction_mean_phys: np.ndarray,
    prediction_std_phys: np.ndarray,
    target_phys: np.ndarray,
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(target_names):
        pred_std = prediction_std_phys[:, index]
        abs_error = np.abs(prediction_mean_phys[:, index] - target_phys[:, index])
        corr = float(np.corrcoef(pred_std, abs_error)[0, 1]) if pred_std.size > 1 else float("nan")
        metrics[name] = {
            "pred_std_mean": float(np.mean(pred_std)),
            "pred_std_median": float(np.median(pred_std)),
            "pred_std_p90": float(np.quantile(pred_std, 0.90)),
            "one_sigma_coverage": float(np.mean(abs_error <= np.maximum(pred_std, 1e-6))),
            "abs_error_std_corr": corr,
        }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained UpperDyn checkpoint on a dataset split.")
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional path to checkpoint. Defaults to best.pt.")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--output", default=None, help="Optional output path for metrics JSON.")
    parser.add_argument("--data-dir", default=None, help="Optional raw NetCDF directory overriding config train/val dir.")
    parser.add_argument("--precomputed-dir", default=None, help="Optional precomputed tensor dataset directory.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.training.seed)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (config.paths.checkpoint_dir / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_mtime = checkpoint_path.stat().st_mtime
    stats = load_stats(config.paths.stats_path) if config.paths.stats_path.exists() else None
    if stats is None and "stats" not in checkpoint and not Path(checkpoint_path).with_name("run_manifest.json").exists():
        raise FileNotFoundError(f"Missing stats file: {config.paths.stats_path}")
    if "config_text" not in checkpoint and Path(args.config).stat().st_mtime > checkpoint_mtime:
        print(
            f"Warning: checkpoint {checkpoint_path} does not contain a config snapshot, "
            f"and {args.config} is newer than the checkpoint.",
            file=sys.stderr,
        )
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    device = resolve_device(config.training.device)
    if "stats" not in checkpoint and not Path(checkpoint_path).with_name("run_manifest.json").exists() and config.paths.stats_path.exists() and config.paths.stats_path.stat().st_mtime > checkpoint_mtime:
        print(
            f"Warning: checkpoint {checkpoint_path} does not contain a stats snapshot, "
            f"and {config.paths.stats_path} is newer than the checkpoint.",
            file=sys.stderr,
        )
    stats = stats_from_dict(checkpoint["stats"]) if "stats" in checkpoint else stats

    if args.precomputed_dir is not None:
        data_dir = Path(args.precomputed_dir)
        dataset = PrecomputedTensorDataset(
            data_dir,
            dynamic_features=config.dataset.dynamic_features,
            static_features=config.dataset.static_features,
            stats=stats,
            target_names=config.dataset.targets,
            mld_target_transform=config.dataset.mld_target_transform,
            dynamic_history_hours=config.dataset.dynamic_history_hours,
        )
    else:
        data_dir = Path(args.data_dir) if args.data_dir is not None else (config.paths.train_dir if args.split == "train" else config.paths.val_dir)
        files = list_netcdf_files(data_dir)
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers if args.num_workers is None else args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_overrides = _infer_model_overrides_from_checkpoint(checkpoint, config)
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
        use_dynamic_context=bool(checkpoint_overrides.get("use_dynamic_context", config.model.use_dynamic_context)),
        dynamic_context_mode=str(checkpoint_overrides.get("dynamic_context_mode", "latent_segments")),
        dynamic_context_hidden_dim=config.model.dynamic_context_hidden_dim,
        dynamic_summary_kernel_size=config.model.dynamic_summary_kernel_size,
        dynamic_summary_windows=config.model.dynamic_summary_windows,
        film_hidden_dim=config.model.film_hidden_dim,
        attention_hidden_dim=config.model.attention_hidden_dim,
        attention_pooling_mode=str(checkpoint_overrides.get("attention_pooling_mode", config.model.attention_pooling_mode)),
        attention_tokens_per_segment=int(checkpoint_overrides.get("attention_tokens_per_segment", config.model.attention_tokens_per_segment)),
        num_attention_experts=config.model.num_attention_experts,
        expert_kernel_sizes=checkpoint_overrides.get("expert_kernel_sizes", config.model.expert_kernel_sizes),
        fusion_hidden_dim=config.model.fusion_hidden_dim,
        backbone_hidden_dim=config.model.backbone_hidden_dim,
        predictive_distribution=str(checkpoint_overrides.get("predictive_distribution", config.model.predictive_distribution)),
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
    ).to(device)
    model.load_state_dict(_load_checkpoint_state(checkpoint))
    model.eval()

    prediction_means: list[np.ndarray] = []
    prediction_stds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    attention_summaries: list[np.ndarray] = []
    gate_summaries: list[np.ndarray] = []
    static_group_summaries: list[np.ndarray] = []
    dynamic_segment_summaries: list[np.ndarray] = []
    regime_gate_summaries: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
            static = batch["static"].to(device=device, dtype=torch.float32)
            target = batch["target"].cpu().numpy()
            prediction_output, diagnostics = model(dynamic, static, return_diagnostics=True)
            prediction_mean, prediction_std = split_prediction_params(prediction_output, len(config.dataset.targets))
            prediction_means.append(prediction_mean.cpu().numpy())
            if prediction_std is not None:
                prediction_stds.append(prediction_std.cpu().numpy())
            targets.append(target)
            attention_summaries.append(diagnostics["combined_attention"].cpu().numpy())
            gate_summaries.append(diagnostics["expert_gates"].cpu().numpy())
            static_group_summaries.append(diagnostics["static_group_gates"].cpu().numpy())
            if "dynamic_segment_gates" in diagnostics:
                dynamic_segment_summaries.append(diagnostics["dynamic_segment_gates"].cpu().numpy())
            elif "dynamic_feature_gates" in diagnostics:
                dynamic_segment_summaries.append(diagnostics["dynamic_feature_gates"].cpu().numpy())
            if "regime_gate" in diagnostics:
                regime_gate_summaries.append(diagnostics["regime_gate"].cpu().numpy())

    prediction_norm = np.concatenate(prediction_means, axis=0)
    target_norm = np.concatenate(targets, axis=0)
    prediction_phys = denormalize_targets(
        prediction_norm,
        stats,
        target_names=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
    )
    target_phys = denormalize_targets(
        target_norm,
        stats,
        target_names=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
    )
    mean_attention = np.concatenate(attention_summaries, axis=0).mean(axis=0)
    mean_gates = np.concatenate(gate_summaries, axis=0).mean(axis=0)
    mean_static_groups = np.concatenate(static_group_summaries, axis=0).mean(axis=0)
    mean_dynamic_segment_gates = (
        np.concatenate(dynamic_segment_summaries, axis=0).mean(axis=0) if dynamic_segment_summaries else None
    )
    mean_regime_gate = np.concatenate(regime_gate_summaries, axis=0).mean() if regime_gate_summaries else None
    qq_quantiles = np.linspace(0.05, 0.95, 19)
    uncertainty_report: dict[str, object] | None = None
    if prediction_stds:
        prediction_std_phys = np.clip(np.concatenate(prediction_stds, axis=0), a_min=1e-6, a_max=None)
        uncertainty_report = {
            "distribution": config.model.predictive_distribution,
            "targets": predictive_uncertainty_metrics(prediction_phys, prediction_std_phys, target_phys, config.dataset.targets),
        }

    report = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "num_samples": int(prediction_phys.shape[0]),
        "targets": per_target_metrics(prediction_phys, target_phys, config.dataset.targets),
        "mld_bins": mld_bin_metrics(prediction_phys[:, 0], target_phys[:, 0]),
        "mld_top_5_percent": top_tail_metrics(prediction_phys[:, 0], target_phys[:, 0], quantile=0.95),
        "qq_summary": {
            "mld": qq_summary(prediction_phys[:, 0], target_phys[:, 0], qq_quantiles),
            "N2": qq_summary(prediction_phys[:, 1], target_phys[:, 1], qq_quantiles),
        },
        "attention": {
            "num_steps": int(mean_attention.shape[0]),
            "top_hours": [int(x) for x in np.argsort(mean_attention)[-10:][::-1]],
            "top_weights": [float(mean_attention[int(x)]) for x in np.argsort(mean_attention)[-10:][::-1]],
        },
        "experts": {
            "num_experts": int(mean_gates.shape[0]),
            "kernel_sizes": [int(value) for value in checkpoint_overrides.get("expert_kernel_sizes", config.model.expert_kernel_sizes)],
            "mean_gates": [float(x) for x in mean_gates],
            "dominant_expert": int(np.argmax(mean_gates)),
        },
        "static_context": {
            "group_names": ["ocean_state", "geography", "seasonality"],
            "mean_group_gates": [float(x) for x in mean_static_groups],
            "dominant_group": int(np.argmax(mean_static_groups)),
        },
    }
    if mean_dynamic_segment_gates is not None:
        if mean_dynamic_segment_gates.shape[0] == len(config.dataset.dynamic_features):
            segment_names = list(config.dataset.dynamic_features)
            mean_key = "mean_feature_gates"
            dominant_key = "dominant_feature"
        else:
            segment_names = [
                f"{0 if idx == 0 else config.model.dynamic_summary_windows[idx - 1]}_{end}h"
                for idx, end in enumerate(config.model.dynamic_summary_windows)
            ]
            mean_key = "mean_segment_gates"
            dominant_key = "dominant_segment"
        report["dynamic_context"] = {
            "context_names": segment_names,
            mean_key: [float(x) for x in mean_dynamic_segment_gates],
            dominant_key: int(np.argmax(mean_dynamic_segment_gates)),
            "summary_windows": list(config.model.dynamic_summary_windows),
            "regime_gate_mean": float(mean_regime_gate) if mean_regime_gate is not None else float("nan"),
        }
    if uncertainty_report is not None:
        report["uncertainty"] = uncertainty_report
    print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config.paths.metrics_path.parent / f"{args.split}_physical_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
