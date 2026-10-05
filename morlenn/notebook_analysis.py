from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config_text
from .data import MonthlyNetCDFDataset, RobustStats, denormalize_targets, list_netcdf_files, stats_from_dict
from .model import UpperDynTCNAttentionModel, split_prediction_params


MLD_CLASS_BINS = [0.0, 50.0, 150.0, 300.0, np.inf]
MLD_CLASS_LABELS = ["0-50", "50-150", "150-300", "300+"]


@dataclass
class UpperDynRunAnalysis:
    run: dict[str, object]
    analysis_frame: pd.DataFrame
    static_feature_columns: list[str]
    dynamic_summary_columns: list[str]


def _denormalize_static(static_norm: np.ndarray, stats: RobustStats, feature_names: list[str]) -> pd.DataFrame:
    static_phys = static_norm * stats.static_iqr[None, :] + stats.static_median[None, :]
    return pd.DataFrame(static_phys, columns=feature_names)


def _summarize_dynamic(dynamic_norm: np.ndarray, stats: RobustStats, feature_names: list[str]) -> pd.DataFrame:
    dynamic_phys = dynamic_norm * stats.dynamic_iqr[None, :, None] + stats.dynamic_median[None, :, None]
    rows: dict[str, np.ndarray] = {}
    recent24 = slice(-24, None)
    recent72 = slice(-72, None)
    for index, feature in enumerate(feature_names):
        values = dynamic_phys[:, index, :]
        rows[f"{feature}__mean"] = values.mean(axis=1)
        rows[f"{feature}__std"] = values.std(axis=1)
        rows[f"{feature}__min"] = values.min(axis=1)
        rows[f"{feature}__max"] = values.max(axis=1)
        rows[f"{feature}__last"] = values[:, -1]
        rows[f"{feature}__recent24_mean"] = values[:, recent24].mean(axis=1)
        rows[f"{feature}__recent72_mean"] = values[:, recent72].mean(axis=1)
        rows[f"{feature}__recent24_std"] = values[:, recent24].std(axis=1)
        rows[f"{feature}__delta_last_recent72"] = values[:, -1] - values[:, recent72].mean(axis=1)
    return pd.DataFrame(rows)


def _fit_model(
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_overrides: dict[str, object] | None = None,
) -> UpperDynTCNAttentionModel:
    checkpoint_overrides = checkpoint_overrides or {}
    return UpperDynTCNAttentionModel(
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
    ).to(device)


def _load_checkpoint_state(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]  # type: ignore[assignment]
    elif "model_state" in checkpoint:
        state = checkpoint["model_state"]  # type: ignore[assignment]
    else:
        raise KeyError("Checkpoint does not contain 'model_state_dict' or 'model_state'.")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model state is not a state_dict.")
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}  # type: ignore[return-value]
    return state  # type: ignore[return-value]


def _infer_model_overrides_from_checkpoint(
    checkpoint: dict[str, object],
    config: ExperimentConfig,
) -> dict[str, object]:
    state = _load_checkpoint_state(checkpoint)
    expert_kernel_sizes = [
        int(state[key].shape[-1])
        for key in sorted(
            (name for name in state if name.startswith("experts.") and name.endswith("temporal_filter.weight")),
            key=lambda name: int(name.split(".")[1]),
        )
    ]
    has_dynamic_context = any(name.startswith("dynamic_context_encoder.") for name in state)
    if isinstance(checkpoint.get("dynamic_context_mode"), str):
        dynamic_context_mode = str(checkpoint["dynamic_context_mode"])
    elif any(
        name.startswith("dynamic_context_encoder.sequence_proj.")
        or name.startswith("dynamic_context_encoder.context_proj.")
        or name.startswith("dynamic_context_encoder.score_proj.")
        for name in state
    ):
        dynamic_context_mode = "latent_attention"
    elif any(name.startswith("dynamic_context_encoder.segment_proj.") or name.startswith("dynamic_context_encoder.segment_gate.") for name in state):
        dynamic_context_mode = "latent_segments"
    elif any(name.startswith("dynamic_context_encoder.feature_gate.") for name in state):
        dynamic_context_mode = "legacy_raw_summaries"
    else:
        dynamic_context_mode = "latent_segments"
    predictive_distribution = "gaussian" if any(name.startswith("mld_std_head.") for name in state) else "deterministic"
    if isinstance(checkpoint.get("attention_pooling_mode"), str):
        attention_pooling_mode = str(checkpoint["attention_pooling_mode"])
    elif any(".segment_proj.0.weight" in name for name in state if name.startswith("experts.")):
        attention_pooling_mode = "segment_tokens"
    elif any(".zone_score_proj.weight" in name for name in state if name.startswith("experts.")):
        attention_pooling_mode = "hierarchical_zone_attention"
    elif any(".sequence_proj.weight" in name for name in state if name.startswith("experts.")):
        attention_pooling_mode = "pointwise_attention"
    else:
        attention_pooling_mode = config.model.attention_pooling_mode
    attention_tokens_per_segment = int(checkpoint.get("attention_tokens_per_segment", config.model.attention_tokens_per_segment))
    return {
        "use_dynamic_context": has_dynamic_context,
        "dynamic_context_mode": dynamic_context_mode,
        "attention_pooling_mode": attention_pooling_mode,
        "attention_tokens_per_segment": attention_tokens_per_segment,
        "expert_kernel_sizes": expert_kernel_sizes or list(config.model.expert_kernel_sizes),
        "predictive_distribution": predictive_distribution,
    }


def _apply_checkpoint_snapshot(
    config: ExperimentConfig,
    stats: RobustStats,
    checkpoint: dict[str, object],
    checkpoint_path: str | Path | None = None,
) -> tuple[ExperimentConfig, RobustStats]:
    if "config_text" in checkpoint:
        config = load_config_text(str(checkpoint["config_text"]))
    if "stats" in checkpoint:
        stats = stats_from_dict(checkpoint["stats"])  # type: ignore[arg-type]
    elif checkpoint_path is not None:
        manifest_path = Path(checkpoint_path).with_name("run_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "config_text" in manifest:
                config = load_config_text(str(manifest["config_text"]))
            if "stats" in manifest:
                stats = stats_from_dict(manifest["stats"])  # type: ignore[arg-type]
    return config, stats


def _safe_corr(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2:
        return float("nan")
    if np.allclose(obs.std(), 0.0) or np.allclose(pred.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(obs, pred)[0, 1])


def _metric_row(scenario: str, group: str, obs: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    err = pred - obs
    obs_std = float(obs.std()) if len(obs) else float("nan")
    pred_std = float(pred.std()) if len(pred) else float("nan")
    return {
        "scenario": scenario,
        "group": group,
        "count": int(len(obs)),
        "rmse": float(np.sqrt(np.mean(err**2))) if len(obs) else float("nan"),
        "bias": float(err.mean()) if len(obs) else float("nan"),
        "mae": float(np.abs(err).mean()) if len(obs) else float("nan"),
        "corr": _safe_corr(obs, pred),
        "obs_std": obs_std,
        "pred_std": pred_std,
        "std_ratio": float(pred_std / obs_std) if len(obs) and obs_std > 0.0 else float("nan"),
    }


def load_upperdyn_run_analysis(
    config: ExperimentConfig,
    stats: RobustStats,
    checkpoint_path: str | Path,
    device: torch.device,
    batch_size: int = 256,
) -> UpperDynRunAnalysis:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    dataset = MonthlyNetCDFDataset(
        files=list_netcdf_files(config.paths.val_dir),
        dynamic_features=config.dataset.dynamic_features,
        static_features=config.dataset.static_features,
        targets=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
        stats=stats,
        replace_nan_with_zero=config.dataset.replace_nan_with_zero,
        clip_mld_max=config.dataset.clip_mld_max,
        strict_static_features=config.dataset.strict_static_features,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_overrides = _infer_model_overrides_from_checkpoint(checkpoint, config)
    model = _fit_model(config, device, checkpoint_overrides=checkpoint_overrides)
    model.load_state_dict(_load_checkpoint_state(checkpoint))
    model.eval()

    pred_mean_all: list[np.ndarray] = []
    pred_std_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    attn_all: list[np.ndarray] = []
    expert_attn_all: list[np.ndarray] = []
    expert_score_all: list[np.ndarray] = []
    latent_norm_all: list[np.ndarray] = []
    latent_abs_mean_all: list[np.ndarray] = []
    gate_all: list[np.ndarray] = []
    static_gate_all: list[np.ndarray] = []
    dynamic_segment_gate_all: list[np.ndarray] = []
    dynamic_context_attention_all: list[np.ndarray] = []
    regime_gate_all: list[np.ndarray] = []
    fusion_gate_all: list[np.ndarray] = []
    zone_weight_all: list[np.ndarray] = []
    static_frames: list[pd.DataFrame] = []
    dynamic_frames: list[pd.DataFrame] = []

    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
            static = batch["static"].to(device=device, dtype=torch.float32)
            pred_output, diagnostics = model(dynamic, static, return_diagnostics=True)

            pred_mean, pred_std = split_prediction_params(pred_output, len(config.dataset.targets))
            pred_mean_all.append(pred_mean.cpu().numpy())
            if pred_std is not None:
                pred_std_all.append(pred_std.cpu().numpy())
            target_all.append(batch["target"].cpu().numpy())
            attn_all.append(diagnostics["combined_attention"].cpu().numpy())
            expert_attn_all.append(diagnostics["expert_attention"].cpu().numpy())
            expert_score_all.append(diagnostics["expert_attention_scores"].cpu().numpy())
            latent_norm_all.append(diagnostics["latent_sequence_l2"].cpu().numpy())
            latent_abs_mean_all.append(diagnostics["latent_sequence_abs_mean"].cpu().numpy())
            gate_all.append(diagnostics["expert_gates"].cpu().numpy())
            static_gate_all.append(diagnostics["static_group_gates"].cpu().numpy())
            if "dynamic_segment_gates" in diagnostics:
                dynamic_segment_gate_all.append(diagnostics["dynamic_segment_gates"].cpu().numpy())
            elif "dynamic_feature_gates" in diagnostics:
                dynamic_segment_gate_all.append(diagnostics["dynamic_feature_gates"].cpu().numpy())
            if "dynamic_context_attention" in diagnostics:
                dynamic_context_attention_all.append(diagnostics["dynamic_context_attention"].cpu().numpy())
            if "regime_gate" in diagnostics:
                regime_gate_all.append(diagnostics["regime_gate"].cpu().numpy())
            if "combined_zone_weights" in diagnostics:
                zone_weight_all.append(diagnostics["combined_zone_weights"].cpu().numpy())
            fusion_gate_all.append(diagnostics["fusion_gate"].cpu().numpy())
            static_frames.append(
                _denormalize_static(batch["static"].cpu().numpy(), stats, config.dataset.static_features)
            )
            dynamic_frames.append(
                _summarize_dynamic(batch["dynamic"].cpu().numpy(), stats, config.dataset.dynamic_features)
            )

    pred_norm = np.concatenate(pred_mean_all, axis=0)
    target_norm = np.concatenate(target_all, axis=0)
    pred_phys = denormalize_targets(
        pred_norm,
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
    pred_std_phys = np.concatenate(pred_std_all, axis=0) if pred_std_all else None

    static_frame = pd.concat(static_frames, ignore_index=True)
    dynamic_frame = pd.concat(dynamic_frames, ignore_index=True)
    analysis_frame = pd.concat([static_frame, dynamic_frame], axis=1)
    analysis_frame["obs_mld"] = target_phys[:, 0]
    analysis_frame["pred_mld"] = pred_phys[:, 0]
    analysis_frame["error_mld"] = analysis_frame["pred_mld"] - analysis_frame["obs_mld"]
    analysis_frame["abs_error_mld"] = analysis_frame["error_mld"].abs()
    analysis_frame["obs_n2"] = target_phys[:, 1]
    analysis_frame["pred_n2"] = pred_phys[:, 1]
    analysis_frame["error_n2"] = analysis_frame["pred_n2"] - analysis_frame["obs_n2"]
    analysis_frame["abs_error_n2"] = analysis_frame["error_n2"].abs()
    analysis_frame["obs_mld_class"] = pd.cut(
        analysis_frame["obs_mld"],
        bins=MLD_CLASS_BINS,
        labels=MLD_CLASS_LABELS,
        right=False,
        include_lowest=True,
    )
    if pred_std_phys is not None:
        analysis_frame["pred_std_mld"] = pred_std_phys[:, 0]
        analysis_frame["pred_std_n2"] = pred_std_phys[:, 1]
        analysis_frame["pred_std_mld_quantile"] = pd.qcut(
            analysis_frame["pred_std_mld"],
            q=5,
            labels=[f"Q{i}" for i in range(1, 6)],
            duplicates="drop",
        )
        analysis_frame["pred_std_n2_quantile"] = pd.qcut(
            analysis_frame["pred_std_n2"],
            q=5,
            labels=[f"Q{i}" for i in range(1, 6)],
            duplicates="drop",
        )

    mask = analysis_frame["obs_mld"] > 0
    analysis_frame = analysis_frame.loc[mask].reset_index(drop=True)
    attn = np.concatenate(attn_all, axis=0)[mask.to_numpy()]
    expert_attn = np.concatenate(expert_attn_all, axis=0)[mask.to_numpy()]
    expert_scores = np.concatenate(expert_score_all, axis=0)[mask.to_numpy()]
    latent_norm = np.concatenate(latent_norm_all, axis=0)[mask.to_numpy()]
    latent_abs_mean = np.concatenate(latent_abs_mean_all, axis=0)[mask.to_numpy()]
    gates = np.concatenate(gate_all, axis=0)[mask.to_numpy()]
    static_gates = np.concatenate(static_gate_all, axis=0)[mask.to_numpy()]
    dynamic_segment_gates = (
        np.concatenate(dynamic_segment_gate_all, axis=0)[mask.to_numpy()] if dynamic_segment_gate_all else None
    )
    dynamic_context_attention = (
        np.concatenate(dynamic_context_attention_all, axis=0)[mask.to_numpy()]
        if dynamic_context_attention_all
        else None
    )
    regime_gates = np.concatenate(regime_gate_all, axis=0)[mask.to_numpy()] if regime_gate_all else None
    fusion_gates = np.concatenate(fusion_gate_all, axis=0)[mask.to_numpy()]
    for expert_idx in range(gates.shape[1]):
        analysis_frame[f"expert_gate_{expert_idx}"] = gates[:, expert_idx]
    static_group_names = ["ocean_state", "geography", "seasonality"]
    for group_idx, group_name in enumerate(static_group_names):
        analysis_frame[f"static_gate_{group_name}"] = static_gates[:, group_idx]
    if dynamic_segment_gates is not None:
        if dynamic_segment_gates.shape[1] == len(config.dataset.dynamic_features):
            segment_names = list(config.dataset.dynamic_features)
        else:
            segment_names = [f"{0 if idx == 0 else config.model.dynamic_summary_windows[idx - 1]}_{end}h" for idx, end in enumerate(config.model.dynamic_summary_windows)]
        for segment_idx, segment_name in enumerate(segment_names):
            analysis_frame[f"dynamic_context_gate_{segment_name}"] = dynamic_segment_gates[:, segment_idx]
    if regime_gates is not None:
        analysis_frame["regime_gate_mean"] = regime_gates.mean(axis=1)
    if zone_weight_all:
        zone_weights = np.concatenate(zone_weight_all, axis=0)[mask.to_numpy()]
        zone_names = [str(name) for name in diagnostics["zone_names"]]
        for zone_idx, zone_name in enumerate(zone_names):
            analysis_frame[f"zone_weight_{zone_name}"] = zone_weights[:, zone_idx]
    analysis_frame["fusion_gate_mean"] = fusion_gates.mean(axis=1)
    analysis_frame["attention_peak_hour"] = attn.shape[1] - 1 - attn.argmax(axis=1)
    analysis_frame["attention_entropy"] = -(attn * np.log(np.clip(attn, 1e-12, None))).sum(axis=1)
    if dynamic_context_attention is not None:
        analysis_frame["dynamic_context_attention_peak_hour"] = (
            dynamic_context_attention.shape[1] - 1 - dynamic_context_attention.argmax(axis=1)
        )
        analysis_frame["dynamic_context_attention_entropy"] = -(
            dynamic_context_attention * np.log(np.clip(dynamic_context_attention, 1e-12, None))
        ).sum(axis=1)

    run = {
        "mld": analysis_frame["obs_mld"].to_numpy(),
        "mlotst": analysis_frame["pred_mld"].to_numpy(),
        "N2": analysis_frame["obs_n2"].to_numpy(),
        "N2otst": analysis_frame["pred_n2"].to_numpy(),
        "attention_per_sample": attn,
        "attention_mean": attn.mean(axis=0),
        "expert_attention_per_sample": expert_attn,
        "expert_attention_mean": expert_attn.mean(axis=0),
        "expert_score_per_sample": expert_scores,
        "expert_score_mean": expert_scores.mean(axis=0),
        "latent_norm_per_sample": latent_norm,
        "latent_norm_mean": latent_norm.mean(axis=0),
        "latent_abs_mean_per_sample": latent_abs_mean,
        "latent_abs_mean_mean": latent_abs_mean.mean(axis=0),
        "expert_gates_mean": gates.mean(axis=0),
        "static_group_gates_mean": static_gates.mean(axis=0),
        "fusion_gate_mean": fusion_gates.mean(axis=0),
        "expert_kernel_sizes": np.asarray(checkpoint_overrides.get("expert_kernel_sizes", config.model.expert_kernel_sizes)),
        "attention_pooling_mode": checkpoint_overrides.get("attention_pooling_mode", config.model.attention_pooling_mode),
        "attention_tokens_per_segment": checkpoint_overrides.get("attention_tokens_per_segment", config.model.attention_tokens_per_segment),
        "checkpoint_meta": checkpoint,
    }
    if dynamic_context_attention is not None:
        run["dynamic_context_attention_per_sample"] = dynamic_context_attention
        run["dynamic_context_attention_mean"] = dynamic_context_attention.mean(axis=0)
    if dynamic_segment_gates is not None:
        if dynamic_segment_gates.shape[1] == len(config.dataset.dynamic_features):
            context_names = np.asarray(list(config.dataset.dynamic_features))
            run["dynamic_feature_gates_mean"] = dynamic_segment_gates.mean(axis=0)
        else:
            context_names = np.asarray(
                [f"{0 if idx == 0 else config.model.dynamic_summary_windows[idx - 1]}_{end}h" for idx, end in enumerate(config.model.dynamic_summary_windows)]
            )
            run["dynamic_segment_gates_mean"] = dynamic_segment_gates.mean(axis=0)
            run["dynamic_segment_names"] = context_names
        run["dynamic_context_gates_mean"] = dynamic_segment_gates.mean(axis=0)
        run["dynamic_context_names"] = context_names
    if regime_gates is not None:
        run["regime_gate_mean"] = regime_gates.mean(axis=0)
    if zone_weight_all:
        zone_weights = np.concatenate(zone_weight_all, axis=0)[mask.to_numpy()]
        zone_names = np.asarray([str(name) for name in diagnostics["zone_names"]])
        run["zone_names"] = zone_names
        run["combined_zone_weights_per_sample"] = zone_weights
        run["combined_zone_weights_mean"] = zone_weights.mean(axis=0)
    if pred_std_phys is not None:
        run["mld_std_phys"] = analysis_frame["pred_std_mld"].to_numpy()
        run["n2_std_phys"] = analysis_frame["pred_std_n2"].to_numpy()
    return UpperDynRunAnalysis(
        run=run,
        analysis_frame=analysis_frame,
        static_feature_columns=list(static_frame.columns),
        dynamic_summary_columns=list(dynamic_frame.columns),
    )


def run_temporal_ablation_study(
    config: ExperimentConfig,
    stats: RobustStats,
    checkpoint_path: str | Path,
    device: torch.device,
    batch_size: int = 256,
    scenarios: dict[str, list[tuple[int, int]]] | None = None,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    dataset = MonthlyNetCDFDataset(
        files=list_netcdf_files(config.paths.val_dir),
        dynamic_features=config.dataset.dynamic_features,
        static_features=config.dataset.static_features,
        targets=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
        stats=stats,
        replace_nan_with_zero=config.dataset.replace_nan_with_zero,
        clip_mld_max=config.dataset.clip_mld_max,
        strict_static_features=config.dataset.strict_static_features,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_overrides = _infer_model_overrides_from_checkpoint(checkpoint, config)
    model = _fit_model(config, device, checkpoint_overrides=checkpoint_overrides)
    model.load_state_dict(_load_checkpoint_state(checkpoint))
    model.eval()

    if scenarios is None:
        scenarios = {
            "baseline": [],
            # Edge-focused tests
            "mask_oldest_edge_0_1h": [(0, 1)],
            "mask_oldest_edge_0_6h": [(0, 6)],
            "mask_oldest_edge_0_24h": [(0, 24)],
            "mask_recent_edge_0_1h": [(720 - 1, 720)],
            "mask_recent_edge_0_6h": [(720 - 6, 720)],
            "mask_recent_edge_0_24h": [(720 - 24, 720)],
            # Absolute windows in hours before collocation: 0 = most recent, 720 = oldest edge.
            "mask_hours_before_24_72h": [(720 - 72, 720 - 24)],
            "mask_hours_before_72_168h": [(720 - 168, 720 - 72)],
            "mask_hours_before_168_360h": [(720 - 360, 720 - 168)],
            "mask_hours_before_360_720h": [(0, 720 - 360)],
        }

    target_all: list[np.ndarray] = []
    pred_all: dict[str, list[np.ndarray]] = {name: [] for name in scenarios}

    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
            static = batch["static"].to(device=device, dtype=torch.float32)
            target_norm = batch["target"].cpu().numpy()
            target_phys = denormalize_targets(
                target_norm,
                stats,
                target_names=config.dataset.targets,
                mld_target_transform=config.dataset.mld_target_transform,
            )
            target_all.append(target_phys)

            for scenario_name, segments in scenarios.items():
                masked_dynamic = dynamic.clone()
                for start, end in segments:
                    start_idx = max(0, min(int(start), masked_dynamic.shape[-1]))
                    end_idx = max(start_idx, min(int(end), masked_dynamic.shape[-1]))
                    masked_dynamic[:, :, start_idx:end_idx] = 0.0
                pred_output = model(masked_dynamic, static)
                pred_mean, _ = split_prediction_params(pred_output, len(config.dataset.targets))
                pred_phys = denormalize_targets(
                    pred_mean.cpu().numpy(),
                    stats,
                    target_names=config.dataset.targets,
                    mld_target_transform=config.dataset.mld_target_transform,
                )
                pred_all[scenario_name].append(pred_phys)

    target_phys_all = np.concatenate(target_all, axis=0)
    obs_mld = target_phys_all[:, 0]

    rows: list[dict[str, object]] = []
    for scenario_name, pred_chunks in pred_all.items():
        pred_phys_all = np.concatenate(pred_chunks, axis=0)
        pred_mld = pred_phys_all[:, 0]
        rows.append(_metric_row(scenario_name, "global", obs_mld, pred_mld))
        for lower, upper, label in zip(MLD_CLASS_BINS[:-1], MLD_CLASS_BINS[1:], MLD_CLASS_LABELS):
            mask = (obs_mld >= lower) & (obs_mld < upper)
            if mask.any():
                rows.append(_metric_row(scenario_name, label, obs_mld[mask], pred_mld[mask]))
    return pd.DataFrame(rows)


def run_dynamic_variable_ablation_study(
    config: ExperimentConfig,
    stats: RobustStats,
    checkpoint_path: str | Path,
    device: torch.device,
    batch_size: int = 256,
    windows: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    dataset = MonthlyNetCDFDataset(
        files=list_netcdf_files(config.paths.val_dir),
        dynamic_features=config.dataset.dynamic_features,
        static_features=config.dataset.static_features,
        targets=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
        stats=stats,
        replace_nan_with_zero=config.dataset.replace_nan_with_zero,
        clip_mld_max=config.dataset.clip_mld_max,
        strict_static_features=config.dataset.strict_static_features,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_overrides = _infer_model_overrides_from_checkpoint(checkpoint, config)
    model = _fit_model(config, device, checkpoint_overrides=checkpoint_overrides)
    model.load_state_dict(_load_checkpoint_state(checkpoint))
    model.eval()

    if windows is None:
        windows = [(0, 24), (24, 72), (72, 168), (168, 360), (360, 720)]

    scenario_specs = [("baseline", None, None)]
    for feature_idx, feature_name in enumerate(config.dataset.dynamic_features):
        for start_hour, end_hour in windows:
            scenario_specs.append(
                (
                    f"{feature_name}__hours_before_{start_hour}_{end_hour}h",
                    feature_idx,
                    (int(start_hour), int(end_hour)),
                )
            )

    target_all: list[np.ndarray] = []
    pred_all: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in scenario_specs}
    gate_all: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in scenario_specs}
    seq_len = None

    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
            static = batch["static"].to(device=device, dtype=torch.float32)
            if seq_len is None:
                seq_len = int(dynamic.shape[-1])
            target_norm = batch["target"].cpu().numpy()
            target_phys = denormalize_targets(
                target_norm,
                stats,
                target_names=config.dataset.targets,
                mld_target_transform=config.dataset.mld_target_transform,
            )
            target_all.append(target_phys)

            for scenario_name, feature_idx, window in scenario_specs:
                masked_dynamic = dynamic.clone()
                if feature_idx is not None and window is not None:
                    start_hour, end_hour = window
                    start_idx = max(0, min(seq_len, seq_len - int(end_hour)))
                    end_idx = max(start_idx, min(seq_len, seq_len - int(start_hour)))
                    masked_dynamic[:, feature_idx, start_idx:end_idx] = 0.0
                pred_output, diagnostics = model(masked_dynamic, static, return_diagnostics=True)
                pred_mean, _ = split_prediction_params(pred_output, len(config.dataset.targets))
                pred_phys = denormalize_targets(
                    pred_mean.cpu().numpy(),
                    stats,
                    target_names=config.dataset.targets,
                    mld_target_transform=config.dataset.mld_target_transform,
                )
                pred_all[scenario_name].append(pred_phys)
                gate_all[scenario_name].append(diagnostics["expert_gates"].cpu().numpy())

    target_phys_all = np.concatenate(target_all, axis=0)
    obs_mld = target_phys_all[:, 0]
    rows: list[dict[str, object]] = []
    for scenario_name, feature_idx, window in scenario_specs:
        pred_phys_all = np.concatenate(pred_all[scenario_name], axis=0)
        expert_gates = np.concatenate(gate_all[scenario_name], axis=0)
        pred_mld = pred_phys_all[:, 0]

        group_specs = [("global", np.ones_like(obs_mld, dtype=bool))]
        group_specs.extend(
            (
                label,
                (obs_mld >= lower) & (obs_mld < upper),
            )
            for lower, upper, label in zip(MLD_CLASS_BINS[:-1], MLD_CLASS_BINS[1:], MLD_CLASS_LABELS)
        )

        for group_name, mask in group_specs:
            if not mask.any():
                continue
            row = _metric_row(scenario_name, group_name, obs_mld[mask], pred_mld[mask])
            row["variable"] = feature_idx if feature_idx is None else config.dataset.dynamic_features[feature_idx]
            row["window"] = "baseline" if window is None else f"{window[0]}_{window[1]}h"
            if feature_idx is None:
                row["feature_index"] = -1
                row["window_start_hour"] = np.nan
                row["window_end_hour"] = np.nan
            else:
                row["feature_index"] = int(feature_idx)
                row["window_start_hour"] = int(window[0])
                row["window_end_hour"] = int(window[1])
            gate_subset = expert_gates[mask]
            for expert_idx in range(gate_subset.shape[1]):
                row[f"expert_gate_{expert_idx}_mean"] = float(gate_subset[:, expert_idx].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def run_input_group_ablation_study(
    config: ExperimentConfig,
    stats: RobustStats,
    checkpoint_path: str | Path,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config, stats = _apply_checkpoint_snapshot(config, stats, checkpoint, checkpoint_path=checkpoint_path)
    dataset = MonthlyNetCDFDataset(
        files=list_netcdf_files(config.paths.val_dir),
        dynamic_features=config.dataset.dynamic_features,
        static_features=config.dataset.static_features,
        targets=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
        stats=stats,
        replace_nan_with_zero=config.dataset.replace_nan_with_zero,
        clip_mld_max=config.dataset.clip_mld_max,
        strict_static_features=config.dataset.strict_static_features,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_overrides = _infer_model_overrides_from_checkpoint(checkpoint, config)
    model = _fit_model(config, device, checkpoint_overrides=checkpoint_overrides)
    model.load_state_dict(_load_checkpoint_state(checkpoint))
    model.eval()

    static_index = {name: idx for idx, name in enumerate(config.dataset.static_features)}
    static_group_indices = {
        "sea_state": [static_index[name] for name in ("analysed_sst", "sss", "adt") if name in static_index],
        "geography": [static_index[name] for name in ("geo_x", "geo_y", "geo_z", "elevation") if name in static_index],
        "seasonality": [static_index[name] for name in ("doy_sin", "doy_cos") if name in static_index],
    }
    all_static_indices = sorted(idx for indices in static_group_indices.values() for idx in indices)

    scenario_specs: list[tuple[str, str, list[int]]] = [
        ("baseline", "none", []),
        ("freeze_dynamic_to_t0", "freeze_to_t0", []),
        ("mask_static_all", "none", all_static_indices),
    ]
    scenario_specs.extend(
        (f"mask_static_{group_name}", "none", indices)
        for group_name, indices in static_group_indices.items()
        if indices
    )

    target_all: list[np.ndarray] = []
    pred_all: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in scenario_specs}
    expert_gate_all: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in scenario_specs}
    static_gate_all: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in scenario_specs}

    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
            static = batch["static"].to(device=device, dtype=torch.float32)
            target_norm = batch["target"].cpu().numpy()
            target_phys = denormalize_targets(
                target_norm,
                stats,
                target_names=config.dataset.targets,
                mld_target_transform=config.dataset.mld_target_transform,
            )
            target_all.append(target_phys)

            for scenario_name, dynamic_mode, static_indices in scenario_specs:
                masked_dynamic = dynamic.clone()
                masked_static = static.clone()
                if dynamic_mode == "freeze_to_t0":
                    # Dataset time axis is ordered oldest -> most recent after loading,
                    # so index -1 corresponds to the collocated instantaneous state (t=0).
                    masked_dynamic = masked_dynamic[:, :, -1:].expand_as(masked_dynamic).clone()
                elif dynamic_mode == "zero":
                    masked_dynamic.zero_()
                if static_indices:
                    masked_static[:, static_indices] = 0.0
                pred_output, diagnostics = model(masked_dynamic, masked_static, return_diagnostics=True)
                pred_mean, _ = split_prediction_params(pred_output, len(config.dataset.targets))
                pred_phys = denormalize_targets(
                    pred_mean.cpu().numpy(),
                    stats,
                    target_names=config.dataset.targets,
                    mld_target_transform=config.dataset.mld_target_transform,
                )
                pred_all[scenario_name].append(pred_phys)
                expert_gate_all[scenario_name].append(diagnostics["expert_gates"].cpu().numpy())
                static_gate_all[scenario_name].append(diagnostics["static_group_gates"].cpu().numpy())

    target_phys_all = np.concatenate(target_all, axis=0)
    obs_mld = target_phys_all[:, 0]
    rows: list[dict[str, object]] = []
    static_gate_names = ["ocean_state", "geography", "seasonality"]

    for scenario_name, dynamic_mode, static_indices in scenario_specs:
        pred_phys_all = np.concatenate(pred_all[scenario_name], axis=0)
        expert_gates = np.concatenate(expert_gate_all[scenario_name], axis=0)
        static_gates = np.concatenate(static_gate_all[scenario_name], axis=0)
        pred_mld = pred_phys_all[:, 0]

        group_specs = [("global", np.ones_like(obs_mld, dtype=bool))]
        group_specs.extend(
            (
                label,
                (obs_mld >= lower) & (obs_mld < upper),
            )
            for lower, upper, label in zip(MLD_CLASS_BINS[:-1], MLD_CLASS_BINS[1:], MLD_CLASS_LABELS)
        )

        for group_name, mask in group_specs:
            if not mask.any():
                continue
            row = _metric_row(scenario_name, group_name, obs_mld[mask], pred_mld[mask])
            row["dynamic_ablation_mode"] = dynamic_mode
            row["masked_static_group"] = (
                "none"
                if not static_indices
                else "all"
                if scenario_name == "mask_static_all"
                else scenario_name.removeprefix("mask_static_")
            )
            gate_subset = expert_gates[mask]
            static_gate_subset = static_gates[mask]
            for expert_idx in range(gate_subset.shape[1]):
                row[f"expert_gate_{expert_idx}_mean"] = float(gate_subset[:, expert_idx].mean())
            for group_idx, static_name in enumerate(static_gate_names):
                row[f"static_gate_{static_name}_mean"] = float(static_gate_subset[:, group_idx].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def grouped_error_metrics(frame: pd.DataFrame, group_col: str, target: str) -> pd.DataFrame:
    obs_col = f"obs_{target}"
    pred_col = f"pred_{target}"
    err_col = f"error_{target}"
    abs_err_col = f"abs_error_{target}"
    std_col = f"pred_std_{target}" if f"pred_std_{target}" in frame.columns else None

    rows: list[dict[str, object]] = []
    for group_value, sub in frame.groupby(group_col, dropna=False):
        obs = sub[obs_col].to_numpy()
        pred = sub[pred_col].to_numpy()
        err = sub[err_col].to_numpy()
        corr = np.corrcoef(obs, pred)[0, 1] if len(sub) > 1 else np.nan
        row: dict[str, object] = {
            "group": group_value,
            "count": int(len(sub)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
            "mae": float(np.mean(np.abs(err))),
            "corr": float(corr),
            "obs_std": float(np.std(obs)),
            "pred_std": float(np.std(pred)),
            "std_ratio": float(np.std(pred) / (np.std(obs) + 1e-6)),
            "median_abs_error": float(np.median(sub[abs_err_col])),
        }
        if std_col is not None:
            sigma = sub[std_col].to_numpy()
            row["predicted_sigma_mean"] = float(np.mean(sigma))
            row["one_sigma_coverage"] = float(np.mean(np.abs(err) <= np.maximum(sigma, 1e-6)))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_correlation_table(
    frame: pd.DataFrame,
    feature_columns: list[str],
    response_column: str,
    top_n: int = 15,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    response = pd.to_numeric(frame[response_column], errors="coerce")
    for feature in feature_columns:
        series = pd.to_numeric(frame[feature], errors="coerce")
        valid = series.notna() & response.notna()
        if valid.sum() < 8 or series[valid].nunique() < 4:
            continue
        pearson = float(series[valid].corr(response[valid], method="pearson"))
        spearman = float(series[valid].corr(response[valid], method="spearman"))
        rows.append(
            {
                "feature": feature,
                "pearson": pearson,
                "spearman": spearman,
                "abs_spearman": abs(spearman),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "pearson", "spearman", "abs_spearman"])
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).head(top_n).reset_index(drop=True)


def feature_profile_table(
    frame: pd.DataFrame,
    feature_columns: list[str],
    group_col: str,
) -> pd.DataFrame:
    grouped = frame.groupby(group_col)[feature_columns].mean(numeric_only=True)
    return grouped.transpose()


def uncertainty_calibration_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    std_col = f"pred_std_{target}"
    if std_col not in frame.columns:
        return pd.DataFrame()
    bin_col = f"{std_col}_quantile_tmp"
    tmp = frame.copy()
    tmp[bin_col] = pd.qcut(tmp[std_col], q=5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop")
    out = grouped_error_metrics(tmp, bin_col, target)
    return out.rename(columns={"group": "sigma_quantile"})


def grouped_gate_table(
    frame: pd.DataFrame,
    group_col: str,
    gate_columns: list[str],
) -> pd.DataFrame:
    if not gate_columns:
        return pd.DataFrame()
    grouped = frame.groupby(group_col, dropna=False)[gate_columns].mean(numeric_only=True)
    return grouped.reset_index()
