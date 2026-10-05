from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


@dataclass
class PathsConfig:
    train_dir: Path
    val_dir: Path
    stats_path: Path
    checkpoint_dir: Path
    metrics_path: Path
    tensorboard_dir: Path


@dataclass
class DatasetConfig:
    dynamic_features: list[str]
    static_features: list[str]
    strict_static_features: list[str]
    targets: list[str]
    mld_target_transform: str
    replace_nan_with_zero: bool
    clip_mld_max: float | None
    train_precomputed_dir: Path | None
    val_precomputed_dir: Path | None
    dynamic_history_hours: int | None
    eddy_max_distance_radius: float | None
    mld_subset_min: float | None
    mld_subset_max: float | None
    eddy_balance: bool
    neighbor_train_path: Path | None
    neighbor_val_path: Path | None
    # Global mode: one precompute + one neighbor pool + coords, split by year at train time.
    precomputed_dir: Path | None
    coords_path: Path | None
    neighbor_path: Path | None
    val_years: list[int]


@dataclass
class ModelConfig:
    tcn_channels: list[int]
    dilations: list[int]
    kernel_size: int
    dropout: float
    static_hidden_dim: int
    static_num_heads: int
    use_dynamic_context: bool
    dynamic_context_mode: str
    dynamic_context_hidden_dim: int
    dynamic_summary_kernel_size: int
    dynamic_summary_windows: list[int]
    film_hidden_dim: int
    attention_hidden_dim: int
    attention_pooling_mode: str
    attention_tokens_per_segment: int
    num_attention_experts: int
    expert_kernel_sizes: list[int]
    fusion_hidden_dim: int
    backbone_hidden_dim: int
    predictive_distribution: str
    min_std: float
    mld_branch_hidden_dim: int
    mld_branch_depth: int
    detach_n2_head_input: bool
    n2_separate_temporal_branch: bool
    n2_separate_attention_branch: bool
    n2_attention_hidden_dim: int
    n2_num_attention_experts: int
    n2_expert_kernel_sizes: list[int]
    n2_fusion_hidden_dim: int
    n2_backbone_hidden_dim: int
    n2_backbone_grad_scale: float
    n2_branch_hidden_dim: int
    n2_branch_depth: int
    mld_regime_split: bool
    num_mld_regimes: int
    regime_gate_hidden_dim: int
    mld_regime_attention_hidden_dim: int | None
    mld_regime_fusion_hidden_dim: int | None
    mld_regime_backbone_hidden_dim: int | None
    mld_regime_branch_hidden_dim: int | None
    mld_regime_branch_depth: int | None
    use_mld_neighbors: bool
    num_neighbors: int
    neighbor_feat_dim: int
    neighbor_hidden_dim: int
    neighbor_modality_dropout: float
    neighbor_increment: bool
    neighbor_increment_full: bool = False


@dataclass
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    device: str
    scheduler_patience: int
    scheduler_factor: float
    min_learning_rate: float
    early_stopping_patience: int
    selection_metric: str
    checkpoint_bias_weight: float
    checkpoint_std_weight: float
    checkpoint_tail_weight: float
    tail_quantile: float
    cuda_visible_devices: str | None
    deterministic: bool
    cudnn_benchmark: bool
    train_metric_batches: int | None
    finite_check_interval: int
    val_interval_steps: int
    max_steps: int | None
    init_from_checkpoint: Path | None
    freeze_backbone: bool


@dataclass
class LossConfig:
    mld_log_weight: float
    n2_task_weight: float
    mld_deep_thresholds: list[float]
    mld_deep_weights: list[float]
    mld_underprediction_weight: float
    mld_physical_weight: float
    mld_physical_delta: float
    mld_rmse_weight: float
    mld_tail_rmse_weight: float
    mld_spread_weight: float
    mld_corr_weight: float
    n2_rmse_weight: float
    n2_spread_weight: float
    n2_corr_weight: float
    n2_warmup_epochs: int
    n2_warmup_start_factor: float
    mld_tail_loss_quantile: float
    huber_delta: float
    uncertainty_weight: float
    mld_regime_thresholds: list[float]
    regime_gate_weight: float
    regime_load_balance_weight: float


@dataclass
class EvaluationConfig:
    legacy_val_dir: Path | None


@dataclass
class ExperimentConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig


def _parse_config(raw: dict) -> ExperimentConfig:
    return ExperimentConfig(
        paths=PathsConfig(
            train_dir=Path(raw["paths"]["train_dir"]),
            val_dir=Path(raw["paths"]["val_dir"]),
            stats_path=Path(raw["paths"]["stats_path"]),
            checkpoint_dir=Path(raw["paths"]["checkpoint_dir"]),
            metrics_path=Path(raw["paths"]["metrics_path"]),
            tensorboard_dir=Path(raw["paths"]["tensorboard_dir"]),
        ),
        dataset=DatasetConfig(
            dynamic_features=list(raw["dataset"]["dynamic_features"]),
            static_features=list(raw["dataset"]["static_features"]),
            strict_static_features=list(raw["dataset"].get("strict_static_features", [])),
            targets=list(raw["dataset"]["targets"]),
            mld_target_transform=str(raw["dataset"].get("mld_target_transform", "identity")),
            replace_nan_with_zero=bool(raw["dataset"]["replace_nan_with_zero"]),
            clip_mld_max=raw["dataset"].get("clip_mld_max"),
            train_precomputed_dir=(
                Path(raw["dataset"]["train_precomputed_dir"])
                if raw["dataset"].get("train_precomputed_dir") is not None
                else None
            ),
            val_precomputed_dir=(
                Path(raw["dataset"]["val_precomputed_dir"])
                if raw["dataset"].get("val_precomputed_dir") is not None
                else None
            ),
            dynamic_history_hours=(
                int(raw["dataset"]["dynamic_history_hours"])
                if raw["dataset"].get("dynamic_history_hours") is not None
                else None
            ),
            eddy_max_distance_radius=(
                float(raw["dataset"]["eddy_max_distance_radius"])
                if raw["dataset"].get("eddy_max_distance_radius") is not None
                else None
            ),
            mld_subset_min=(
                float(raw["dataset"]["mld_subset_min"])
                if raw["dataset"].get("mld_subset_min") is not None
                else None
            ),
            mld_subset_max=(
                float(raw["dataset"]["mld_subset_max"])
                if raw["dataset"].get("mld_subset_max") is not None
                else None
            ),
            eddy_balance=bool(raw["dataset"].get("eddy_balance", False)),
            neighbor_train_path=(
                Path(raw["dataset"]["neighbor_train_path"])
                if raw["dataset"].get("neighbor_train_path") is not None
                else None
            ),
            neighbor_val_path=(
                Path(raw["dataset"]["neighbor_val_path"])
                if raw["dataset"].get("neighbor_val_path") is not None
                else None
            ),
            precomputed_dir=(
                Path(raw["dataset"]["precomputed_dir"])
                if raw["dataset"].get("precomputed_dir") is not None
                else None
            ),
            coords_path=(
                Path(raw["dataset"]["coords_path"])
                if raw["dataset"].get("coords_path") is not None
                else None
            ),
            neighbor_path=(
                Path(raw["dataset"]["neighbor_path"])
                if raw["dataset"].get("neighbor_path") is not None
                else None
            ),
            val_years=[int(value) for value in raw["dataset"].get("val_years", [])],
        ),
        model=ModelConfig(
            tcn_channels=[int(value) for value in raw["model"]["tcn_channels"]],
            dilations=[int(value) for value in raw["model"]["dilations"]],
            kernel_size=int(raw["model"]["kernel_size"]),
            dropout=float(raw["model"]["dropout"]),
            static_hidden_dim=int(raw["model"]["static_hidden_dim"]),
            static_num_heads=int(raw["model"].get("static_num_heads", 4)),
            use_dynamic_context=bool(raw["model"].get("use_dynamic_context", False)),
            dynamic_context_mode=str(raw["model"].get("dynamic_context_mode", "latent_segments")),
            dynamic_context_hidden_dim=int(
                raw["model"].get("dynamic_context_hidden_dim", raw["model"]["static_hidden_dim"])
            ),
            dynamic_summary_kernel_size=int(raw["model"].get("dynamic_summary_kernel_size", 25)),
            dynamic_summary_windows=[int(value) for value in raw["model"].get("dynamic_summary_windows", [24, 72, 168, 360, 720])],
            film_hidden_dim=int(raw["model"].get("film_hidden_dim", raw["model"]["static_hidden_dim"])),
            attention_hidden_dim=int(raw["model"]["attention_hidden_dim"]),
            attention_pooling_mode=str(raw["model"].get("attention_pooling_mode", "segment_tokens")),
            attention_tokens_per_segment=int(raw["model"].get("attention_tokens_per_segment", 4)),
            num_attention_experts=int(raw["model"].get("num_attention_experts", 3)),
            expert_kernel_sizes=[int(value) for value in raw["model"].get("expert_kernel_sizes", [13, 49, 97])],
            fusion_hidden_dim=int(raw["model"]["fusion_hidden_dim"]),
            backbone_hidden_dim=int(raw["model"]["backbone_hidden_dim"]),
            predictive_distribution=str(raw["model"].get("predictive_distribution", "deterministic")),
            min_std=float(raw["model"].get("min_std", 1e-3)),
            mld_branch_hidden_dim=int(raw["model"].get("mld_branch_hidden_dim", raw["model"]["backbone_hidden_dim"])),
            mld_branch_depth=int(raw["model"].get("mld_branch_depth", 0)),
            detach_n2_head_input=bool(raw["model"].get("detach_n2_head_input", False)),
            n2_separate_temporal_branch=bool(raw["model"].get("n2_separate_temporal_branch", False)),
            n2_separate_attention_branch=bool(raw["model"].get("n2_separate_attention_branch", False)),
            n2_attention_hidden_dim=int(raw["model"].get("n2_attention_hidden_dim", raw["model"]["attention_hidden_dim"])),
            n2_num_attention_experts=int(raw["model"].get("n2_num_attention_experts", raw["model"].get("num_attention_experts", 3))),
            n2_expert_kernel_sizes=[
                int(value)
                for value in raw["model"].get(
                    "n2_expert_kernel_sizes",
                    raw["model"].get("expert_kernel_sizes", [13, 49, 97]),
                )
            ],
            n2_fusion_hidden_dim=int(raw["model"].get("n2_fusion_hidden_dim", raw["model"]["fusion_hidden_dim"])),
            n2_backbone_hidden_dim=int(raw["model"].get("n2_backbone_hidden_dim", raw["model"]["backbone_hidden_dim"])),
            n2_backbone_grad_scale=float(raw["model"].get("n2_backbone_grad_scale", 1.0)),
            n2_branch_hidden_dim=int(raw["model"].get("n2_branch_hidden_dim", raw["model"]["backbone_hidden_dim"])),
            n2_branch_depth=int(raw["model"].get("n2_branch_depth", 0)),
            mld_regime_split=bool(raw["model"].get("mld_regime_split", False)),
            num_mld_regimes=int(raw["model"].get("num_mld_regimes", 2)),
            regime_gate_hidden_dim=int(raw["model"].get("regime_gate_hidden_dim", 64)),
            mld_regime_attention_hidden_dim=(
                int(raw["model"]["mld_regime_attention_hidden_dim"])
                if raw["model"].get("mld_regime_attention_hidden_dim") is not None
                else None
            ),
            mld_regime_fusion_hidden_dim=(
                int(raw["model"]["mld_regime_fusion_hidden_dim"])
                if raw["model"].get("mld_regime_fusion_hidden_dim") is not None
                else None
            ),
            mld_regime_backbone_hidden_dim=(
                int(raw["model"]["mld_regime_backbone_hidden_dim"])
                if raw["model"].get("mld_regime_backbone_hidden_dim") is not None
                else None
            ),
            mld_regime_branch_hidden_dim=(
                int(raw["model"]["mld_regime_branch_hidden_dim"])
                if raw["model"].get("mld_regime_branch_hidden_dim") is not None
                else None
            ),
            mld_regime_branch_depth=(
                int(raw["model"]["mld_regime_branch_depth"])
                if raw["model"].get("mld_regime_branch_depth") is not None
                else None
            ),
            use_mld_neighbors=bool(raw["model"].get("use_mld_neighbors", False)),
            num_neighbors=int(raw["model"].get("num_neighbors", 5)),
            neighbor_feat_dim=int(raw["model"].get("neighbor_feat_dim", 20)),
            neighbor_hidden_dim=int(raw["model"].get("neighbor_hidden_dim", 96)),
            neighbor_modality_dropout=float(raw["model"].get("neighbor_modality_dropout", 0.5)),
            neighbor_increment=bool(raw["model"].get("neighbor_increment", False)),
            neighbor_increment_full=bool(raw["model"].get("neighbor_increment_full", False)),
        ),
        training=TrainingConfig(
            seed=int(raw["training"]["seed"]),
            epochs=int(raw["training"]["epochs"]),
            batch_size=int(raw["training"]["batch_size"]),
            num_workers=int(raw["training"]["num_workers"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            weight_decay=float(raw["training"]["weight_decay"]),
            grad_clip=float(raw["training"]["grad_clip"]),
            device=str(raw["training"]["device"]),
            scheduler_patience=int(raw["training"]["scheduler_patience"]),
            scheduler_factor=float(raw["training"]["scheduler_factor"]),
            min_learning_rate=float(raw["training"]["min_learning_rate"]),
            early_stopping_patience=int(raw["training"]["early_stopping_patience"]),
            selection_metric=str(raw["training"]["selection_metric"]),
            checkpoint_bias_weight=float(raw["training"]["checkpoint_bias_weight"]),
            checkpoint_std_weight=float(raw["training"]["checkpoint_std_weight"]),
            checkpoint_tail_weight=float(raw["training"].get("checkpoint_tail_weight", 0.0)),
            tail_quantile=float(raw["training"].get("tail_quantile", 0.95)),
            cuda_visible_devices=(
                str(raw["training"]["cuda_visible_devices"])
                if raw["training"].get("cuda_visible_devices") is not None
                else None
            ),
            deterministic=bool(raw["training"].get("deterministic", True)),
            cudnn_benchmark=bool(raw["training"].get("cudnn_benchmark", False)),
            train_metric_batches=(
                int(raw["training"]["train_metric_batches"])
                if raw["training"].get("train_metric_batches") is not None
                else None
            ),
            finite_check_interval=int(raw["training"].get("finite_check_interval", 1)),
            val_interval_steps=int(raw["training"].get("val_interval_steps", 0)),
            max_steps=(
                int(raw["training"]["max_steps"])
                if raw["training"].get("max_steps") is not None
                else None
            ),
            init_from_checkpoint=(
                Path(raw["training"]["init_from_checkpoint"])
                if raw["training"].get("init_from_checkpoint") is not None
                else None
            ),
            freeze_backbone=bool(raw["training"].get("freeze_backbone", False)),
        ),
        loss=LossConfig(
            mld_log_weight=float(
                raw["loss"]["mld_log_weight"] if "mld_log_weight" in raw["loss"] else raw["loss"]["mld_task_weight"]
            ),
            n2_task_weight=float(raw["loss"]["n2_task_weight"]),
            mld_deep_thresholds=[float(value) for value in raw["loss"].get("mld_deep_thresholds", [])],
            mld_deep_weights=[float(value) for value in raw["loss"].get("mld_deep_weights", [])],
            mld_underprediction_weight=float(raw["loss"].get("mld_underprediction_weight", 1.0)),
            mld_physical_weight=float(raw["loss"].get("mld_physical_weight", 0.0)),
            mld_physical_delta=float(raw["loss"].get("mld_physical_delta", 25.0)),
            mld_rmse_weight=float(raw["loss"].get("mld_rmse_weight", 0.0)),
            mld_tail_rmse_weight=float(raw["loss"].get("mld_tail_rmse_weight", 0.0)),
            mld_spread_weight=float(raw["loss"].get("mld_spread_weight", 0.0)),
            mld_corr_weight=float(raw["loss"].get("mld_corr_weight", 0.0)),
            n2_rmse_weight=float(raw["loss"].get("n2_rmse_weight", 0.0)),
            n2_spread_weight=float(raw["loss"].get("n2_spread_weight", 0.0)),
            n2_corr_weight=float(raw["loss"].get("n2_corr_weight", 0.0)),
            n2_warmup_epochs=int(raw["loss"].get("n2_warmup_epochs", 0)),
            n2_warmup_start_factor=float(raw["loss"].get("n2_warmup_start_factor", 1.0)),
            mld_tail_loss_quantile=float(raw["loss"].get("mld_tail_loss_quantile", 0.90)),
            huber_delta=float(raw["loss"]["huber_delta"]),
            uncertainty_weight=float(raw["loss"].get("uncertainty_weight", 0.0)),
            mld_regime_thresholds=[float(value) for value in raw["loss"].get("mld_regime_thresholds", [])],
            regime_gate_weight=float(raw["loss"].get("regime_gate_weight", 0.0)),
            regime_load_balance_weight=float(raw["loss"].get("regime_load_balance_weight", 0.0)),
        ),
        evaluation=EvaluationConfig(
            legacy_val_dir=(
                Path(raw["evaluation"]["legacy_val_dir"])
                if raw.get("evaluation", {}).get("legacy_val_dir") is not None
                else None
            ),
        ),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    cuda_visible_devices = raw.get("training", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return _parse_config(raw)


def load_config_text(text: str) -> ExperimentConfig:
    return _parse_config(tomllib.loads(text))
