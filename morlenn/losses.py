from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .data import denormalize_targets_tensor
from .model import split_prediction_params


class WeightedUpperDynLoss(nn.Module):
    def __init__(
        self,
        target_median: list[float] | torch.Tensor,
        target_iqr: list[float] | torch.Tensor,
        mld_log_weight: float = 0.6,
        n2_task_weight: float = 1.0,
        mld_deep_thresholds: list[float] | tuple[float, ...] = (),
        mld_deep_weights: list[float] | tuple[float, ...] = (),
        mld_underprediction_weight: float = 1.0,
        mld_physical_weight: float = 1.0,
        mld_physical_delta: float = 25.0,
        mld_rmse_weight: float = 0.0,
        mld_tail_rmse_weight: float = 0.0,
        mld_spread_weight: float = 0.0,
        mld_corr_weight: float = 0.0,
        n2_rmse_weight: float = 0.0,
        n2_spread_weight: float = 0.0,
        n2_corr_weight: float = 0.0,
        n2_warmup_epochs: int = 0,
        n2_warmup_start_factor: float = 1.0,
        mld_tail_loss_quantile: float = 0.90,
        huber_delta: float = 1.0,
        uncertainty_weight: float = 0.0,
        predictive_distribution: str = "deterministic",
        min_std: float = 1e-3,
        target_names: list[str] | tuple[str, ...] = ("mld", "N2"),
        mld_target_transform: str = "identity",
        mld_regime_thresholds: list[float] | tuple[float, ...] = (),
        regime_gate_weight: float = 0.0,
        regime_load_balance_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if len(mld_deep_thresholds) != len(mld_deep_weights):
            raise ValueError("mld_deep_thresholds and mld_deep_weights must have the same length.")
        if predictive_distribution not in {"deterministic", "gaussian"}:
            raise ValueError("predictive_distribution must be either 'deterministic' or 'gaussian'.")
        self.mld_log_weight = mld_log_weight
        self.n2_task_weight = n2_task_weight
        self.mld_deep_thresholds = list(mld_deep_thresholds)
        self.mld_deep_weights = list(mld_deep_weights)
        self.mld_underprediction_weight = mld_underprediction_weight
        self.mld_physical_weight = mld_physical_weight
        self.mld_physical_delta = mld_physical_delta
        self.mld_rmse_weight = mld_rmse_weight
        self.mld_tail_rmse_weight = mld_tail_rmse_weight
        self.mld_spread_weight = mld_spread_weight
        self.mld_corr_weight = mld_corr_weight
        self.n2_rmse_weight = n2_rmse_weight
        self.n2_spread_weight = n2_spread_weight
        self.n2_corr_weight = n2_corr_weight
        self.n2_warmup_epochs = int(n2_warmup_epochs)
        self.n2_warmup_start_factor = float(n2_warmup_start_factor)
        self.current_epoch = 1
        self.mld_tail_loss_quantile = mld_tail_loss_quantile
        self.huber_delta = huber_delta
        self.uncertainty_weight = uncertainty_weight
        self.predictive_distribution = predictive_distribution
        self.min_std = min_std
        self.target_names = list(target_names)
        self.mld_target_transform = mld_target_transform
        self.mld_regime_thresholds = sorted(float(value) for value in mld_regime_thresholds)
        self.regime_gate_weight = float(regime_gate_weight)
        self.regime_load_balance_weight = float(regime_load_balance_weight)
        target_median_tensor = torch.as_tensor(target_median, dtype=torch.float32)
        target_iqr_tensor = torch.as_tensor(target_iqr, dtype=torch.float32)
        self.register_buffer("target_median", target_median_tensor)
        self.register_buffer("target_iqr", target_iqr_tensor)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return denormalize_targets_tensor(
            values,
            target_median=self.target_median,
            target_iqr=self.target_iqr,
            target_names=self.target_names,
            mld_target_transform=self.mld_target_transform,
        )

    def _mld_sample_weights(self, target_mld_norm: torch.Tensor) -> torch.Tensor:
        target_vector = torch.stack([target_mld_norm, torch.zeros_like(target_mld_norm)], dim=-1)
        target_mld_phys = self._denormalize(target_vector)[:, 0]
        weights = torch.ones_like(target_mld_phys)
        for threshold, weight in zip(self.mld_deep_thresholds, self.mld_deep_weights):
            weights = torch.where(target_mld_phys > threshold, torch.full_like(weights, weight), weights)
        return weights

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def _target_regime(self, target_mld_phys: torch.Tensor) -> torch.Tensor:
        regime = torch.zeros_like(target_mld_phys, dtype=torch.long)
        for threshold in self.mld_regime_thresholds:
            regime = regime + (target_mld_phys > threshold).long()
        return regime

    def _n2_weight_factor(self) -> float:
        if self.n2_warmup_epochs <= 0:
            return 1.0
        progress = min(max(self.current_epoch, 1), self.n2_warmup_epochs) / float(self.n2_warmup_epochs)
        return self.n2_warmup_start_factor + (1.0 - self.n2_warmup_start_factor) * progress

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        regime_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction_mean, prediction_std = split_prediction_params(prediction, len(self.target_names))
        prediction_mld = prediction_mean[:, 0]
        prediction_n2 = prediction_mean[:, 1]
        target_mld = target[:, 0]
        target_n2 = target[:, 1]

        mld_weights = self._mld_sample_weights(target_mld)
        mld_log_huber = F.huber_loss(
            prediction_mld,
            target_mld,
            delta=self.huber_delta,
            reduction="none",
        )
        mld_log_loss = self._weighted_mean(mld_log_huber, mld_weights)
        n2_loss = F.huber_loss(
            prediction_n2,
            target_n2,
            delta=self.huber_delta,
            reduction="mean",
        )
        n2_squared_error = (prediction_n2 - target_n2).square()
        n2_rmse_loss = n2_squared_error.mean().clamp_min(1e-6).sqrt()
        prediction_n2_std = prediction_n2.std(unbiased=False)
        target_n2_std = target_n2.std(unbiased=False).clamp_min(1e-6)
        n2_spread_ratio = prediction_n2_std / target_n2_std
        n2_spread_loss = (n2_spread_ratio - 1.0).square()
        prediction_n2_centered = prediction_n2 - prediction_n2.mean()
        target_n2_centered = target_n2 - target_n2.mean()
        n2_corr_denom = (
            prediction_n2_centered.square().mean().sqrt()
            * target_n2_centered.square().mean().sqrt()
        ).clamp_min(1e-6)
        n2_corr = (prediction_n2_centered * target_n2_centered).mean() / n2_corr_denom
        n2_corr_loss = 1.0 - n2_corr.clamp(-1.0, 1.0)

        prediction_phys = self._denormalize(prediction_mean)
        target_phys = self._denormalize(target)
        prediction_mld_phys = prediction_phys[:, 0]
        target_mld_phys = target_phys[:, 0]
        mld_huber_phys = F.huber_loss(
            prediction_mld_phys,
            target_mld_phys,
            delta=self.mld_physical_delta,
            reduction="none",
        )
        underprediction_weights = torch.where(
            prediction_mld_phys < target_mld_phys,
            torch.full_like(mld_weights, self.mld_underprediction_weight),
            torch.ones_like(mld_weights),
        )
        mld_physical_weights = mld_weights * underprediction_weights
        mld_physical_loss = self._weighted_mean(mld_huber_phys, mld_physical_weights)
        mld_squared_error_phys = (prediction_mld_phys - target_mld_phys).square()
        mld_rmse_loss = self._weighted_mean(mld_squared_error_phys, mld_weights).clamp_min(1e-6).sqrt()
        prediction_mld_std = prediction_mld_phys.std(unbiased=False)
        target_mld_std = target_mld_phys.std(unbiased=False).clamp_min(1e-6)
        mld_spread_ratio = prediction_mld_std / target_mld_std
        mld_spread_loss = (mld_spread_ratio - 1.0).square()
        prediction_mld_centered = prediction_mld_phys - prediction_mld_phys.mean()
        target_mld_centered = target_mld_phys - target_mld_phys.mean()
        mld_corr_denom = (
            prediction_mld_centered.square().mean().sqrt()
            * target_mld_centered.square().mean().sqrt()
        ).clamp_min(1e-6)
        mld_corr = (prediction_mld_centered * target_mld_centered).mean() / mld_corr_denom
        mld_corr_loss = 1.0 - mld_corr.clamp(-1.0, 1.0)
        tail_threshold = torch.quantile(target_mld_phys.detach(), self.mld_tail_loss_quantile)
        tail_mask = target_mld_phys.detach() >= tail_threshold
        if tail_mask.any():
            mld_tail_rmse_loss = (
                self._weighted_mean(mld_squared_error_phys[tail_mask], mld_weights[tail_mask])
                .clamp_min(1e-6)
                .sqrt()
            )
        else:
            mld_tail_rmse_loss = torch.zeros((), device=prediction.device)

        n2_weight_factor = self._n2_weight_factor()
        total = (
            self.mld_physical_weight * mld_physical_loss
            + self.mld_rmse_weight * mld_rmse_loss
            + self.mld_tail_rmse_weight * mld_tail_rmse_loss
            + self.mld_spread_weight * mld_spread_loss
            + self.mld_corr_weight * mld_corr_loss
            + self.mld_log_weight * mld_log_loss
            + n2_weight_factor * self.n2_task_weight * n2_loss
            + n2_weight_factor * self.n2_rmse_weight * n2_rmse_loss
            + n2_weight_factor * self.n2_spread_weight * n2_spread_loss
            + n2_weight_factor * self.n2_corr_weight * n2_corr_loss
        )
        regime_gate_loss = torch.zeros((), device=prediction.device)
        regime_load_balance_loss = torch.zeros((), device=prediction.device)
        regime_accuracy = torch.zeros((), device=prediction.device)
        if regime_logits is not None and self.regime_gate_weight > 0.0 and self.mld_regime_thresholds:
            target_regime = self._target_regime(target_mld_phys.detach())
            regime_gate_loss = F.cross_entropy(regime_logits, target_regime)
            with torch.no_grad():
                regime_accuracy = (regime_logits.argmax(dim=-1) == target_regime).float().mean()
            total = total + self.regime_gate_weight * regime_gate_loss
            if self.regime_load_balance_weight > 0.0:
                num_regimes = regime_logits.shape[-1]
                gate_probs = torch.softmax(regime_logits, dim=-1)
                importance = gate_probs.mean(dim=0)
                assignment = torch.zeros(num_regimes, device=prediction.device).scatter_add_(
                    0,
                    regime_logits.argmax(dim=-1),
                    torch.ones(regime_logits.shape[0], device=prediction.device),
                ) / float(regime_logits.shape[0])
                regime_load_balance_loss = num_regimes * torch.sum(importance * assignment)
                total = total + self.regime_load_balance_weight * regime_load_balance_loss

        mld_uncertainty_loss = torch.zeros((), device=prediction.device)
        n2_uncertainty_loss = torch.zeros((), device=prediction.device)
        uncertainty_loss = torch.zeros((), device=prediction.device)
        if prediction_std is not None:
            prediction_std_phys = prediction_std.clamp_min(self.min_std)
            gaussian_nll = F.gaussian_nll_loss(
                prediction_phys,
                target_phys,
                prediction_std_phys.square(),
                full=False,
                eps=self.min_std**2,
                reduction="none",
            )
            mld_uncertainty_loss = self._weighted_mean(gaussian_nll[:, 0], mld_weights)
            n2_uncertainty_loss = gaussian_nll[:, 1].mean()
            uncertainty_loss = mld_uncertainty_loss + n2_weight_factor * self.n2_task_weight * n2_uncertainty_loss
            if self.predictive_distribution == "gaussian" and self.uncertainty_weight > 0.0:
                total = total + self.uncertainty_weight * uncertainty_loss
        metrics = {
            "loss": float(total.detach()),
            "mld_log_loss": float(mld_log_loss.detach()),
            "mld_physical_loss": float(mld_physical_loss.detach()),
            "mld_rmse_loss": float(mld_rmse_loss.detach()),
            "mld_tail_rmse_loss": float(mld_tail_rmse_loss.detach()),
            "mld_spread_loss": float(mld_spread_loss.detach()),
            "mld_spread_ratio": float(mld_spread_ratio.detach()),
            "mld_corr_loss": float(mld_corr_loss.detach()),
            "mld_corr_batch": float(mld_corr.detach()),
            "n2_loss": float(n2_loss.detach()),
            "n2_rmse_loss": float(n2_rmse_loss.detach()),
            "n2_spread_loss": float(n2_spread_loss.detach()),
            "n2_spread_ratio": float(n2_spread_ratio.detach()),
            "n2_corr_loss": float(n2_corr_loss.detach()),
            "n2_corr_batch": float(n2_corr.detach()),
            "n2_weight_factor": float(n2_weight_factor),
            "uncertainty_loss": float(uncertainty_loss.detach()),
            "mld_uncertainty_loss": float(mld_uncertainty_loss.detach()),
            "n2_uncertainty_loss": float(n2_uncertainty_loss.detach()),
            "mld_weight_mean": float(mld_weights.mean().detach()),
            "mld_physical_weight_mean": float(mld_physical_weights.mean().detach()),
            "regime_gate_loss": float(regime_gate_loss.detach()),
            "regime_load_balance_loss": float(regime_load_balance_loss.detach()),
            "regime_accuracy": float(regime_accuracy.detach()),
        }
        if prediction_std is not None:
            metrics["mld_pred_std_phys_mean"] = float(prediction_std_phys[:, 0].mean().detach())
            metrics["n2_pred_std_phys_mean"] = float(prediction_std_phys[:, 1].mean().detach())
        return total, metrics
