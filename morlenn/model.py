from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def split_prediction_params(prediction: torch.Tensor, num_targets: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    if prediction.shape[-1] == num_targets:
        return prediction, None
    if prediction.shape[-1] == 2 * num_targets:
        return prediction[..., :num_targets], prediction[..., num_targets:]
    raise ValueError(
        f"Unexpected prediction shape {tuple(prediction.shape)} for num_targets={num_targets}. "
        "Expected last dimension to be num_targets or 2 * num_targets."
    )


def inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(value_tensor)))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.padding > 0:
            x = x[..., :-self.padding]
        return x


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.norm1(x))
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.activation(self.norm2(x))
        x = self.conv2(x)
        x = self.dropout(x)
        return x + residual


class StaticFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticContextEncoder(nn.Module):
    def __init__(self, num_static: int, hidden_dim: int, dropout: float, num_heads: int) -> None:
        super().__init__()
        if num_static == 9:
            self.group_names = ("ocean_state", "geography", "seasonality")
            group_dims = (3, 4, 2)
            self.group_slices = ((0, 3), (3, 7), (7, 9))
        elif num_static == 10:
            self.group_names = ("ocean_state", "geography", "seasonality")
            group_dims = (4, 4, 2)
            self.group_slices = ((0, 4), (4, 8), (8, 10))
        elif num_static == 11:
            self.group_names = ("ocean_state", "geography", "seasonality")
            group_dims = (5, 4, 2)
            self.group_slices = ((0, 5), (5, 9), (9, 11))
        elif num_static == 18:
            self.group_names = ("ocean_state", "eddy", "geography", "seasonality")
            group_dims = (6, 6, 4, 2)
            self.group_slices = ((0, 6), (6, 12), (12, 16), (16, 18))
        elif num_static == 19:
            self.group_names = ("ocean_state", "eddy", "geography", "seasonality")
            group_dims = (7, 6, 4, 2)
            self.group_slices = ((0, 7), (7, 13), (13, 17), (17, 19))
        else:
            self.group_names = ("static",)
            group_dims = (num_static,)
            self.group_slices = ((0, num_static),)
        self.group_encoders = nn.ModuleList(
            [StaticFeatureEncoder(group_dim, hidden_dim, dropout) for group_dim in group_dims]
        )
        self.group_embeddings = nn.Parameter(torch.zeros(len(self.group_slices), hidden_dim))
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.group_gate = nn.Sequential(
            nn.Linear(hidden_dim * len(self.group_slices), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.group_slices)),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        group_tokens: list[torch.Tensor] = []
        for (start, end), encoder in zip(self.group_slices, self.group_encoders):
            group_tokens.append(encoder(x[:, start:end]))
        tokens = torch.stack(group_tokens, dim=1) + self.group_embeddings.unsqueeze(0)
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.feed_forward(tokens))

        gates = torch.softmax(self.group_gate(tokens.reshape(tokens.shape[0], -1)), dim=-1)
        pooled = torch.sum(tokens * gates.unsqueeze(-1), dim=1)
        strongest = tokens.max(dim=1).values
        context = self.output(torch.cat([pooled, strongest], dim=-1))
        diagnostics = {
            "group_gates": gates,
            "group_tokens": tokens,
        }
        return context, diagnostics


class LegacyDynamicSummaryEncoder(nn.Module):
    def __init__(
        self,
        num_dynamic: int,
        hidden_dim: int,
        dropout: float,
        summary_kernel_size: int,
        summary_windows: list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        if summary_kernel_size % 2 == 0:
            raise ValueError("dynamic summary kernel_size must be odd.")
        if not summary_windows:
            raise ValueError("dynamic_summary_windows must not be empty.")
        self.summary_windows = tuple(int(window) for window in summary_windows)
        self.std_window = self.summary_windows[min(1, len(self.summary_windows) - 1)]
        self.component_names = (
            ["weighted_integral"]
            + [f"recent_mean_{window}h" for window in self.summary_windows]
            + ["last", f"recent_std_{self.std_window}h"]
        )
        self.summary_padding = summary_kernel_size // 2
        self.temporal_score = nn.Conv1d(
            num_dynamic,
            num_dynamic,
            kernel_size=summary_kernel_size,
            padding=self.summary_padding,
            groups=num_dynamic,
        )
        nn.init.zeros_(self.temporal_score.weight)
        nn.init.zeros_(self.temporal_score.bias)

        num_components = len(self.component_names)
        summary_dim = num_dynamic * num_components
        self.feature_gate = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_dynamic),
        )
        self.output = nn.Sequential(
            nn.Linear(summary_dim + num_dynamic, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        dynamic: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[str]]]:
        temporal_input = (
            dynamic
        )
        temporal_scores = self.temporal_score(temporal_input)
        temporal_weights = torch.softmax(temporal_scores, dim=-1)
        weighted_integral = torch.sum(dynamic * temporal_weights, dim=-1)

        components: list[torch.Tensor] = [weighted_integral]
        num_steps = dynamic.shape[-1]
        for window in self.summary_windows:
            steps = min(window, num_steps)
            components.append(dynamic[:, :, -steps:].mean(dim=-1))
        components.append(dynamic[:, :, -1])
        std_steps = min(self.std_window, num_steps)
        components.append(dynamic[:, :, -std_steps:].std(dim=-1, unbiased=False))

        summary_stack = torch.stack(components, dim=1)
        summary_flat = summary_stack.reshape(dynamic.shape[0], -1)
        feature_gates = torch.softmax(self.feature_gate(summary_flat), dim=-1)
        gated_summary = (summary_stack * feature_gates.unsqueeze(1)).reshape(dynamic.shape[0], -1)
        context_out = self.output(torch.cat([gated_summary, feature_gates], dim=-1))
        diagnostics: dict[str, torch.Tensor | list[str]] = {
            "feature_gates": feature_gates,
            "weighted_integral": weighted_integral,
            "component_names": self.component_names,
        }
        return context_out, diagnostics


class LatentSegmentContextEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
        summary_kernel_size: int,
        summary_windows: list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        if summary_kernel_size % 2 == 0:
            raise ValueError("dynamic summary kernel_size must be odd.")
        if not summary_windows:
            raise ValueError("dynamic_summary_windows must not be empty.")
        self.summary_windows = tuple(sorted(int(window) for window in summary_windows))
        if self.summary_windows[0] <= 0:
            raise ValueError("dynamic_summary_windows must contain strictly positive boundaries.")
        self.summary_segments: tuple[tuple[int, int], ...] = tuple(
            (0 if index == 0 else self.summary_windows[index - 1], boundary)
            for index, boundary in enumerate(self.summary_windows)
        )
        self.segment_names = [f"{start}_{end}h" for start, end in self.summary_segments]
        self.summary_padding = summary_kernel_size // 2
        self.temporal_score = nn.Conv1d(
            channels,
            channels,
            kernel_size=summary_kernel_size,
            padding=self.summary_padding,
            groups=channels,
        )
        nn.init.constant_(self.temporal_score.weight, 1.0 / summary_kernel_size)
        nn.init.zeros_(self.temporal_score.bias)
        self.segment_proj = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.segment_gate = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[str]]]:
        sequence_t = sequence.transpose(1, 2)
        smoothed = self.temporal_score(sequence_t).transpose(1, 2)
        latent = 0.5 * (sequence + smoothed)

        segment_tokens: list[torch.Tensor] = []
        num_steps = latent.shape[1]
        for start, end in self.summary_segments:
            start_clamped = min(start, num_steps)
            end_clamped = min(end, num_steps)
            steps = end_clamped - start_clamped
            if steps <= 0:
                segment = latent[:, :1, :]
            else:
                segment = latent[:, -end_clamped : -start_clamped if start_clamped > 0 else None, :]
            mean_token = segment.mean(dim=1)
            max_token = segment.max(dim=1).values
            last_token = segment[:, -1, :]
            segment_tokens.append(self.segment_proj(torch.cat([mean_token, max_token, last_token], dim=-1)))

        token_tensor = torch.stack(segment_tokens, dim=1)
        gate_input = torch.cat([token_tensor, context.unsqueeze(1).expand(-1, token_tensor.shape[1], -1)], dim=-1)
        segment_gates = torch.softmax(self.segment_gate(gate_input).squeeze(-1), dim=-1)
        pooled = torch.sum(token_tensor * segment_gates.unsqueeze(-1), dim=1)
        strongest = token_tensor.max(dim=1).values
        context_out = self.output(torch.cat([pooled, strongest], dim=-1))
        diagnostics: dict[str, torch.Tensor | list[str]] = {
            "segment_gates": segment_gates,
            "segment_tokens": token_tensor,
            "segment_names": self.segment_names,
        }
        return context_out, diagnostics


class LatentAttentionContextEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_proj = nn.Linear(channels, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        self.output = nn.Sequential(
            nn.Linear(channels * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context_term = self.context_proj(context).unsqueeze(1)
        scores = self.score_proj(self.activation(self.sequence_proj(sequence) + context_term)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        summary = torch.cat(
            [
                sequence.mean(dim=1),
                sequence.max(dim=1).values,
                sequence[:, -1, :],
                pooled,
            ],
            dim=-1,
        )
        context_out = self.output(summary)
        diagnostics = {
            "context_attention": weights,
            "context_attention_scores": scores,
        }
        return context_out, diagnostics


class FiLMGenerator(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, num_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels * 2),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(context)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        return gamma, beta


class ScaleAwareAttentionExpert(nn.Module):
    def __init__(self, channels: int, context_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("expert kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.filter_padding = kernel_size // 2
        self.temporal_filter = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.filter_padding,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.temporal_filter.weight, 1.0 / kernel_size)
        self.sequence_proj = nn.Linear(channels, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_t = sequence.transpose(1, 2)
        smoothed = self.temporal_filter(sequence_t).transpose(1, 2)
        expert_sequence = 0.5 * (sequence + smoothed)
        context_term = self.context_proj(context).unsqueeze(1)
        scores = self.score_proj(self.activation(self.sequence_proj(expert_sequence) + context_term)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(expert_sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights, scores


class SegmentTokenAttentionExpert(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        hidden_dim: int,
        kernel_size: int,
        summary_windows: list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("expert kernel_size must be odd.")
        if not summary_windows:
            raise ValueError("summary_windows must not be empty for segment-token experts.")
        self.kernel_size = kernel_size
        self.filter_padding = kernel_size // 2
        self.summary_windows = tuple(sorted(int(window) for window in summary_windows))
        self.summary_segments: tuple[tuple[int, int], ...] = tuple(
            (0 if index == 0 else self.summary_windows[index - 1], boundary)
            for index, boundary in enumerate(self.summary_windows)
        )
        self.temporal_filter = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.filter_padding,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.temporal_filter.weight, 1.0 / kernel_size)
        self.segment_proj = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.value_proj = nn.Linear(hidden_dim, channels)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_t = sequence.transpose(1, 2)
        smoothed = self.temporal_filter(sequence_t).transpose(1, 2)
        expert_sequence = 0.5 * (sequence + smoothed)

        num_steps = expert_sequence.shape[1]
        segment_hidden: list[torch.Tensor] = []
        segment_values: list[torch.Tensor] = []
        valid_ranges: list[tuple[int, int]] = []
        for start, end in self.summary_segments:
            start_clamped = min(start, num_steps)
            end_clamped = min(end, num_steps)
            steps = end_clamped - start_clamped
            if steps <= 0:
                continue
            left = num_steps - end_clamped
            right = num_steps - start_clamped
            segment = expert_sequence[:, left:right, :]
            token_input = torch.cat([segment.mean(dim=1), segment.max(dim=1).values, segment[:, -1, :]], dim=-1)
            hidden = self.segment_proj(token_input)
            segment_hidden.append(hidden)
            segment_values.append(self.value_proj(hidden))
            valid_ranges.append((left, right))

        if not segment_hidden:
            pooled = expert_sequence[:, -1, :]
            weights = torch.zeros(expert_sequence.shape[0], num_steps, device=expert_sequence.device, dtype=expert_sequence.dtype)
            weights[:, -1] = 1.0
            scores = torch.zeros_like(weights)
            return pooled, weights, scores

        hidden_tensor = torch.stack(segment_hidden, dim=1)
        value_tensor = torch.stack(segment_values, dim=1)
        context_term = self.context_proj(context).unsqueeze(1)
        token_scores = self.score_proj(self.activation(hidden_tensor + context_term)).squeeze(-1)
        token_weights = torch.softmax(token_scores, dim=1)
        pooled = torch.sum(value_tensor * token_weights.unsqueeze(-1), dim=1)

        expanded_scores = torch.zeros(
            expert_sequence.shape[0],
            num_steps,
            device=expert_sequence.device,
            dtype=expert_sequence.dtype,
        )
        expanded_weights = torch.zeros_like(expanded_scores)
        for token_idx, (left, right) in enumerate(valid_ranges):
            width = max(right - left, 1)
            expanded_scores[:, left:right] = token_scores[:, token_idx].unsqueeze(-1)
            expanded_weights[:, left:right] = token_weights[:, token_idx].unsqueeze(-1) / float(width)
        return pooled, expanded_weights, expanded_scores


class HierarchicalZoneAttentionExpert(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        hidden_dim: int,
        kernel_size: int,
        summary_windows: list[int] | tuple[int, ...],
        tokens_per_segment: int,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("expert kernel_size must be odd.")
        if not summary_windows:
            raise ValueError("summary_windows must not be empty for hierarchical experts.")
        if tokens_per_segment < 1:
            raise ValueError("tokens_per_segment must be at least 1.")
        self.kernel_size = kernel_size
        self.filter_padding = kernel_size // 2
        self.summary_windows = tuple(sorted(int(window) for window in summary_windows))
        self.summary_segments: tuple[tuple[int, int], ...] = tuple(
            (0 if index == 0 else self.summary_windows[index - 1], boundary)
            for index, boundary in enumerate(self.summary_windows)
        )
        self.tokens_per_segment = int(tokens_per_segment)
        self.temporal_filter = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.filter_padding,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.temporal_filter.weight, 1.0 / kernel_size)
        self.token_proj = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.value_proj = nn.Linear(hidden_dim, channels)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.intra_score_proj = nn.Linear(hidden_dim, 1)
        self.zone_score_proj = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()

    def _chunk_ranges(self, left: int, right: int) -> list[tuple[int, int]]:
        width = max(right - left, 1)
        num_tokens = min(self.tokens_per_segment, width)
        boundaries = torch.linspace(left, right, steps=num_tokens + 1, dtype=torch.float32).round().to(torch.int64).tolist()
        ranges: list[tuple[int, int]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start_i = int(start)
            end_i = max(start_i + 1, int(end))
            end_i = min(end_i, right)
            if start_i < right:
                ranges.append((start_i, end_i))
        return ranges or [(left, right)]

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_t = sequence.transpose(1, 2)
        smoothed = self.temporal_filter(sequence_t).transpose(1, 2)
        expert_sequence = 0.5 * (sequence + smoothed)

        num_steps = expert_sequence.shape[1]
        context_hidden = self.context_proj(context)
        zone_hidden_list: list[torch.Tensor] = []
        zone_value_list: list[torch.Tensor] = []
        zone_token_scores: list[torch.Tensor] = []
        zone_chunk_ranges: list[list[tuple[int, int]]] = []

        for start, end in self.summary_segments:
            start_clamped = min(start, num_steps)
            end_clamped = min(end, num_steps)
            if end_clamped <= start_clamped:
                continue
            left = num_steps - end_clamped
            right = num_steps - start_clamped
            chunk_ranges = self._chunk_ranges(left, right)
            token_hidden_list: list[torch.Tensor] = []
            token_value_list: list[torch.Tensor] = []
            for chunk_left, chunk_right in chunk_ranges:
                chunk = expert_sequence[:, chunk_left:chunk_right, :]
                token_input = torch.cat([chunk.mean(dim=1), chunk.max(dim=1).values, chunk[:, -1, :]], dim=-1)
                hidden = self.token_proj(token_input)
                token_hidden_list.append(hidden)
                token_value_list.append(self.value_proj(hidden))

            token_hidden = torch.stack(token_hidden_list, dim=1)
            token_values = torch.stack(token_value_list, dim=1)
            token_scores = self.intra_score_proj(self.activation(token_hidden + context_hidden.unsqueeze(1))).squeeze(-1)
            token_weights = torch.softmax(token_scores, dim=1)
            zone_hidden = torch.sum(token_hidden * token_weights.unsqueeze(-1), dim=1)
            zone_value = torch.sum(token_values * token_weights.unsqueeze(-1), dim=1)

            zone_hidden_list.append(zone_hidden)
            zone_value_list.append(zone_value)
            zone_token_scores.append(token_scores)
            zone_chunk_ranges.append(chunk_ranges)

        if not zone_hidden_list:
            pooled = expert_sequence[:, -1, :]
            weights = torch.zeros(expert_sequence.shape[0], num_steps, device=expert_sequence.device, dtype=expert_sequence.dtype)
            weights[:, -1] = 1.0
            scores = torch.zeros_like(weights)
            zone_weights = torch.zeros(expert_sequence.shape[0], 0, device=expert_sequence.device, dtype=expert_sequence.dtype)
            zone_scores = torch.zeros_like(zone_weights)
            return pooled, weights, scores, zone_weights, zone_scores

        zone_hidden_tensor = torch.stack(zone_hidden_list, dim=1)
        zone_value_tensor = torch.stack(zone_value_list, dim=1)
        zone_scores = self.zone_score_proj(self.activation(zone_hidden_tensor + context_hidden.unsqueeze(1))).squeeze(-1)
        zone_weights = torch.softmax(zone_scores, dim=1)
        pooled = torch.sum(zone_value_tensor * zone_weights.unsqueeze(-1), dim=1)

        expanded_scores = torch.zeros(
            expert_sequence.shape[0], num_steps, device=expert_sequence.device, dtype=expert_sequence.dtype
        )
        expanded_weights = torch.zeros_like(expanded_scores)
        for zone_idx, chunk_ranges in enumerate(zone_chunk_ranges):
            token_scores = zone_token_scores[zone_idx]
            token_weights = torch.softmax(token_scores, dim=1)
            for token_idx, (left, right) in enumerate(chunk_ranges):
                width = max(right - left, 1)
                combined_score = zone_scores[:, zone_idx] + token_scores[:, token_idx]
                expanded_scores[:, left:right] = combined_score.unsqueeze(-1)
                expanded_weights[:, left:right] = (
                    zone_weights[:, zone_idx].unsqueeze(-1) * token_weights[:, token_idx].unsqueeze(-1) / float(width)
                )
        return pooled, expanded_weights, expanded_scores, zone_weights, zone_scores


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class GatedFusionBlock(nn.Module):
    def __init__(self, dynamic_dim: int, static_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dynamic_proj = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim),
            nn.GELU(),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(dynamic_dim + static_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, dynamic_summary: torch.Tensor, static_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic_state = self.dynamic_proj(dynamic_summary)
        static_state = self.static_proj(static_context)
        gate = torch.sigmoid(self.gate(torch.cat([dynamic_summary, static_context], dim=-1)))
        mixed = gate * dynamic_state + (1.0 - gate) * static_state
        fused = self.out(torch.cat([mixed, dynamic_state - static_state], dim=-1))
        return fused, gate


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.detach() + float(scale) * (x - x.detach())


def make_attention_experts(
    mode: str,
    final_channels: int,
    static_hidden_dim: int,
    attention_hidden_dim: int,
    expert_kernel_sizes: list[int] | tuple[int, ...],
    dynamic_summary_windows: list[int] | tuple[int, ...],
    attention_tokens_per_segment: int,
) -> nn.ModuleList:
    if mode == "pointwise_attention":
        return nn.ModuleList(
            [
                ScaleAwareAttentionExpert(final_channels, static_hidden_dim, attention_hidden_dim, kernel)
                for kernel in expert_kernel_sizes
            ]
        )
    if mode == "segment_tokens":
        return nn.ModuleList(
            [
                SegmentTokenAttentionExpert(
                    final_channels,
                    static_hidden_dim,
                    attention_hidden_dim,
                    kernel,
                    dynamic_summary_windows,
                )
                for kernel in expert_kernel_sizes
            ]
        )
    if mode == "hierarchical_zone_attention":
        return nn.ModuleList(
            [
                HierarchicalZoneAttentionExpert(
                    final_channels,
                    static_hidden_dim,
                    attention_hidden_dim,
                    kernel,
                    dynamic_summary_windows,
                    attention_tokens_per_segment,
                )
                for kernel in expert_kernel_sizes
            ]
        )
    raise ValueError(f"Unsupported attention_pooling_mode: {mode}")


def make_temporal_stack(
    num_dynamic: int,
    tcn_channels: list[int] | tuple[int, ...],
    dilations: list[int] | tuple[int, ...],
    kernel_size: int,
    dropout: float,
    static_hidden_dim: int,
    film_hidden_dim: int,
) -> tuple[nn.Sequential, nn.ModuleList, nn.ModuleList]:
    stem = nn.Sequential(
        nn.Conv1d(num_dynamic, tcn_channels[0], kernel_size=1),
        nn.GELU(),
    )
    blocks: list[nn.Module] = []
    film_layers: list[nn.Module] = []
    in_channels = tcn_channels[0]
    for out_channels, dilation in zip(tcn_channels, dilations):
        blocks.append(
            TemporalResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
        )
        film_layers.append(FiLMGenerator(static_hidden_dim, film_hidden_dim, out_channels))
        in_channels = out_channels
    return stem, nn.ModuleList(blocks), nn.ModuleList(film_layers)


def compute_attention_summary(
    experts: nn.ModuleList,
    gating_network: nn.Module,
    expert_kernel_sizes: list[int],
    sequence: torch.Tensor,
    regime_context: torch.Tensor,
    temporal_mean: torch.Tensor,
    temporal_max: torch.Tensor,
    temporal_last: torch.Tensor,
    dynamic_context: torch.Tensor | None,
    attention_pooling_mode: str,
) -> dict[str, torch.Tensor | None]:
    gate_parts = [regime_context, temporal_mean, temporal_max, temporal_last]
    if dynamic_context is not None:
        gate_parts.insert(1, dynamic_context)
    gate_input = torch.cat(gate_parts, dim=-1)
    expert_gate_logits = gating_network(gate_input)
    expert_gates = torch.softmax(expert_gate_logits, dim=-1)

    expert_pooled: list[torch.Tensor] = []
    expert_attention: list[torch.Tensor] = []
    expert_scores: list[torch.Tensor] = []
    expert_zone_weights: list[torch.Tensor] = []
    expert_zone_scores: list[torch.Tensor] = []
    for expert in experts:
        if attention_pooling_mode == "hierarchical_zone_attention":
            pooled, weights, scores, zone_weights, zone_scores = expert(sequence, regime_context)
            expert_zone_weights.append(zone_weights)
            expert_zone_scores.append(zone_scores)
        else:
            pooled, weights, scores = expert(sequence, regime_context)
        expert_pooled.append(pooled)
        expert_attention.append(weights)
        expert_scores.append(scores)

    expert_pooled_tensor = torch.stack(expert_pooled, dim=1)
    expert_attention_tensor = torch.stack(expert_attention, dim=1)
    expert_score_tensor = torch.stack(expert_scores, dim=1)
    expert_zone_weight_tensor = torch.stack(expert_zone_weights, dim=1) if expert_zone_weights else None
    expert_zone_score_tensor = torch.stack(expert_zone_scores, dim=1) if expert_zone_scores else None
    mixture_pooled = torch.sum(expert_pooled_tensor * expert_gates.unsqueeze(-1), dim=1)
    combined_attention = torch.sum(expert_attention_tensor * expert_gates.unsqueeze(-1), dim=1)
    dynamic_summary_parts = [temporal_mean, temporal_max, temporal_last, mixture_pooled, expert_gates]
    if dynamic_context is not None:
        dynamic_summary_parts.append(dynamic_context)
    return {
        "dynamic_summary": torch.cat(dynamic_summary_parts, dim=-1),
        "combined_attention": combined_attention,
        "expert_attention": expert_attention_tensor,
        "expert_attention_scores": expert_score_tensor,
        "expert_gates": expert_gates,
        "expert_kernel_sizes": torch.as_tensor(expert_kernel_sizes, device=sequence.device),
        "mixture_pooled": mixture_pooled,
        "expert_zone_weights": expert_zone_weight_tensor,
        "expert_zone_scores": expert_zone_score_tensor,
    }


class MldRegimeBranch(nn.Module):
    """A self-contained MLD path (attention -> fusion -> backbone -> head).

    One branch is instantiated per additional MLD regime. The branch shares the
    upstream TCN sequence with the rest of the model but owns its attention
    experts, gating, fusion and head, so that distinct MLD regimes can read the
    temporal features in their own way. Combined softly downstream by a regime
    gate, so the rare deep branch still receives (weighted) gradient everywhere.
    """

    def __init__(
        self,
        final_channels: int,
        static_hidden_dim: int,
        attention_hidden_dim: int,
        expert_kernel_sizes: list[int] | tuple[int, ...],
        dynamic_summary_windows: list[int] | tuple[int, ...],
        attention_tokens_per_segment: int,
        attention_pooling_mode: str,
        dynamic_context_hidden_dim: int,
        fusion_hidden_dim: int,
        backbone_hidden_dim: int,
        branch_hidden_dim: int,
        branch_depth: int,
        dropout: float,
        predictive_distribution: str,
        min_std: float,
    ) -> None:
        super().__init__()
        self.attention_pooling_mode = attention_pooling_mode
        self.expert_kernel_sizes = list(expert_kernel_sizes)
        self.num_experts = len(self.expert_kernel_sizes)
        self.dynamic_context_hidden_dim = dynamic_context_hidden_dim
        self.predictive_distribution = predictive_distribution
        self.min_std = min_std
        self.experts = make_attention_experts(
            attention_pooling_mode,
            final_channels,
            static_hidden_dim,
            attention_hidden_dim,
            self.expert_kernel_sizes,
            dynamic_summary_windows,
            attention_tokens_per_segment,
        )
        gate_input_dim = static_hidden_dim + dynamic_context_hidden_dim + final_channels * 3
        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, self.num_experts),
        )
        dynamic_summary_dim = final_channels * 4 + self.num_experts + dynamic_context_hidden_dim
        self.fusion = GatedFusionBlock(dynamic_summary_dim, static_hidden_dim, fusion_hidden_dim, dropout)
        self.backbone = nn.Sequential(
            ResidualMLPBlock(fusion_hidden_dim, dropout),
            ResidualMLPBlock(fusion_hidden_dim, dropout),
            nn.Linear(fusion_hidden_dim, backbone_hidden_dim),
            nn.GELU(),
        )
        branch_hidden_dim = int(branch_hidden_dim or backbone_hidden_dim)
        if branch_depth > 0:
            branch_layers: list[nn.Module] = [
                nn.Linear(backbone_hidden_dim, branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            for _ in range(branch_depth):
                branch_layers.append(ResidualMLPBlock(branch_hidden_dim, dropout))
            self.branch = nn.Sequential(*branch_layers)
            head_input_dim = branch_hidden_dim
        else:
            self.branch = nn.Identity()
            head_input_dim = backbone_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim),
            nn.GELU(),
            nn.Linear(head_input_dim, 1),
        )
        if predictive_distribution == "gaussian":
            self.std_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(head_input_dim, head_input_dim),
                nn.GELU(),
                nn.Linear(head_input_dim, 1),
            )
            self.std_head[-1].bias.data.fill_(inverse_softplus(25.0 - min_std))
        else:
            self.std_head = None

    def forward(
        self,
        sequence: torch.Tensor,
        regime_context: torch.Tensor,
        temporal_mean: torch.Tensor,
        temporal_max: torch.Tensor,
        temporal_last: torch.Tensor,
        dynamic_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        summary = compute_attention_summary(
            self.experts,
            self.gating_network,
            self.expert_kernel_sizes,
            sequence,
            regime_context,
            temporal_mean,
            temporal_max,
            temporal_last,
            dynamic_context,
            self.attention_pooling_mode,
        )
        fused, _ = self.fusion(summary["dynamic_summary"], regime_context)  # type: ignore[arg-type]
        fused = self.backbone(fused)
        features = self.branch(fused)
        mean = self.head(features).squeeze(-1)
        std = None
        if self.std_head is not None:
            std = (self.min_std + F.softplus(self.std_head(features))).squeeze(-1)
        return mean, std


class MldNeighborBranch(nn.Module):
    """Late-fusion encoder over the K nearest past-MLD analogs.

    Each neighbor is [norm_mld, <19 static features>]. The branch encodes every
    neighbor, scores them by an attention conditioned on the TARGET's static
    context (so it learns regime-match relevance — which statics matter is
    learned, not hand-weighted), pools, and returns a latent that is added to
    the MLD features. A learned null embedding covers the no-neighbor case, so
    the channel is optional at inference.
    """

    def __init__(self, feat_dim: int, context_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.null = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, neighbors: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        enc = self.encode(neighbors)  # [B, K, H]
        ctx = self.context_proj(context).unsqueeze(1)  # [B, 1, H]
        score = self.score(torch.tanh(enc + ctx)).squeeze(-1)  # [B, K]
        score = score.masked_fill(mask < 0.5, -1e9)
        weights = torch.softmax(score, dim=1)  # [B, K]
        pooled = torch.sum(enc * weights.unsqueeze(-1), dim=1)  # [B, H]
        has_neighbor = mask.sum(dim=1, keepdim=True) > 0.5  # [B, 1]
        latent = torch.where(has_neighbor, pooled, self.null.unsqueeze(0).expand_as(pooled))
        return self.out(latent)  # [B, H]


class UpperDynTCNAttentionModel(nn.Module):
    def __init__(
        self,
        num_dynamic: int,
        num_static: int,
        num_targets: int,
        tcn_channels: list[int] | tuple[int, ...] = (32, 64, 64, 128, 128, 128),
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 5,
        dropout: float = 0.10,
        static_hidden_dim: int = 64,
        static_num_heads: int = 4,
        use_dynamic_context: bool = False,
        dynamic_context_mode: str = "latent_segments",
        dynamic_context_hidden_dim: int = 64,
        dynamic_summary_kernel_size: int = 25,
        dynamic_summary_windows: list[int] | tuple[int, ...] = (24, 72, 168, 360, 720),
        film_hidden_dim: int = 64,
        attention_hidden_dim: int = 64,
        attention_pooling_mode: str = "segment_tokens",
        attention_tokens_per_segment: int = 4,
        num_attention_experts: int = 3,
        expert_kernel_sizes: list[int] | tuple[int, ...] = (13, 49, 97),
        fusion_hidden_dim: int = 128,
        backbone_hidden_dim: int = 64,
        predictive_distribution: str = "deterministic",
        min_std: float = 1e-3,
        mld_branch_hidden_dim: int | None = None,
        mld_branch_depth: int = 0,
        detach_n2_head_input: bool = False,
        n2_separate_temporal_branch: bool = False,
        n2_separate_attention_branch: bool = False,
        n2_attention_hidden_dim: int | None = None,
        n2_num_attention_experts: int | None = None,
        n2_expert_kernel_sizes: list[int] | tuple[int, ...] | None = None,
        n2_fusion_hidden_dim: int | None = None,
        n2_backbone_hidden_dim: int | None = None,
        n2_backbone_grad_scale: float = 1.0,
        n2_branch_hidden_dim: int | None = None,
        n2_branch_depth: int = 0,
        mld_regime_split: bool = False,
        num_mld_regimes: int = 2,
        regime_gate_hidden_dim: int = 64,
        mld_regime_attention_hidden_dim: int | None = None,
        mld_regime_fusion_hidden_dim: int | None = None,
        mld_regime_backbone_hidden_dim: int | None = None,
        mld_regime_branch_hidden_dim: int | None = None,
        mld_regime_branch_depth: int | None = None,
        use_mld_neighbors: bool = False,
        num_neighbors: int = 5,
        neighbor_feat_dim: int = 20,
        neighbor_hidden_dim: int = 96,
        neighbor_modality_dropout: float = 0.5,
        neighbor_increment: bool = False,
        neighbor_increment_full: bool = False,
    ) -> None:
        super().__init__()
        if len(tcn_channels) != len(dilations):
            raise ValueError("tcn_channels and dilations must have the same length.")
        if num_targets != 2:
            raise ValueError("This implementation expects exactly two targets: MLD and N2.")
        if num_attention_experts < 1:
            raise ValueError("num_attention_experts must be at least 1.")
        if len(expert_kernel_sizes) != num_attention_experts:
            raise ValueError("expert_kernel_sizes must have the same length as num_attention_experts.")
        if predictive_distribution not in {"deterministic", "gaussian"}:
            raise ValueError("predictive_distribution must be either 'deterministic' or 'gaussian'.")
        n2_num_attention_experts = int(n2_num_attention_experts or num_attention_experts)
        n2_expert_kernel_sizes = list(n2_expert_kernel_sizes or expert_kernel_sizes)
        if len(n2_expert_kernel_sizes) != n2_num_attention_experts:
            raise ValueError("n2_expert_kernel_sizes must have the same length as n2_num_attention_experts.")

        self.num_targets = num_targets
        self.num_attention_experts = num_attention_experts
        self.expert_kernel_sizes = list(expert_kernel_sizes)
        self.n2_num_attention_experts = n2_num_attention_experts
        self.n2_expert_kernel_sizes = list(n2_expert_kernel_sizes)
        self.predictive_distribution = predictive_distribution
        self.min_std = min_std
        self.detach_n2_head_input = detach_n2_head_input
        self.n2_backbone_grad_scale = 0.0 if detach_n2_head_input else float(n2_backbone_grad_scale)
        self.n2_separate_temporal_branch = bool(n2_separate_temporal_branch)
        self.n2_separate_attention_branch = bool(n2_separate_attention_branch)
        self.use_dynamic_context = use_dynamic_context
        self.dynamic_context_mode = dynamic_context_mode
        self.attention_pooling_mode = attention_pooling_mode
        self.attention_tokens_per_segment = attention_tokens_per_segment
        self.dynamic_stem, self.tcn_blocks, self.film_layers = make_temporal_stack(
            num_dynamic=num_dynamic,
            tcn_channels=tcn_channels,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
            static_hidden_dim=static_hidden_dim,
            film_hidden_dim=film_hidden_dim,
        )
        if self.n2_separate_temporal_branch:
            self.n2_dynamic_stem, self.n2_tcn_blocks, self.n2_film_layers = make_temporal_stack(
                num_dynamic=num_dynamic,
                tcn_channels=tcn_channels,
                dilations=dilations,
                kernel_size=kernel_size,
                dropout=dropout,
                static_hidden_dim=static_hidden_dim,
                film_hidden_dim=film_hidden_dim,
            )
        else:
            self.n2_dynamic_stem = None
            self.n2_tcn_blocks = None
            self.n2_film_layers = None
        self.dynamic_context_hidden_dim = dynamic_context_hidden_dim if use_dynamic_context else 0
        self.static_encoder = StaticContextEncoder(
            num_static=num_static,
            hidden_dim=static_hidden_dim,
            dropout=dropout,
            num_heads=static_num_heads,
        )
        self.regime_fusion = (
            GatedFusionBlock(
                dynamic_dim=dynamic_context_hidden_dim,
                static_dim=static_hidden_dim,
                hidden_dim=static_hidden_dim,
                dropout=dropout,
            )
            if self.use_dynamic_context
            else None
        )

        final_channels = tcn_channels[-1]
        if self.use_dynamic_context:
            if self.dynamic_context_mode == "legacy_raw_summaries":
                self.dynamic_context_encoder: LegacyDynamicSummaryEncoder | LatentSegmentContextEncoder | LatentAttentionContextEncoder | None = LegacyDynamicSummaryEncoder(
                    num_dynamic=num_dynamic,
                    hidden_dim=dynamic_context_hidden_dim,
                    dropout=dropout,
                    summary_kernel_size=dynamic_summary_kernel_size,
                    summary_windows=dynamic_summary_windows,
                )
            elif self.dynamic_context_mode == "latent_segments":
                self.dynamic_context_encoder = LatentSegmentContextEncoder(
                    channels=final_channels,
                    context_dim=static_hidden_dim,
                    hidden_dim=dynamic_context_hidden_dim,
                    dropout=dropout,
                    summary_kernel_size=dynamic_summary_kernel_size,
                    summary_windows=dynamic_summary_windows,
                )
            elif self.dynamic_context_mode == "latent_attention":
                self.dynamic_context_encoder = LatentAttentionContextEncoder(
                    channels=final_channels,
                    context_dim=static_hidden_dim,
                    hidden_dim=dynamic_context_hidden_dim,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported dynamic_context_mode: {self.dynamic_context_mode}")
        else:
            self.dynamic_context_encoder = None
        self.experts = make_attention_experts(
            self.attention_pooling_mode,
            final_channels,
            static_hidden_dim,
            attention_hidden_dim,
            self.expert_kernel_sizes,
            dynamic_summary_windows,
            attention_tokens_per_segment,
        )
        gate_input_dim = static_hidden_dim + self.dynamic_context_hidden_dim + final_channels * 3
        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_attention_experts),
        )
        n2_attention_hidden_dim = int(n2_attention_hidden_dim or attention_hidden_dim)
        if self.n2_separate_attention_branch:
            self.n2_experts = make_attention_experts(
                self.attention_pooling_mode,
                final_channels,
                static_hidden_dim,
                n2_attention_hidden_dim,
                self.n2_expert_kernel_sizes,
                dynamic_summary_windows,
                attention_tokens_per_segment,
            )
            self.n2_gating_network = nn.Sequential(
                nn.Linear(gate_input_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, self.n2_num_attention_experts),
            )
        else:
            self.n2_experts = None
            self.n2_gating_network = None

        dynamic_summary_dim = final_channels * 4 + num_attention_experts + self.dynamic_context_hidden_dim
        self.fusion = GatedFusionBlock(dynamic_summary_dim, static_hidden_dim, fusion_hidden_dim, dropout)
        self.backbone = nn.Sequential(
            ResidualMLPBlock(fusion_hidden_dim, dropout),
            ResidualMLPBlock(fusion_hidden_dim, dropout),
            nn.Linear(fusion_hidden_dim, backbone_hidden_dim),
            nn.GELU(),
        )
        n2_fusion_hidden_dim = int(n2_fusion_hidden_dim or fusion_hidden_dim)
        n2_backbone_hidden_dim = int(n2_backbone_hidden_dim or backbone_hidden_dim)
        if self.n2_separate_attention_branch:
            n2_dynamic_summary_dim = final_channels * 4 + self.n2_num_attention_experts + self.dynamic_context_hidden_dim
            self.n2_fusion = GatedFusionBlock(n2_dynamic_summary_dim, static_hidden_dim, n2_fusion_hidden_dim, dropout)
            self.n2_backbone = nn.Sequential(
                ResidualMLPBlock(n2_fusion_hidden_dim, dropout),
                ResidualMLPBlock(n2_fusion_hidden_dim, dropout),
                nn.Linear(n2_fusion_hidden_dim, n2_backbone_hidden_dim),
                nn.GELU(),
            )
            n2_branch_input_dim = n2_backbone_hidden_dim
        else:
            self.n2_fusion = None
            self.n2_backbone = None
            n2_branch_input_dim = backbone_hidden_dim
        mld_branch_hidden_dim = int(mld_branch_hidden_dim or backbone_hidden_dim)
        if mld_branch_depth > 0:
            mld_branch_layers: list[nn.Module] = [
                nn.Linear(backbone_hidden_dim, mld_branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            for _ in range(mld_branch_depth):
                mld_branch_layers.append(ResidualMLPBlock(mld_branch_hidden_dim, dropout))
            self.mld_branch = nn.Sequential(*mld_branch_layers)
            mld_head_input_dim = mld_branch_hidden_dim
        else:
            self.mld_branch = nn.Identity()
            mld_head_input_dim = backbone_hidden_dim
        self.mld_head = nn.Sequential(
            nn.Linear(mld_head_input_dim, mld_head_input_dim),
            nn.GELU(),
            nn.Linear(mld_head_input_dim, 1),
        )
        n2_branch_hidden_dim = int(n2_branch_hidden_dim or n2_branch_input_dim)
        if n2_branch_depth > 0:
            n2_branch_layers: list[nn.Module] = [
                nn.Linear(n2_branch_input_dim, n2_branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            for _ in range(n2_branch_depth):
                n2_branch_layers.append(ResidualMLPBlock(n2_branch_hidden_dim, dropout))
            self.n2_branch = nn.Sequential(*n2_branch_layers)
            n2_head_input_dim = n2_branch_hidden_dim
        else:
            self.n2_branch = nn.Identity()
            n2_head_input_dim = n2_branch_input_dim
        self.n2_head = nn.Sequential(
            nn.Linear(n2_head_input_dim, n2_head_input_dim),
            nn.GELU(),
            nn.Linear(n2_head_input_dim, 1),
        )
        if self.predictive_distribution == "gaussian":
            self.mld_std_head = nn.Sequential(
                nn.Linear(mld_head_input_dim, mld_head_input_dim),
                nn.GELU(),
                nn.Linear(mld_head_input_dim, 1),
            )
            self.n2_std_head = nn.Sequential(
                nn.Linear(n2_head_input_dim, n2_head_input_dim),
                nn.GELU(),
                nn.Linear(n2_head_input_dim, 1),
            )
            self.mld_std_head[-1].bias.data.fill_(inverse_softplus(25.0 - self.min_std))
            self.n2_std_head[-1].bias.data.fill_(inverse_softplus(0.02 - self.min_std))
        else:
            self.mld_std_head = None
            self.n2_std_head = None

        self.mld_regime_split = bool(mld_regime_split)
        self.num_mld_regimes = int(num_mld_regimes) if self.mld_regime_split else 1
        if self.mld_regime_split:
            if self.num_mld_regimes < 2:
                raise ValueError("num_mld_regimes must be at least 2 when mld_regime_split is enabled.")
            regime_attention_hidden_dim = int(mld_regime_attention_hidden_dim or attention_hidden_dim)
            regime_fusion_hidden_dim = int(mld_regime_fusion_hidden_dim or fusion_hidden_dim)
            regime_backbone_hidden_dim = int(mld_regime_backbone_hidden_dim or backbone_hidden_dim)
            regime_branch_hidden_dim = int(mld_regime_branch_hidden_dim or mld_branch_hidden_dim)
            regime_branch_depth = int(
                mld_regime_branch_depth if mld_regime_branch_depth is not None else mld_branch_depth
            )
            # Regime 0 reuses the existing MLD path above; regimes 1..K-1 get their own branches.
            self.mld_extra_regime_branches = nn.ModuleList(
                [
                    MldRegimeBranch(
                        final_channels=final_channels,
                        static_hidden_dim=static_hidden_dim,
                        attention_hidden_dim=regime_attention_hidden_dim,
                        expert_kernel_sizes=self.expert_kernel_sizes,
                        dynamic_summary_windows=dynamic_summary_windows,
                        attention_tokens_per_segment=attention_tokens_per_segment,
                        attention_pooling_mode=self.attention_pooling_mode,
                        dynamic_context_hidden_dim=self.dynamic_context_hidden_dim,
                        fusion_hidden_dim=regime_fusion_hidden_dim,
                        backbone_hidden_dim=regime_backbone_hidden_dim,
                        branch_hidden_dim=regime_branch_hidden_dim,
                        branch_depth=regime_branch_depth,
                        dropout=dropout,
                        predictive_distribution=self.predictive_distribution,
                        min_std=self.min_std,
                    )
                    for _ in range(self.num_mld_regimes - 1)
                ]
            )
            regime_gate_input_dim = static_hidden_dim + final_channels * 3
            self.regime_gate = nn.Sequential(
                nn.Linear(regime_gate_input_dim, regime_gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(regime_gate_hidden_dim, self.num_mld_regimes),
            )
        else:
            self.mld_extra_regime_branches = None
            self.regime_gate = None

        self.use_mld_neighbors = bool(use_mld_neighbors)
        self.num_neighbors = int(num_neighbors)
        self.neighbor_modality_dropout = float(neighbor_modality_dropout)
        self.neighbor_increment = bool(neighbor_increment)
        self.neighbor_increment_full = bool(neighbor_increment_full)
        if self.use_mld_neighbors:
            self.neighbor_branch: MldNeighborBranch | None = MldNeighborBranch(
                feat_dim=int(neighbor_feat_dim),
                context_dim=static_hidden_dim,
                hidden_dim=int(neighbor_hidden_dim),
                dropout=dropout,
            )
            # Zero-init fusion -> correction starts at 0, so loading a forcing-only
            # checkpoint reproduces it exactly before any training. v2 increment mode
            # outputs a scalar Delta added to the MLD mean (in normalized space);
            # v3 (increment_full) outputs 4 Deltas (MLD mean/std, N2 mean/std);
            # v1 mode adds a vector to the MLD features.
            if self.neighbor_increment_full:
                fusion_out = 4
            elif self.neighbor_increment:
                fusion_out = 1
            else:
                fusion_out = mld_head_input_dim
            self.neighbor_fusion: nn.Linear | None = nn.Linear(int(neighbor_hidden_dim), fusion_out)
            nn.init.zeros_(self.neighbor_fusion.weight)
            nn.init.zeros_(self.neighbor_fusion.bias)
        else:
            self.neighbor_branch = None
            self.neighbor_fusion = None

    def _attention_summary(
        self,
        experts: nn.ModuleList,
        gating_network: nn.Module,
        expert_kernel_sizes: list[int],
        sequence: torch.Tensor,
        regime_context: torch.Tensor,
        temporal_mean: torch.Tensor,
        temporal_max: torch.Tensor,
        temporal_last: torch.Tensor,
        dynamic_context: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        return compute_attention_summary(
            experts,
            gating_network,
            expert_kernel_sizes,
            sequence,
            regime_context,
            temporal_mean,
            temporal_max,
            temporal_last,
            dynamic_context,
            self.attention_pooling_mode,
        )

    def _apply_film(self, temporal: torch.Tensor, context: torch.Tensor, layer: FiLMGenerator) -> torch.Tensor:
        gamma, beta = layer(context)
        return temporal * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

    def _encode_temporal(
        self,
        dynamic: torch.Tensor,
        context: torch.Tensor,
        stem: nn.Module,
        blocks: nn.ModuleList,
        film_layers: nn.ModuleList,
    ) -> torch.Tensor:
        temporal = stem(dynamic)
        for block, film_layer in zip(blocks, film_layers):
            temporal = block(temporal)
            temporal = self._apply_film(temporal, context, film_layer)
        return temporal.transpose(1, 2)

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        return_attention: bool = False,
        return_diagnostics: bool = False,
        return_regime_logits: bool = False,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        static_context, static_diagnostics = self.static_encoder(static)
        dynamic_context = None
        dynamic_diagnostics = None
        regime_gate = None

        if (
            self.use_dynamic_context
            and self.dynamic_context_mode == "legacy_raw_summaries"
            and self.dynamic_context_encoder is not None
            and self.regime_fusion is not None
        ):
            dynamic_context, dynamic_diagnostics = self.dynamic_context_encoder(dynamic)
            regime_context, regime_gate = self.regime_fusion(dynamic_context, static_context)
        else:
            regime_context = static_context

        mld_film_context = regime_context if self.dynamic_context_mode == "legacy_raw_summaries" else static_context
        sequence = self._encode_temporal(
            dynamic=dynamic,
            context=mld_film_context,
            stem=self.dynamic_stem,
            blocks=self.tcn_blocks,
            film_layers=self.film_layers,
        )
        n2_sequence = sequence
        if (
            self.use_dynamic_context
            and self.dynamic_context_mode in {"latent_segments", "latent_attention"}
            and self.dynamic_context_encoder is not None
            and self.regime_fusion is not None
        ):
            dynamic_context, dynamic_diagnostics = self.dynamic_context_encoder(sequence, static_context)
            regime_context, regime_gate = self.regime_fusion(dynamic_context, static_context)
        temporal_mean = sequence.mean(dim=1)
        temporal_max = sequence.max(dim=1).values
        temporal_last = sequence[:, -1, :]
        if self.n2_separate_temporal_branch:
            if self.n2_dynamic_stem is None or self.n2_tcn_blocks is None or self.n2_film_layers is None:
                raise RuntimeError("N2 separate temporal branch is not fully initialized.")
            n2_context = scale_gradient(mld_film_context, self.n2_backbone_grad_scale)
            n2_sequence = self._encode_temporal(
                dynamic=dynamic,
                context=n2_context,
                stem=self.n2_dynamic_stem,
                blocks=self.n2_tcn_blocks,
                film_layers=self.n2_film_layers,
            )
        n2_temporal_mean = n2_sequence.mean(dim=1)
        n2_temporal_max = n2_sequence.max(dim=1).values
        n2_temporal_last = n2_sequence[:, -1, :]

        mld_attention = self._attention_summary(
            experts=self.experts,
            gating_network=self.gating_network,
            expert_kernel_sizes=self.expert_kernel_sizes,
            sequence=sequence,
            regime_context=regime_context,
            temporal_mean=temporal_mean,
            temporal_max=temporal_max,
            temporal_last=temporal_last,
            dynamic_context=dynamic_context,
        )
        fused, fusion_gate = self.fusion(mld_attention["dynamic_summary"], regime_context)  # type: ignore[arg-type]
        fused = self.backbone(fused)
        mld_features = self.mld_branch(fused)
        n2_attention = None
        n2_fusion_gate = None
        if self.n2_separate_attention_branch:
            if self.n2_experts is None or self.n2_gating_network is None or self.n2_fusion is None or self.n2_backbone is None:
                raise RuntimeError("N2 separate attention branch is not fully initialized.")
            n2_sequence_for_attention = (
                n2_sequence
                if self.n2_separate_temporal_branch
                else scale_gradient(sequence, self.n2_backbone_grad_scale)
            )
            n2_regime_context = scale_gradient(regime_context, self.n2_backbone_grad_scale)
            n2_dynamic_context = (
                scale_gradient(dynamic_context, self.n2_backbone_grad_scale)
                if dynamic_context is not None
                else None
            )
            n2_attention = self._attention_summary(
                experts=self.n2_experts,
                gating_network=self.n2_gating_network,
                expert_kernel_sizes=self.n2_expert_kernel_sizes,
                sequence=n2_sequence_for_attention,
                regime_context=n2_regime_context,
                temporal_mean=n2_temporal_mean if self.n2_separate_temporal_branch else scale_gradient(temporal_mean, self.n2_backbone_grad_scale),
                temporal_max=n2_temporal_max if self.n2_separate_temporal_branch else scale_gradient(temporal_max, self.n2_backbone_grad_scale),
                temporal_last=n2_temporal_last if self.n2_separate_temporal_branch else scale_gradient(temporal_last, self.n2_backbone_grad_scale),
                dynamic_context=n2_dynamic_context,
            )
            n2_features, n2_fusion_gate = self.n2_fusion(n2_attention["dynamic_summary"], n2_regime_context)  # type: ignore[arg-type]
            n2_features = self.n2_backbone(n2_features)
        else:
            n2_features = scale_gradient(fused, self.n2_backbone_grad_scale)
        n2_features = self.n2_branch(n2_features)

        # MLD neighbor assimilation: a learned correction from the K nearest analogs.
        neighbor_delta = None
        neighbor_delta4 = None
        if (
            self.use_mld_neighbors
            and self.neighbor_branch is not None
            and self.neighbor_fusion is not None
            and neighbors is not None
            and neighbor_mask is not None
        ):
            active_mask = neighbor_mask
            if self.training and self.neighbor_modality_dropout > 0.0:
                drop = torch.rand(neighbors.shape[0], device=neighbors.device) < self.neighbor_modality_dropout
                active_mask = neighbor_mask * (~drop).to(neighbor_mask.dtype).unsqueeze(1)
            neighbor_latent = self.neighbor_branch(neighbors, active_mask, static_context)
            fused_neighbor = self.neighbor_fusion(neighbor_latent)
            if self.neighbor_increment_full:
                # v3: 4 increments [MLD mean, MLD std-logit, N2 mean, N2 std-logit].
                neighbor_delta4 = fused_neighbor
            elif self.neighbor_increment:
                # v2: scalar increment added to the MLD mean (background = forcing).
                neighbor_delta = fused_neighbor.squeeze(-1)
            else:
                # v1: vector correction added to the MLD features.
                mld_features = mld_features + fused_neighbor

        is_gaussian = self.predictive_distribution == "gaussian"
        # Regime 0 reuses the shared MLD path computed above. Std heads produce a logit
        # so the neighbour std-increment is applied *inside* softplus (std stays > min_std
        # but can shrink near a float -> assimilation increases confidence).
        mld_mean_0 = self.mld_head(mld_features).squeeze(-1)
        mld_std_logit = self.mld_std_head(mld_features).squeeze(-1) if is_gaussian else None
        n2_mean = self.n2_head(n2_features).squeeze(-1)
        n2_std_logit = self.n2_std_head(n2_features).squeeze(-1) if is_gaussian else None
        if neighbor_delta is not None:
            mld_mean_0 = mld_mean_0 + neighbor_delta
        if neighbor_delta4 is not None:
            mld_mean_0 = mld_mean_0 + neighbor_delta4[:, 0]
            n2_mean = n2_mean + neighbor_delta4[:, 2]
            if is_gaussian:
                mld_std_logit = mld_std_logit + neighbor_delta4[:, 1]
                n2_std_logit = n2_std_logit + neighbor_delta4[:, 3]
        mld_std_0 = (self.min_std + F.softplus(mld_std_logit)) if is_gaussian else None
        n2_std = (self.min_std + F.softplus(n2_std_logit)) if is_gaussian else None

        regime_logits = None
        regime_weights = None
        if self.mld_regime_split and self.mld_extra_regime_branches is not None and self.regime_gate is not None:
            regime_means = [mld_mean_0]
            regime_stds = [mld_std_0] if is_gaussian else None
            for branch in self.mld_extra_regime_branches:
                branch_mean, branch_std = branch(
                    sequence=sequence,
                    regime_context=regime_context,
                    temporal_mean=temporal_mean,
                    temporal_max=temporal_max,
                    temporal_last=temporal_last,
                    dynamic_context=dynamic_context,
                )
                regime_means.append(branch_mean)
                if is_gaussian and regime_stds is not None:
                    regime_stds.append(branch_std)
            regime_mean_stack = torch.stack(regime_means, dim=-1)  # [B, K]
            regime_gate_input = torch.cat([static_context, temporal_mean, temporal_max, temporal_last], dim=-1)
            regime_logits = self.regime_gate(regime_gate_input)  # [B, K]
            regime_weights = torch.softmax(regime_logits, dim=-1)
            mld_mean = torch.sum(regime_mean_stack * regime_weights, dim=-1)
            if is_gaussian and regime_stds is not None:
                regime_std_stack = torch.stack(regime_stds, dim=-1)  # [B, K]
                # Variance of the gate-weighted mixture of Gaussians.
                mixture_second_moment = torch.sum(
                    regime_weights * (regime_std_stack.square() + regime_mean_stack.square()), dim=-1
                )
                mld_var = (mixture_second_moment - mld_mean.square()).clamp_min(self.min_std**2)
                mld_std = mld_var.sqrt()
            else:
                mld_std = mld_std_0
        else:
            mld_mean = mld_mean_0
            mld_std = mld_std_0

        prediction_mean = torch.stack([mld_mean, n2_mean], dim=-1)
        prediction = prediction_mean
        prediction_std = None
        if is_gaussian:
            prediction_std = torch.stack([mld_std, n2_std], dim=-1)
            prediction = torch.cat([prediction_mean, prediction_std], dim=-1)

        if return_regime_logits:
            return prediction, regime_logits

        if return_diagnostics:
            diagnostics = {
                "combined_attention": mld_attention["combined_attention"],
                "expert_attention": mld_attention["expert_attention"],
                "expert_attention_scores": mld_attention["expert_attention_scores"],
                "expert_gates": mld_attention["expert_gates"],
                "expert_kernel_sizes": mld_attention["expert_kernel_sizes"],
                "temporal_mean": temporal_mean,
                "temporal_max": temporal_max,
                "temporal_last": temporal_last,
                "mixture_pooled": mld_attention["mixture_pooled"],
                "dynamic_summary": mld_attention["dynamic_summary"],
                "static_group_gates": static_diagnostics["group_gates"],
                "fusion_gate": fusion_gate,
                "latent_sequence_l2": sequence.norm(dim=-1),
                "latent_sequence_abs_mean": sequence.abs().mean(dim=-1),
                "attention_pooling_mode": self.attention_pooling_mode,
                "attention_tokens_per_segment": torch.as_tensor(self.attention_tokens_per_segment, device=prediction.device),
                "predictive_distribution": self.predictive_distribution,
                "use_dynamic_context": self.use_dynamic_context,
                "dynamic_context_mode": self.dynamic_context_mode,
                "n2_separate_temporal_branch": torch.as_tensor(
                    int(self.n2_separate_temporal_branch),
                    device=prediction.device,
                ),
                "n2_separate_attention_branch": torch.as_tensor(
                    int(self.n2_separate_attention_branch),
                    device=prediction.device,
                ),
            }
            if n2_attention is not None:
                diagnostics["n2_combined_attention"] = n2_attention["combined_attention"]
                diagnostics["n2_expert_attention"] = n2_attention["expert_attention"]
                diagnostics["n2_expert_attention_scores"] = n2_attention["expert_attention_scores"]
                diagnostics["n2_expert_gates"] = n2_attention["expert_gates"]
                diagnostics["n2_expert_kernel_sizes"] = n2_attention["expert_kernel_sizes"]
                diagnostics["n2_mixture_pooled"] = n2_attention["mixture_pooled"]
                diagnostics["n2_dynamic_summary"] = n2_attention["dynamic_summary"]
                if n2_fusion_gate is not None:
                    diagnostics["n2_fusion_gate"] = n2_fusion_gate
            if mld_attention["expert_zone_weights"] is not None and mld_attention["expert_zone_scores"] is not None:
                diagnostics["expert_zone_weights"] = mld_attention["expert_zone_weights"]
                diagnostics["expert_zone_scores"] = mld_attention["expert_zone_scores"]
                diagnostics["zone_names"] = [
                    f"{0 if idx == 0 else self.experts[0].summary_windows[idx - 1]}_{end}h"  # type: ignore[attr-defined]
                    for idx, end in enumerate(self.experts[0].summary_windows)  # type: ignore[attr-defined]
                ]
                diagnostics["combined_zone_weights"] = torch.sum(
                    mld_attention["expert_zone_weights"] * mld_attention["expert_gates"].unsqueeze(-1), dim=1  # type: ignore[operator,union-attr]
                )
            if n2_attention is not None and n2_attention["expert_zone_weights"] is not None and n2_attention["expert_zone_scores"] is not None:
                diagnostics["n2_expert_zone_weights"] = n2_attention["expert_zone_weights"]
                diagnostics["n2_expert_zone_scores"] = n2_attention["expert_zone_scores"]
                diagnostics["n2_combined_zone_weights"] = torch.sum(
                    n2_attention["expert_zone_weights"] * n2_attention["expert_gates"].unsqueeze(-1), dim=1  # type: ignore[operator,union-attr]
                )
            if dynamic_diagnostics is not None and self.dynamic_context_encoder is not None:
                if "segment_gates" in dynamic_diagnostics:
                    diagnostics["dynamic_segment_gates"] = dynamic_diagnostics["segment_gates"]
                    diagnostics["dynamic_segment_names"] = dynamic_diagnostics["segment_names"]
                    diagnostics["dynamic_summary_windows"] = torch.as_tensor(
                        self.dynamic_context_encoder.summary_windows,
                        device=prediction.device,
                    )
                if "feature_gates" in dynamic_diagnostics:
                    diagnostics["dynamic_feature_gates"] = dynamic_diagnostics["feature_gates"]
                    diagnostics["dynamic_weighted_integral"] = dynamic_diagnostics["weighted_integral"]
                    diagnostics["dynamic_summary_windows"] = torch.as_tensor(
                        self.dynamic_context_encoder.summary_windows,
                        device=prediction.device,
                    )
                if "context_attention" in dynamic_diagnostics:
                    diagnostics["dynamic_context_attention"] = dynamic_diagnostics["context_attention"]
                    diagnostics["dynamic_context_attention_scores"] = dynamic_diagnostics["context_attention_scores"]
            if regime_gate is not None:
                diagnostics["regime_gate"] = regime_gate
            if regime_logits is not None and regime_weights is not None:
                diagnostics["mld_regime_logits"] = regime_logits
                diagnostics["mld_regime_weights"] = regime_weights
            if prediction_std is not None:
                diagnostics["prediction_std"] = prediction_std
            return prediction, diagnostics
        if return_attention:
            return prediction, combined_attention
        return prediction
