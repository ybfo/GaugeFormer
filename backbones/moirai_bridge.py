"""Thin, non-invasive bridge to the official Moirai-1.1 implementation.

The bridge intentionally does not copy or modify Salesforce's source.  It
uses the public ``MoiraiForecast._convert`` layout helper and a scoped forward
hook on ``MoiraiModule.in_proj``.  The hook exposes independent per-variate
patch tokens while leaving the official distributional forward pass intact.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn


class Moirai1Bridge(nn.Module):
    """Return official Moirai forecasts together with independent patch tokens."""

    def __init__(
        self,
        forecast: nn.Module,
        *,
        patch_size: int = 16,
        training_point_samples: int = 20,
        evaluation_point_samples: int = 100,
    ) -> None:
        super().__init__()
        if patch_size not in tuple(int(value) for value in forecast.module.patch_sizes):
            raise ValueError("patch_size is absent from the official Moirai module")
        if training_point_samples < 1 or evaluation_point_samples < 1:
            raise ValueError("point sample counts must be positive")
        self.forecast = forecast
        self.patch_size = int(patch_size)
        self.training_point_samples = int(training_point_samples)
        self.evaluation_point_samples = int(evaluation_point_samples)

    @property
    def foundation_width(self) -> int:
        return int(self.forecast.module.d_model)

    @staticmethod
    def _shared_query_indices(query_index: torch.Tensor, batch: int) -> torch.Tensor:
        if query_index.ndim == 1:
            result = query_index
        elif query_index.ndim == 2 and query_index.shape[0] == batch:
            result = query_index[0]
            if not torch.equal(query_index, result.unsqueeze(0).expand_as(query_index)):
                raise ValueError(
                    "The official packed Moirai bridge requires one shared query set per batch"
                )
        else:
            raise ValueError("query_index must have shape [Q] or [B, Q]")
        if result.numel() == 0:
            raise ValueError("At least one query channel is required")
        if torch.unique(result).numel() != result.numel():
            raise ValueError("query_index contains a duplicate channel")
        return result.to(dtype=torch.long)

    @staticmethod
    def _format_point(
        forecast: nn.Module,
        patch_size: int,
        packed: torch.Tensor,
        target_dim: int,
    ) -> torch.Tensor:
        formatted = forecast._format_preds(patch_size, packed, target_dim)
        if target_dim == 1 and formatted.ndim == 3:
            formatted = formatted.unsqueeze(-1)
        if formatted.ndim != 4:
            raise RuntimeError("Unexpected official Moirai prediction layout")
        return formatted

    @staticmethod
    def _pool_original_channels(
        independent_tokens: torch.Tensor,
        sample_id: torch.Tensor,
        variate_id: torch.Tensor,
        prediction_mask: torch.Tensor,
        packed_to_original: torch.Tensor,
        channel_count: int,
    ) -> torch.Tensor:
        batch, _, width = independent_tokens.shape
        pooled = independent_tokens.new_zeros(batch, channel_count, width)
        valid_context = (sample_id > 0) & ~prediction_mask
        for packed_id, original_id in enumerate(packed_to_original.tolist()):
            selected = valid_context & (variate_id == packed_id)
            weights = selected.to(independent_tokens.dtype).unsqueeze(-1)
            pooled[:, original_id] = (independent_tokens * weights).sum(1) / (
                weights.sum(1).clamp_min(1.0)
            )
        return pooled

    def forward(
        self,
        canonical_values: torch.Tensor,
        observation_mask: torch.Tensor,
        query_index: torch.Tensor,
        *,
        future_canonical: torch.Tensor | None = None,
        future_observation_mask: torch.Tensor | None = None,
        point_mode: str | None = None,
        point_samples: int | None = None,
    ) -> dict[str, Any]:
        if canonical_values.ndim != 3:
            raise ValueError("canonical_values must have shape [B, C, L]")
        if observation_mask.shape != canonical_values.shape:
            raise ValueError("observation_mask must match canonical_values")
        batch, channels, context_length = canonical_values.shape
        if context_length != int(self.forecast.hparams.context_length):
            raise ValueError(
                "Context length differs from the official forecast wrapper"
            )
        query = self._shared_query_indices(query_index, batch).to(
            device=canonical_values.device
        )
        if torch.any((query < 0) | (query >= channels)):
            raise ValueError("query_index is outside the channel set")
        query_set = set(query.tolist())
        covariate = torch.tensor(
            [index for index in range(channels) if index not in query_set],
            device=canonical_values.device,
            dtype=torch.long,
        )
        packed_to_original = torch.cat((query, covariate))

        time_major = canonical_values.transpose(1, 2)
        mask_time_major = observation_mask.transpose(1, 2)
        target_context = time_major.index_select(-1, query)
        target_observed = mask_time_major.index_select(-1, query)
        past_covariates = (
            time_major.index_select(-1, covariate) if covariate.numel() else None
        )
        past_covariates_observed = (
            mask_time_major.index_select(-1, covariate) if covariate.numel() else None
        )
        past_is_pad = ~observation_mask.any(dim=1)

        target_future = None
        target_future_observed = None
        future_is_pad = None
        if future_canonical is not None:
            if future_canonical.ndim != 3 or future_canonical.shape[:2] != (
                batch,
                channels,
            ):
                raise ValueError("future_canonical must have shape [B, C, H]")
            future_time_major = future_canonical.transpose(1, 2)
            target_future = future_time_major.index_select(-1, query)
            if future_observation_mask is None:
                target_future_observed = torch.ones_like(
                    target_future, dtype=torch.bool
                )
            else:
                if future_observation_mask.shape != future_canonical.shape:
                    raise ValueError(
                        "future_observation_mask must match future_canonical"
                    )
                target_future_observed = future_observation_mask.transpose(
                    1, 2
                ).index_select(-1, query)
            future_is_pad = ~target_future_observed.any(dim=-1)

        context = (
            self.forecast.hparams_context(
                target_dim=int(query.numel()),
                past_feat_dynamic_real_dim=int(covariate.numel()),
            )
            if hasattr(self.forecast, "hparams_context")
            else nullcontext()
        )
        with context:
            converted = self.forecast._convert(
                self.patch_size,
                past_target=target_context,
                past_observed_target=target_observed,
                past_is_pad=past_is_pad,
                future_target=target_future,
                future_observed_target=target_future_observed,
                future_is_pad=future_is_pad,
                past_feat_dynamic_real=past_covariates,
                past_observed_feat_dynamic_real=past_covariates_observed,
            )
        target, observed, sample_id, time_id, variate_id, prediction_mask = converted
        captured: list[torch.Tensor] = []

        def capture_independent(
            _module: nn.Module,
            _arguments: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured.append(output)

        handle = self.forecast.module.in_proj.register_forward_hook(capture_independent)
        try:
            distribution = self.forecast.module(
                target,
                observed,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                torch.full_like(time_id, self.patch_size),
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Official Moirai input projection was not called once")
        channel_tokens = self._pool_original_channels(
            captured[0],
            sample_id,
            variate_id,
            prediction_mask,
            packed_to_original,
            channels,
        )

        mode = point_mode or "sample_median"
        if mode == "mean":
            if point_samples is not None:
                raise ValueError("point_samples is only valid for sample_median")
            packed_point = distribution.mean.unsqueeze(0)
            formatted = self._format_point(
                self.forecast, self.patch_size, packed_point, int(query.numel())
            )
            base = formatted[:, 0]
        elif mode == "sample_median":
            samples = point_samples
            if samples is None:
                samples = (
                    self.training_point_samples
                    if self.training
                    else self.evaluation_point_samples
                )
            if samples < 1:
                raise ValueError("point_samples must be positive")
            packed_samples = distribution.sample(torch.Size((samples,)))
            formatted = self._format_point(
                self.forecast, self.patch_size, packed_samples, int(query.numel())
            )
            base = formatted.median(dim=1).values
        else:
            raise ValueError("point_mode must be 'mean' or 'sample_median'")

        return {
            "base_forecast_canonical": base.transpose(1, 2),
            "channel_tokens": channel_tokens,
            "distribution": distribution,
            "packed_target": target,
            "packed_observed_mask": observed,
            "packed_sample_id": sample_id,
            "packed_time_id": time_id,
            "packed_variate_id": variate_id,
            "packed_prediction_mask": prediction_mask,
            "packed_patch_size": torch.full_like(time_id, self.patch_size),
            "packed_to_original": packed_to_original,
            "query_index_shared": query,
            "point_mode": mode,
            "point_sample_count": 0 if mode == "mean" else int(samples),
        }
