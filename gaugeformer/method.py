"""Frozen GaugeFormer level-memory forecasting procedure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .local import expert_predictions as local_predictions
from .memory import GaugeDynamicsMemory, OnlineRouterConfig


Array = np.ndarray
HORIZON = 24
BACKCAST_ORIGINS = (48, 56, 64, 72)
STABLE_NAMES = (
    "persistence",
    "damped_trend_12_0.8",
    "damped_trend_24_0.9",
)
LEVEL_NAMES = (
    "recent_median_6",
    "recent_median_12",
    "recent_median_24",
    "holt_0.3_0.1_0.8",
    "holt_0.3_0.1_0.95",
)
FULL_CANDIDATE_NAMES = ("backbone_raw", *STABLE_NAMES, *LEVEL_NAMES)
VARIANT_CANDIDATES = {
    "full": FULL_CANDIDATE_NAMES,
    "backbone_only": ("backbone_raw",),
    "no_level_experts": ("backbone_raw", *STABLE_NAMES),
    "no_trend_experts": (
        "backbone_raw",
        "persistence",
        "recent_median_6",
        "recent_median_12",
        "recent_median_24",
    ),
    "no_origin_win_guard": FULL_CANDIDATE_NAMES,
    "no_shrinkage": FULL_CANDIDATE_NAMES,
    "no_warmup": FULL_CANDIDATE_NAMES,
}


@dataclass(frozen=True)
class GaugeFormerConfig:
    """Immutable method and one-factor-ablation configuration."""

    variant: str = "full"
    origins: tuple[int, ...] = BACKCAST_ORIGINS
    margin: float = 0.0
    minimum_origin_win_fraction: float = 0.55
    blend: float = 0.5
    warmup_windows: int = 16

    def __post_init__(self) -> None:
        if self.variant not in VARIANT_CANDIDATES:
            raise ValueError(f"unknown GaugeFormer variant: {self.variant}")
        if self.origins != BACKCAST_ORIGINS:
            raise ValueError("GaugeFormer backcast origins differ from the freeze")
        if self.margin != 0.0:
            raise ValueError("GaugeFormer margin differs from the freeze")
        if self.variant == "no_origin_win_guard":
            expected_fraction = 0.0
        else:
            expected_fraction = 0.55
        if self.minimum_origin_win_fraction != expected_fraction:
            raise ValueError("origin-win threshold differs from the selected variant")
        expected_blend = 1.0 if self.variant == "no_shrinkage" else 0.5
        if self.blend != expected_blend:
            raise ValueError("blend differs from the selected variant")
        expected_warmup = 1 if self.variant == "no_warmup" else 16
        if self.warmup_windows != expected_warmup:
            raise ValueError("warmup differs from the selected variant")

    @classmethod
    def for_variant(cls, variant: str) -> "GaugeFormerConfig":
        return cls(
            variant=variant,
            minimum_origin_win_fraction=(
                0.0 if variant == "no_origin_win_guard" else 0.55
            ),
            blend=1.0 if variant == "no_shrinkage" else 0.5,
            warmup_windows=1 if variant == "no_warmup" else 16,
        )

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return VARIANT_CANDIDATES[self.variant]

    @property
    def router_config(self) -> OnlineRouterConfig:
        return OnlineRouterConfig(
            margin=self.margin,
            minimum_origin_win_fraction=self.minimum_origin_win_fraction,
            blend=self.blend,
            warmup_windows=self.warmup_windows,
        )


@dataclass(frozen=True)
class GaugeFormerOutput:
    prediction: Array
    selected_candidate: Array
    eligible: Array
    candidate_names: tuple[str, ...]
    backcast_losses: Array


def _validate_context_query(context: Array, query: Array) -> tuple[Array, Array]:
    values = np.asarray(context, dtype=np.float64)
    query_index = np.asarray(query, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 96:
        raise ValueError("context must have shape [channels, 96]")
    if query_index.ndim != 1 or len(query_index) < 1:
        raise ValueError("query must be a nonempty one-dimensional array")
    if query_index.min() < 0 or query_index.max() >= values.shape[0]:
        raise ValueError("query lies outside the channel axis")
    if len(np.unique(query_index)) != len(query_index):
        raise ValueError("query indices must be unique")
    if not np.isfinite(values).all():
        raise ValueError("context contains a nonfinite value")
    return values, query_index


def full_expert_predictions(context, query, *, horizon=HORIZON):
    return local_predictions(context, query, horizon=horizon)


def candidate_predictions(
    context: Array,
    query: Array,
    raw_backbone: Array,
    candidate_names: tuple[str, ...] = FULL_CANDIDATE_NAMES,
) -> Array:
    values, query_index = _validate_context_query(context, query)
    raw = np.asarray(raw_backbone, dtype=np.float32)
    if raw.shape != (len(query_index), HORIZON) or not np.isfinite(raw).all():
        raise ValueError(
            "raw backbone forecast has the wrong shape or a nonfinite value"
        )
    if not candidate_names or candidate_names[0] != "backbone_raw":
        raise ValueError("candidate zero must be raw backbone")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate names must be unique")
    if not set(candidate_names).issubset(FULL_CANDIDATE_NAMES):
        raise ValueError("candidate names leave the fixed family")
    alternative_names, alternatives = full_expert_predictions(values, query_index)
    full_names = ("backbone_raw", *alternative_names)
    full = np.concatenate((raw[None], alternatives), axis=0)
    positions = [full_names.index(name) for name in candidate_names]
    return full[positions]


def context_backcast_losses(
    context: Array,
    query: Array,
    raw_backbone_backcasts: Array,
    candidate_names: tuple[str, ...] = FULL_CANDIDATE_NAMES,
    origins: tuple[int, ...] = BACKCAST_ORIGINS,
) -> Array:
    """Compute causal candidate losses using only realized context prefixes."""

    values, query_index = _validate_context_query(context, query)
    if origins != BACKCAST_ORIGINS:
        raise ValueError("backcast origins differ from the fixed coordinates")
    raw = np.asarray(raw_backbone_backcasts, dtype=np.float32)
    expected = (len(origins), len(query_index), HORIZON)
    if raw.shape != expected or not np.isfinite(raw).all():
        raise ValueError(
            "raw backbone backcasts have the wrong shape or a nonfinite value"
        )
    losses = []
    for position, origin in enumerate(origins):
        observed = values[query_index, origin : origin + HORIZON]
        if observed.shape != (len(query_index), HORIZON):
            raise ValueError("backcast origin does not expose one complete horizon")
        # Candidate experts must see the actual prefix length, not the left padding
        # used only by the fixed-length backbone bridge.
        alternative_names, alternatives = full_expert_predictions(
            values[:, :origin], query_index
        )
        full_names = ("backbone_raw", *alternative_names)
        full = np.concatenate((raw[position : position + 1], alternatives), axis=0)
        positions = [full_names.index(name) for name in candidate_names]
        candidates = full[positions]
        losses.append(
            np.abs(candidates.astype(np.float64) - observed[None]).mean(axis=-1)
        )
    result = np.stack(losses)
    if result.shape != (len(origins), len(candidate_names), len(query_index)):
        raise RuntimeError("GaugeFormer backcast-loss schema differs")
    return result


class GaugeFormerMemory:
    """One causal online memory stream; reset it at every system/sequence boundary."""

    def __init__(self, config: GaugeFormerConfig | None = None) -> None:
        self.config = config or GaugeFormerConfig()
        self._memory = (
            None
            if self.config.variant == "backbone_only"
            else GaugeDynamicsMemory(self.config.router_config)
        )
        self.observed_windows = 0

    def predict(
        self,
        context: Array,
        query: Array,
        raw_backbone_future: Array,
        raw_backbone_backcasts: Array,
    ) -> GaugeFormerOutput:
        values, query_index = _validate_context_query(context, query)
        candidates = candidate_predictions(
            values,
            query_index,
            raw_backbone_future,
            self.config.candidate_names,
        )
        losses = context_backcast_losses(
            values,
            query_index,
            raw_backbone_backcasts,
            self.config.candidate_names,
            self.config.origins,
        )
        self.observed_windows += 1
        if self._memory is None:
            selected = np.zeros(len(query_index), dtype=np.int64)
            eligible = np.zeros(len(query_index), dtype=bool)
            prediction = candidates[0].astype(np.float64)
        else:
            prediction, selected, eligible = self._memory.update_and_predict(
                candidates, losses
            )
        return GaugeFormerOutput(
            prediction=prediction.astype(np.float32),
            selected_candidate=selected.astype(np.int64),
            eligible=eligible.astype(bool),
            candidate_names=self.config.candidate_names,
            backcast_losses=losses.astype(np.float32),
        )

    def evidence_summary(self) -> dict[str, object]:
        if self._memory is None:
            return {
                "observed_windows": self.observed_windows,
                "observed_origin_panels": self.observed_windows
                * len(self.config.origins),
                "variant": "backbone_only",
            }
        return self._memory.evidence_summary(self.config.candidate_names)
