"""Cumulative historical loss, repeated wins, and accepted residual shrinkage."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
Array = np.ndarray
def select_online_blend(
    future_candidates: Array,
    cumulative_losses: Array,
    cumulative_origin_wins: Array,
    *,
    observed_origin_panels: int,
    observed_windows: int,
    margin: float,
    minimum_win_fraction: float,
    blend: float,
    warmup_windows: int,
) -> tuple[Array, Array, Array]:
    """Select from cumulative context-backcast evidence in an unlabeled stream."""

    candidates = np.asarray(future_candidates, dtype=np.float64)
    losses = np.asarray(cumulative_losses, dtype=np.float64)
    wins = np.asarray(cumulative_origin_wins, dtype=np.float64)
    if candidates.ndim != 3:
        raise ValueError("future_candidates must have shape [experts, query, horizon]")
    if losses.shape != candidates.shape[:2] or wins.shape != losses.shape:
        raise ValueError("cumulative evidence must have shape [experts, query]")
    if candidates.shape[0] < 2:
        raise ValueError("candidate zero plus at least one alternative are required")
    if observed_origin_panels < 1 or observed_windows < 1 or warmup_windows < 1:
        raise ValueError("online evidence counts and warmup must be positive")
    if not 0 <= margin < 1 or not 0 <= minimum_win_fraction <= 1:
        raise ValueError("margin and win fraction lie outside their ranges")
    if not 0 <= blend <= 1:
        raise ValueError("blend must lie in [0,1]")
    if not np.isfinite(candidates).all() or not np.isfinite(losses).all():
        raise ValueError("candidates and cumulative losses must be finite")

    query = np.arange(candidates.shape[1])
    best = np.argmin(losses[1:], axis=0) + 1
    best_loss = losses[best, query]
    base_loss = losses[0]
    win_fraction = wins[best, query] / observed_origin_panels
    eligible = (
        (observed_windows >= warmup_windows)
        & (best_loss <= base_loss * (1.0 - margin))
        & (win_fraction >= minimum_win_fraction)
    )
    selected = np.where(eligible, best, 0)
    alternative = candidates[selected, query]
    weight = eligible.astype(np.float64)[:, None] * blend
    prediction = candidates[0] + weight * (alternative - candidates[0])
    return prediction, selected, eligible

@dataclass(frozen=True)
class OnlineRouterConfig:
    margin: float = 0.0
    minimum_origin_win_fraction: float = 0.55
    blend: float = 0.5
    warmup_windows: int = 16


class GaugeDynamicsMemory:
    """Causal context-backcast memory with exact backbone fallback."""

    def __init__(self, config: OnlineRouterConfig | None = None) -> None:
        self.config = config or OnlineRouterConfig()
        self.cumulative_losses: np.ndarray | None = None
        self.cumulative_origin_wins: np.ndarray | None = None
        self.observed_origin_panels = 0
        self.observed_windows = 0

    def update_and_predict(
        self,
        future_candidates: np.ndarray,
        backcast_losses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates = np.asarray(future_candidates, dtype=np.float64)
        losses = np.asarray(backcast_losses, dtype=np.float64)
        if candidates.ndim != 3:
            raise ValueError("future_candidates must have shape [experts, query, horizon]")
        if losses.ndim != 3 or losses.shape[1:] != candidates.shape[:2]:
            raise ValueError("backcast_losses must have shape [origins, experts, query]")
        if losses.shape[0] < 1:
            raise ValueError("at least one realized context backcast is required")
        if not np.isfinite(candidates).all() or not np.isfinite(losses).all():
            raise ValueError("router inputs must be finite")
        if self.cumulative_losses is None:
            self.cumulative_losses = np.zeros(candidates.shape[:2], dtype=np.float64)
            self.cumulative_origin_wins = np.zeros_like(self.cumulative_losses)
        if self.cumulative_losses.shape != candidates.shape[:2]:
            raise ValueError("candidate or query schema changed inside an online stream")

        self.cumulative_losses += losses.sum(axis=0)
        self.cumulative_origin_wins += (
            losses < losses[:, 0:1, :]
        ).sum(axis=0)
        self.observed_origin_panels += losses.shape[0]
        self.observed_windows += 1
        return select_online_blend(
            candidates,
            self.cumulative_losses,
            self.cumulative_origin_wins,
            observed_origin_panels=self.observed_origin_panels,
            observed_windows=self.observed_windows,
            margin=self.config.margin,
            minimum_win_fraction=self.config.minimum_origin_win_fraction,
            blend=self.config.blend,
            warmup_windows=self.config.warmup_windows,
        )

    def evidence_summary(self, candidate_names: tuple[str, ...]) -> dict[str, object]:
        if self.cumulative_losses is None or self.cumulative_origin_wins is None:
            raise RuntimeError("online memory has not observed a window")
        if len(candidate_names) != self.cumulative_losses.shape[0]:
            raise ValueError("candidate names differ from online memory")
        raw = np.maximum(self.cumulative_losses[0], 1e-10)
        return {
            "observed_windows": self.observed_windows,
            "observed_origin_panels": self.observed_origin_panels,
            "cumulative_loss_ratio_vs_raw": {
                name: float(np.mean(self.cumulative_losses[index] / raw))
                for index, name in enumerate(candidate_names)
            },
            "origin_win_fraction_vs_raw": {
                name: float(
                    np.mean(
                        self.cumulative_origin_wins[index]
                        / self.observed_origin_panels
                    )
                )
                for index, name in enumerate(candidate_names)
            },
        }

