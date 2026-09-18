"""The eight interpretable local trajectories used in GaugeFormer."""
from __future__ import annotations
import numpy as np
Array = np.ndarray
def _persistence(past: Array, horizon: int) -> Array:
    return np.repeat(past[:, -1:], horizon, axis=-1)


def _damped_trend(past: Array, horizon: int, width: int, damping: float) -> Array:
    recent = past[:, -min(width, past.shape[-1]) :]
    slope = np.median(np.diff(recent, axis=-1), axis=-1, keepdims=True)
    powers = np.arange(1, horizon + 1, dtype=np.float64)
    cumulative = np.cumsum(np.power(damping, powers))[None, :]
    return past[:, -1:] + slope * cumulative

def _validate(past: Array, horizon: int) -> Array:
    values = np.asarray(past, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("past must have shape [channels, time] with time >= 2")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not np.isfinite(values).all():
        raise ValueError("past contains a nonfinite value")
    return values


def recent_median(past: Array, horizon: int, width: int) -> Array:
    values = _validate(past, horizon)
    if width < 1:
        raise ValueError("width must be positive")
    level = np.median(values[:, -min(width, values.shape[1]) :], axis=-1)
    return np.repeat(level[:, None], horizon, axis=-1)


def damped_holt(
    past: Array,
    horizon: int,
    *,
    alpha: float,
    beta: float,
    damping: float,
) -> Array:
    values = _validate(past, horizon)
    if not all(0.0 < value < 1.0 for value in (alpha, beta, damping)):
        raise ValueError("Holt coefficients must lie in (0, 1)")
    level = values[:, 0].copy()
    trend = values[:, 1] - values[:, 0]
    for index in range(1, values.shape[1]):
        previous = level
        level = alpha * values[:, index] + (1.0 - alpha) * (
            level + damping * trend
        )
        trend = beta * (level - previous) + (1.0 - beta) * damping * trend
    powers = np.arange(1, horizon + 1, dtype=np.float64)
    multiplier = np.cumsum(np.power(damping, powers))
    return level[:, None] + trend[:, None] * multiplier[None]


STABLE_NAMES = ('persistence', 'damped_trend_12_0.8', 'damped_trend_24_0.9')
LEVEL_NAMES = ('recent_median_6', 'recent_median_12', 'recent_median_24',
               'holt_0.3_0.1_0.8', 'holt_0.3_0.1_0.95')

def expert_predictions(context, query, *, horizon=24):
    values = _validate(context, horizon)
    query = np.asarray(query, dtype=np.int64)
    stable = np.stack((_persistence(values, horizon),
        _damped_trend(values, horizon, 12, .8),
        _damped_trend(values, horizon, 24, .9)))[:, query]
    levels = np.stack([recent_median(values, horizon, w) for w in (6, 12, 24)] +
        [damped_holt(values, horizon, alpha=.3, beta=.1, damping=rho) for rho in (.8, .95)])[:, query].astype(np.float32)
    # The evaluated implementation rounds the level bank to float32 before
    # concatenation with the float64 increment bank; retain this order.
    return (*STABLE_NAMES, *LEVEL_NAMES), np.concatenate((stable, levels), axis=0)

def selected_experts(contexts, query):
    values = np.asarray(contexts, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite [windows, channels, time] contexts')
    count, channels, length = values.shape
    _, predictions = expert_predictions(values.reshape(count*channels, length),
                                         np.arange(count*channels))
    return predictions.reshape(8, count, channels, 24).transpose(1, 0, 2, 3)[:, :, query]
