"""Paired block inference for immutable multi-method result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


def validate_block_reconstruction(
    metric: Mapping[str, object], atol: float = 1e-10
) -> None:
    """Verify that block sufficient statistics reproduce the reported SD-NMAE."""

    records = metric.get("statistical_blocks")
    if not isinstance(records, list) or not records:
        raise ValueError("result has no statistical blocks")
    weights = np.asarray(
        [int(record["elements_per_channel"]) for record in records], dtype=np.float64
    )
    values = np.asarray([float(record["sd_nmae"]) for record in records])
    if np.any(weights <= 0) or not np.isfinite(values).all():
        raise ValueError("invalid block sufficient statistics")
    reconstructed = float(np.average(values, weights=weights))
    if not np.isclose(reconstructed, float(metric["sd_nmae"]), atol=atol, rtol=0.0):
        raise ValueError(
            f"block reconstruction {reconstructed} differs from metric {metric['sd_nmae']}"
        )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in their original order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a vector in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, values[order] * (len(values) - np.arange(len(values))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()
