"""Paired block inference for immutable multi-method result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


def validate_block_reconstruction(metric: Mapping[str, object], atol: float = 1e-10) -> None:
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


def aligned_blocks(
    first: Mapping[str, object], second: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned block scores and weights, rejecting any key/count mismatch."""

    validate_block_reconstruction(first)
    validate_block_reconstruction(second)
    left = {record["block_id"]: record for record in first["statistical_blocks"]}
    right = {record["block_id"]: record for record in second["statistical_blocks"]}
    if set(left) != set(right):
        raise ValueError("paired methods do not contain identical statistical blocks")
    keys = sorted(left)
    left_weight = np.asarray(
        [int(left[key]["elements_per_channel"]) for key in keys], dtype=np.int64
    )
    right_weight = np.asarray(
        [int(right[key]["elements_per_channel"]) for key in keys], dtype=np.int64
    )
    if not np.array_equal(left_weight, right_weight):
        raise ValueError("paired methods have different block element counts")
    left_score = np.asarray([float(left[key]["sd_nmae"]) for key in keys])
    right_score = np.asarray([float(right[key]["sd_nmae"]) for key in keys])
    return left_score, right_score, left_weight


@dataclass(frozen=True)
class PairedBootstrapConfig:
    replicates: int = 20_000
    seed: int = 20260813
    confidence: float = 0.95


def paired_block_bootstrap(
    proposed: Mapping[str, object],
    baseline: Mapping[str, object],
    config: PairedBootstrapConfig = PairedBootstrapConfig(),
) -> dict[str, float | int]:
    """Bootstrap baseline-minus-proposed SD-NMAE with common physical blocks."""

    proposed_score, baseline_score, weight = aligned_blocks(proposed, baseline)
    delta = baseline_score - proposed_score
    observed = float(np.average(delta, weights=weight))
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.replicates, dtype=np.float64)
    for offset in range(config.replicates):
        index = rng.integers(0, len(delta), size=len(delta))
        samples[offset] = np.average(delta[index], weights=weight[index])
    alpha = (1.0 - config.confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    # Two-sided sign-bootstrap p-value with the standard finite-replicate correction.
    p_value = min(
        1.0,
        2.0
        * (min(np.count_nonzero(samples <= 0), np.count_nonzero(samples >= 0)) + 1)
        / (config.replicates + 1),
    )
    return {
        "baseline_minus_proposed": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(p_value),
        "blocks": int(len(delta)),
        "replicates": int(config.replicates),
        "seed": int(config.seed),
    }


def _validate_paired_grid(
    proposed: Mapping[Hashable, Mapping[str, Mapping[str, object]]],
    baseline: Mapping[Hashable, Mapping[str, Mapping[str, object]]],
) -> tuple[list[Hashable], list[str], dict[tuple[Hashable, str], tuple[np.ndarray, np.ndarray]]]:
    """Validate a seed-by-system result grid and cache paired block deltas."""

    if not proposed or set(proposed) != set(baseline):
        raise ValueError("paired methods must contain identical non-empty seed sets")
    seeds = sorted(proposed, key=str)
    reference_systems = set(proposed[seeds[0]])
    if not reference_systems:
        raise ValueError("every seed must contain at least one system")
    cells: dict[tuple[Hashable, str], tuple[np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        if set(proposed[seed]) != reference_systems:
            raise ValueError("proposed result grid has inconsistent systems across seeds")
        if set(baseline[seed]) != reference_systems:
            raise ValueError("baseline result grid has inconsistent systems across seeds")
        for system in sorted(reference_systems):
            proposed_score, baseline_score, weight = aligned_blocks(
                proposed[seed][system], baseline[seed][system]
            )
            cells[(seed, system)] = (baseline_score - proposed_score, weight)
    return seeds, sorted(reference_systems), cells


def hierarchical_paired_block_bootstrap(
    proposed: Mapping[Hashable, Mapping[str, Mapping[str, object]]],
    baseline: Mapping[Hashable, Mapping[str, Mapping[str, object]]],
    config: PairedBootstrapConfig = PairedBootstrapConfig(),
) -> dict[str, float | int | str]:
    """Infer a system-balanced improvement across seeds and physical blocks.

    Each replicate draws the paired training seeds with replacement.  Within
    every sampled seed and each fixed benchmark system, physical time blocks
    are drawn with replacement and combined using their element counts.  Cell
    scores are then averaged equally across systems and sampled seeds, exactly
    matching the paper's system-balanced primary estimand while propagating
    both optimization-seed and temporal-block uncertainty.
    """

    if config.replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < config.confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    seeds, systems, cells = _validate_paired_grid(proposed, baseline)
    observed_cells = [
        float(np.average(delta, weights=weight))
        for delta, weight in cells.values()
    ]
    observed = float(np.mean(observed_cells))
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.replicates, dtype=np.float64)
    for replicate in range(config.replicates):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        total = 0.0
        count = 0
        for seed_index in sampled_seeds:
            seed = seeds[int(seed_index)]
            for system in systems:
                delta, weight = cells[(seed, system)]
                block_index = rng.integers(0, len(delta), size=len(delta))
                total += float(
                    np.average(delta[block_index], weights=weight[block_index])
                )
                count += 1
        samples[replicate] = total / count
    alpha = (1.0 - config.confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    p_value = min(
        1.0,
        2.0
        * (min(np.count_nonzero(samples <= 0), np.count_nonzero(samples >= 0)) + 1)
        / (config.replicates + 1),
    )
    return {
        "estimand": "system-balanced mean baseline-minus-proposed SD-NMAE across training seeds",
        "baseline_minus_proposed": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(p_value),
        "training_seeds": int(len(seeds)),
        "systems": int(len(systems)),
        "seed_system_cells": int(len(cells)),
        "physical_blocks_across_cells": int(
            sum(len(delta) for delta, _ in cells.values())
        ),
        "replicates": int(config.replicates),
        "seed": int(config.seed),
    }


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
