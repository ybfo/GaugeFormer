"""Frozen, metadata-only construction of schema and gauge stress views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import WindowStore
from .perturbations import PerturbedStore, deterministic_seed


DEFAULT_PROTOCOL_VERSION = "gf-cs-2026-08-12-v5"


@dataclass(frozen=True)
class StressCondition:
    name: str
    kind: str
    replicate: int = 0
    amount: float | int | None = None
    canonical_baseline_preprocessing: bool = False


def load_query_panel(
    path: Path,
    system_id: str,
    expected_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["protocol_version"] != expected_protocol_version:
        raise ValueError("query-panel protocol version mismatch")
    return np.asarray(payload["systems"][system_id]["indices"], dtype=np.int64)


def alternative_gauge(base: WindowStore) -> tuple[np.ndarray, np.ndarray]:
    """Return one fixed legal non-identity gauge per documented declared unit."""

    scale = np.asarray(base.unit_scale, dtype=np.float64).copy()
    offset = np.asarray(base.unit_offset, dtype=np.float64).copy()
    alternatives = {
        "bar": (1e3, 0.0),  # kPa
        "mmHg": (1.0, 0.0),  # Pa
        "degree Celsius": (1.0, 0.0),  # K
        "C": (1.0, 0.0),  # K (common metadata abbreviation)
        "W": (1e3, 0.0),  # kW
        "kW": (1.0, 0.0),  # W
        "Wh": (3.6e6, 0.0),  # kWh
        "kWh": (3600.0, 0.0),  # Wh
        "kvarh": (3600.0, 0.0),  # varh
        "L/min": (1.0, 0.0),  # m^3/s
        "mm/s": (1.0, 0.0),  # m/s
        "%": (1.0, 0.0),  # fraction
        "km": (1.0, 0.0),  # m
        "rpm": (1.0, 0.0),  # rad/s
        "A": (1e-3, 0.0),  # mA
        "V": (1e-3, 0.0),  # mV
        "N m": (1e3, 0.0),  # kN m
        "s": (60.0, 0.0),  # min
    }
    for index, channel in enumerate(base.metadata["channels"]):
        chosen = alternatives.get(channel["declared_unit"])
        if chosen is not None and not np.allclose(
            chosen, (float(scale[index]), float(offset[index]))
        ):
            scale[index], offset[index] = chosen
    return scale.astype(np.float32), offset.astype(np.float32)


def make_stress_view(
    base: WindowStore,
    panel: np.ndarray,
    condition: StressCondition,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> PerturbedStore:
    channels = np.arange(len(base.metadata["channels"]), dtype=np.int64)
    seed = deterministic_seed(
        protocol_version,
        base.metadata["system_id"],
        condition.name,
        condition.replicate,
    )
    rng = np.random.default_rng(seed)
    source = channels.copy()
    inserted_positions: list[int] = []
    declared_scale = None
    declared_offset = None
    if condition.kind == "permutation":
        source = rng.permutation(channels)
    elif condition.kind == "deletion":
        nonquery = np.setdiff1d(channels, panel, assume_unique=False)
        number = min(
            len(nonquery),
            int(np.ceil(float(condition.amount) * len(nonquery))),
        )
        removed = rng.choice(nonquery, size=number, replace=False) if number else []
        source = np.asarray(
            [index for index in channels if index not in set(map(int, removed))],
            dtype=np.int64,
        )
    elif condition.kind == "insertion":
        number = int(condition.amount)
        if number < 1:
            raise ValueError("insertion count must be positive")
        candidates = np.setdiff1d(channels, panel, assume_unique=False)
        if not len(candidates):
            candidates = channels
        inserted_sources = rng.choice(candidates, size=number, replace=True)
        source = np.concatenate((channels, inserted_sources.astype(np.int64)))
        inserted_positions = list(range(len(channels), len(source)))
    elif condition.kind == "gauge":
        declared_scale, declared_offset = alternative_gauge(base)
    elif condition.kind != "reference":
        raise ValueError(f"unsupported stress condition: {condition.kind}")
    return PerturbedStore(
        base,
        source_indices=source,
        query_source_indices=panel,
        declared_scale=(declared_scale[source] if declared_scale is not None else None),
        declared_offset=(
            declared_offset[source] if declared_offset is not None else None
        ),
        inserted_positions=inserted_positions,
        perturbation_seed=seed,
        canonical_baseline_preprocessing=condition.canonical_baseline_preprocessing,
    )


def robustness_conditions(replicates: int = 3) -> list[StressCondition]:
    conditions = [StressCondition("reference", "reference")]
    conditions.extend(
        [
            StressCondition("legal_gauge_direct", "gauge"),
            StressCondition(
                "legal_gauge_canonical_preprocessing",
                "gauge",
                canonical_baseline_preprocessing=True,
            ),
        ]
    )
    for fraction in (0.1, 0.3, 0.5):
        for replicate in range(replicates):
            conditions.append(
                StressCondition(
                    f"deletion_{int(100 * fraction)}",
                    "deletion",
                    replicate=replicate,
                    amount=fraction,
                )
            )
    for number in (1, 2, 4):
        for replicate in range(replicates):
            conditions.append(
                StressCondition(
                    f"insertion_{number}",
                    "insertion",
                    replicate=replicate,
                    amount=number,
                )
            )
    return conditions
