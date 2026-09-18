"""Shared immutable few-shot subset and source-episode definitions."""

from __future__ import annotations

import hashlib

import numpy as np


PROTOCOL = "gf-cs-2026-08-12-v5"


def hashed_subset(
    length: int,
    fraction: float,
    system: str,
    protocol_version: str = PROTOCOL,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("few-shot fraction must lie in (0, 1]")
    count = max(1, round(length * fraction))
    order = sorted(
        range(length),
        key=lambda index: hashlib.sha256(
            f"{protocol_version}:{system}:{fraction}:{index}".encode("utf-8")
        ).digest(),
    )
    return np.asarray(order[:count], dtype=np.int64)


def subset_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
