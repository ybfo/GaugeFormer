"""Validation of globally selected few-shot adaptation configurations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .protocol import file_sha256


FROZEN_BASE_RATES = {
    "gaugeformer": 3e-4,
    "unitime": 1e-4,
    "moirai": 5e-7,
    "timer_xl": 5e-6,
    "gtm": 2e-5,
    "cpiri": 2e-4,
}


def validate_selection(
    manifest: Path,
    method: str,
    fraction: float,
    fine_tuning_windows: int,
    learning_rate: float,
    protocol_version: str | None = None,
) -> str:
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if (
        protocol_version is not None
        and payload.get("protocol_version") != protocol_version
    ):
        raise ValueError("few-shot selection protocol version differs")
    chosen = payload["selected"]
    if payload.get("method") != method or not np.isclose(
        float(payload.get("fraction")), fraction
    ):
        raise ValueError("few-shot selection identity differs")
    if payload.get("selection_seed") != 17 or payload.get("test_access") is not False:
        raise ValueError("few-shot selection provenance is invalid")
    if int(chosen["fine_tuning_windows"]) != fine_tuning_windows:
        raise ValueError("fine-tuning budget differs from frozen selection")
    expected_rate = FROZEN_BASE_RATES[method] * float(
        chosen["learning_rate_multiplier"]
    )
    if not np.isclose(learning_rate, expected_rate, rtol=1e-12, atol=0.0):
        raise ValueError("learning rate differs from frozen selection")
    return file_sha256(manifest)
