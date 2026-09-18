"""Protocol guards separating development work from confirmatory inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONFIRMATORY_PROTOCOL = "gf-cs-2026-08-13-v6-development"
V7_CONFIRMATORY_PROTOCOL = "gf-cs-2026-08-20-v7"
STRICT_CONFIRMATORY_PROTOCOLS = {CONFIRMATORY_PROTOCOL, V7_CONFIRMATORY_PROTOCOL}
EXPECTED_CONFIRMATORY_CHECKPOINTS = 525
EXPECTED_CONFIRMATORY_RESULTS = 516
EXPECTED_V7_DEVELOPMENT_ARTIFACTS = 1723


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_checkpoint_protocol(checkpoint: dict) -> str:
    """Read the protocol identity from a joint or adapted checkpoint."""

    candidates = [checkpoint.get("protocol_version")]
    for field in ("train_config", "config"):
        nested = checkpoint.get(field)
        if isinstance(nested, dict):
            candidates.append(nested.get("protocol_version"))
    versions = {item for item in candidates if isinstance(item, str) and item}
    if len(versions) != 1:
        raise ValueError("checkpoint has no unique recorded protocol version")
    return versions.pop()
