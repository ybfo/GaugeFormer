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


def require_confirmatory_authorization(
    split: str,
    manifest_path: Path | None,
    protocol_version: str,
) -> dict | None:
    """Require a frozen, self-consistent authorization record for test access."""

    if split != "test":
        if manifest_path is not None:
            raise ValueError("confirmatory authorization is not accepted for development")
        return None
    if manifest_path is None:
        raise PermissionError(
            "test access requires --confirmatory-manifest after every development "
            "choice and checkpoint hash has been frozen"
        )
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != protocol_version:
        raise PermissionError("confirmatory manifest protocol version differs")
    if payload.get("status") != "frozen_before_first_test_inference":
        raise PermissionError("confirmatory manifest is not in the frozen state")
    if payload.get("test_inference_authorized") is not True:
        raise PermissionError("confirmatory manifest does not authorize test inference")
    for field in (
        "frozen_at_utc",
        "development_artifact_index_sha256",
        "checkpoint_index_sha256",
        "source_bundle_sha256",
        "processed_data_index_sha256",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise PermissionError(f"confirmatory manifest is missing {field}")
    if protocol_version in STRICT_CONFIRMATORY_PROTOCOLS:
        runtime_identity = payload.get("official_runtime_identity_sha256")
        if not isinstance(runtime_identity, str) or len(runtime_identity) != 64:
            raise PermissionError(
                "confirmatory manifest is missing official runtime identity"
            )
        if payload.get("checkpoint_count") != EXPECTED_CONFIRMATORY_CHECKPOINTS:
            raise PermissionError("confirmatory checkpoint count differs")
        if (
            payload.get("expected_confirmatory_result_count")
            != EXPECTED_CONFIRMATORY_RESULTS
        ):
            raise PermissionError("confirmatory result count differs")
        if payload.get("test_inference_history_at_freeze") != []:
            raise PermissionError(
                "confirmatory manifest was not frozen before the first test inference"
            )
        checkpoint_index = payload.get("checkpoint_index")
        development_index = payload.get("development_artifact_index")
        if not isinstance(checkpoint_index, list) or len(checkpoint_index) != payload.get(
            "checkpoint_count"
        ):
            raise PermissionError("confirmatory checkpoint index is incomplete")
        if not isinstance(development_index, list) or len(
            development_index
        ) != payload.get("development_artifact_count"):
            raise PermissionError("confirmatory development index is incomplete")
        if protocol_version == V7_CONFIRMATORY_PROTOCOL:
            if payload.get("development_artifact_count") != EXPECTED_V7_DEVELOPMENT_ARTIFACTS:
                raise PermissionError("v7 development artifact count differs")
            for field in (
                "statistical_analysis_protocol_sha256",
                "comparison_selection_sha256",
                "confirmatory_source_freeze_sha256",
            ):
                value = payload.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise PermissionError(f"v7 confirmatory manifest is missing {field}")
    payload["authorization_manifest"] = str(path)
    payload["authorization_manifest_sha256"] = file_sha256(path)
    return payload


def require_authorized_checkpoint(
    authorization: dict | None, checkpoint_path: Path
) -> dict | None:
    """Bind test inference to exactly one checkpoint in the frozen index."""

    if authorization is None:
        return None
    path = Path(checkpoint_path)
    actual_sha256 = file_sha256(path)
    actual_bytes = path.stat().st_size
    matches = [
        record
        for record in authorization.get("checkpoint_index", [])
        if record.get("sha256") == actual_sha256
        and int(record.get("bytes", -1)) == actual_bytes
    ]
    if len(matches) != 1:
        raise PermissionError(
            "test checkpoint is not uniquely present in the frozen authorization index"
        )
    return matches[0]
