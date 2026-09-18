"""Immutable sample-level prediction artifacts for confirmatory evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .metrics import BlockSDNMAEAccumulator, ChannelMetricAccumulator


FORMAT = "gaugeformer-sample-predictions-npz-v1"


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise FloatingPointError("sample artifact contains a nonfinite tensor")
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass
class _Record:
    metadata: dict[str, Any]
    channel_names: tuple[str, ...]
    predictions: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)
    window_indices: list[np.ndarray] = field(default_factory=list)
    block_ids: list[np.ndarray] = field(default_factory=list)


class SampleArtifactCollector:
    """Collect ordered prediction batches and write one non-pickle NPZ bundle."""

    def __init__(
        self,
        protocol_version: str,
        split: str,
        method: str,
        artifact_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.protocol_version = str(protocol_version)
        self.split = str(split)
        self.method = str(method)
        self.artifact_metadata = json.loads(
            json.dumps(
                dict(artifact_metadata or {}), sort_keys=True, separators=(",", ":")
            )
        )
        self._records: dict[str, _Record] = {}

    def update(
        self,
        record_id: str,
        prediction: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
        window_indices: Sequence[int] | np.ndarray,
        block_ids: Sequence[object] | np.ndarray,
        channel_names: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> None:
        predicted = _array(prediction)
        observed = _array(target)
        if predicted.shape != observed.shape or predicted.ndim != 3:
            raise ValueError("sample predictions must match [window,horizon,channel] targets")
        indices = np.asarray(window_indices, dtype=np.int64)
        blocks = np.asarray([str(item) for item in block_ids], dtype=np.str_)
        names = tuple(str(item) for item in channel_names)
        if (
            indices.shape != (predicted.shape[0],)
            or blocks.shape != (predicted.shape[0],)
            or len(names) != predicted.shape[2]
            or len(set(map(int, indices))) != len(indices)
        ):
            raise ValueError("sample identifiers or channels do not match the prediction tensor")
        normalized_metadata = json.loads(
            json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
        )
        record = self._records.get(record_id)
        if record is None:
            record = _Record(normalized_metadata, names)
            self._records[record_id] = record
        elif record.metadata != normalized_metadata or record.channel_names != names:
            raise ValueError(f"sample metadata changed within a record: {record_id}")
        record.predictions.append(predicted)
        record.targets.append(observed)
        record.window_indices.append(indices)
        record.block_ids.append(blocks)

    def save(self, path: Path) -> dict[str, Any]:
        if not self._records:
            raise ValueError("cannot write an empty sample artifact")
        arrays: dict[str, np.ndarray] = {}
        manifest_records: list[dict[str, Any]] = []
        total_windows = 0
        for ordinal, record_id in enumerate(sorted(self._records)):
            record = self._records[record_id]
            prefix = f"r{ordinal:03d}"
            prediction = np.concatenate(record.predictions, axis=0)
            target = np.concatenate(record.targets, axis=0)
            indices = np.concatenate(record.window_indices)
            blocks = np.concatenate(record.block_ids)
            if len(set(map(int, indices))) != len(indices):
                raise ValueError(f"sample window indices are not unique: {record_id}")
            arrays[f"{prefix}_prediction"] = prediction
            arrays[f"{prefix}_target"] = target
            arrays[f"{prefix}_window_index"] = indices
            arrays[f"{prefix}_block_id"] = blocks
            arrays[f"{prefix}_channel_name"] = np.asarray(
                record.channel_names, dtype=np.str_
            )
            manifest_records.append(
                {
                    "record_id": record_id,
                    "prefix": prefix,
                    "windows": int(prediction.shape[0]),
                    "horizon": int(prediction.shape[1]),
                    "channels": int(prediction.shape[2]),
                    "metadata": record.metadata,
                }
            )
            total_windows += int(prediction.shape[0])
        manifest = {
            "status": "complete",
            "format": FORMAT,
            "protocol_version": self.protocol_version,
            "split": self.split,
            "method": self.method,
            "metadata": self.artifact_metadata,
            "record_count": len(manifest_records),
            "records": manifest_records,
        }
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        arrays["manifest_utf8"] = np.frombuffer(manifest_bytes, dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        return {
            "filename": path.name,
            "format": FORMAT,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "record_count": len(manifest_records),
            "window_rows": total_windows,
        }


def load(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Load and structurally validate a sample artifact without pickle objects."""

    with np.load(path, allow_pickle=False) as archive:
        if "manifest_utf8" not in archive.files:
            raise ValueError("sample artifact manifest is absent")
        manifest = json.loads(archive["manifest_utf8"].tobytes().decode("utf-8"))
        manifest_records = manifest.get("records") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest, dict)
            or
            manifest.get("status") != "complete"
            or manifest.get("format") != FORMAT
            or not isinstance(manifest.get("metadata"), dict)
            or not isinstance(manifest_records, list)
            or manifest.get("record_count") != len(manifest_records)
            or not manifest_records
        ):
            raise ValueError("sample artifact manifest differs")
        expected = {"manifest_utf8"}
        records: dict[str, dict[str, np.ndarray]] = {}
        prefixes: set[str] = set()
        for item in manifest_records:
            if not isinstance(item, dict):
                raise ValueError("sample record manifest differs")
            record_id = item.get("record_id")
            prefix = item.get("prefix")
            dimensions = tuple(item.get(key) for key in ("windows", "horizon", "channels"))
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in records
                or not isinstance(prefix, str)
                or not prefix
                or prefix in prefixes
                or not all(isinstance(value, int) and value > 0 for value in dimensions)
                or not isinstance(item.get("metadata"), dict)
            ):
                raise ValueError("sample record manifest differs")
            prefixes.add(prefix)
            keys = {
                name: f"{prefix}_{name}"
                for name in (
                    "prediction",
                    "target",
                    "window_index",
                    "block_id",
                    "channel_name",
                )
            }
            expected.update(keys.values())
            if not all(key in archive.files for key in keys.values()):
                raise ValueError(f"sample arrays are absent: {prefix}")
            record = {name: np.asarray(archive[key]).copy() for name, key in keys.items()}
            prediction = record["prediction"]
            target = record["target"]
            window_index = record["window_index"]
            block_id = record["block_id"]
            channel_name = record["channel_name"]
            if (
                prediction.dtype != np.float32
                or target.dtype != np.float32
                or prediction.shape != target.shape
                or prediction.ndim != 3
                or window_index.dtype != np.int64
                or window_index.shape != (prediction.shape[0],)
                or len(np.unique(window_index)) != len(window_index)
                or block_id.dtype.kind != "U"
                or block_id.shape != (prediction.shape[0],)
                or channel_name.dtype.kind != "U"
                or channel_name.shape != (prediction.shape[2],)
                or any(not str(value) for value in block_id)
                or any(not str(value) for value in channel_name)
                or len(np.unique(channel_name)) != len(channel_name)
                or tuple(prediction.shape)
                != dimensions
                or not np.isfinite(prediction).all()
                or not np.isfinite(target).all()
            ):
                raise ValueError(f"sample tensor structure differs: {prefix}")
            records[record_id] = record
        if set(archive.files) != expected or len(records) != len(manifest_records):
            raise ValueError("sample artifact contains missing, duplicate, or unexpected arrays")
    return manifest, records


def recompute_metric(
    record: Mapping[str, np.ndarray],
    system_id: str,
    training_standard_deviation: Sequence[float],
    batch_size: int,
) -> dict[str, Any]:
    """Recompute the exact streaming metric schema from stored samples."""

    prediction = np.asarray(record["prediction"], dtype=np.float32)
    target = np.asarray(record["target"], dtype=np.float32)
    blocks = np.asarray(record["block_id"]).astype(str)
    names = [str(item) for item in np.asarray(record["channel_name"]).tolist()]
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    metric = ChannelMetricAccumulator(names, np.asarray(training_standard_deviation))
    block = BlockSDNMAEAccumulator(names, np.asarray(training_standard_deviation))
    for offset in range(0, len(prediction), batch_size):
        stop = offset + batch_size
        predicted = torch.from_numpy(prediction[offset:stop])
        observed = torch.from_numpy(target[offset:stop])
        metric.update(predicted, observed)
        block.update(predicted, observed, blocks[offset:stop])
    result = metric.compute(system_id, len(prediction))
    result["statistical_blocks"] = block.compute()
    return result
