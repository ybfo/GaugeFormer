"""Immutable evaluation views for schema, gauge, and cadence shifts."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import WindowStore, _window_starts, block_ids_for_window_starts


def deterministic_seed(*parts: object) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def training_location_scale(root: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(root / "values.npy", mmap_mode="r")
    split_codes = np.load(root / "split_codes.npy", mmap_mode="r")
    training = np.asarray(values[np.asarray(split_codes) == 0], dtype=np.float64)
    return training.mean(axis=0), np.maximum(training.std(axis=0), 1e-8)


class PerturbedStore:
    """A channel/gauge view that never mutates its frozen backing store."""

    def __init__(
        self,
        base: WindowStore,
        source_indices: Sequence[int] | None = None,
        query_source_indices: Sequence[int] | None = None,
        declared_scale: np.ndarray | None = None,
        declared_offset: np.ndarray | None = None,
        inserted_positions: Sequence[int] = (),
        perturbation_seed: int = 0,
        canonical_baseline_preprocessing: bool = False,
    ) -> None:
        self.base = base
        original_channels = len(base.metadata["channels"])
        self.source_indices = np.asarray(
            source_indices
            if source_indices is not None
            else np.arange(original_channels),
            dtype=np.int64,
        )
        if np.any(self.source_indices < 0) or np.any(
            self.source_indices >= original_channels
        ):
            raise ValueError("source channel index lies outside the backing store")
        queries = np.asarray(
            query_source_indices
            if query_source_indices is not None
            else base.query_indices,
            dtype=np.int64,
        )
        positions = []
        for query in queries:
            matches = np.flatnonzero(self.source_indices == query)
            if not len(matches):
                raise ValueError("every query channel must be retained")
            positions.append(int(matches[0]))
        self.query_source_indices = queries
        self.query_indices = np.asarray(positions, dtype=np.int64)
        self.inserted_positions = frozenset(int(item) for item in inserted_positions)
        if any(
            item < 0 or item >= len(self.source_indices)
            for item in self.inserted_positions
        ):
            raise ValueError("inserted position lies outside the view")
        if any(item in self.query_indices for item in self.inserted_positions):
            raise ValueError("an inserted sensor cannot be a query")
        self.perturbation_seed = int(perturbation_seed)
        self.root = base.root
        self.context_length = base.context_length
        self.horizon = base.horizon
        self.starts = base.starts
        self.metadata = copy.deepcopy(base.metadata)
        channels = []
        for position, source in enumerate(self.source_indices):
            channel = copy.deepcopy(base.metadata["channels"][int(source)])
            if position in self.inserted_positions:
                channel["name"] = f"inserted_{position}_{channel['name']}"
                channel["query"] = False
            channels.append(channel)
        self.metadata["channels"] = channels
        self.metadata["query_indices"] = self.query_indices.tolist()
        self.dimensions = np.asarray(base.dimensions)[self.source_indices].copy()
        original_scale = np.asarray(base.unit_scale)[self.source_indices].astype(
            np.float64
        )
        original_offset = np.asarray(base.unit_offset)[self.source_indices].astype(
            np.float64
        )
        requested_scale = (
            np.asarray(declared_scale, dtype=np.float64)
            if declared_scale is not None
            else original_scale.copy()
        )
        requested_offset = (
            np.asarray(declared_offset, dtype=np.float64)
            if declared_offset is not None
            else original_offset.copy()
        )
        if (
            requested_scale.shape != original_scale.shape
            or requested_offset.shape != original_offset.shape
        ):
            raise ValueError(
                "declared gauge must contain one scale and offset per view channel"
            )
        if np.any(requested_scale <= 0) or not np.isfinite(requested_scale).all():
            raise ValueError("declared gauge scales must be finite and positive")
        self.original_scale = original_scale.astype(np.float32)
        self.original_offset = original_offset.astype(np.float32)
        self.canonical_baseline_preprocessing = canonical_baseline_preprocessing
        if canonical_baseline_preprocessing:
            self.unit_scale = np.ones_like(requested_scale, dtype=np.float32)
            self.unit_offset = np.zeros_like(requested_offset, dtype=np.float32)
        else:
            self.unit_scale = requested_scale.astype(np.float32)
            self.unit_offset = requested_offset.astype(np.float32)
        self._identity_coordinate = bool(
            np.array_equal(self.unit_scale, self.original_scale)
            and np.array_equal(self.unit_offset, self.original_offset)
        )
        interval = float(self.metadata["sampling_interval_seconds"])
        self.sampling_interval = np.full(len(channels), interval, dtype=np.float32)
        center, spread = training_location_scale(base.root)
        center = center[self.source_indices]
        spread = spread[self.source_indices]
        if canonical_baseline_preprocessing:
            self.normalization_center = center * original_scale + original_offset
            self.normalization_spread = spread * np.abs(original_scale)
        else:
            # A direct unit-shift stress deliberately leaves the baseline's
            # fitted numerical normalization in its training gauge.
            self.normalization_center = center
            self.normalization_spread = spread

    def __len__(self) -> int:
        return len(self.base)

    def _to_view_coordinate(self, values: np.ndarray) -> np.ndarray:
        if self._identity_coordinate:
            return values.astype(np.float32, copy=True)
        canonical = (
            values.astype(np.float64) * self.original_scale[:, None]
            + self.original_offset[:, None]
        )
        return (
            (canonical - self.unit_offset[:, None]) / self.unit_scale[:, None]
        ).astype(np.float32)

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        context, future = self.base.get(index)
        context = self._to_view_coordinate(np.asarray(context)[self.source_indices])
        future = self._to_view_coordinate(np.asarray(future)[self.source_indices])
        for position in self.inserted_positions:
            context_rng = np.random.default_rng(
                deterministic_seed(self.perturbation_seed, index, position, "context")
            )
            future_rng = np.random.default_rng(
                deterministic_seed(self.perturbation_seed, index, position, "future")
            )
            context[position] = context[
                position, context_rng.permutation(context.shape[1])
            ]
            future[position] = future[position, future_rng.permutation(future.shape[1])]
        return context, future

    def original_query_future(self, index: int) -> np.ndarray:
        _, future = self.base.get(index)
        return np.asarray(future)[self.query_source_indices]

    def prediction_to_original(self, prediction: torch.Tensor) -> torch.Tensor:
        """Map a baseline output [B,H,Q] from view gauge to original gauge."""

        positions = self.query_indices
        source = self.query_source_indices
        view_scale = torch.as_tensor(
            self.unit_scale[positions], device=prediction.device, dtype=prediction.dtype
        ).view(1, 1, -1)
        view_offset = torch.as_tensor(
            self.unit_offset[positions],
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, 1, -1)
        original_scale = torch.as_tensor(
            np.asarray(self.base.unit_scale)[source],
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, 1, -1)
        original_offset = torch.as_tensor(
            np.asarray(self.base.unit_offset)[source],
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, 1, -1)
        canonical = prediction * view_scale + view_offset
        return (canonical - original_offset) / original_scale

    @property
    def output_scale(self) -> np.ndarray:
        return np.asarray(self.base.unit_scale)[self.query_source_indices]

    @property
    def output_offset(self) -> np.ndarray:
        return np.asarray(self.base.unit_offset)[self.query_source_indices]

    def block_ids(
        self, indices: np.ndarray, maximum_block_seconds: float = 86400.0
    ) -> np.ndarray:
        return self.base.block_ids(indices, maximum_block_seconds)
