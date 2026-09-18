"""Streaming, channel-balanced metrics shared by every paper model.

The accumulator intentionally keeps only sufficient statistics.  It can be
used on the complete public partitions without retaining forecasts in memory,
while preserving the frozen aggregation order (time within channel, then
channels within system).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import torch


@dataclass
class ChannelMetricAccumulator:
    channel_names: list[str]
    training_standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        channels = len(self.channel_names)
        scale = np.asarray(self.training_standard_deviation, dtype=np.float64)
        if scale.shape != (channels,):
            raise ValueError("training scale must contain one value per channel")
        self.training_standard_deviation = np.maximum(scale, 1e-8)
        self.absolute_sum = np.zeros(channels, dtype=np.float64)
        self.squared_sum = np.zeros(channels, dtype=np.float64)
        self.target_sum = np.zeros(channels, dtype=np.float64)
        self.target_squared_sum = np.zeros(channels, dtype=np.float64)
        self.element_count = np.zeros(channels, dtype=np.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate tensors shaped ``[batch, horizon, channels]``."""

        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction and target must be matching [B,H,C] tensors")
        if prediction.shape[-1] != len(self.channel_names):
            raise ValueError("channel count does not match accumulator")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("prediction contains NaN or infinity")
        if not torch.isfinite(target).all():
            raise FloatingPointError("target contains NaN or infinity")
        error = (prediction - target).detach().double().cpu().numpy()
        observed = target.detach().double().cpu().numpy()
        self.absolute_sum += np.abs(error).sum(axis=(0, 1))
        self.squared_sum += np.square(error).sum(axis=(0, 1))
        self.target_sum += observed.sum(axis=(0, 1))
        self.target_squared_sum += np.square(observed).sum(axis=(0, 1))
        self.element_count += prediction.shape[0] * prediction.shape[1]

    def compute(self, system_id: str, windows: int) -> dict:
        if np.any(self.element_count == 0):
            raise ValueError("cannot compute metrics before every channel is observed")
        mae = self.absolute_sum / self.element_count
        rmse = np.sqrt(self.squared_sum / self.element_count)
        target_mean = self.target_sum / self.element_count
        sst = self.target_squared_sum - self.element_count * np.square(target_mean)
        r2 = 1.0 - self.squared_sum / np.maximum(sst, 1e-12)
        sd_nmae = mae / self.training_standard_deviation
        sd_nrmse = rmse / self.training_standard_deviation
        return {
            "system_id": system_id,
            "windows": int(windows),
            "channel_names": self.channel_names,
            "mae_by_channel": mae.tolist(),
            "rmse_by_channel": rmse.tolist(),
            "r2_by_channel": r2.tolist(),
            "training_standard_deviation": self.training_standard_deviation.tolist(),
            "sd_nmae_by_channel": sd_nmae.tolist(),
            "sd_nrmse_by_channel": sd_nrmse.tolist(),
            "sd_nmae": float(sd_nmae.mean()),
            "sd_nrmse": float(sd_nrmse.mean()),
            "mean_r2": float(r2.mean()),
        }


@dataclass
class BlockSDNMAEAccumulator:
    """Accumulate channel-balanced SD-NMAE for paired physical blocks."""

    channel_names: list[str]
    training_standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        scale = np.asarray(self.training_standard_deviation, dtype=np.float64)
        if scale.shape != (len(self.channel_names),):
            raise ValueError("training scale must contain one value per channel")
        self.training_standard_deviation = np.maximum(scale, 1e-8)
        self._absolute_sum: dict[Hashable, np.ndarray] = {}
        self._count: dict[Hashable, int] = {}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        block_ids: list[Hashable] | np.ndarray,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction and target must be matching [B,H,C] tensors")
        if prediction.shape[-1] != len(self.channel_names):
            raise ValueError("channel count does not match accumulator")
        if len(block_ids) != prediction.shape[0]:
            raise ValueError("one block id is required for every batch element")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise FloatingPointError("prediction or target contains NaN or infinity")
        absolute = (prediction - target).detach().abs().double().cpu().numpy()
        for row, block_id in enumerate(block_ids):
            row_sum = absolute[row].sum(axis=0)
            if block_id not in self._absolute_sum:
                self._absolute_sum[block_id] = np.zeros_like(row_sum)
                self._count[block_id] = 0
            self._absolute_sum[block_id] += row_sum
            self._count[block_id] += prediction.shape[1]

    def compute(self) -> list[dict[str, object]]:
        records = []
        for block_id in sorted(self._absolute_sum, key=str):
            channel_mae = self._absolute_sum[block_id] / self._count[block_id]
            records.append(
                {
                    "block_id": str(block_id),
                    "sd_nmae": float(
                        np.mean(channel_mae / self.training_standard_deviation)
                    ),
                    "elements_per_channel": int(self._count[block_id]),
                }
            )
        return records
