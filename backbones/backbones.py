"""Train paper-linked official baseline architectures on the frozen corpus.

This file is deliberately an experiment wrapper, not a reimplementation: each
core model class is imported from the archived official source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gaugeformer.data import QuerySubsetStore, WindowStore, load_query_partition
from gaugeformer.metrics import BlockSDNMAEAccumulator, ChannelMetricAccumulator
from gaugeformer.training import (
    atomic_json,
    environment_record,
    training_standard_deviation,
)


from experiments.common import PROJECT

SYSTEMS = ("building_energy", "hydraulic", "metropt", "pmsm", "steel_industry")


@dataclass(frozen=True)
class BaselineTrainConfig:
    protocol_version: str = "gf-cs-2026-08-12-v5"
    seed: int = 17
    batch_size: int = 16
    max_windows: int = 100_000
    validation_every_windows: int = 10_000
    patience_evaluations: int = 6
    learning_rate: float = 1e-4
    amp: bool = False
    amp_dtype: str = "bfloat16"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_official(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sampled_indices(length: int, count: int | None) -> np.ndarray:
    if count is None or count >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=count, dtype=np.int64)


def draw_window_batch(
    store: WindowStore,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, futures = zip(*(store.get(int(index)) for index in indices))
    context = torch.from_numpy(np.stack(contexts).copy()).transpose(1, 2).to(device)
    future = torch.from_numpy(np.stack(futures).copy()).transpose(1, 2).to(device)
    return context, future


def query_prediction(
    prediction: torch.Tensor,
    future: torch.Tensor,
    query_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.as_tensor(query_indices, device=prediction.device, dtype=torch.long)
    return prediction.index_select(-1, query), future.index_select(-1, query)


@lru_cache(maxsize=None)
def training_location_scale(root: str) -> tuple[np.ndarray, np.ndarray]:
    store_root = Path(root)
    values = np.load(store_root / "values.npy", mmap_mode="r")
    split = np.load(store_root / "split_codes.npy", mmap_mode="r")
    training = np.asarray(values[np.asarray(split) == 0], dtype=np.float64)
    center = training.mean(axis=0)
    spread = np.maximum(training.std(axis=0), 1e-8)
    return center.astype(np.float32), spread.astype(np.float32)


def standardize_context(
    context: torch.Tensor, store: WindowStore
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(store, "normalization_center") and hasattr(
        store, "normalization_spread"
    ):
        center_np = np.asarray(store.normalization_center, dtype=np.float32)
        spread_np = np.asarray(store.normalization_spread, dtype=np.float32)
    else:
        center_np, spread_np = training_location_scale(str(store.root.resolve()))
    center = torch.from_numpy(center_np).to(context.device).view(1, 1, -1)
    spread = torch.from_numpy(spread_np).to(context.device).view(1, 1, -1)
    return (context - center) / spread, center, spread


def standardized_future(
    future: torch.Tensor, center: torch.Tensor, spread: torch.Tensor
) -> torch.Tensor:
    return (future - center) / spread


class OfficialWrapper(nn.Module):
    name: str

    def predict(self, context: torch.Tensor, store: WindowStore) -> torch.Tensor:
        raise NotImplementedError

    def training_loss(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        store: WindowStore,
        generator: np.random.Generator,
    ) -> torch.Tensor:
        raise NotImplementedError

    def predict_queries(
        self, context: torch.Tensor, store: WindowStore
    ) -> torch.Tensor:
        prediction = self.predict(context, store)
        query, _ = query_prediction(prediction, prediction, store.query_indices)
        return query

    def configure_training(
        self, config: BaselineTrainConfig, total_steps: int
    ) -> tuple[
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
        float,
        dict,
    ]:
        raise NotImplementedError

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return self.state_dict()

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=True)


class TimerXLWrapper(OfficialWrapper):
    name = "Timer-XL"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        source = PROJECT / "third_party" / "openltm"
        sys.path.insert(0, str(source))
        module = import_official("official_timer_xl", source / "models" / "timer_xl.py")
        config = SimpleNamespace(
            input_token_len=96,
            output_token_len=96,
            d_model=1024,
            n_heads=8,
            e_layers=8,
            d_ff=2048,
            dropout=0.1,
            activation="relu",
            covariate=False,
            flash_attention=False,
            output_attention=False,
            use_norm=True,
        )
        self.model = module.Model(config)
        if pretrained:
            checkpoint_path = (
                PROJECT / "checkpoints" / "official" / "timer_xl" / "checkpoint.pth"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing official Timer-XL checkpoint: {checkpoint_path}"
                )
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)

    def predict_standardized(
        self, scaled: torch.Tensor, store: WindowStore
    ) -> torch.Tensor:
        empty = scaled.new_zeros(scaled.shape[0], scaled.shape[1], 1)
        output = self.model(scaled, empty, empty)
        next_token = output[:, -96:, :]
        return next_token[:, : store.horizon, :]

    def predict(self, context: torch.Tensor, store: WindowStore) -> torch.Tensor:
        scaled, center, spread = standardize_context(context, store)
        return self.predict_standardized(scaled, store) * spread + center

    def training_loss(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        store: WindowStore,
        generator: np.random.Generator,
    ) -> torch.Tensor:
        scaled, center, spread = standardize_context(context, store)
        future_scaled = standardized_future(future, center, spread)
        prediction, target = query_prediction(
            self.predict_standardized(scaled, store),
            future_scaled,
            store.query_indices,
        )
        return F.mse_loss(prediction, target)

    def configure_training(
        self, config: BaselineTrainConfig, total_steps: int
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict
    ]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=config.learning_rate, weight_decay=0.0
        )
        steps_per_epoch = max(1, math.ceil(total_steps / 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-8
        )
        return (
            optimizer,
            scheduler,
            math.inf,
            {
                "optimizer": "Adam",
                "objective": "MSE on training-partition standardized values",
                "learning_rate": config.learning_rate,
                "weight_decay": 0.0,
                "scheduler": "CosineAnnealingLR stepped at each of ten mapped epochs",
                "eta_min": 1e-8,
                "scheduler_interval_steps": steps_per_epoch,
                "gradient_clip": None,
            },
        )


class GTMWrapper(OfficialWrapper):
    name = "GTM"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        source = PROJECT / "third_party" / "gtm"
        sys.path.insert(0, str(source))
        module = import_official("official_gtm", source / "models" / "GTM.py")
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=96,
            pred_len=24,
            patch_len=96,
            stride=96,
            d_model=768,
            n_heads=8,
            d_layers=12,
            d_ff=32,
            dropout=0.1,
            factor=3,
            activation="gelu",
            enc_in=1,
            num_gran=5,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model = module.Model(config)
        if pretrained:
            checkpoint_path = (
                PROJECT
                / "checkpoints"
                / "official"
                / "gtm"
                / "pre_train_checkpoint.pth"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing official GTM pre-training checkpoint: {checkpoint_path}"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint = checkpoint["model"]
            state = {
                key.removeprefix("module."): value for key, value in checkpoint.items()
            }
            self.model.load_state_dict(state, strict=True)

    @staticmethod
    def granularity(
        seconds: float, batch: int, device: torch.device
    ) -> list[torch.Tensor]:
        units = [0.001, 1.0, 60.0, 3600.0, 86400.0]
        vector = [0.0] * 5
        valid = [
            (abs(seconds / unit - round(seconds / unit)), i, round(seconds / unit))
            for i, unit in enumerate(units)
        ]
        _, index, value = min(valid, key=lambda item: (item[0], abs(item[2])))
        vector[index] = float(value)
        return [torch.full((batch,), item, device=device) for item in vector]

    def predict_standardized(
        self, scaled: torch.Tensor, store: WindowStore
    ) -> torch.Tensor:
        batch, _, channels = scaled.shape
        flat = scaled.transpose(1, 2).reshape(batch * channels, 96, 1)
        time_gra = self.granularity(
            store.metadata["sampling_interval_seconds"],
            batch * channels,
            scaled.device,
        )
        output = self.model(flat, time_gra)
        return output.reshape(batch, channels, store.horizon).transpose(1, 2)

    def predict(self, context: torch.Tensor, store: WindowStore) -> torch.Tensor:
        scaled, center, spread = standardize_context(context, store)
        return self.predict_standardized(scaled, store) * spread + center

    def training_loss(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        store: WindowStore,
        generator: np.random.Generator,
    ) -> torch.Tensor:
        scaled, center, spread = standardize_context(context, store)
        future_scaled = standardized_future(future, center, spread)
        prediction, target = query_prediction(
            self.predict_standardized(scaled, store),
            future_scaled,
            store.query_indices,
        )
        return F.mse_loss(prediction, target)

    def configure_training(
        self, config: BaselineTrainConfig, total_steps: int
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict
    ]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        steps_per_epoch = max(1, math.ceil(total_steps / 30))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=0.0
        )
        return (
            optimizer,
            scheduler,
            math.inf,
            {
                "optimizer": "Adam",
                "objective": "MSE on training-partition standardized values",
                "learning_rate": config.learning_rate,
                "weight_decay": 0.0,
                "scheduler": "repository cosine schedule stepped at each of thirty mapped epochs",
                "scheduler_interval_steps": steps_per_epoch,
                "gradient_clip": None,
            },
        )


class UniTimeWrapper(OfficialWrapper):
    name = "UniTime"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        source = PROJECT / "third_party" / "unitime"
        sys.path.insert(0, str(source))
        module = import_official("official_unitime", source / "models" / "unitime.py")
        model_path = PROJECT / "checkpoints" / "official" / "gpt2_small"
        if pretrained and not (model_path / "config.json").exists():
            raise FileNotFoundError(f"Missing official GPT-2 backbone: {model_path}")
        if not pretrained:
            raise ValueError(
                "UniTime smoke and formal runs require its published GPT-2 initialization"
            )
        args = SimpleNamespace(
            mask_rate=0.5,
            patch_len=16,
            max_token_num=128,
            max_backcast_len=96,
            max_forecast_len=24,
            logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
            model_path=str(model_path),
            lm_layer_num=6,
            lm_ft_type="full",
            dec_trans_layer_num=2,
            ts_embed_dropout=0.3,
            dec_head_dropout=0.1,
        )
        self.model = module.UniTime(args)

    @staticmethod
    def description(store: WindowStore) -> str:
        quantities = sorted({item["quantity"] for item in store.metadata["channels"]})
        return (
            "multivariate physical sensor measurements sampled every "
            f"{store.metadata['sampling_interval_seconds']:g} seconds; measured "
            f"quantities include {', '.join(quantities)}."
        )

    def forward_standardized(
        self, scaled: torch.Tensor, mask: torch.Tensor, store: WindowStore
    ) -> torch.Tensor:
        return self.model(
            ["engineering", 96, 16, self.description(store)],
            scaled,
            mask,
        )

    def predict(self, context: torch.Tensor, store: WindowStore) -> torch.Tensor:
        scaled, center, spread = standardize_context(context, store)
        mask = torch.ones_like(scaled)
        output = self.forward_standardized(scaled.clone(), mask, store)
        return output[:, 96:120, :] * spread + center

    def training_loss(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        store: WindowStore,
        generator: np.random.Generator,
    ) -> torch.Tensor:
        scaled, global_center, global_spread = standardize_context(context, store)
        keep = torch.rand_like(scaled) >= 0.5
        masked = scaled.masked_fill(~keep, 0.0)
        output = self.forward_standardized(
            masked.clone(),
            keep.to(context.dtype),
            store,
        )
        query = torch.as_tensor(
            store.query_indices, device=context.device, dtype=torch.long
        )
        future_scaled = (future - global_center) / global_spread
        reconstruction_error = (output[:, :96, :] - scaled).square()
        forecast_error = (
            output[:, 96:120, :].index_select(-1, query)
            - future_scaled.index_select(-1, query)
        ).square()
        return (reconstruction_error.sum() + forecast_error.sum()) / (
            reconstruction_error.numel() + forecast_error.numel()
        )

    def configure_training(
        self, config: BaselineTrainConfig, total_steps: int
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict
    ]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=config.learning_rate, weight_decay=0.0
        )
        steps_per_epoch = max(1, math.ceil(total_steps / 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-6
        )
        return (
            optimizer,
            scheduler,
            5.0,
            {
                "optimizer": "AdamW",
                "objective": "official masked reconstruction-and-forecast MSE in training-partition standardized values; future terms restricted to query channels",
                "learning_rate": config.learning_rate,
                "weight_decay": 0.0,
                "scheduler": "CosineAnnealingLR mapped from the released T_max=20 over 10 training epochs",
                "scheduler_interval_steps": steps_per_epoch,
                "eta_min": 1e-6,
                "gradient_clip": 5.0,
                "max_token_num": 128,
            },
        )


class MoiraiWrapper(OfficialWrapper):
    name = "Moirai"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if not pretrained:
            raise ValueError("Moirai requires the official pretrained initialization")
        source = PROJECT / "third_party" / "moirai" / "src"
        sys.path.insert(0, str(source))
        from uni2ts.loss.packed import PackedNLLLoss
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        model_path = PROJECT / "checkpoints" / "official" / "moirai_1.1_R_small"
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(f"Missing official Moirai checkpoint: {model_path}")
        module = MoiraiModule.from_pretrained(str(model_path))
        self.forecast = MoiraiForecast(
            module=module,
            prediction_length=24,
            context_length=96,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.training_loss_func = PackedNLLLoss()

    @staticmethod
    def target_and_past_covariate_indices(
        store: WindowStore, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.as_tensor(store.query_indices, device=device, dtype=torch.long)
        target_set = set(store.query_indices.tolist())
        covariate = torch.as_tensor(
            [
                index
                for index in range(len(store.metadata["channels"]))
                if index not in target_set
            ],
            device=device,
            dtype=torch.long,
        )
        return target, covariate

    def predict_queries(
        self, context: torch.Tensor, store: WindowStore
    ) -> torch.Tensor:
        batch = context.shape[0]
        target_index, covariate_index = self.target_and_past_covariate_indices(
            store, context.device
        )
        target_context = context.index_select(-1, target_index)
        past_covariates = (
            context.index_select(-1, covariate_index)
            if covariate_index.numel()
            else None
        )
        with self.forecast.hparams_context(
            target_dim=target_context.shape[-1],
            past_feat_dynamic_real_dim=int(covariate_index.numel()),
        ):
            samples = self.forecast(
                past_target=target_context,
                past_observed_target=torch.ones_like(target_context, dtype=torch.bool),
                past_is_pad=torch.zeros(
                    batch,
                    context.shape[1],
                    device=context.device,
                    dtype=torch.bool,
                ),
                past_feat_dynamic_real=past_covariates,
                past_observed_feat_dynamic_real=(
                    torch.ones_like(past_covariates, dtype=torch.bool)
                    if past_covariates is not None
                    else None
                ),
                num_samples=100,
            )
        return samples.median(dim=1).values

    def predict(self, context: torch.Tensor, store: WindowStore) -> torch.Tensor:
        raise RuntimeError("Moirai uses predict_queries for arbitrary target subsets")

    def training_loss(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        store: WindowStore,
        generator: np.random.Generator,
    ) -> torch.Tensor:
        batch = context.shape[0]
        target_index, covariate_index = self.target_and_past_covariate_indices(
            store, context.device
        )
        target_context = context.index_select(-1, target_index)
        target_future = future.index_select(-1, target_index)
        past_covariates = (
            context.index_select(-1, covariate_index)
            if covariate_index.numel()
            else None
        )
        with self.forecast.hparams_context(
            target_dim=target_context.shape[-1],
            past_feat_dynamic_real_dim=int(covariate_index.numel()),
        ):
            converted = self.forecast._convert(
                16,
                past_target=target_context,
                past_observed_target=torch.ones_like(target_context, dtype=torch.bool),
                past_is_pad=torch.zeros(
                    batch, 96, device=context.device, dtype=torch.bool
                ),
                future_target=target_future,
                future_observed_target=torch.ones_like(target_future, dtype=torch.bool),
                future_is_pad=torch.zeros(
                    batch, 24, device=context.device, dtype=torch.bool
                ),
                past_feat_dynamic_real=past_covariates,
                past_observed_feat_dynamic_real=(
                    torch.ones_like(past_covariates, dtype=torch.bool)
                    if past_covariates is not None
                    else None
                ),
            )
        target, observed, sample_id, time_id, variate_id, prediction_mask = converted
        distribution = self.forecast.module(
            target,
            observed,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            torch.full_like(time_id, 16),
        )
        return self.training_loss_func(
            pred=distribution,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed,
            sample_id=sample_id,
            variate_id=variate_id,
        )

    def configure_training(
        self, config: BaselineTrainConfig, total_steps: int
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict
    ]:
        from uni2ts.model.moirai import MoiraiFinetune

        official = MoiraiFinetune(
            min_patches=2,
            min_mask_ratio=0.15,
            max_mask_ratio=0.5,
            max_dim=128,
            num_training_steps=total_steps,
            num_warmup_steps=0,
            module=self.forecast.module,
            num_samples=100,
            beta1=0.9,
            beta2=0.98,
            lr=config.learning_rate,
            weight_decay=0.1,
            context_length=96,
            prediction_length=24,
            patch_size=16,
            finetune_pattern="full",
        )
        bundle = official.configure_optimizers()
        optimizer = bundle["optimizer"]
        scheduler = bundle["lr_scheduler"]["scheduler"]
        return (
            optimizer,
            scheduler,
            1.0,
            {
                "optimizer": "official MoiraiFinetune AdamW parameter groups",
                "objective": "official packed negative log-likelihood on query-channel future masks",
                "learning_rate": config.learning_rate,
                "weight_decay": 0.1,
                "betas": [0.9, 0.98],
                "epsilon": 1e-6,
                "scheduler": "official constant schedule with zero warm-up",
                "scheduler_interval_steps": 1,
                "gradient_clip": 1.0,
                "patch_size": 16,
                "num_samples": 100,
            },
        )


class CPiRiWrapper(OfficialWrapper):
    name = "CPiRi"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if not pretrained:
            raise ValueError(
                "CPiRi requires the official frozen Sundial initialization"
            )
        source = PROJECT / "third_party" / "cpiri"
        checkpoint = (
            PROJECT
            / "checkpoints"
            / "official"
            / "sundial_base_128m"
            / "model.safetensors"
        )
        expected = source / "baselines" / "Sundial" / "ckpt" / "model.safetensors"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing official Sundial checkpoint: {checkpoint}"
            )
        expected.parent.mkdir(parents=True, exist_ok=True)
        if not expected.exists():
            expected.symlink_to(checkpoint)
        sys.path.insert(0, str(source))
        old_cwd = Path.cwd()
        try:
            os.chdir(source)
            module = import_official(
                "official_cpiri", source / "baselines" / "CPiRi" / "arch.py"
            )
            self.model = module.CPiRi(
                input_len=96,
                input_dim=1,
                output_len=24,
                spital_num_layers=4,
                revin=False,
                from_pretrain=True,
            )
        finally:
            os.chdir(old_cwd)

    def predict(self, context: torch.Tensor, store: WindowStore) -> torch.Tensor:
        scaled, center, spread = standardize_context(context, store)
        output = self.model(scaled.unsqueeze(-1), train=False).squeeze(-1)
        return output * spread + center

    def training_loss(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        store: WindowStore,
        generator: np.random.Generator,
    ) -> torch.Tensor:
        scaled, center, spread = standardize_context(context, store)
        future_scaled = (future - center) / spread
        channels = scaled.shape[-1]
        permutation = torch.as_tensor(
            generator.permutation(channels), device=context.device
        )
        shuffled = scaled.index_select(-1, permutation)
        inverse = torch.argsort(permutation)
        prediction = self.model(shuffled.unsqueeze(-1), train=True).squeeze(-1)
        prediction = prediction.index_select(-1, inverse)
        prediction, target = query_prediction(
            prediction, future_scaled, store.query_indices
        )
        return F.l1_loss(prediction, target)

    def configure_training(
        self, config: BaselineTrainConfig, total_steps: int
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict
    ]:
        optimizer = torch.optim.Adam(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=1e-5,
        )
        epoch_fractions = (1 / 60, 10 / 60, 25 / 60, 40 / 60)
        milestones = sorted(
            {
                max(1, min(total_steps - 1, round(total_steps * fraction)))
                for fraction in epoch_fractions
                if total_steps > 1
            }
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        )
        return (
            optimizer,
            scheduler,
            3.0,
            {
                "optimizer": "Adam",
                "objective": "official MAE through the differentiable Sundial sampler on per-channel training Z-scores",
                "learning_rate": config.learning_rate,
                "weight_decay": 1e-5,
                "scheduler": "MultiStepLR",
                "scheduler_interval_steps": 1,
                "milestones_steps": milestones,
                "milestones_source_epochs": [1, 10, 25, 40],
                "source_total_epochs": 60,
                "gamma": 0.5,
                "gradient_clip": 3.0,
            },
        )

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        trainable = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: value
            for name, value in self.state_dict().items()
            if name in trainable
        }

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected CPiRi checkpoint keys: {unexpected}")
        trainable = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if trainable - set(state):
            raise RuntimeError(
                f"Missing trainable CPiRi keys: {sorted(trainable - set(state))}"
            )


def build_model(name: str, pretrained: bool = True) -> OfficialWrapper:
    if name == "timer_xl":
        return TimerXLWrapper(pretrained=pretrained)
    if name == "gtm":
        return GTMWrapper(pretrained=pretrained)
    if name == "unitime":
        return UniTimeWrapper(pretrained=pretrained)
    if name == "moirai":
        return MoiraiWrapper(pretrained=pretrained)
    if name == "cpiri":
        return CPiRiWrapper(pretrained=pretrained)
    raise ValueError(f"Unsupported trainable baseline: {name}")


@torch.inference_mode()
def evaluate(
    model: OfficialWrapper,
    store: WindowStore,
    device: torch.device,
    batch_size: int,
    window_limit: int | None = None,
    protocol_version: str = "gf-cs-2026-08-12-v5",
) -> dict:
    model.eval()
    indices = sampled_indices(len(store), window_limit)
    query_indices = store.query_indices
    names = [store.metadata["channels"][int(i)]["name"] for i in query_indices]
    scale = training_standard_deviation(store)[query_indices]
    accumulator = ChannelMetricAccumulator(names, scale)
    block_accumulator = BlockSDNMAEAccumulator(names, scale)
    seed_material = (
        f"{protocol_version}:{model.name}:{store.metadata['system_id']}"
    ).encode("utf-8")
    evaluation_seed = int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8], "big"
    ) % (2**31)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(evaluation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(evaluation_seed)
        for offset in range(0, len(indices), batch_size):
            part = indices[offset : offset + batch_size]
            context, future = draw_window_batch(store, part, device)
            prediction = model.predict_queries(context, store)
            _, target = query_prediction(future, future, query_indices)
            accumulator.update(prediction, target)
            block_accumulator.update(
                prediction,
                target,
                store.block_ids(part),
            )
    result = accumulator.compute(store.metadata["system_id"], len(indices))
    result["query_indices"] = query_indices.tolist()
    result["probabilistic_evaluation_seed"] = evaluation_seed
    result["statistical_blocks"] = block_accumulator.compute()
    return result


def train(
    method: str,
    output: Path,
    config: BaselineTrainConfig,
    system_ids: tuple[str, ...],
    dev_limit: int | None,
    pretrained: bool,
    save_checkpoint: bool,
    full_development_after_selection: bool,
    query_indices_by_system: Mapping[str, Sequence[int]] | None = None,
    query_partition_manifest_sha256: str | None = None,
) -> dict:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if query_indices_by_system is not None:
        missing = sorted(set(system_ids) - set(query_indices_by_system))
        if missing:
            raise ValueError(f"missing supervised query indices for: {missing}")

    def make_store(name: str, split: str) -> WindowStore | QuerySubsetStore:
        base = WindowStore(PROJECT / "data" / "processed" / name, split)
        if query_indices_by_system is None:
            return base
        return QuerySubsetStore(base, query_indices_by_system[name])

    train_stores = {name: make_store(name, "train") for name in system_ids}
    dev_stores = {name: make_store(name, "dev") for name in system_ids}
    model = build_model(method, pretrained=pretrained).to(device)
    steps = math.ceil(config.max_windows / config.batch_size)
    optimizer, scheduler, gradient_clip, official_training = model.configure_training(
        config, steps
    )
    scheduler_interval = int(official_training.get("scheduler_interval_steps", 1))
    if scheduler_interval < 1:
        raise ValueError("scheduler_interval_steps must be positive")
    amp = bool(config.amp and device.type == "cuda")
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.amp_dtype]
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(amp and amp_dtype == torch.float16)
    )
    rng = np.random.default_rng(config.seed)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history = []
    best = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_windows = 0
    best_full_score = math.inf
    best_full_development: dict | None = None
    stale = 0
    windows = 0
    next_validation = min(config.validation_every_windows, config.max_windows)

    while windows < config.max_windows:
        model.train()
        system_id = system_ids[int(rng.integers(len(system_ids)))]
        store = train_stores[system_id]
        current = min(config.batch_size, config.max_windows - windows)
        index = rng.integers(0, len(store), size=current)
        context, future = draw_window_batch(store, index, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            loss = model.training_loss(context, future, store, rng)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {method} loss after {windows + current} windows: "
                f"{float(loss.detach())}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        completed_step = math.ceil((windows + current) / config.batch_size)
        if (
            completed_step % scheduler_interval == 0
            or windows + current >= config.max_windows
        ):
            scheduler.step()
        windows += current
        history.append(
            {
                "type": "train",
                "windows": windows,
                "system": system_id,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if windows < next_validation and windows < config.max_windows:
            continue
        validation = {
            name: evaluate(
                model,
                store,
                device,
                config.batch_size,
                dev_limit,
                config.protocol_version,
            )
            for name, store in dev_stores.items()
        }
        score = float(np.mean([item["sd_nmae"] for item in validation.values()]))
        history.append(
            {
                "type": "development",
                "windows": windows,
                "mean_sd_nmae": score,
                "systems": validation,
            }
        )
        if score < best:
            best = score
            stale = 0
            best_windows = windows
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.checkpoint_state().items()
            }
            best_full_development = validation if dev_limit is None else None
            best_full_score = score if dev_limit is None else math.inf
        else:
            stale += 1
        atomic_json(output / "history.json", {"events": history})
        if stale >= config.patience_evaluations:
            break
        next_validation += config.validation_every_windows

    if best_state is None:
        raise RuntimeError("Training completed without a finite development checkpoint")
    model.load_checkpoint_state(best_state)
    if best_full_development is None and full_development_after_selection:
        full_development = {
            name: evaluate(
                model, store, device, config.batch_size, None, config.protocol_version
            )
            for name, store in dev_stores.items()
        }
        full_development_score = float(
            np.mean([item["sd_nmae"] for item in full_development.values()])
        )
    elif best_full_development is not None:
        full_development = best_full_development
        full_development_score = best_full_score
    else:
        full_development = None
        full_development_score = None
    checkpoint_payload = {
        "model": best_state,
        "method": method,
        "config": asdict(config),
        "systems": system_ids,
        "supervised_query_indices": (
            None
            if query_indices_by_system is None
            else {
                name: list(map(int, query_indices_by_system[name]))
                for name in system_ids
            }
        ),
        "query_partition_manifest_sha256": query_partition_manifest_sha256,
        "windows_seen": best_windows,
        "selection_development_sd_nmae": best,
        "full_development_sd_nmae": full_development_score,
        "checkpoint_scope": "trainable-only" if method == "cpiri" else "full",
    }
    if save_checkpoint:
        torch.save(checkpoint_payload, output / "best.pt")
    result = {
        "status": "complete",
        "protocol_version": config.protocol_version,
        "method": model.name,
        "official_source": str((PROJECT / "third_party").resolve()),
        "train_config": asdict(config),
        "official_training": official_training,
        "source_systems": list(system_ids),
        "supervised_query_indices": checkpoint_payload["supervised_query_indices"],
        "query_partition_manifest_sha256": checkpoint_payload[
            "query_partition_manifest_sha256"
        ],
        "best_development_sd_nmae": (
            full_development_score if full_development_score is not None else best
        ),
        "selection_development_sd_nmae": best,
        "best_windows_seen": best_windows,
        "development_systems": full_development,
        "full_development_recomputed": full_development is not None,
        "checkpoint_saved": save_checkpoint,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "training_throughput_windows_per_second": (
            float(best_windows / max(time.time() - started, 1e-9))
        ),
        "environment": environment_record(),
        "partition_access": "training and development only",
        "pretrained_initialization": bool(
            pretrained and method in {"timer_xl", "unitime", "moirai", "gtm", "cpiri"}
        ),
    }
    atomic_json(output / "run.json", result)
    return result
