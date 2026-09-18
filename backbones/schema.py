"""Official TEFN/TGGC with a documented 26-slot schema I/O adapter.

Training randomly injects real channels into the shared slot bank, ensuring
that every parameter slot can be trained even when Building is held out.
Evaluation uses the canonical first-C slots. Dummy outputs never enter loss.
"""
from __future__ import annotations
import importlib.util
import types
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.common import PROJECT

HERE = PROJECT / "backbones"
from experiments.common import ROOT

WIDTH = 26
BASELINE_ROOT = ROOT / "baselines_native_zscore_v2"


@lru_cache(maxsize=None)
def native_location_scale(root):
    root = Path(root)
    values = np.load(root / "values.npy", mmap_mode="r")
    split = np.load(root / "split_codes.npy", mmap_mode="r")
    train = np.asarray(values[np.asarray(split) == 0], dtype=np.float64)
    center, spread = train.mean(axis=0), train.std(axis=0)
    # TGGC's official ForecastDataset and TEFN's StandardScaler use scale=1
    # on training-constant features. Retain that behavior, including covariates.
    spread = np.where(spread == 0, 1.0, spread)
    return center.astype(np.float32), spread.astype(np.float32)


def standardize_native(context, store):
    if hasattr(store, "normalization_center"):
        center_np, spread_np = store.normalization_center, store.normalization_spread
        spread_np = np.where(np.asarray(spread_np) == 0, 1.0, spread_np)
    else:
        center_np, spread_np = native_location_scale(str(store.root.resolve()))
    center = torch.as_tensor(
        center_np, device=context.device, dtype=context.dtype
    ).view(1, 1, -1)
    spread = torch.as_tensor(
        spread_np, device=context.device, dtype=context.dtype
    ).view(1, 1, -1)
    return (context - center) / spread, center, spread


def official_module(method):
    path = (
        HERE
        / "vendor"
        / method.upper()
        / ("TEFN.py" if method == "tefn" else "base_model.py")
    )
    module = types.ModuleType(f"extension_official_{method}")
    source = path.read_text(encoding="utf-8")
    if method == "tggc":
        # Exactly one allocation in the release unconditionally calls .cuda().
        # Device placement is handled by model.to(device); equations are unchanged.
        if source.count(".cuda()") != 1:
            raise RuntimeError("TGGC device portability patch no longer matches source")
        source = source.replace(".cuda()", "")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class SchemaBaseline(nn.Module):
    def __init__(self, method, width=WIDTH):
        super().__init__()
        self.method = method
        self.width = width
        source = official_module(method)
        if method == "tefn":
            self.core = source.Model(
                SimpleNamespace(
                    task_name="long_term_forecast",
                    seq_len=96,
                    label_len=48,
                    pred_len=24,
                    enc_in=width,
                    e_layers=0,
                    dropout=0.1,
                    use_norm=True,
                    use_T_model=True,
                    use_C_model=True,
                    fusion_method="add",
                    use_probabilistic_layer=False,
                    kernel_activation=None,
                    use_residual=True,
                )
            )
        elif method == "tggc":
            self.core = source.Model(
                units=width,
                stack_cnt=2,
                time_step=96,
                multi_layer=5,
                horizon=24,
                dropout_rate=0.4,
                leaky_rate=0.02,
                coe_a=1.2,
                coe_b=1.0,
                order=4,
                gconv="gegen",
                non_linear="linear",
                Fouropt="FB",
                attention_set="linear",
                modes=5,
                activation="softmax",
                device="cpu",
            )
            # Random Fourier indices in the release are plain Python attributes.
            # Persist them as buffers so loading a checkpoint cannot redraw them.
            for block in self.core.stock_block:
                block.Fourier.register_buffer(
                    "saved_frequency_indices",
                    torch.tensor(block.Fourier.index, dtype=torch.int64),
                )
        else:
            raise ValueError(method)

    def load_state_dict(self, state, strict=True, **kwargs):
        result = super().load_state_dict(state, strict=strict, **kwargs)
        if self.method == "tggc":
            for block in self.core.stock_block:
                block.Fourier.index = (
                    block.Fourier.saved_frequency_indices.cpu().tolist()
                )
        return result

    def pack(self, values, generator=None):
        c = values.shape[-1]
        if c > self.width:
            raise ValueError("Real channel count exceeds frozen schema width")
        slots = (
            np.arange(c) if generator is None else generator.permutation(self.width)[:c]
        )
        slots = torch.as_tensor(slots, dtype=torch.long, device=values.device)
        packed = values.new_zeros(values.shape[0], values.shape[1], self.width)
        packed[..., slots] = values
        return packed, slots

    def forward_scaled(self, scaled, generator=None):
        packed, slots = self.pack(scaled, generator)
        if self.method == "tefn":
            forecast = self.core(packed, None, None, None)
            reconstruction = None
        else:
            forecast, _, reconstruction = self.core(packed)
        return (
            forecast[..., slots],
            None if reconstruction is None else reconstruction[..., slots],
        )

    def loss(self, x, y, store, generator):
        scaled, center, spread = standardize_native(x, store)
        target = (y - center) / spread
        pred, reconstruction = self.forward_scaled(scaled, generator)
        q = store.query_indices
        result = F.mse_loss(pred[..., q], target[..., q])
        if reconstruction is not None:
            # Official TGGC objective includes reconstruction of observed context.
            result = result + F.mse_loss(reconstruction, scaled)
        return result

    def predict_queries(self, x, store):
        scaled, center, spread = standardize_native(x, store)
        if self.method == "tggc" and len(x) > 1:
            # The release averages graph estimates over a batch. Evaluate each
            # window independently to avoid using later windows' context.
            pred = torch.cat(
                [self.forward_scaled(row[None])[0] for row in scaled], dim=0
            )
        else:
            pred, _ = self.forward_scaled(scaled)
        return (pred * spread + center)[..., store.query_indices]

    def optimizer(self, learning_rate, max_windows):
        if self.method == "tefn":
            optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
            interval = max(1, int(np.ceil(max_windows / 32 / 10)))
        else:
            optim = torch.optim.RMSprop(self.parameters(), lr=learning_rate, eps=1e-8)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
            # Official 50 epochs, exponential decay every 15 epochs.
            interval = max(1, int(np.ceil(max_windows / 32 * 15 / 50)))
        return optim, scheduler, interval


BASE_LR = {"tefn": 0.05, "tggc": 0.001}
