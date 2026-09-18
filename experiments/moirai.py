"""Exact stochastic seed and prefix adapters for the evaluated Moirai pair."""
from __future__ import annotations
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import torch
from backbones.moirai_bridge import Moirai1Bridge
from .common import PROJECT, ORIGINS as BACKCAST_ORIGINS

FORMAL_PROTOCOL = "gf-cs-2026-08-24-v13-formal-development"
TEST_PROTOCOL = "gf-cs-2026-08-24-v13-confirmatory"
FUTURE_SAMPLES = 100
BACKCAST_SAMPLES = 20


def build_frozen_moirai_bridge(
    checkpoint_path: Path, device: torch.device
) -> tuple[Moirai1Bridge, dict]:
    source = PROJECT / "third_party" / "moirai" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    official = PROJECT / "checkpoints" / "official" / "moirai_1.1_R_small"
    module = MoiraiModule.from_pretrained(str(official))
    forecast = MoiraiForecast(
        module=module,
        prediction_length=24,
        context_length=96,
        patch_size=16,
        num_samples=FUTURE_SAMPLES,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("method") != "moirai":
        raise ValueError("paired base checkpoint is not Moirai")
    state = {
        key.removeprefix("forecast."): value for key, value in payload["model"].items()
    }
    forecast.load_state_dict(state, strict=True)
    bridge = Moirai1Bridge(
        forecast,
        patch_size=16,
        training_point_samples=BACKCAST_SAMPLES,
        evaluation_point_samples=FUTURE_SAMPLES,
    ).to(device)
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    bridge.eval()
    return bridge, payload


def _evaluation_seed(
    checkpoint_sha256: str,
    family: str,
    system: str,
    label: str,
    samples: int,
    protocol: str = FORMAL_PROTOCOL,
) -> int:
    material = (
        f"{protocol}|{checkpoint_sha256}|{family}|{system}|{label}|{samples}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


@contextmanager
def _fork_rng(device: torch.device, seed: int) -> Iterator[None]:
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def _left_padded_prefix(
    context: torch.Tensor, origin: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.ndim != 3 or context.shape[-1] != 96 or origin not in BACKCAST_ORIGINS:
        raise ValueError("formal Moirai prefix differs")
    values = torch.zeros_like(context)
    mask = torch.zeros_like(context, dtype=torch.bool)
    values[..., -origin:] = context[..., :origin]
    mask[..., -origin:] = True
    return values, mask


@torch.inference_mode()
def _moirai_prediction(
    bridge: torch.nn.Module,
    contexts: torch.Tensor,
    query: torch.Tensor,
    device: torch.device,
    *,
    checkpoint_sha256: str,
    family: str,
    system: str,
    label: str,
    samples: int,
    origin: int | None,
    protocol: str = FORMAL_PROTOCOL,
) -> torch.Tensor:
    if origin is None:
        values = contexts
        mask = torch.ones_like(values, dtype=torch.bool)
    else:
        values, mask = _left_padded_prefix(contexts, origin)
    seed = _evaluation_seed(checkpoint_sha256, family, system, label, samples, protocol)
    with _fork_rng(device, seed):
        return bridge(
            values,
            mask,
            query,
            point_mode="sample_median",
            point_samples=samples,
        )["base_forecast_canonical"]
