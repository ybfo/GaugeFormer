"""Shared normalization and serialization utilities."""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from .data import WindowStore
def training_standard_deviation(store: WindowStore) -> np.ndarray:
    split_codes = np.load(store.root / "split_codes.npy", mmap_mode="r")
    train = np.asarray(store.values[np.asarray(split_codes) == 0], dtype=np.float64)
    return np.maximum(np.std(train, axis=0), 1e-8)


def environment_record() -> dict:
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cudnn": torch.backends.cudnn.version(),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)

