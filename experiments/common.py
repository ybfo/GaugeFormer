"""Portable paths, fixed protocol, checkpoints and metric utilities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

PROJECT = Path(
    os.environ.get("GAUGEFORMER_ROOT", Path(__file__).resolve().parents[1])
).resolve()
HERE = Path(__file__).resolve().parent
from gaugeformer.data import WindowStore, QuerySubsetStore, load_query_partition
from gaugeformer.metrics import ChannelMetricAccumulator, BlockSDNMAEAccumulator
from gaugeformer.training import training_standard_deviation

PROTOCOL = "gf-extension-2026-09-14-multibackbone-tpami-v1"
SYSTEMS = ("building_energy", "hydraulic", "metropt", "pmsm", "steel_industry")
SEEDS = (17, 29, 43, 71, 101)
ORIGINS = (48, 56, 64, 72)
ROOT = PROJECT / "results" / "runs"

QUERY_FILE = PROJECT / "configs" / "queries.json"
REGISTRY = PROJECT / "configs" / "checkpoints.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_setup():
    torch.set_num_threads(int(os.environ.get("GAUGEFORMER_THREADS", "4")))
    return torch.device(
        os.environ.get(
            "GAUGEFORMER_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    )


def cell_id(method, family, seed, target=None, fraction=None):
    parts = [method, family, f"seed{seed}"]
    if target:
        parts.append(target)
    if fraction is not None:
        parts.append(f"fraction{fraction:g}")
    return "_".join(parts)


def cells(method):
    # Complete source-only results have first priority.
    for seed in SEEDS:
        for target in SYSTEMS:
            yield dict(
                method=method, family="loso", seed=seed, target=target, fraction=None
            )
    for family in ("joint", "unseen_target"):
        for seed in SEEDS:
            yield dict(
                method=method, family=family, seed=seed, target=None, fraction=None
            )
    for fraction in (0.01, 0.05):
        for seed in SEEDS:
            for target in SYSTEMS:
                yield dict(
                    method=method,
                    family="fewshot",
                    seed=seed,
                    target=target,
                    fraction=fraction,
                )


def make_store(system, split, family, *, training=False):
    store = WindowStore(PROJECT / "data" / "processed" / system, split)
    if family == "unseen_target":
        partition = "supervised_forecast" if training else "unseen_target"
        query = load_query_partition(QUERY_FILE, partition, "gf-cs-2026-08-12-v5")
        store = QuerySubsetStore(store, query[system])
    return store


def batch(store, indices, device):
    pairs = [store.get(int(i)) for i in indices]
    x = (
        torch.from_numpy(np.stack([p[0] for p in pairs]).copy())
        .transpose(1, 2)
        .to(device)
    )
    y = (
        torch.from_numpy(np.stack([p[1] for p in pairs]).copy())
        .transpose(1, 2)
        .to(device)
    )
    return x, y


def metric_state(store):
    q = np.asarray(store.query_indices, dtype=np.int64)
    names = [store.metadata["channels"][int(i)]["name"] for i in q]
    scale = training_standard_deviation(store)[q]
    return (
        ChannelMetricAccumulator(names, scale),
        BlockSDNMAEAccumulator(names, scale),
        names,
    )


def select_indices(store, limit=None):
    if limit is None or len(store) <= limit:
        return np.arange(len(store), dtype=np.int64)
    return np.linspace(0, len(store) - 1, limit, dtype=np.int64)


def checkpoint_record(method, family, seed, target=None, fraction=None):
    records = read(REGISTRY)["records"]
    matches = [
        r
        for r in records
        if (r["method"], r["family"], r["seed"], r["target_system"], r["fraction"])
        == (method, family, seed, target, fraction)
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one registered checkpoint")
    r = matches[0]
    path = PROJECT / r["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Download {r['asset']} first: {path}")
    if path.stat().st_size != r["bytes"] or digest(path) != r["sha256"]:
        raise ValueError(f"Checkpoint integrity check failed: {path}")
    return path, dict(
        r, checkpoint={"relative_path": r["path"], "sha256": r["original_sha256"]}
    )


def completed(path):
    path = Path(path)
    if not path.exists():
        return False
    record = read(path)
    if record.get("status") != "complete" or record.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"Invalid completion record: {path}")
    for item in record.get("artifacts", []):
        artifact = PROJECT / item["path"]
        if not artifact.is_file() or digest(artifact) != item["sha256"]:
            raise RuntimeError(f"Completed artifact changed: {artifact}")
    return True


def artifact(path):
    path = Path(path)
    return dict(
        path=path.relative_to(PROJECT).as_posix(),
        bytes=path.stat().st_size,
        sha256=digest(path),
    )
