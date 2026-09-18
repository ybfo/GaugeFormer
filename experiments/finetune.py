"""Few-shot update of one published baseline without development/test access."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from gaugeformer.data import WindowStore
from gaugeformer.fewshot import hashed_subset, subset_sha256
from gaugeformer.training import atomic_json, environment_record
from gaugeformer.protocol import file_sha256, recorded_checkpoint_protocol
from gaugeformer.fewshot_selection import validate_selection

from backbones.backbones import (
    BaselineTrainConfig,
    build_model,
    draw_window_batch,
    seed_everything,
)


from .common import PROJECT

DEFAULT_PROTOCOL_VERSION = "gf-cs-2026-08-20-v7"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("unitime", "moirai", "timer_xl", "gtm", "cpiri"),
        required=True,
    )
    parser.add_argument("--protocol-version", default=DEFAULT_PROTOCOL_VERSION)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-system", required=True)
    parser.add_argument("--fraction", type=float, choices=(0.01, 0.05), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args()

    selection_sha256 = None
    if args.selection_manifest is not None:
        selection_sha256 = validate_selection(
            args.selection_manifest,
            args.method,
            args.fraction,
            args.max_windows,
            args.learning_rate,
            protocol_version=args.protocol_version,
        )

    seed_everything(args.seed)
    source_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if recorded_checkpoint_protocol(source_checkpoint) != args.protocol_version:
        raise ValueError("source checkpoint protocol version differs")
    model = build_model(args.method, pretrained=True)
    model.load_checkpoint_state(source_checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    store = WindowStore(PROJECT / "data" / "processed" / args.target_system, "train")
    selected = hashed_subset(
        len(store),
        args.fraction,
        args.target_system,
        args.protocol_version,
    )
    config = BaselineTrainConfig(
        protocol_version=args.protocol_version,
        seed=args.seed,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        validation_every_windows=args.max_windows,
        learning_rate=args.learning_rate,
        amp=False,
    )
    total_steps = math.ceil(args.max_windows / args.batch_size)
    optimizer, scheduler, gradient_clip, official_training = model.configure_training(
        config, total_steps
    )
    scheduler_interval = int(official_training.get("scheduler_interval_steps", 1))
    rng = np.random.default_rng(args.seed)
    started = time.time()
    windows = 0
    while windows < args.max_windows:
        model.train()
        current = min(args.batch_size, args.max_windows - windows)
        indices = rng.choice(selected, size=current, replace=True)
        context, future = draw_window_batch(store, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(context, future, store, rng)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {args.method} few-shot loss after {windows} windows"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        completed_step = math.ceil((windows + current) / args.batch_size)
        if (
            completed_step % scheduler_interval == 0
            or windows + current >= args.max_windows
        ):
            scheduler.step()
        windows += current

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapted = {
        "protocol_version": args.protocol_version,
        "model": {
            name: value.detach().cpu()
            for name, value in model.checkpoint_state().items()
        },
        "method": args.method,
        "config": asdict(config),
        "systems": list(source_checkpoint["systems"]) + [args.target_system],
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "selection_manifest_sha256": selection_sha256,
        "target_system": args.target_system,
        "fewshot_fraction": args.fraction,
        "fewshot_indices_sha256": subset_sha256(selected),
        "windows_seen": args.max_windows,
        "checkpoint_scope": "trainable-only" if args.method == "cpiri" else "full",
    }
    adapted_path = args.output_dir / "best.pt"
    torch.save(adapted, adapted_path)
    adapted_sha256 = file_sha256(adapted_path)
    result = {
        "status": "complete",
        "protocol_version": args.protocol_version,
        "method": model.name,
        "target_system": args.target_system,
        "fraction": args.fraction,
        "seed": args.seed,
        "selected_training_windows": int(len(selected)),
        "selected_indices_sha256": subset_sha256(selected),
        "fine_tuning_windows_seen": args.max_windows,
        "learning_rate": args.learning_rate,
        "official_training": official_training,
        "elapsed_seconds": time.time() - started,
        "partition_access": "target training partition only; no development or test inference",
        "environment": environment_record(),
        "source_checkpoint_sha256": adapted["source_checkpoint_sha256"],
        "selection_manifest_sha256": selection_sha256,
        "checkpoint_sha256": adapted_sha256,
        "checkpoint_bytes": adapted_path.stat().st_size,
    }
    atomic_json(args.output_dir / "run.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
