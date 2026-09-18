"""Complete training, source-validation selection and frozen few-shot grid."""
from __future__ import annotations
import argparse
import copy
import math
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .common import *
from backbones.schema import SchemaBaseline, BASE_LR, BASELINE_ROOT
from gaugeformer.fewshot import hashed_subset, subset_sha256


@torch.inference_mode()
def validate(model, stores, device, limit=2048):
    model.eval()
    metrics = {}
    for system, store in stores.items():
        indices = select_indices(store, limit)
        acc, _, _ = metric_state(store)
        for offset in range(0, len(indices), 32):
            x, y = batch(store, indices[offset:offset + 32], device)
            acc.update(model.predict_queries(x, store), y[..., store.query_indices])
        metrics[system] = acc.compute(system, len(indices))
    return float(np.mean([m["sd_nmae"] for m in metrics.values()])), metrics


def train_path(method, family, seed, target=None, fraction=None):
    return BASELINE_ROOT / "training" / cell_id(method, family, seed, target, fraction)


def fit(method, family, seed, target=None, fraction=None, *, max_windows=100000,
        lr=None, source=None, output=None, smoke=False):
    device = device_setup()
    output = Path(output) if output else train_path(method, family, seed, target, fraction)
    if completed(output / "complete.json"):
        return output
    if max_windows != 100000 and family != "fewshot" and not smoke:
        raise ValueError("Full source training must use the frozen 100000-window ceiling")
    seed_all(seed)
    model = SchemaBaseline(method)
    binding = None
    if source:
        source = Path(source)
        if not completed(source / "complete.json"):
            raise ValueError("Source training is incomplete")
        state = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        if state["method"] != method or state["seed"] != seed or target in state["source_systems"]:
            raise ValueError("Few-shot source checkpoint identity or LOSO isolation differs")
        model.load_state_dict(state["model"])
        binding = artifact(source / "best.pt")
    model.to(device)
    training_systems = (target,) if family == "fewshot" else tuple(s for s in SYSTEMS if s != target)
    train_stores = {s: make_store(s, "train", family, training=True) for s in training_systems}
    dev_stores = {} if family == "fewshot" else {s: make_store(s, "dev", family, training=True) for s in training_systems}
    subsets = {}
    if family == "fewshot":
        if source is None or fraction not in (0.01, 0.05):
            raise ValueError("Few-shot fitting requires its LOSO source and fixed fraction")
        subsets[target] = hashed_subset(len(train_stores[target]), fraction, target, "gf-cs-2026-08-20-v7")
    learning_rate = BASE_LR[method] if lr is None else lr
    optimizer, scheduler, interval = model.optimizer(learning_rate, max_windows)
    rng = np.random.default_rng(seed)
    history = []
    best, best_state, best_windows, stale = math.inf, None, 0, 0
    windows, step = 0, 0
    start = time.time()
    boundary = min(10000, max_windows)
    next_validation = boundary
    last_progress = 0
    output.mkdir(parents=True, exist_ok=True)
    while windows < max_windows:
        model.train()
        system = training_systems[int(rng.integers(len(training_systems)))]
        store = train_stores[system]
        size = min(32, max_windows - windows)
        indices = (rng.choice(subsets[system], size=size, replace=True) if system in subsets
                   else rng.integers(0, len(store), size=size))
        x, y = batch(store, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y, store, rng)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {windows} windows")
        loss.backward()
        # Validation only; no gradient clipping changes the official optimizer.
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite gradient at {windows} windows")
        optimizer.step()
        windows += size
        step += 1
        if step % interval == 0:
            scheduler.step()
        if time.time() - last_progress >= 30:
            write(output / "progress.json", dict(windows=windows, max_windows=max_windows,
                  elapsed_seconds=time.time()-start, loss=float(loss.detach()), pid=os.getpid()))
            print(f"TRAIN {output.name} {windows}/{max_windows} loss={float(loss.detach()):.6f}", flush=True)
            last_progress = time.time()
        if windows < next_validation and windows < max_windows:
            continue
        if family == "fewshot":
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            best_windows = windows
        else:
            score, metrics = validate(model, dev_stores, device, limit=32 if smoke else 2048)
            history.append(dict(windows=windows, source_dev_sd_nmae=score, systems=metrics))
            print(f"VALIDATE {output.name} windows={windows} source_dev={score:.6f}", flush=True)
            if score < best:
                best, best_windows, stale = score, windows, 0
                best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            else:
                stale += 1
            write(output / "history.json", history)
            if stale >= 6:
                break
        next_validation += boundary
    if best_state is None:
        raise RuntimeError("No finite checkpoint was selected")
    state = dict(protocol_version=PROTOCOL, method=method, family=family, seed=seed,
                 target_system=target, fraction=fraction, model=best_state,
                 source_systems=list(training_systems), source_checkpoint=binding,
                 learning_rate=learning_rate, selected_windows=best_windows,
                 normalization="official training-partition z-score; training-constant channels use scale 1",
                 schema_width=26, random_slot_training=True,
                 supervised_queries={s: st.query_indices.tolist() for s, st in train_stores.items()},
                 fewshot_subsets={s: dict(count=len(v), sha256=subset_sha256(v)) for s, v in subsets.items()})
    temporary = output / "best.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, output / "best.pt")
    write(output / "complete.json", dict(status="complete", protocol_version=PROTOCOL, method=method,
          family=family, seed=seed, target_system=target, fraction=fraction, smoke=smoke,
          max_windows=max_windows, windows_processed=windows, selected_windows=best_windows,
          source_dev_sd_nmae=None if family == "fewshot" else best, learning_rate=learning_rate,
          source_systems=list(training_systems), test_access=False,
          source_checkpoint=binding, artifacts=[artifact(output / "best.pt")],
          elapsed_seconds=time.time()-start))
    print(f"TRAIN COMPLETE {output.name} {time.time()-start:.1f}s", flush=True)
    return output


def calibrate(method, fraction):
    output = BASELINE_ROOT / "calibration" / f"{method}_fraction{fraction:g}"
    if completed(output / "complete.json"):
        return read(output / "complete.json")
    device = device_setup()
    records = []
    for multiplier in (0.03, 0.1):
        for windows in (1000, 5000):
            metrics = {}
            for target in SYSTEMS:
                path = output / f"lr{multiplier:g}_windows{windows}" / target
                fit(method, "fewshot", 17, target, fraction, max_windows=windows,
                    lr=BASE_LR[method] * multiplier, source=train_path(method, "loso", 17, target), output=path)
                score_path = path / "development.json"
                if score_path.exists():
                    metric = read(score_path)
                else:
                    seed_all(17)
                    model = SchemaBaseline(method)
                    state = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(state["model"])
                    model.to(device)
                    score, values = validate(model, {target: make_store(target, "dev", "fewshot")}, device)
                    metric = dict(sd_nmae=score, metrics=values, checkpoint=artifact(path / "best.pt"))
                    write(score_path, metric)
                    del model
                metrics[target] = metric
            records.append(dict(multiplier=multiplier, windows=windows, targets=metrics,
                                score=float(np.mean([m["sd_nmae"] for m in metrics.values()]))))
    selected = min(records, key=lambda r: (r["score"], r["windows"], r["multiplier"]))
    result = dict(status="complete", protocol_version=PROTOCOL, method=method, fraction=fraction,
                  records=records, selected=selected, selection_seed=17, test_access=False)
    write(output / "complete.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("tefn", "tggc"), required=True)
    p.add_argument("--family", choices=("joint", "loso", "unseen_target", "fewshot", "calibrate"), required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--target", choices=SYSTEMS)
    p.add_argument("--fraction", type=float)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.family == "calibrate":
        calibrate(args.method, args.fraction)
    elif args.family == "fewshot":
        selection = read(BASELINE_ROOT / "calibration" / f"{args.method}_fraction{args.fraction:g}" / "complete.json")["selected"]
        fit(args.method, args.family, args.seed, args.target, args.fraction,
            max_windows=selection["windows"], lr=BASE_LR[args.method]*selection["multiplier"],
            source=train_path(args.method, "loso", args.seed, args.target))
    else:
        fit(args.method, args.family, args.seed, args.target, args.fraction,
            max_windows=64 if args.smoke else 100000, smoke=args.smoke,
            output=(BASELINE_ROOT / "smoke" / f"train_{args.method}" if args.smoke else None))


if __name__ == "__main__":
    main()
