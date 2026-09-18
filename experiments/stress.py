"""Main unit/reindexing/schema studies with their original two panel definitions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import numpy as np
import torch
from .common import ROOT, SYSTEMS, make_store, select_indices, write
from .evaluate import load_model, forecasts, BACKBONES, BATCH_SIZE
from .correction import CorrectionSuite
from .diagnostics import (
    query_panel,
    score_predictions,
    original_targets,
    predict_official_view,
    predict_paired_moirai_view,
    FORMAL_PROTOCOL,
)
from gaugeformer.stress import StressCondition, make_stress_view, robustness_conditions


def repeated_conditions():
    result = [
        StressCondition("reference", "reference"),
        StressCondition("legal_gauge_direct", "gauge"),
        StressCondition(
            "legal_gauge_canonical_preprocessing",
            "gauge",
            canonical_baseline_preprocessing=True,
        ),
    ]
    for r in range(3):
        result.append(StressCondition("permutation", "permutation", replicate=r))
        result.extend(
            StressCondition(f"deletion_{int(n*100)}", "deletion", replicate=r, amount=n)
            for n in (0.3, 0.5)
        )
        result.extend(
            StressCondition(f"insertion_{n}", "insertion", replicate=r, amount=n)
            for n in (1, 4)
        )
    return result


@torch.inference_mode()
def predict(method, model, binding, view, indices, panel, condition):
    device = next(model.parameters()).device
    if method == "moirai" and panel == "diagnostic":
        values, target, blocks, routing = predict_paired_moirai_view(
            model, view, indices, device, binding["original_sha256"]
        )
        return {
            "raw": values["moirai"].numpy(),
            "full": values["gaugeformer_v13"].numpy(),
        }
    if method not in BACKBONES or (
        panel == "diagnostic" and condition.kind in ("deletion", "insertion")
    ):
        p, _, _ = predict_official_view(
            method, model, view, indices, device, binding["original_sha256"]
        )
        return {"raw": p.numpy()}
    suite = CorrectionSuite()
    values = {"raw": [], "full": []}
    for offset in range(0, len(indices), 16):
        part = indices[offset : offset + 16]
        contexts = np.stack([view.get(int(i))[0] for i in part]).astype(np.float32)
        raw, back = forecasts(
            method,
            model,
            contexts,
            view,
            binding,
            "joint",
            offset,
            split="dev",
            label="robustness",
        )
        outputs, _ = suite.predict(contexts, view.query_indices, raw, back)
        for variant, pred in outputs.items():
            values[variant].append(
                view.prediction_to_original(
                    torch.from_numpy(pred).transpose(1, 2)
                ).numpy()
            )
    return {v: np.concatenate(p) for v, p in values.items()}


@torch.inference_mode()
def run(method, seed, panel, system=None, smoke=False):
    if panel == "diagnostic" and seed != 17:
        raise ValueError("Diagnostic panel uses seed 17")
    model, binding = load_model(method, "joint", seed)
    folder = ROOT / "stress" / panel / f"{method}_seed{seed}"
    if smoke:
        folder = folder / "smoke"
    if (folder / "complete.json").exists():
        raise FileExistsError(folder)
    results = {}
    for name in (system,) if system else SYSTEMS:
        base = make_store(name, "dev", "joint")
        queries = (
            query_panel(name)
            if panel == "diagnostic"
            else np.asarray(base.query_indices)
        )
        indices = select_indices(
            base, 32 if smoke else 1024 if panel == "diagnostic" else 128
        )
        if panel == "diagnostic":
            conditions = robustness_conditions(3)
            if smoke:
                conditions = conditions[:4]
            specs = [(c, indices, "condition") for c in conditions]
            short = select_indices(base, 32)
            specs.append(
                (StressCondition("reference", "reference"), short, "reindexing")
            )
            specs.extend(
                (
                    StressCondition("permutation", "permutation", replicate=r),
                    short,
                    "reindexing",
                )
                for r in range(2 if smoke else 100)
            )
        else:
            specs = [
                (c, indices, "condition")
                for c in (repeated_conditions()[:4] if smoke else repeated_conditions())
            ]
        references = {}
        for condition, idx, kind in specs:
            view = make_stress_view(
                base,
                queries,
                condition,
                **(
                    {"protocol_version": FORMAL_PROTOCOL}
                    if panel == "diagnostic"
                    else {}
                ),
            )
            outputs = predict(method, model, binding, view, idx, panel, condition)
            target = original_targets(view, idx)
            blocks = view.block_ids(idx)
            record = {}
            for variant, values in outputs.items():
                metric = score_predictions(
                    torch.from_numpy(values), target, blocks, base, queries
                )
                if condition.kind == "reference":
                    references[kind, variant] = values
                delta = np.abs(values - references[kind, variant])
                metric["max_absolute_discrepancy"] = float(delta.max())
                metric["max_relative_discrepancy"] = float(
                    (delta / np.maximum(np.abs(references[kind, variant]), 1)).max()
                )
                record[variant] = metric
            identity = f"{name}/{kind}/{condition.name}/rep{condition.replicate}"
            results[identity] = dict(condition=asdict(condition), metrics=record)
            print(identity, flush=True)
    write(
        folder / "complete.json",
        dict(
            status="complete",
            method=method,
            seed=seed,
            panel=panel,
            smoke=smoke,
            checkpoint=binding,
            results=results,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=(*BACKBONES, "unitime", "cpiri"), required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43, 71, 101), default=17)
    p.add_argument("--panel", choices=("diagnostic", "repeated"), required=True)
    p.add_argument("--system", choices=SYSTEMS)
    p.add_argument("--smoke", action="store_true")
    run(**vars(p.parse_args()))
