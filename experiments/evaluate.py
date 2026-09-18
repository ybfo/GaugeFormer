"""Evaluate the main-paper models, matched pairs, and component controls."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import time
import numpy as np
import torch
from .common import (
    PROJECT,
    ROOT,
    SYSTEMS,
    ORIGINS,
    batch,
    cell_id,
    checkpoint_record,
    device_setup,
    make_store,
    metric_state,
    seed_all,
    select_indices,
    write,
    artifact,
)
from .correction import CorrectionSuite, CONTROLS, ABLATIONS, LOCAL_CONTROLS
from .moirai import (
    build_frozen_moirai_bridge,
    _moirai_prediction,
    _evaluation_seed,
    _fork_rng,
    FORMAL_PROTOCOL,
    TEST_PROTOCOL,
)
from gaugeformer.sample_artifacts import SampleArtifactCollector

BACKBONES = ("moirai", "timer_xl", "gtm")
METHODS = (*BACKBONES, "unitime", "cpiri", "tefn", "tggc")
BATCH_SIZE = dict(moirai=16, timer_xl=16, gtm=16, unitime=8, cpiri=32, tefn=32, tggc=32)
DEV_PANELS = dict(
    building_energy=160, hydraulic=441, metropt=2048, pmsm=2048, steel_industry=288
)


def load_model(method, family, seed, target=None, fraction=None, device=None):
    device = device or device_setup()
    path, binding = checkpoint_record(method, family, seed, target, fraction)
    seed_all(seed)
    if method == "moirai":
        model, _ = build_frozen_moirai_bridge(path, device)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["method"] != method:
            raise ValueError("Checkpoint method differs")
        if method in ("tefn", "tggc"):
            from backbones.schema import SchemaBaseline

            model = SchemaBaseline(method)
            model.load_state_dict(state["model"])
        else:
            from backbones.backbones import build_model

            model = build_model(method, pretrained=method in ("unitime", "cpiri"))
            model.load_checkpoint_state(state["model"])
        model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if method == "gtm":
        from backbones.gtm_cache import install

        install(model)
    return model, binding


def causal_prefix(context, origin):
    if context.ndim != 3 or context.shape[1] != 96 or origin not in ORIGINS:
        raise ValueError("Historical prefix shape or origin differs")
    return torch.cat(
        (context[:, :1].expand(-1, 96 - origin, -1), context[:, :origin]), dim=1
    )


@torch.inference_mode()
def forecasts(
    method,
    model,
    contexts,
    store,
    binding,
    family,
    offset,
    *,
    split="test",
    historical=True,
    label="validation",
    device_context=None,
):
    """Use observed contexts only; current evaluation targets are not an argument."""
    if method == "moirai":
        # Preserve the evaluated channel-first tensor layout and seed labels.
        x = (
            device_context.transpose(1, 2)
            if device_context is not None
            else torch.from_numpy(contexts).to(next(model.parameters()).device)
        )
        kwargs = dict(
            bridge=model,
            contexts=x,
            query=torch.as_tensor(store.query_indices, device=x.device),
            device=x.device,
            checkpoint_sha256=binding["original_sha256"],
            family=family,
            system=store.metadata["system_id"],
            protocol=TEST_PROTOCOL if split == "test" else FORMAL_PROTOCOL,
        )
        current_label = (
            f"future_batch_{offset}" if split == "test" else f"{label}_future_{offset}"
        )
        if label == "formal_ablation":
            current_label = f"formal_ablation_future_batch_{offset}"
        raw = _moirai_prediction(
            **kwargs, label=current_label, samples=100, origin=None
        )
        back = None
        if historical:
            back = torch.stack(
                [
                    _moirai_prediction(
                        **kwargs,
                        label=f"backcast_{o}_batch_{offset}"
                        if split == "test"
                        else f"formal_ablation_backcast_{o}_batch_{offset}"
                        if label == "formal_ablation"
                        else f"{label}_historical_{o}_{offset}",
                        samples=20,
                        origin=o,
                    )
                    for o in ORIGINS
                ],
                dim=1,
            )
    else:
        x = (
            device_context
            if device_context is not None
            else torch.from_numpy(contexts.copy())
            .transpose(1, 2)
            .to(next(model.parameters()).device)
        )
        raw = model.predict_queries(x, store).transpose(1, 2)
        back = (
            torch.stack(
                [
                    model.predict_queries(causal_prefix(x, o), store).transpose(1, 2)
                    for o in ORIGINS
                ],
                dim=1,
            )
            if historical
            else None
        )
    return raw.cpu().numpy(), None if back is None else back.cpu().numpy()


@torch.inference_mode()
def run(
    method,
    family,
    seed,
    target=None,
    fraction=None,
    mode="pair",
    limit=None,
    system=None,
):
    if family in ("loso", "fewshot") and target not in SYSTEMS:
        raise ValueError("LOSO/fewshot requires --target")
    if family in ("joint", "unseen_target") and target is not None:
        raise ValueError("Joint/withheld-query checkpoints do not have a target system")
    if (family == "fewshot") != (fraction in (0.01, 0.05)):
        raise ValueError("Use --fraction .01 or .05 only for fewshot")
    if mode != "pair" and (method not in BACKBONES or family not in ("loso", "joint")):
        raise ValueError(
            "Main controls/components use the three backbones in LOSO/joint"
        )
    if mode == "diagnostic-components" and (method, family, seed) != (
        "moirai",
        "joint",
        17,
    ):
        raise ValueError(
            "The system-resolved component diagnostic uses joint Moirai seed 17"
        )
    split = "dev" if mode in ("components", "diagnostic-components") else "test"
    identity = cell_id(method, family, seed, target, fraction)
    output = ROOT / mode / identity
    if system:
        output = output / system
    if limit:
        output = output / f"smoke_{limit}"
    if (output / "complete.json").exists():
        raise FileExistsError(f"Result already exists: {output}")
    device = device_setup()
    model, binding = load_model(method, family, seed, target, fraction, device)
    variants = (
        ("raw", "full") + ABLATIONS
        if mode in ("components", "diagnostic-components")
        else CONTROLS + LOCAL_CONTROLS
        if mode == "controls"
        else ("raw", "full")
        if method in BACKBONES
        else ("raw",)
    )
    collectors = {
        v: SampleArtifactCollector(
            TEST_PROTOCOL if split == "test" else FORMAL_PROTOCOL,
            split,
            method if v == "raw" else f"{method}_{v}",
            dict(
                training_seed=seed,
                family=family,
                target_system=target,
                fraction=fraction,
                original_checkpoint_sha256=binding["original_sha256"],
            ),
        )
        for v in variants
    }
    metrics = {v: {} for v in variants}
    routing = {}
    started = time.time()
    systems = (target,) if target else SYSTEMS
    if system:
        if system not in systems:
            raise ValueError("Requested system is outside this checkpoint cell")
        systems = (system,)
    for name in systems:
        store = make_store(name, split, family)
        indices = select_indices(
            store, limit or (DEV_PANELS[name] if split == "dev" else None)
        )
        states = {v: metric_state(store) for v in variants}
        suite = CorrectionSuite(variants) if method in BACKBONES else None
        q = np.asarray(store.query_indices, dtype=np.int64)
        # CPiRi's stochastic inference retains the original per-cell RNG stream.
        rng = (
            _fork_rng(
                device,
                _evaluation_seed(
                    binding["original_sha256"],
                    family,
                    name,
                    "baseline_future",
                    0,
                    TEST_PROTOCOL,
                ),
            )
            if method in ("unitime", "cpiri")
            else nullcontext()
        )
        with rng:
            for offset in range(0, len(indices), BATCH_SIZE[method]):
                part = indices[offset : offset + BATCH_SIZE[method]]
                pairs = [store.get(int(i)) for i in part]
                contexts = np.stack([p[0] for p in pairs]).astype(np.float32)
                raw, historical = forecasts(
                    method,
                    model,
                    contexts,
                    store,
                    binding,
                    family,
                    offset,
                    split=split,
                    historical=suite is not None,
                    label="formal_ablation"
                    if mode == "diagnostic-components"
                    else "validation",
                )
                outputs = (
                    suite.predict(contexts, q, raw, historical)[0]
                    if suite
                    else {"raw": raw}
                )
                # Evaluation targets enter only scoring after the adaptation decision.
                observed = torch.from_numpy(
                    np.stack([p[1] for p in pairs])[:, q]
                ).transpose(1, 2)
                blocks = store.block_ids(part)
                for v, values in outputs.items():
                    pred = torch.from_numpy(values).transpose(1, 2)
                    acc, bacc, names = states[v]
                    acc.update(pred, observed)
                    bacc.update(pred, observed, blocks)
                    collectors[v].update(
                        name,
                        pred,
                        observed,
                        part,
                        blocks,
                        names,
                        dict(system_id=name, family=family, training_seed=seed),
                    )
                if offset % 2048 == 0:
                    print(
                        f"{identity} {name}: {offset+len(part)}/{len(indices)}",
                        flush=True,
                    )
        for v, (acc, bacc, _) in states.items():
            metrics[v][name] = acc.compute(name, len(indices))
            metrics[v][name]["statistical_blocks"] = bacc.compute()
        if suite:
            routing[name] = suite.summary()
    files = []
    for variant, collector in collectors.items():
        path = output / f"{variant}.npz"
        collector.save(path)
        files.append(artifact(path))
    result = dict(
        status="complete",
        method=method,
        family=family,
        seed=seed,
        target_system=target,
        fraction=fraction,
        split=split,
        mode=mode,
        limited_panel=limit,
        metrics=metrics,
        routing=routing,
        checkpoint=binding,
        artifacts=files,
        elapsed_seconds=time.time() - started,
    )
    write(output / "complete.json", result)
    print(output / "complete.json", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument(
        "--family", choices=("loso", "fewshot", "joint", "unseen_target"), required=True
    )
    p.add_argument("--seed", type=int, choices=(17, 29, 43, 71, 101), default=17)
    p.add_argument("--target", choices=SYSTEMS)
    p.add_argument("--fraction", type=float, choices=(0.01, 0.05))
    p.add_argument(
        "--mode",
        choices=("pair", "controls", "components", "diagnostic-components"),
        default="pair",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Smoke check only; full reported streams are the default",
    )
    p.add_argument(
        "--system", choices=SYSTEMS, help="Evaluate one system of a joint checkpoint"
    )
    run(**vars(p.parse_args()))
