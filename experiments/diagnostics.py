"""The seed-17 diagnostic's original query panels and coupled forecast draws."""
from __future__ import annotations
from collections import Counter
from typing import Any
import numpy as np
import torch
from .common import PROJECT, SYSTEMS, ORIGINS as BACKCAST_ORIGINS
from .evaluate import BATCH_SIZE
from .moirai import _evaluation_seed, _fork_rng, _moirai_prediction, FORMAL_PROTOCOL
from gaugeformer import GaugeFormerConfig, GaugeFormerMemory
from gaugeformer.data import WindowStore
from gaugeformer.stress import load_query_panel
from gaugeformer.training import training_standard_deviation
from gaugeformer.metrics import BlockSDNMAEAccumulator, ChannelMetricAccumulator

QUERY_MANIFEST = PROJECT / "configs/diagnostic_queries.json"
QUERY_PROTOCOL = "gf-cs-2026-08-12-v5"
FUTURE_SAMPLES = 100
BACKCAST_SAMPLES = 20
BATCH_SIZE = dict(BATCH_SIZE, gaugeformer_v13=16)
OTHER_METHODS = ("unitime", "timer_xl", "gtm", "cpiri")
VARIANTS = (
    "full",
    "backbone_only",
    "no_level_experts",
    "no_trend_experts",
    "no_origin_win_guard",
    "no_shrinkage",
    "no_warmup",
)


def query_panel(system: str) -> np.ndarray:
    panel = load_query_panel(QUERY_MANIFEST, system, QUERY_PROTOCOL)
    if panel.ndim != 1 or not len(panel) or len(np.unique(panel)) != len(panel):
        raise RuntimeError("formal robustness query panel differs")
    return panel


def original_targets(view: Any, indices: np.ndarray) -> torch.Tensor:
    values = np.stack(
        [view.original_query_future(int(index)) for index in indices]
    ).transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def score_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    block_ids: np.ndarray,
    base: WindowStore,
    panel: np.ndarray,
) -> dict:
    names = [base.metadata["channels"][int(index)]["name"] for index in panel]
    scale = training_standard_deviation(base)[panel]
    metric = ChannelMetricAccumulator(names, scale)
    blocks = BlockSDNMAEAccumulator(names, scale)
    metric.update(prediction, target)
    blocks.update(prediction, target, block_ids)
    result = metric.compute(base.metadata["system_id"], prediction.shape[0])
    result["query_source_indices"] = panel.tolist()
    result["statistical_blocks"] = blocks.compute()
    return result


@torch.inference_mode()
def predict_official_view(
    method: str,
    model: torch.nn.Module,
    view: Any,
    indices: np.ndarray,
    device: torch.device,
    checkpoint_hash: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if method not in OTHER_METHODS:
        raise ValueError("official-view predictor supports four non-Moirai baselines")
    predictions = []
    targets = []
    block_ids: list[str] = []
    seed = _evaluation_seed(
        checkpoint_hash,
        "joint",
        str(view.metadata["system_id"]),
        "formal_downstream_official_baseline",
        0,
    )
    with _fork_rng(device, seed):
        for offset in range(0, len(indices), BATCH_SIZE[method]):
            part = indices[offset : offset + BATCH_SIZE[method]]
            contexts = np.stack([view.get(int(index))[0] for index in part])
            context = torch.from_numpy(
                np.ascontiguousarray(contexts, dtype=np.float32)
            ).to(device)
            prediction = model.predict_queries(context.transpose(1, 2), view)
            predictions.append(view.prediction_to_original(prediction).cpu())
            targets.append(original_targets(view, part))
            block_ids.extend(map(str, view.block_ids(part)))
    return (
        torch.cat(predictions),
        torch.cat(targets),
        np.asarray(block_ids, dtype=np.str_),
    )


@torch.inference_mode()
def predict_paired_moirai_view(
    bridge: torch.nn.Module,
    view: Any,
    indices: np.ndarray,
    device: torch.device,
    checkpoint_hash: str,
    *,
    variant: str = "full",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, np.ndarray, dict[str, object]]:
    if variant not in VARIANTS:
        raise ValueError("formal GaugeFormer-v13 variant differs")
    memory = GaugeFormerMemory(GaugeFormerConfig.for_variant(variant))
    predictions: dict[str, list[torch.Tensor]] = {
        "moirai": [],
        "gaugeformer_v13": [],
    }
    targets = []
    block_ids: list[str] = []
    choice_histogram: Counter[str] = Counter()
    eligible_queries = 0
    total_query_windows = 0
    system = str(view.metadata["system_id"])
    for offset in range(0, len(indices), BATCH_SIZE["gaugeformer_v13"]):
        part = indices[offset : offset + BATCH_SIZE["gaugeformer_v13"]]
        contexts = np.stack([view.get(int(index))[0] for index in part]).astype(
            np.float32
        )
        context = torch.from_numpy(contexts).to(device)
        query_array = np.asarray(view.query_indices, dtype=np.int64)
        query = torch.from_numpy(query_array).to(device)
        raw_future = (
            _moirai_prediction(
                bridge,
                context,
                query,
                device,
                checkpoint_sha256=checkpoint_hash,
                family="joint",
                system=system,
                label=f"formal_downstream_future_batch_{offset}",
                samples=FUTURE_SAMPLES,
                origin=None,
            )
            .cpu()
            .numpy()
        )
        raw_backcasts = np.stack(
            [
                _moirai_prediction(
                    bridge,
                    context,
                    query,
                    device,
                    checkpoint_sha256=checkpoint_hash,
                    family="joint",
                    system=system,
                    label=f"formal_downstream_backcast_{origin}_batch_{offset}",
                    samples=BACKCAST_SAMPLES,
                    origin=origin,
                )
                .cpu()
                .numpy()
                for origin in BACKCAST_ORIGINS
            ],
            axis=1,
        )
        method_prediction = []
        for row, values in enumerate(contexts):
            output = memory.predict(
                values,
                query_array,
                raw_future[row],
                raw_backcasts[row],
            )
            method_prediction.append(output.prediction)
            choice_histogram.update(
                output.candidate_names[int(position)]
                for position in output.selected_candidate
            )
            eligible_queries += int(output.eligible.sum())
            total_query_windows += len(query_array)
        view_predictions = {
            "moirai": torch.from_numpy(raw_future).transpose(1, 2),
            "gaugeformer_v13": torch.from_numpy(
                np.stack(method_prediction).astype(np.float32)
            ).transpose(1, 2),
        }
        for method, prediction in view_predictions.items():
            predictions[method].append(view.prediction_to_original(prediction).cpu())
        targets.append(original_targets(view, part))
        block_ids.extend(map(str, view.block_ids(part)))
    routing = {
        "variant": variant,
        "choice_histogram": dict(sorted(choice_histogram.items())),
        "eligible_query_fraction": (
            float(eligible_queries / total_query_windows)
            if total_query_windows
            else 0.0
        ),
        "memory": memory.evidence_summary(),
    }
    return (
        {method: torch.cat(parts) for method, parts in predictions.items()},
        torch.cat(targets),
        np.asarray(block_ids, dtype=np.str_),
        routing,
    )
