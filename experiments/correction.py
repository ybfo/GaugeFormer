"""Shared eight-expert computation for unchanged GF and declared diagnostics."""
from __future__ import annotations
from collections import Counter
import numpy as np
from .common import ORIGINS
from gaugeformer.local import selected_experts
from gaugeformer import GaugeFormerConfig
from gaugeformer.method import FULL_CANDIDATE_NAMES
from gaugeformer.memory import GaugeDynamicsMemory, OnlineRouterConfig
from gaugeformer.memory import select_online_blend

CONTROLS = (
    "raw",
    "full",
    "persistence",
    "history_local",
    "persistence_blend",
    "history_blend",
)
LOCAL_CONTROLS = (*FULL_CANDIDATE_NAMES[2:], "mean_bank")
ABLATIONS = (
    "no_level_experts",
    "no_trend_experts",
    "no_origin_win_guard",
    "no_shrinkage",
    "no_warmup",
    "no_memory",
)


def bank(contexts, query, raw, historical):
    values = np.asarray(contexts, dtype=np.float64)
    query = np.asarray(query, dtype=np.int64)
    raw = np.asarray(raw, dtype=np.float32)
    historical = np.asarray(historical, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[-1] != 96
        or raw.shape != (len(values), len(query), 24)
        or historical.shape != (len(values), 4, len(query), 24)
    ):
        raise ValueError("Incorrect context/backbone shape")
    candidates = np.concatenate((raw[:, None], selected_experts(values, query)), axis=1)
    losses = []
    for i, origin in enumerate(ORIGINS):
        predictions = np.concatenate(
            (historical[:, i : i + 1], selected_experts(values[..., :origin], query)),
            axis=1,
        )
        losses.append(
            np.abs(predictions - values[:, None, query, origin : origin + 24]).mean(
                axis=-1
            )
        )
    return candidates, np.stack(losses, axis=1)


class CorrectionSuite:
    def __init__(self, variants=("raw", "full")):
        self.variants = tuple(variants)
        self.states, self.positions = {}, {}
        self.n = 0
        self.history_losses = None
        self.counts = {v: Counter() for v in variants}
        for variant in variants:
            if (variant in CONTROLS and variant != "full") or variant in LOCAL_CONTROLS:
                continue
            cfg = OnlineRouterConfig(blend=0.5)
            names = FULL_CANDIDATE_NAMES
            if variant in ABLATIONS and variant != "no_memory":
                formal = GaugeFormerConfig.for_variant(variant)
                cfg, names = formal.router_config, formal.candidate_names
            elif variant not in ("full", "no_memory"):
                raise ValueError(variant)
            self.states[variant] = GaugeDynamicsMemory(cfg)
            self.positions[variant] = [FULL_CANDIDATE_NAMES.index(n) for n in names]

    def predict(self, contexts, query, raw, historical):
        candidates, losses = bank(contexts, query, raw, historical)
        output = {v: [] for v in self.variants}
        masks = {v: [] for v in self.variants}
        for cand, loss in zip(candidates, losses):
            self.n += 1
            if self.history_losses is None:
                self.history_losses = np.zeros_like(loss[0])
            self.history_losses += loss.sum(axis=0)
            best = 1 + self.history_losses[1:].argmin(axis=0)
            local = cand[best, np.arange(cand.shape[1])]
            for variant in self.variants:
                mask = np.ones(cand.shape[1], dtype=bool)
                if variant == "raw":
                    pred, mask = cand[0], np.zeros(cand.shape[1], dtype=bool)
                elif variant == "persistence":
                    pred = cand[1]
                elif variant == "history_local":
                    pred = local
                elif variant == "persistence_blend":
                    pred = cand[0] + 0.5 * (cand[1] - cand[0])
                elif variant == "history_blend":
                    pred = cand[0] + 0.5 * (local - cand[0])
                elif variant == "mean_bank":
                    pred = cand[1:].mean(axis=0)
                elif variant in LOCAL_CONTROLS:
                    pred = cand[FULL_CANDIDATE_NAMES.index(variant)]
                elif variant == "no_memory":
                    # Keep the same stream warm-up count; discard prior losses/wins only.
                    pred, _, mask = select_online_blend(
                        cand,
                        loss.sum(axis=0),
                        (loss < loss[:, 0:1]).sum(axis=0),
                        observed_origin_panels=4,
                        observed_windows=self.n,
                        margin=0.0,
                        minimum_win_fraction=0.55,
                        blend=0.5,
                        warmup_windows=16,
                    )
                else:
                    positions = self.positions[variant]
                    pred, _, mask = self.states[variant].update_and_predict(
                        cand[positions], loss[:, positions]
                    )
                if not np.isfinite(pred).all():
                    raise FloatingPointError(f"Nonfinite correction: {variant}")
                output[variant].append(np.asarray(pred, dtype=np.float32))
                masks[variant].append(mask)
                self.counts[variant]["eligible"] += int(mask.sum())
                self.counts[variant]["query_windows"] += len(mask)
        return {v: np.stack(p) for v, p in output.items()}, {
            v: np.stack(p) for v, p in masks.items()
        }

    def summary(self):
        return {
            v: dict(
                observed_windows=self.n,
                eligible_query_fraction=c["eligible"] / max(c["query_windows"], 1),
            )
            for v, c in self.counts.items()
        }
