"""Shared physical-block inference conditional on the five fitted seeds."""
import hashlib
import numpy as np
from .common import SYSTEMS, SEEDS
from gaugeformer.statistics import holm_adjust

PROTOCOL = "gf-extension-2026-09-14-multibackbone-tpami-v1"


def regime_name(family, fraction):
    return f"fewshot_{fraction}" if family == "fewshot" else family


def bootstrap(grid, family, fraction, proposed, baseline):
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(f"{PROTOCOL}|{family}|{fraction}|blocks".encode()).digest()[
                :8
            ],
            "big",
        )
    )
    replicates = np.zeros(20000)
    observed = []
    blocks_total = 0
    for system in SYSTEMS:
        deltas = []
        reference_ids = reference_weights = None
        for seed in SEEDS:
            left = grid[(family, fraction, proposed, seed, system)]["metric"]
            right = grid[(family, fraction, baseline, seed, system)]["metric"]
            if (
                left["channel_names"] != right["channel_names"]
                or left["windows"] != right["windows"]
            ):
                raise RuntimeError("Paired forecast panel differs")
            np.testing.assert_array_equal(
                left["training_standard_deviation"],
                right["training_standard_deviation"],
            )
            a = sorted(left["statistical_blocks"], key=lambda b: b["block_id"])
            b = sorted(right["statistical_blocks"], key=lambda b: b["block_id"])
            ids = [r["block_id"] for r in a]
            weights = np.array([r["elements_per_channel"] for r in a], dtype=np.float64)
            assert ids == [r["block_id"] for r in b]
            np.testing.assert_array_equal(
                weights, [r["elements_per_channel"] for r in b]
            )
            if reference_ids is None:
                reference_ids, reference_weights = ids, weights
            else:
                assert ids == reference_ids
                np.testing.assert_array_equal(weights, reference_weights)
            deltas.append(
                np.array([r["sd_nmae"] for r in b])
                - np.array([r["sd_nmae"] for r in a])
            )
            observed.append(right["sd_nmae"] - left["sd_nmae"])
        delta = np.mean(deltas, axis=0)
        n = len(delta)
        blocks_total += n
        for offset in range(0, 20000, 500):
            draws = rng.integers(0, n, size=(min(500, 20000 - offset), n))
            replicates[offset : offset + len(draws)] += (
                (delta[draws] * weights[draws]).sum(axis=1)
                / weights[draws].sum(axis=1)
                / len(SYSTEMS)
            )
    low, high = np.quantile(replicates, [0.025, 0.975])
    p = min(
        1.0,
        2
        * (
            min(np.count_nonzero(replicates <= 0), np.count_nonzero(replicates >= 0))
            + 1
        )
        / 20001,
    )
    return dict(
        regime=regime_name(family, fraction),
        proposed=proposed,
        baseline=baseline,
        gain=float(np.mean(observed)),
        ci_lower=float(low),
        ci_upper=float(high),
        p_value=p,
        physical_blocks=blocks_total,
        replicates=20000,
    )
