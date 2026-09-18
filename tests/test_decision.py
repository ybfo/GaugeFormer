import numpy as np
import pytest
from gaugeformer import GaugeFormer, GaugeFormerConfig
from gaugeformer.local import expert_predictions, selected_experts
from gaugeformer.memory import GaugeDynamicsMemory, OnlineRouterConfig


def test_warmup_and_exact_backbone_fallback():
    memory = GaugeDynamicsMemory(OnlineRouterConfig(blend=0.5))
    candidates = np.array([[[10.0, 10.0]], [[2.0, 2.0]]])
    losses = np.array([[[5.0], [1.0]]] * 4)
    for _ in range(15):
        pred, choice, eligible = memory.update_and_predict(candidates, losses)
        np.testing.assert_array_equal(pred, candidates[0])
        assert not eligible.any()
    pred, choice, eligible = memory.update_and_predict(candidates, losses)
    np.testing.assert_array_equal(pred, [[6.0, 6.0]])
    assert eligible.all() and choice[0] == 1


def test_mean_advantage_and_repeated_wins_are_separate():
    memory = GaugeDynamicsMemory(OnlineRouterConfig(blend=0.5, warmup_windows=1))
    candidates = np.array([[[3.0]], [[1.0]]])
    losses = np.array([[[10.0], [0.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]]])
    _, _, eligible = memory.update_and_predict(candidates, losses)
    assert memory.cumulative_losses[1, 0] < memory.cumulative_losses[0, 0]
    assert not eligible[0]


def test_batched_rules_equal_independent_rules():
    x = np.random.default_rng(71).normal(size=(5, 4, 96))
    q = np.array([3, 0])
    a = selected_experts(x, q)
    b = np.stack([expert_predictions(row, q)[1] for row in x])
    np.testing.assert_array_equal(a, b)


def test_historical_predictions_see_only_their_prefix():
    from gaugeformer.method import context_backcast_losses

    x = np.random.default_rng(17).normal(size=(3, 96))
    q = np.array([0, 2])
    historical = np.zeros((4, 2, 24), np.float32)
    first = context_backcast_losses(x, q, historical)
    changed = x.copy()
    changed[:, 80:] += 100
    second = context_backcast_losses(changed, q, historical)
    # The first two tasks finish at 72 and 80; later observations cannot affect them.
    np.testing.assert_array_equal(first[:2], second[:2])


def test_positive_affine_and_permutation_consistency():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(4, 96))
    q = np.array([0, 2, 3])
    a = 2.0
    c = 5.0
    raw = rng.normal(size=(3, 24)).astype(np.float32)
    back = rng.normal(size=(4, 3, 24)).astype(np.float32)
    one = GaugeFormer(GaugeFormerConfig.for_variant("no_warmup"))
    two = GaugeFormer(GaugeFormerConfig.for_variant("no_warmup"))
    out = one.predict(x, q, raw, back)
    transformed = two.predict(a * x + c, q, a * raw + c, a * back + c)
    np.testing.assert_array_equal(out.eligible, transformed.eligible)
    np.testing.assert_allclose(
        transformed.prediction, a * out.prediction + c, atol=2e-6
    )
    perm = np.array([2, 0, 3, 1])
    new_q = np.argsort(perm)[q]
    three = GaugeFormer(GaugeFormerConfig.for_variant("no_warmup"))
    reordered = three.predict(x[perm], new_q, raw, back)
    np.testing.assert_array_equal(out.prediction, reordered.prediction)


def test_rejects_invalid_query_schema():
    with pytest.raises(ValueError, match="unique"):
        GaugeFormer().predict(
            np.zeros((2, 96)), np.array([0, 0]), np.zeros((2, 24)), np.zeros((4, 2, 24))
        )
