"""Cache constant metadata predicates; retain all GTM tensor arithmetic."""
from __future__ import annotations
import types

import torch


def null_rows(template, rows):
    if len(template) != 5 or any(
        t.device.type != "cpu" or t.numel() != 1 for t in template
    ):
        raise ValueError("Expected the five scalar CPU granularity templates")
    # Match the official float32 conversion and left-to-right Python sum.
    row = torch.stack(template).reshape(5).to(torch.float32)
    return list(range(rows)) if bool(sum(row) == 0) else []


def install(wrapper):
    if hasattr(wrapper, "_constant_granularity_stats"):
        return wrapper._constant_granularity_stats
    if wrapper.training or any(p.requires_grad for p in wrapper.parameters()):
        raise RuntimeError("Constant metadata cache is restricted to frozen inference")
    original = wrapper.predict_standardized
    stats = dict(
        optimization="gtm_constant_granularity_predicate_cache_v1",
        avoided_scalar_checks=0,
        metadata_templates=0,
        tensor_arithmetic_unchanged=True,
        cuda_graph_used=False,
    )
    cache = {}
    active = None

    def make_cached_null(original_null):
        def cached_null(self, granularity):
            if active is None:
                return original_null(granularity)
            rows, indices = active
            if tuple(granularity.shape) != (rows, 5):
                raise RuntimeError("GTM granularity rows differ from wrapper metadata")
            stats["avoided_scalar_checks"] += rows
            return granularity, indices

        return cached_null

    for layer in wrapper.model.decoder.layers:
        filt = layer.filter
        filt.replace_null = types.MethodType(make_cached_null(filt.replace_null), filt)

    def predict(self, scaled, store):
        nonlocal active
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Cached GTM path requires inference mode")
        seconds = float(store.metadata["sampling_interval_seconds"])
        rows = scaled.shape[0] * scaled.shape[-1]
        key = (seconds, rows)
        if key not in cache:
            # The published wrapper repeats one metadata vector for every
            # flattened channel/window row. Derive that same vector on CPU once.
            template = self.granularity(seconds, 1, torch.device("cpu"))
            cache[key] = null_rows(template, rows)
            stats["metadata_templates"] += 1
        if active is not None:
            raise RuntimeError("Concurrent calls on one GTM instance are unsupported")
        active = (rows, cache[key])
        try:
            return original(scaled, store)
        finally:
            active = None

    wrapper.predict_standardized = types.MethodType(predict, wrapper)
    wrapper._constant_granularity_stats = stats
    return stats
