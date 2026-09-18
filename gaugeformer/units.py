"""Affine changes of measurement gauge.

The canonical value is ``x_c = scale * x + offset``.  This covers common
engineering conversions including kPa--Pa and degree Celsius--kelvin.  Physical
dimensions are represented separately as integer exponent vectors over the SI
basis (mass, length, time, current, temperature, amount, luminous intensity).
"""

from __future__ import annotations

import torch


def _broadcast_parameter(parameter: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    result = parameter
    while result.ndim < values.ndim:
        result = result.unsqueeze(-1)
    return result


def affine_to_canonical(
    values: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """Convert measurements from their declared gauge to canonical SI values."""

    scale = _broadcast_parameter(scale, values)
    offset = _broadcast_parameter(offset, values)
    if torch.any(scale == 0):
        raise ValueError("A unit scale must be non-zero")
    return scale * values + offset


def affine_from_canonical(
    values: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """Convert canonical SI values to a requested measurement gauge."""

    scale = _broadcast_parameter(scale, values)
    offset = _broadcast_parameter(offset, values)
    if torch.any(scale == 0):
        raise ValueError("A unit scale must be non-zero")
    return (values - offset) / scale
