# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Small-batch safety guard.

Budgets are relative to the batch, so prediction error that averages out
over hundreds of episodes can push a small batch past its official cap,
which zeroes the whole tier. Below ``THRESHOLD`` episodes the operating
caps shrink linearly toward all-light, and the Premium K1 overlay is
skipped outright: the realized AX31 tail has produced single-item
cost explosions that no predicted-budget check can bound.

At or above ``THRESHOLD`` every value passes through unchanged, so
large-batch behaviour stays byte-identical to the frozen champion.
"""

from __future__ import annotations

THRESHOLD = 48


def batch_factor(n: int) -> float:
    """Linear ramp 0 -> 1 over [0, THRESHOLD]; exactly 1.0 at or above."""

    if n >= THRESHOLD:
        return 1.0
    if n <= 0:
        return 0.0
    return n / THRESHOLD


def effective_cap(
    predicted_cap: float, official_cap: float, n: int
) -> float:
    """Shrink the predicted cap's headroom above all-light by batch_factor."""

    if n >= THRESHOLD:
        return predicted_cap
    factor = batch_factor(n)
    shrunk = 1.0 + (predicted_cap - 1.0) * factor
    return min(shrunk, official_cap)


__all__ = (
    "THRESHOLD",
    "batch_factor",
    "effective_cap",
)
