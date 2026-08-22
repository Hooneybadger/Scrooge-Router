# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Two-price buy/settle allocation with exact-content-SHA group rollback.

Buy uses one predicted-cost surface; settle re-bills the same decisions on a
higher tail surface. Operating-cap breaches roll back whole content groups
in ascending marginal predicted uplift / settle incremental cost. Fast and
Balanced never buy K1. Predicted light totals are supplied by the caller;
this module never reads an actual light bill.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence, Tuple

import numpy as np

from research.lab.quality_heads import (
    allocate_k1_on_top,
    allocate_two_action,
    models_three_action,
    models_two_action,
)


RATIO_ATOL = 1e-12
_LIGHT = 0
_AX31 = 1
_K1 = 2


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def predicted_ratio(spend: float, light_total: float) -> float:
    if float(light_total) <= 0.0:
        raise ValueError("predicted light total must be positive")
    return float(spend) / float(light_total)


def selection_spend(models: Sequence[str], costs: np.ndarray) -> float:
    columns = np.asarray(
        [{"ax31-light": 0, "ax31": 1, "axk1-think": 2}[model] for model in models],
        dtype=np.int64,
    )
    matrix = np.asarray(costs, dtype=np.float64)
    return float(matrix[np.arange(matrix.shape[0]), columns].sum())


def group_members(group_keys: Sequence[str]) -> dict[str, list[int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        members[str(key)].append(int(index))
    return dict(members)


def _group_density(
    indexes: Sequence[int],
    upgrade: np.ndarray,
    pred_uplift: np.ndarray,
    settle_inc: np.ndarray,
) -> float:
    chosen = [index for index in indexes if bool(upgrade[index])]
    if not chosen:
        return float("inf")
    uplift = float(pred_uplift[chosen].sum())
    increment = float(settle_inc[chosen].sum())
    if increment <= 0.0:
        return float("inf") if uplift > 0.0 else 0.0
    return uplift / increment


def rollback_groups(
    upgrade: np.ndarray,
    pred_uplift: np.ndarray,
    settle_inc: np.ndarray,
    settle_current: np.ndarray,
    predicted_light_total: float,
    cap: float,
    group_keys: Sequence[str],
) -> np.ndarray:
    """Turn off whole content groups until settle spend / pred light <= cap."""

    chosen = np.asarray(upgrade, dtype=bool).copy()
    pred = _as_float(pred_uplift)
    increment = _as_float(settle_inc)
    current = _as_float(settle_current)
    members = group_members(group_keys)
    light = float(predicted_light_total)
    cap_f = float(cap)

    def spend() -> float:
        return float((current + increment * chosen.astype(np.float64)).sum())

    if predicted_ratio(spend(), light) <= cap_f + RATIO_ATOL:
        return chosen

    ranked = sorted(
        (
            (
                _group_density(indexes, chosen, pred, increment),
                key,
            )
            for key, indexes in members.items()
            if any(chosen[index] for index in indexes)
        ),
        key=lambda row: (row[0], row[1]),
    )
    for _density, key in ranked:
        if predicted_ratio(spend(), light) <= cap_f + RATIO_ATOL:
            break
        for index in members[key]:
            chosen[index] = False
    if predicted_ratio(spend(), light) > cap_f + RATIO_ATOL:
        chosen[:] = False
    return chosen


def allocate_single_price(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    costs: np.ndarray,
    predicted_light_total: float,
    cap: float,
    tie_keys: Sequence[str],
    *,
    k1_enabled: bool,
) -> Tuple[str, ...]:
    """Greedy density on one predicted-cost surface. Predicted light only."""

    upgrade_a = allocate_two_action(
        pred_qa, costs, predicted_light_total, cap, tie_keys
    )
    if not k1_enabled:
        return models_two_action(upgrade_a)
    upgrade_k = allocate_k1_on_top(
        pred_qk, upgrade_a, costs, predicted_light_total, cap, tie_keys
    )
    return models_three_action(upgrade_a, upgrade_k)


def allocate_two_price(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    buy_costs: np.ndarray,
    settle_costs: np.ndarray,
    predicted_light_total: float,
    cap: float,
    tie_keys: Sequence[str],
    group_keys: Sequence[str],
    *,
    k1_enabled: bool,
) -> Tuple[str, ...]:
    """Buy on q_buy, re-bill on q_settle, rollback exact-SHA groups if needed."""

    buy = np.asarray(buy_costs, dtype=np.float64)
    settle = np.asarray(settle_costs, dtype=np.float64)
    if buy.shape != settle.shape or buy.shape[1] != 3:
        raise ValueError("buy and settle costs must be (n, 3)")
    if len(group_keys) != buy.shape[0] or len(tie_keys) != buy.shape[0]:
        raise ValueError("group_keys and tie_keys must align with costs")

    upgrade_a = allocate_two_action(
        pred_qa, buy, predicted_light_total, cap, tie_keys
    )
    upgrade_a = rollback_groups(
        upgrade_a,
        pred_qa,
        settle[:, _AX31] - settle[:, _LIGHT],
        settle[:, _LIGHT],
        predicted_light_total,
        cap,
        group_keys,
    )
    if not k1_enabled:
        return models_two_action(upgrade_a)

    upgrade_k = allocate_k1_on_top(
        pred_qk, upgrade_a, buy, predicted_light_total, cap, tie_keys
    )
    current_after_a = np.where(upgrade_a, settle[:, _AX31], settle[:, _LIGHT])
    upgrade_k = rollback_groups(
        upgrade_k,
        pred_qk,
        settle[:, _K1] - settle[:, _AX31],
        current_after_a,
        predicted_light_total,
        cap,
        group_keys,
    )
    upgrade_k = upgrade_k & upgrade_a
    if predicted_ratio(
        float(
            np.where(
                upgrade_k,
                settle[:, _K1],
                np.where(upgrade_a, settle[:, _AX31], settle[:, _LIGHT]),
            ).sum()
        ),
        predicted_light_total,
    ) > float(cap) + RATIO_ATOL:
        upgrade_a = rollback_groups(
            upgrade_a,
            pred_qa,
            settle[:, _AX31] - settle[:, _LIGHT],
            settle[:, _LIGHT],
            predicted_light_total,
            cap,
            group_keys,
        )
        upgrade_k = upgrade_k & upgrade_a
    return models_three_action(upgrade_a, upgrade_k)


def allocate_all_tiers_single_price(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    costs: np.ndarray,
    predicted_light_total: float,
    caps: Mapping[str, float],
    tie_keys: Sequence[str],
) -> dict[str, Tuple[str, ...]]:
    return {
        "fast": allocate_single_price(
            pred_qa, pred_qk, costs, predicted_light_total, caps["fast"], tie_keys,
            k1_enabled=False,
        ),
        "balanced": allocate_single_price(
            pred_qa, pred_qk, costs, predicted_light_total, caps["balanced"], tie_keys,
            k1_enabled=False,
        ),
        "premium": allocate_single_price(
            pred_qa, pred_qk, costs, predicted_light_total, caps["premium"], tie_keys,
            k1_enabled=True,
        ),
    }


def allocate_all_tiers_two_price(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    buy_costs: np.ndarray,
    settle_by_tier: Mapping[str, np.ndarray],
    predicted_light_total: float,
    caps: Mapping[str, float],
    tie_keys: Sequence[str],
    group_keys: Sequence[str],
) -> dict[str, Tuple[str, ...]]:
    return {
        "fast": allocate_two_price(
            pred_qa, pred_qk, buy_costs, settle_by_tier["fast"],
            predicted_light_total, caps["fast"], tie_keys, group_keys,
            k1_enabled=False,
        ),
        "balanced": allocate_two_price(
            pred_qa, pred_qk, buy_costs, settle_by_tier["balanced"],
            predicted_light_total, caps["balanced"], tie_keys, group_keys,
            k1_enabled=False,
        ),
        "premium": allocate_two_price(
            pred_qa, pred_qk, buy_costs, settle_by_tier["premium"],
            predicted_light_total, caps["premium"], tie_keys, group_keys,
            k1_enabled=True,
        ),
    }


__all__ = (
    "RATIO_ATOL",
    "allocate_all_tiers_single_price",
    "allocate_all_tiers_two_price",
    "allocate_single_price",
    "allocate_two_price",
    "group_members",
    "predicted_ratio",
    "rollback_groups",
    "selection_spend",
)
