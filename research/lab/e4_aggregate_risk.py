# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E4 — point buy with aggregate conformal rollback.

Quality stays locked at E1 ``baseline_continuous_uplift`` OOF. Buy prices are
OOF point predicted incremental costs. Item ``point*exp(zσ)`` is never a
unit price. Sigma enters only the rollback ranking of one candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.protocol import TIERS
from research.lab.e1_objectives import (
    BASELINE_NAME as E1_BASELINE,
    GATE_VIEW_DROP,
    GATE_VIEW_KINDS,
    canonical_json_text,
    current_quality_matrix,
    oof_candidate_predictions,
    sha256_text,
    stress_views,
    write_json_atomic,
)
from research.lab.e1c_regime_residual import FOLD_SEEDS, relabel_folds
from research.lab.e2_cost_uncertainty import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    COVERAGE_SLACK,
    FAMILY_GUARD_PATH,
    GATE_QUALITY_GAIN,
    GATE_QUALITY_SLACK,
    KAPPA_CLIP,
    LOG_EPS,
    MIXTURE_DRAWS,
    MIXTURE_N,
    MIXTURE_SEED,
    QUALITY_SIGNAL,
    STRESS_95_CAPS,
    attach_predicted_light,
    clamp_predicted_costs,
    cost_accuracy,
    deterministic_ratio_views,
    evaluate_allocation,
    family_mixture_ratios,
    grouped_ratio_bootstrap,
    oof_cost_surfaces,
    oof_quality_baseline,
    predicted_light_total,
    _bootstrap_with_caps,
    _hard_caps_ok,
    _json_float,
    _mixture_with_caps,
    _quantile,
)
from research.lab.e3_two_price import RATIO_ATOL, group_members, predicted_ratio, selection_spend
from research.lab.modeling import OFFICIAL_CAPS, sort_mapping
from research.lab.public_pool import ROOT, PublicPool, load_public_pool
from research.lab.quality_heads import (
    allocate_two_action,
    content_tie_keys,
    greedy_upgrade_mask,
    models_three_action,
    models_two_action,
)


EXPERIMENT = "e4-aggregate-risk"
REPORT_TYPE = "scrooge-e4-aggregate-risk-v1"
SCHEMA_VERSION = 1
BASELINE_NAME = "safe_family_point_baseline"
CANDIDATE_ORDER: Tuple[str, ...] = (
    "safe_family_point_baseline",
    "aggregate_conformal_rollback",
    "aggregate_sigma_priority",
)
BOUND_QUANTILE = 0.99
FINITE_SAMPLE_GUARD = 1.05
RESIDUAL_FAMILY = "other"
USED_ACTUAL_LIGHT_IN_ALLOCATOR = False
AUDIT_RELATIVE_PATH = "build/compare-e4-aggregate-risk/episode-audit.json"
BUDGET_BRAKE_PATH = (
    ROOT / "src" / "ossp_router" / "resources" / "budget-brake-router.v1.json"
)
_LIGHT = 0
_AX31 = 1
_K1 = 2

ROLLBACK_KEY_A = (
    "efficiency = sum_g Q / sum_g Δc_point ; rank = (efficiency, group_key); "
    "lowest efficiency groups roll back first"
)
ROLLBACK_KEY_B = (
    "efficiency = sum_g Q / sum_g Δc_point (same buy price as A; σ is not a "
    "price); risk_i = σ_{chosen model,i} + 1[family=other]; "
    "rank = (efficiency, -mean(risk_i), group_key); equal efficiency drops "
    "the higher-risk group first"
)
BOUND_RULE = (
    "Per outer fold and tier, collect inner-fold actual_selected_spend / "
    "point_predicted_selected_spend on outer-train only. "
    "bound = clip(max(q99(ratios), max(ratios)*1.05), 1.0, 3.0). "
    "Held-out rollback requires bound * point_spend <= operating_cap * "
    "predicted_light."
)


def _json_scope() -> dict[str, Any]:
    family = json.loads(FAMILY_GUARD_PATH.read_text(encoding="utf-8"))
    brake = json.loads(BUDGET_BRAKE_PATH.read_text(encoding="utf-8"))
    guard = family["family_guard"]
    multipliers = {str(key): float(value) for key, value in guard["multipliers"].items()}
    block = brake["budget_brake"]
    runtime_caps = {str(key): float(value) for key, value in family["predicted_caps"].items()}
    operating = {tier: float(STRESS_95_CAPS[tier]) for tier in TIERS}
    return {
        "family_guard_path": str(FAMILY_GUARD_PATH.relative_to(ROOT)),
        "family_guard_scope": str(guard["scope"]),
        "finite_sample_guard_ratio": float(
            brake["cost"]["calibration"]["finite_sample_guard_ratio"]
        ),
        "k1": {
            "count_cap": int(block["count_cap"]),
            "denylist_families": list(block["denylist_families"]),
            "runaway_absolute": float(block["runaway_absolute"]),
            "runaway_light_fraction": float(block["runaway_light_fraction"]),
        },
        "multipliers": multipliers,
        "operating_caps": operating,
        "provenance": (
            "multipliers and K1 limits read from current bundled artifacts. "
            "Allocator predicted caps are the predeclared 95% operating caps "
            "1.1875/1.90/3.80, not family-guard predicted_caps "
            f"{runtime_caps}."
        ),
        "residual_family": RESIDUAL_FAMILY,
        "runtime_predicted_caps": runtime_caps,
        "runtime_premium_brake_ratio": float(block["brake_ratio"]),
    }


RUNTIME_SCOPE = _json_scope()
OPERATING_CAPS = dict(RUNTIME_SCOPE["operating_caps"])
FAMILY_MULTIPLIERS = dict(RUNTIME_SCOPE["multipliers"])
K1_SCOPE = dict(RUNTIME_SCOPE["k1"])


CANDIDATE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "safe_family_point_baseline": {
        "allocation": "single_price_point",
        "buy": "OOF point incremental cost; Fast/Balanced AX31 increment * family multiplier",
        "family_multiplier_in_buy": True,
        "rollback": "none",
        "settle": "same point accounting; official scorer uses raw actual costs",
        "summary": (
            "Budget-valid comparator. Point buy + current family-guard "
            "increment philosophy (artifact other=2.5 on Fast/Balanced) + "
            "95% operating caps. Not a bit-identical runtime replay."
        ),
    },
    "aggregate_conformal_rollback": {
        "allocation": "point_buy_aggregate_rollback",
        "buy": "OOF point incremental cost; no family multiplier",
        "family_multiplier_in_buy": False,
        "rollback": ROLLBACK_KEY_A,
        "settle": "point predicted spend * conformal bound vs operating cap",
        "summary": (
            "Buy positive-uplift groups on point costs, then roll back whole "
            "content-SHA groups until the train-calibrated aggregate upper "
            "is inside the operating cap."
        ),
    },
    "aggregate_sigma_priority": {
        "allocation": "point_buy_aggregate_rollback",
        "buy": "same OOF point incremental cost as aggregate_conformal_rollback",
        "family_multiplier_in_buy": False,
        "rollback": ROLLBACK_KEY_B,
        "settle": "same aggregate bound as aggregate_conformal_rollback",
        "summary": (
            "Same buy prices and same bound. Sigma and residual-family "
            "indicators change rollback order only."
        ),
    },
}


def apply_family_increment_multiplier(
    costs: np.ndarray,
    families: Sequence[str],
    multipliers: Mapping[str, float] = FAMILY_MULTIPLIERS,
) -> np.ndarray:
    """Inflate the AX31-from-light increment for known residual families."""

    out = np.asarray(costs, dtype=np.float64).copy()
    for index, family in enumerate(families):
        multiplier = float(multipliers.get(family, 1.0))
        if multiplier <= 1.0:
            continue
        light = float(out[index, _LIGHT])
        increment = max(float(out[index, _AX31]) - light, 0.0)
        out[index, _AX31] = light + increment * multiplier
    return clamp_predicted_costs(out)


def item_risk(
    sigma: np.ndarray,
    families: Sequence[str],
    models: Sequence[str],
) -> np.ndarray:
    columns = np.asarray(
        [{"ax31-light": 0, "ax31": 1, "axk1-think": 2}[model] for model in models],
        dtype=np.int64,
    )
    values = np.asarray(sigma, dtype=np.float64)[np.arange(len(models)), columns]
    residual = np.asarray(
        [1.0 if family == RESIDUAL_FAMILY else 0.0 for family in families],
        dtype=np.float64,
    )
    return values + residual


def k1_eligible(
    upgrade_a: np.ndarray,
    families: Sequence[str],
    costs: np.ndarray,
    item_light: np.ndarray,
    *,
    scope: Mapping[str, Any] = K1_SCOPE,
) -> np.ndarray:
    increment = np.asarray(costs[:, _K1] - costs[:, _AX31], dtype=np.float64)
    light = np.asarray(item_light, dtype=np.float64)
    denylist = set(scope["denylist_families"])
    allowed = np.asarray(upgrade_a, dtype=bool).copy()
    abs_cap = float(scope["runaway_absolute"])
    frac_cap = float(scope["runaway_light_fraction"])
    for index, family in enumerate(families):
        if not allowed[index]:
            continue
        if family in denylist:
            allowed[index] = False
            continue
        inc = float(increment[index])
        if inc > abs_cap:
            allowed[index] = False
            continue
        if inc > frac_cap * max(float(light[index]), LOG_EPS):
            allowed[index] = False
    return allowed


def trim_k1_count(
    upgrade_k: np.ndarray,
    pred_qk: np.ndarray,
    tie_keys: Sequence[str],
    *,
    count_cap: int = int(K1_SCOPE["count_cap"]),
) -> np.ndarray:
    chosen = np.flatnonzero(upgrade_k)
    if chosen.size <= int(count_cap):
        return np.asarray(upgrade_k, dtype=bool)
    ranked = sorted(
        (int(index) for index in chosen),
        key=lambda index: (-float(pred_qk[index]), tie_keys[index]),
    )
    keep = set(ranked[: int(count_cap)])
    out = np.zeros(upgrade_k.shape[0], dtype=bool)
    for index in keep:
        out[index] = True
    return out


def allocate_k1_scoped(
    pred_qk: np.ndarray,
    upgrade_a: np.ndarray,
    costs: np.ndarray,
    predicted_light_total: float,
    cap: float,
    tie_keys: Sequence[str],
    families: Sequence[str],
    item_light: np.ndarray,
) -> np.ndarray:
    eligible = k1_eligible(upgrade_a, families, costs, item_light)
    increment = costs[:, _K1] - costs[:, _AX31]
    current = np.where(upgrade_a, costs[:, _AX31], costs[:, _LIGHT])
    upgrade_k = greedy_upgrade_mask(
        pred_qk,
        increment,
        current,
        predicted_light_total,
        cap,
        eligible=eligible,
        tie_keys=tie_keys,
    )
    upgrade_k = trim_k1_count(upgrade_k, pred_qk, tie_keys)
    return upgrade_k & np.asarray(upgrade_a, dtype=bool)


def _group_efficiency(
    indexes: Sequence[int],
    chosen: np.ndarray,
    uplift: np.ndarray,
    increment: np.ndarray,
) -> float:
    members = [index for index in indexes if bool(chosen[index])]
    if not members:
        return float("inf")
    uplift_sum = float(uplift[members].sum())
    increment_sum = float(increment[members].sum())
    if increment_sum <= 0.0:
        return float("inf") if uplift_sum > 0.0 else 0.0
    return uplift_sum / increment_sum


def _group_risk(indexes: Sequence[int], chosen: np.ndarray, risk: np.ndarray) -> float:
    members = [index for index in indexes if bool(chosen[index])]
    if not members:
        return 0.0
    return float(np.mean(risk[members]))


def rollback_until_bound(
    chosen: np.ndarray,
    uplift: np.ndarray,
    increment: np.ndarray,
    current: np.ndarray,
    predicted_light_total: float,
    cap: float,
    bound: float,
    group_keys: Sequence[str],
    *,
    risk: Optional[np.ndarray] = None,
    sigma_priority: bool = False,
) -> Tuple[np.ndarray, int]:
    """Drop whole content-SHA groups until bound * point_spend <= cap * light."""

    selected = np.asarray(chosen, dtype=bool).copy()
    light = float(predicted_light_total)
    cap_f = float(cap)
    bound_f = float(bound)
    members = group_members(group_keys)

    def pred_spend() -> float:
        return float((current + increment * selected.astype(np.float64)).sum())

    def upper_ratio() -> float:
        return predicted_ratio(bound_f * pred_spend(), light)

    if upper_ratio() <= cap_f + RATIO_ATOL:
        return selected, 0

    def rank_key(key: str) -> tuple[Any, ...]:
        efficiency = _group_efficiency(members[key], selected, uplift, increment)
        if sigma_priority:
            if risk is None:
                raise ValueError("sigma-priority rollback requires risk")
            return (efficiency, -_group_risk(members[key], selected, risk), key)
        return (efficiency, key)

    ranked = sorted(
        (key for key, indexes in members.items() if any(selected[index] for index in indexes)),
        key=rank_key,
    )
    n_rolled = 0
    for key in ranked:
        if upper_ratio() <= cap_f + RATIO_ATOL:
            break
        for index in members[key]:
            selected[index] = False
        n_rolled += 1
    if upper_ratio() > cap_f + RATIO_ATOL:
        if bool(selected.any()):
            n_rolled += 1
        selected[:] = False
    return selected, n_rolled


def conformal_bound(ratios: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(list(ratios), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        bound = float(np.clip(FINITE_SAMPLE_GUARD, KAPPA_CLIP[0], KAPPA_CLIP[1]))
        return {
            "bound": _json_float(bound),
            "finite_sample_upper": _json_float(FINITE_SAMPLE_GUARD),
            "n": 0,
            "q99": None,
            "ratios": [],
        }
    q99 = float(_quantile(values, BOUND_QUANTILE))
    finite = float(values.max() * FINITE_SAMPLE_GUARD)
    bound = float(np.clip(max(q99, finite), KAPPA_CLIP[0], KAPPA_CLIP[1]))
    return {
        "bound": _json_float(bound),
        "finite_sample_upper": _json_float(finite),
        "n": int(values.size),
        "q99": _json_float(q99),
        "ratios": [_json_float(value) for value in values],
    }


def _buy_masks(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    costs: np.ndarray,
    predicted_light_total: float,
    cap: float,
    tie_keys: Sequence[str],
    families: Sequence[str],
    *,
    k1_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    upgrade_a = allocate_two_action(
        pred_qa, costs, predicted_light_total, cap, tie_keys
    )
    if not k1_enabled:
        return upgrade_a, np.zeros(upgrade_a.shape[0], dtype=bool)
    upgrade_k = allocate_k1_scoped(
        pred_qk,
        upgrade_a,
        costs,
        predicted_light_total,
        cap,
        tie_keys,
        families,
        costs[:, _LIGHT],
    )
    return upgrade_a, upgrade_k


@dataclass(frozen=True)
class TierAllocation:
    bound: float | None
    models: Tuple[str, ...]
    n_ax31_bought: int
    n_ax31_final: int
    n_groups_rolled: int
    n_k1_bought: int
    n_k1_final: int
    pred_spend: float
    pred_upper: float | None


def allocate_tier(
    name: str,
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    buy_costs: np.ndarray,
    predicted_light_total: float,
    cap: float,
    tie_keys: Sequence[str],
    group_keys: Sequence[str],
    families: Sequence[str],
    *,
    k1_enabled: bool,
    bound: float | None,
    sigma: Optional[np.ndarray] = None,
    sigma_priority: bool = False,
) -> TierAllocation:
    costs = np.asarray(buy_costs, dtype=np.float64)
    upgrade_a, upgrade_k = _buy_masks(
        pred_qa,
        pred_qk,
        costs,
        predicted_light_total,
        cap,
        tie_keys,
        families,
        k1_enabled=k1_enabled,
    )
    n_ax31_bought = int(upgrade_a.sum())
    n_k1_bought = int(upgrade_k.sum())
    n_groups_rolled = 0
    if bound is not None:
        risk_a = None
        if sigma_priority:
            prelim = models_two_action(upgrade_a)
            risk_a = item_risk(sigma, families, prelim)
        upgrade_a, rolled_a = rollback_until_bound(
            upgrade_a,
            pred_qa,
            costs[:, _AX31] - costs[:, _LIGHT],
            costs[:, _LIGHT],
            predicted_light_total,
            cap,
            bound,
            group_keys,
            risk=risk_a,
            sigma_priority=sigma_priority,
        )
        n_groups_rolled += rolled_a
        if k1_enabled:
            upgrade_k = upgrade_k & upgrade_a
            upgrade_k = allocate_k1_scoped(
                pred_qk,
                upgrade_a,
                costs,
                predicted_light_total,
                cap,
                tie_keys,
                families,
                costs[:, _LIGHT],
            )
            risk_k = None
            if sigma_priority:
                prelim_k = models_three_action(upgrade_a, upgrade_k)
                risk_k = item_risk(sigma, families, prelim_k)
            current = np.where(upgrade_a, costs[:, _AX31], costs[:, _LIGHT])
            upgrade_k, rolled_k = rollback_until_bound(
                upgrade_k,
                pred_qk,
                costs[:, _K1] - costs[:, _AX31],
                current,
                predicted_light_total,
                cap,
                bound,
                group_keys,
                risk=risk_k,
                sigma_priority=sigma_priority,
            )
            upgrade_k = upgrade_k & upgrade_a
            n_groups_rolled += rolled_k
            still = bound * float(
                np.where(
                    upgrade_k,
                    costs[:, _K1],
                    np.where(upgrade_a, costs[:, _AX31], costs[:, _LIGHT]),
                ).sum()
            )
            if predicted_ratio(still, predicted_light_total) > float(cap) + RATIO_ATOL:
                upgrade_a, rolled_again = rollback_until_bound(
                    upgrade_a,
                    pred_qa,
                    costs[:, _AX31] - costs[:, _LIGHT],
                    costs[:, _LIGHT],
                    predicted_light_total,
                    cap,
                    bound,
                    group_keys,
                    risk=risk_a,
                    sigma_priority=sigma_priority,
                )
                upgrade_k = upgrade_k & upgrade_a
                n_groups_rolled += rolled_again
    models = (
        models_three_action(upgrade_a, upgrade_k)
        if k1_enabled
        else models_two_action(upgrade_a)
    )
    spend = selection_spend(models, costs)
    upper = None if bound is None else float(bound) * spend
    return TierAllocation(
        bound=None if bound is None else float(bound),
        models=models,
        n_ax31_bought=n_ax31_bought,
        n_ax31_final=int(sum(model != "ax31-light" for model in models)),
        n_groups_rolled=n_groups_rolled,
        n_k1_bought=n_k1_bought,
        n_k1_final=int(sum(model == "axk1-think" for model in models)),
        pred_spend=spend,
        pred_upper=upper,
    )


def allocate_candidate(
    name: str,
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    point: np.ndarray,
    alloc_light: np.ndarray,
    predicted_light: float,
    families: Sequence[str],
    tie_keys: Sequence[str],
    group_keys: Sequence[str],
    bounds: Mapping[str, float],
    sigma: np.ndarray,
) -> dict[str, TierAllocation]:
    if float(predicted_light) <= 0.0:
        raise ValueError("allocator refused an actual or non-positive light total")
    if name == BASELINE_NAME:
        guarded = apply_family_increment_multiplier(point, families)
        buy = attach_predicted_light(guarded, alloc_light)
        use_bound = False
        sigma_priority = False
    else:
        buy = attach_predicted_light(point, alloc_light)
        use_bound = True
        sigma_priority = name == "aggregate_sigma_priority"
    out = {}
    for tier in TIERS:
        out[tier] = allocate_tier(
            name,
            pred_qa,
            pred_qk,
            buy,
            predicted_light,
            OPERATING_CAPS[tier],
            tie_keys,
            group_keys,
            families,
            k1_enabled=tier == "premium",
            bound=bounds[tier] if use_bound else None,
            sigma=sigma,
            sigma_priority=sigma_priority,
        )
    return out


def _models_by_tier(alloc: Mapping[str, TierAllocation]) -> dict[str, Tuple[str, ...]]:
    return {tier: alloc[tier].models for tier in TIERS}


def _inner_quality(
    episodes: Sequence[Any], scores: np.ndarray, folds: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    features = current_quality_matrix(episodes)
    predicted = oof_candidate_predictions(features, scores, folds)
    return predicted[E1_BASELINE]


def calibrate_bounds(
    pool: PublicPool,
    bundle: Mapping[str, Any],
    *,
    actual_costs: Optional[np.ndarray] = None,
) -> dict[int, dict[str, Any]]:
    """Tier bounds from outer-train inner OOF selections only."""

    actual = (
        bundle["actual_costs"]
        if actual_costs is None
        else np.asarray(actual_costs, dtype=np.float64)
    )
    rows: dict[int, dict[str, Any]] = {}
    for block in bundle["inner_train"]:
        fold = int(block["fold"])
        train_index = np.asarray(block["train_index"], dtype=np.int64)
        point = np.asarray(block["point"], dtype=np.float64)
        scale = float(block["denom_scale"])
        alloc_light = point[:, _LIGHT] * scale
        buy = attach_predicted_light(point, alloc_light)
        episodes = tuple(pool.episodes[index] for index in train_index)
        families = tuple(pool.families[index] for index in train_index)
        texts = tuple(pool.texts[index] for index in train_index)
        groups = tuple(pool.exact_keys[index] for index in train_index)
        ties = content_tie_keys(texts)
        inner_folds = tuple(int(pool.folds[index]) for index in train_index)
        pred_qa, pred_qk = _inner_quality(episodes, pool.scores[train_index], inner_folds)
        actual_train = actual[train_index]
        fold_ids = np.asarray(inner_folds, dtype=np.int64)
        tier_rows = {}
        for tier in TIERS:
            ratios = []
            for inner in sorted(np.unique(fold_ids)):
                mask = fold_ids == inner
                if not np.any(mask):
                    continue
                indexes = np.flatnonzero(mask)
                local_light = float(alloc_light[mask].sum())
                if local_light <= 0.0:
                    continue
                local = allocate_tier(
                    "inner_buy",
                    pred_qa[mask],
                    pred_qk[mask],
                    buy[mask],
                    local_light,
                    OPERATING_CAPS[tier],
                    tuple(ties[index] for index in indexes),
                    tuple(groups[index] for index in indexes),
                    tuple(families[index] for index in indexes),
                    k1_enabled=tier == "premium",
                    bound=None,
                    sigma=None,
                    sigma_priority=False,
                )
                pred = local.pred_spend
                act = selection_spend(local.models, actual_train[mask])
                if pred > LOG_EPS:
                    ratios.append(act / pred)
            fitted = conformal_bound(ratios)
            fitted["tier"] = tier
            tier_rows[tier] = fitted
        rows[fold] = tier_rows
    return rows


def _assert_k1_contracts(models_by_tier: Mapping[str, Sequence[str]]) -> None:
    if any(model == "axk1-think" for model in models_by_tier["fast"]):
        raise RuntimeError("Fast selected K1")
    if any(model == "axk1-think" for model in models_by_tier["balanced"]):
        raise RuntimeError("Balanced selected K1")
    n_k1 = sum(model == "axk1-think" for model in models_by_tier["premium"])
    if n_k1 > int(K1_SCOPE["count_cap"]):
        raise RuntimeError("Premium K1 exceeded count_cap")


def _seed_safety(
    name: str,
    pooled: Mapping[str, Any],
    per_fold: Sequence[Mapping[str, Any]],
    quality_views: Sequence[Mapping[str, Any]],
    ratio_views: Sequence[Mapping[str, Any]],
    stress: Mapping[str, Any],
    coverage: Mapping[str, Any],
    baseline_q: float | None,
    baseline_valid: bool,
) -> dict[str, Any]:
    view_fail = [
        f"{row['kind']}:{row['name']}"
        for row in quality_views
        if row["kind"] in GATE_VIEW_KINDS
        and row["gated"]
        and row["delta"] is not None
        and row["delta"] < -GATE_VIEW_DROP
    ]
    ratio_fail = [
        f"{row['kind']}:{row['name']}"
        for row in ratio_views
        if row["kind"] in GATE_VIEW_KINDS and row["hard_cap_overrun"]
    ]
    fold_caps = all(_hard_caps_ok(row) for row in per_fold)
    pooled_caps = _hard_caps_ok(pooled)
    stress_fail = []
    for kind in ("bootstrap", "family_mixture"):
        for tier in TIERS:
            block = stress[kind][tier]
            q999 = block.get("q99_9")
            if q999 is None or float(q999) >= STRESS_95_CAPS[tier]:
                stress_fail.append(f"{kind}_q99_9:{tier}")
    coverage_ok = True
    if name != BASELINE_NAME:
        coverage_ok = all(coverage[tier]["slack_ok"] for tier in TIERS)
    quality = float(pooled["quality_weighted"])
    delta = None if baseline_q is None else quality - baseline_q
    quality_ok = True
    if baseline_valid and delta is not None:
        quality_ok = delta >= -GATE_QUALITY_SLACK
    preferred = bool(delta is not None and delta >= GATE_QUALITY_GAIN)
    independent = bool(
        pooled_caps and fold_caps and not ratio_fail and not stress_fail and coverage_ok
    )
    passed = bool(independent and not view_fail and quality_ok)
    if name == BASELINE_NAME:
        passed = independent
    return {
        "candidate": name,
        "coverage_ok": coverage_ok,
        "delta_vs_safe_baseline": None if delta is None else _json_float(delta),
        "fold_caps_ok": fold_caps,
        "independent_safety_ok": independent,
        "pass": passed,
        "pooled_caps_ok": pooled_caps,
        "preferred_quality_gain": preferred,
        "quality_ok": quality_ok,
        "quality_weighted": _json_float(quality),
        "ratio_view_failures": ratio_fail,
        "stress_failures": stress_fail,
        "view_failures": view_fail,
    }


def evaluate_seed(pool: PublicPool, *, actual_costs: Optional[np.ndarray] = None) -> dict[str, Any]:
    pred_qa, pred_qk = oof_quality_baseline(pool)
    bundle = oof_cost_surfaces(pool)
    actual = (
        bundle["actual_costs"]
        if actual_costs is None
        else np.asarray(actual_costs, dtype=np.float64)
    )
    bounds = calibrate_bounds(pool, bundle, actual_costs=actual)
    alloc_light = bundle["point"][:, _LIGHT] * bundle["denom_scale"]
    pred_light = predicted_light_total(bundle["point"][:, _LIGHT], bundle["denom_scale"])
    if pred_light <= 0.0:
        raise RuntimeError("predicted light total is not positive")
    ties = content_tie_keys(pool.texts)
    groups = pool.exact_keys
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    heldout_bounds = {
        tier: {
            int(fold): float(bounds[int(fold)][tier]["bound"]) for fold in bounds
        }
        for tier in TIERS
    }
    # OOF decisions use the bound trained when that fold was held out.
    models_by_name: dict[str, dict[str, Tuple[str, ...]]] = {}
    allocations: dict[str, dict[str, TierAllocation]] = {}
    for name in CANDIDATE_ORDER:
        models = {tier: ["ax31-light"] * len(pool.episodes) for tier in TIERS}
        rolled = {tier: 0 for tier in TIERS}
        bought = {tier: {"ax31": 0, "k1": 0} for tier in TIERS}
        final = {tier: {"ax31": 0, "k1": 0} for tier in TIERS}
        for fold in range(int(fold_ids.max()) + 1):
            mask = fold_ids == fold
            indexes = [index for index, flag in enumerate(mask) if flag]
            local_bounds = {tier: heldout_bounds[tier][fold] for tier in TIERS}
            local_light = predicted_light_total(
                bundle["point"][mask, _LIGHT], bundle["denom_scale"][mask]
            )
            local = allocate_candidate(
                name,
                pred_qa[mask],
                pred_qk[mask],
                bundle["point"][mask],
                alloc_light[mask],
                local_light,
                tuple(pool.families[index] for index in indexes),
                tuple(ties[index] for index in indexes),
                tuple(groups[index] for index in indexes),
                local_bounds,
                bundle["sigma"][mask],
            )
            for tier in TIERS:
                for offset, index in enumerate(indexes):
                    models[tier][index] = local[tier].models[offset]
                rolled[tier] += local[tier].n_groups_rolled
                bought[tier]["ax31"] += local[tier].n_ax31_bought
                bought[tier]["k1"] += local[tier].n_k1_bought
                final[tier]["ax31"] += local[tier].n_ax31_final
                final[tier]["k1"] += local[tier].n_k1_final
        models_t = {tier: tuple(models[tier]) for tier in TIERS}
        _assert_k1_contracts(models_t)
        models_by_name[name] = models_t
        allocations[name] = {
            tier: TierAllocation(
                bound=None
                if name == BASELINE_NAME
                else float(np.mean([heldout_bounds[tier][fold] for fold in bounds])),
                models=models_t[tier],
                n_ax31_bought=bought[tier]["ax31"],
                n_ax31_final=final[tier]["ax31"],
                n_groups_rolled=rolled[tier],
                n_k1_bought=bought[tier]["k1"],
                n_k1_final=final[tier]["k1"],
                pred_spend=selection_spend(
                    models_t[tier], attach_predicted_light(bundle["point"], alloc_light)
                ),
                pred_upper=None,
            )
            for tier in TIERS
        }

    results = {}
    for name in CANDIDATE_ORDER:
        models = models_by_name[name]
        pooled = evaluate_allocation(pool, models)
        per_fold = []
        for fold in range(int(fold_ids.max()) + 1):
            indexes = [index for index, value in enumerate(pool.folds) if value == fold]
            local_models = {
                tier: tuple(models[tier][index] for index in indexes) for tier in TIERS
            }
            local = evaluate_allocation(pool, local_models, indexes=indexes)
            per_fold.append(
                {
                    "fold": fold,
                    "n": len(indexes),
                    "official_final_score": local["official_final_score"],
                    "quality_weighted": local["quality_weighted"],
                    "tiers": local["tiers"],
                }
            )
        point_buy = attach_predicted_light(bundle["point"], alloc_light)
        coverage = {}
        for tier in TIERS:
            if name == BASELINE_NAME:
                coverage[tier] = {
                    "empirical": None,
                    "n_folds": 0,
                    "nominal": BOUND_QUANTILE,
                    "shortfall": 0.0,
                    "slack_ok": True,
                    "skipped": "baseline has no conformal bound",
                }
            else:
                fold_bounds = [heldout_bounds[tier][fold] for fold in sorted(bounds)]
                # coverage uses each fold's own bound
                covered = []
                for fold in sorted(bounds):
                    mask = fold_ids == fold
                    indexes = [index for index, flag in enumerate(mask) if flag]
                    subset = [models[tier][index] for index in indexes]
                    pred = selection_spend(subset, point_buy[mask])
                    act = selection_spend(subset, actual[mask])
                    covered.append(bool(act <= heldout_bounds[tier][fold] * pred + 1e-12))
                empirical = float(np.mean(covered)) if covered else 1.0
                shortfall = max(0.0, BOUND_QUANTILE - empirical)
                coverage[tier] = {
                    "empirical": _json_float(empirical),
                    "fold_bounds": [_json_float(value) for value in fold_bounds],
                    "n_folds": len(covered),
                    "nominal": BOUND_QUANTILE,
                    "shortfall": _json_float(shortfall),
                    "slack_ok": bool(shortfall <= COVERAGE_SLACK + 1e-15),
                }
        results[name] = {
            "coverage": coverage,
            "definition": dict(CANDIDATE_DEFINITIONS[name]),
            "name": name,
            "per_fold": per_fold,
            "pooled": pooled,
            "predicted_light_total": _json_float(pred_light),
            "rollback": {
                tier: {
                    "bound_mean": allocations[name][tier].bound,
                    "n_ax31_bought": allocations[name][tier].n_ax31_bought,
                    "n_ax31_final": allocations[name][tier].n_ax31_final,
                    "n_groups_rolled": allocations[name][tier].n_groups_rolled,
                    "n_k1_bought": allocations[name][tier].n_k1_bought,
                    "n_k1_final": allocations[name][tier].n_k1_final,
                    "pred_spend": _json_float(allocations[name][tier].pred_spend),
                }
                for tier in TIERS
            },
        }

    quality_views = {
        name: stress_views(pool, models_by_name[BASELINE_NAME], models_by_name[name])
        for name in CANDIDATE_ORDER
    }
    ratio_views = {
        name: deterministic_ratio_views(pool, models_by_name[name])
        for name in CANDIDATE_ORDER
    }
    stress = {}
    for name in CANDIDATE_ORDER:
        bootstrap = {}
        mixture = {}
        for tier in TIERS:
            block = grouped_ratio_bootstrap(
                models_by_name[name][tier],
                actual,
                actual[:, _LIGHT],
                pool.group_keys,
                draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
            bootstrap[tier] = _bootstrap_with_caps(
                block, OFFICIAL_CAPS[tier], STRESS_95_CAPS[tier]
            )
            mix = family_mixture_ratios(
                models_by_name[name][tier],
                actual,
                actual[:, _LIGHT],
                pool.families,
                draws=MIXTURE_DRAWS,
                seed=MIXTURE_SEED,
                batch=MIXTURE_N,
            )
            mixture[tier] = _mixture_with_caps(mix, OFFICIAL_CAPS[tier], STRESS_95_CAPS[tier])
        stress[name] = {"bootstrap": bootstrap, "family_mixture": mixture}

    baseline_row = _seed_safety(
        BASELINE_NAME,
        results[BASELINE_NAME]["pooled"],
        results[BASELINE_NAME]["per_fold"],
        quality_views[BASELINE_NAME],
        ratio_views[BASELINE_NAME],
        stress[BASELINE_NAME],
        results[BASELINE_NAME]["coverage"],
        None,
        False,
    )
    baseline_valid = bool(baseline_row["independent_safety_ok"])
    baseline_q = float(results[BASELINE_NAME]["pooled"]["quality_weighted"])
    gate_rows = [baseline_row]
    for name in CANDIDATE_ORDER:
        if name == BASELINE_NAME:
            continue
        gate_rows.append(
            _seed_safety(
                name,
                results[name]["pooled"],
                results[name]["per_fold"],
                quality_views[name],
                ratio_views[name],
                stress[name],
                results[name]["coverage"],
                baseline_q if baseline_valid else None,
                baseline_valid,
            )
        )
    calibration = {
        str(fold): {tier: dict(bounds[fold][tier]) for tier in TIERS}
        for fold in sorted(bounds)
    }
    return {
        "actual_costs_sha_note": "held-out actual costs enter scoring/stress only",
        "baseline_valid": baseline_valid,
        "calibration": calibration,
        "cost_accuracy": cost_accuracy(bundle),
        "fold_seed": int(pool.identity["fold_seed"]),
        "gate_rows": gate_rows,
        "models_by_name": models_by_name,
        "pred_qa": pred_qa,
        "pred_qk": pred_qk,
        "predicted_light_total": _json_float(pred_light),
        "quality_views": quality_views,
        "ratio_views": ratio_views,
        "results": results,
        "sigma": bundle["sigma"],
        "stress": stress,
        "used_actual_light_in_allocator": USED_ACTUAL_LIGHT_IN_ALLOCATOR,
    }


def promotion_gate(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_valid_all = all(row["baseline_valid"] for row in seed_reports)
    by_name = {name: [] for name in CANDIDATE_ORDER}
    for seed in seed_reports:
        for row in seed["gate_rows"]:
            by_name[row["candidate"]].append(row)
    rows = []
    for name in CANDIDATE_ORDER:
        seed_rows = by_name[name]
        worst_q = min(float(row["quality_weighted"]) for row in seed_rows)
        deltas = [
            row["delta_vs_safe_baseline"]
            for row in seed_rows
            if row["delta_vs_safe_baseline"] is not None
        ]
        worst_delta = min(deltas) if deltas else None
        all_pass = all(row["pass"] for row in seed_rows)
        if name != BASELINE_NAME and not baseline_valid_all:
            all_pass = False
        rows.append(
            {
                "candidate": name,
                "pass_all_seeds": all_pass,
                "seed_rows": seed_rows,
                "worst_delta": None if worst_delta is None else _json_float(worst_delta),
                "worst_quality": _json_float(worst_q),
            }
        )
    winners = [
        row["candidate"]
        for row in rows
        if row["pass_all_seeds"] and row["candidate"] != BASELINE_NAME
    ]
    recommended = None
    if baseline_valid_all and winners:
        recommended = max(
            winners,
            key=lambda name: (
                any(
                    seed["preferred_quality_gain"]
                    for seed in by_name[name]
                ),
                float(np.mean([seed["quality_weighted"] for seed in by_name[name]])),
                name,
            ),
        )
    decision = (
        "record-e4-no-valid-reference"
        if not baseline_valid_all
        else (
            f"record-e4-promote-{recommended}"
            if recommended
            else "record-e4-no-promote"
        )
    )
    return {
        "baseline": BASELINE_NAME,
        "baseline_valid_all_seeds": baseline_valid_all,
        "candidates": rows,
        "decision": decision,
        "passed": bool(recommended),
        "recommended": recommended,
        "thresholds": {
            "bound_quantile": BOUND_QUANTILE,
            "coverage_slack": COVERAGE_SLACK,
            "family_mixture_gated": True,
            "operating_caps": OPERATING_CAPS,
            "quality_gain_preferred": GATE_QUALITY_GAIN,
            "quality_slack": GATE_QUALITY_SLACK,
            "slice_ratio_hard_cap": True,
            "stress_95": STRESS_95_CAPS,
            "view_drop": GATE_VIEW_DROP,
            "worst_seed_not_mean": True,
        },
    }


def episode_audit_document(
    seed_pools: Mapping[int, PublicPool],
    seed_reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    blocks = {}
    for seed, pool in seed_pools.items():
        report = seed_reports[seed]
        rows = []
        for index, episode in enumerate(pool.episodes):
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "family": pool.families[index],
                    "fold": int(pool.folds[index]),
                    "group_key": pool.exact_keys[index],
                    "language": pool.languages[index],
                    "length_view": pool.length_views[index],
                    "pred_qa": _json_float(report["pred_qa"][index]),
                    "pred_qk": _json_float(report["pred_qk"][index]),
                    "selected": {
                        name: {
                            tier: str(report["models_by_name"][name][tier][index])
                            for tier in TIERS
                        }
                        for name in CANDIDATE_ORDER
                    },
                    "sigma": [_json_float(value) for value in report["sigma"][index]],
                    "split": pool.split_labels[index],
                }
            )
        blocks[str(seed)] = {"n_rows": len(rows), "rows": rows}
    return {
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "quality_signal": QUALITY_SIGNAL,
        "seeds": blocks,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "allocator": report["allocator"],
            "audit": report["audit"],
            "candidates": report["candidates"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "feature": report["feature"],
            "fold_seeds": report["fold_seeds"],
            "identity": report["identity"],
            "limitations": report["limitations"],
            "promotion_gate": report["promotion_gate"],
            "quality_signal": report["quality_signal"],
            "report_type": report["report_type"],
            "runtime_scope": report["runtime_scope"],
            "schema_version": report["schema_version"],
            "seed_results": report["seed_results"],
        }
    )


def decision_core_sha256(report: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        decision_core_payload(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(encoded)


def _slim_seed(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_valid": report["baseline_valid"],
        "calibration": report["calibration"],
        "cost_accuracy": report["cost_accuracy"],
        "fold_seed": report["fold_seed"],
        "gate_rows": report["gate_rows"],
        "predicted_light_total": report["predicted_light_total"],
        "quality_views": report["quality_views"],
        "ratio_views": report["ratio_views"],
        "results": report["results"],
        "stress": {
            name: {
                "bootstrap": {
                    tier: {
                        key: value
                        for key, value in block.items()
                        if key != "samples"
                    }
                    for tier, block in report["stress"][name]["bootstrap"].items()
                },
                "family_mixture": report["stress"][name]["family_mixture"],
            }
            for name in CANDIDATE_ORDER
        },
        "used_actual_light_in_allocator": report["used_actual_light_in_allocator"],
    }


def assemble(
    pool: PublicPool | None = None,
    *,
    seeds: Sequence[int] = FOLD_SEEDS,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    base = pool or load_public_pool()
    seed_pools = {}
    seed_reports = {}
    for seed in seeds:
        current = relabel_folds(base, int(seed))
        seed_pools[int(seed)] = current
        seed_reports[int(seed)] = evaluate_seed(current)
    ordered = [seed_reports[seed] for seed in sorted(seed_reports)]
    gate = promotion_gate(ordered)
    decision = gate["decision"]
    if decision == "record-e4-no-valid-reference":
        decision_reason = (
            "no-valid-reference: safe_family_point_baseline missed a pooled, "
            "fold, bootstrap, family-mixture, or slice-ratio gate on at least "
            "one seed. A later candidate cannot be integrated without a "
            "budget-valid quality reference. STOP. Keep the current runtime."
        )
    elif decision == "record-e4-no-promote":
        decision_reason = (
            "no-promote: no aggregate candidate cleared every seed's exact "
            "safety and quality gates. Research candidate search stops here. "
            "Keep the current runtime."
        )
    else:
        decision_reason = (
            f"promote {gate['recommended']} on repeated-seed aggregate-risk "
            "gates. Runtime export still requires an independent review."
        )
    audit = episode_audit_document(seed_pools, seed_reports)
    audit_sha = sha256_text(canonical_json_text(audit))
    seed_payload = {str(seed): _slim_seed(seed_reports[seed]) for seed in seed_reports}
    limitations = [
        "This experiment repairs item-q90 over-conservatism with an aggregate "
        "bound. It does not loosen official or 95% operating caps.",
        "Quality is frozen at E1 baseline_continuous_uplift OOF. E1B/E1C "
        "failed heads are not used.",
        BOUND_RULE,
        ROLLBACK_KEY_A,
        ROLLBACK_KEY_B,
        "Aggregate candidates do not apply family increment multipliers. "
        "The safe baseline does not apply an aggregate conformal bound. "
        "That is the only accounting difference besides rollback order.",
        "Family-multiplier + aggregate-bound together would double-count "
        "residual-family risk; that combination is not a candidate.",
        RUNTIME_SCOPE["provenance"],
        "safe_family_point_baseline is not a bit-identical replay of the "
        "shipped family-guard + budget-brake router. Runtime uses tighter "
        f"predicted_caps {RUNTIME_SCOPE['runtime_predicted_caps']} and "
        f"Premium brake_ratio {RUNTIME_SCOPE['runtime_premium_brake_ratio']}.",
        "Allocator denominators are OOF predicted light totals. Actual light "
        "never enters buy/bound/rollback.",
        "Exact public costs are the official scorer and stress only. "
        "Predicted costs are raise-only monotone.",
        "family/language/length/split/fold actual-ratio hard-cap slices use "
        "slice actual light. This slice gate may be harsh on small n>=20 "
        "buckets; it is still applied because this is the last experiment.",
        "Sequential Phase-4 follow-up to the E2 review (A/B only). "
        "Type-I is not family-wise controlled across E1–E4.",
        "A pass is not a runtime export.",
    ]
    report = {
        "allocator": {
            "buy": "OOF point predicted incremental cost",
            "denominator": "OOF predicted light total",
            "group": "exact content SHA-256",
            "k1": K1_SCOPE,
            "operating_caps": OPERATING_CAPS,
            "rollback_keys": {
                "aggregate_conformal_rollback": ROLLBACK_KEY_A,
                "aggregate_sigma_priority": ROLLBACK_KEY_B,
            },
            "used_actual_light_in_allocator": USED_ACTUAL_LIGHT_IN_ALLOCATOR,
        },
        "audit": {
            "n_rows": sum(block["n_rows"] for block in audit["seeds"].values()),
            "relative_path": AUDIT_RELATIVE_PATH,
            "sha256": audit_sha,
        },
        "candidates": {name: dict(CANDIDATE_DEFINITIONS[name]) for name in CANDIDATE_ORDER},
        "decision": decision,
        "decision_reason": decision_reason,
        "experiment": EXPERIMENT,
        "feature": {
            "bound_rule": BOUND_RULE,
            "quality_signal": QUALITY_SIGNAL,
            "runtime_artifact_changed": False,
            "sigma_in_price": False,
            "used_actual_light_in_allocator": USED_ACTUAL_LIGHT_IN_ALLOCATOR,
        },
        "fold_seeds": [int(seed) for seed in sorted(seed_reports)],
        "identity": dict(base.identity),
        "limitations": limitations,
        "promotion_gate": gate,
        "quality_signal": QUALITY_SIGNAL,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "runtime_scope": RUNTIME_SCOPE,
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_payload,
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    return sort_mapping(report), audit


__all__ = (
    "AUDIT_RELATIVE_PATH",
    "BASELINE_NAME",
    "CANDIDATE_ORDER",
    "FOLD_SEEDS",
    "OPERATING_CAPS",
    "USED_ACTUAL_LIGHT_IN_ALLOCATOR",
    "allocate_candidate",
    "apply_family_increment_multiplier",
    "assemble",
    "calibrate_bounds",
    "conformal_bound",
    "item_risk",
    "rollback_until_bound",
    "write_json_atomic",
)
