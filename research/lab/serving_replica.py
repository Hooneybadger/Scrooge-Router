# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Serving-path replica for the shipped budget-brake router.

Fast/Balanced go through the frozen family-guard predictions and
``select_fast_balanced`` with the shipped runaway fraction. Premium
serving goes through ``budget_brake_router.make_submission``, which
applies the E10 residual-majority hedge (parent 2.5 + ``other``
denylist when residual fraction ≥ 0.75) and the E13 family-majority
Fast cap (1.11 → 1.07 when one family is ≥ 0.75). ``allocate_all``
kwargs stay explicit research levers and do not turn those hedges on
by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router import budget_brake_router, family_guard_router
from ossp_router.feasibility_ladder import _select_premium_configured, select_fast_balanced
from ossp_router.protocol import MODEL_IDS, TIERS, InputBatch, OutcomeBatch, RoutingPolicy
from research.lab.modeling import official_score
from research.lab.public_pool import subset_inputs
from research.lab.validation import public_arrays


class ProtocolError(RuntimeError):
    """Sealed protocol or serving-replica contract failure."""


MODEL_INDEX = {model: index for index, model in enumerate(MODEL_IDS)}
LIGHT = MODEL_IDS[0]
AX31 = MODEL_IDS[1]
K1 = MODEL_IDS[2]
TIER_WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
OFFICIAL_CAPS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
NEAR_FRAC = 0.95
INFLATION = 1.054
TVBALL_EPSILON = 0.014204545454545449
VIEW_MIN_N = 20
RESIDUAL_FAMILY = family_guard_router.RESIDUAL_FAMILY
PINNED_DEV_FINAL_SCORE = 0.670710227273
PIN_TOLERANCE = 5e-13
SHIPPED_FAST_CAP = 1.11
SHIPPED_BALANCED_CAP = 1.45
SHIPPED_PREMIUM_CAP = 3.25
SHIPPED_BRAKE_RATIO = 3.8
SMALL_VIEW_N = 200


def json_float(value: Any) -> float:
    return float(np.float64(value))


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or y.size < 3:
        return 0.0
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom <= 1e-15:
        return 0.0
    return float((x * y).sum() / denom)


def residual_fraction(families: Sequence[str]) -> float:
    if not families:
        return 0.0
    return float(sum(family == RESIDUAL_FAMILY for family in families)) / float(
        len(families)
    )


def max_family_fraction(families: Sequence[str]) -> float:
    if not families:
        return 0.0
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    return float(max(counts.values())) / float(len(families))


def top2_family_fraction(families: Sequence[str]) -> float:
    if not families:
        return 0.0
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    ranked = sorted(counts.values(), reverse=True)
    top = ranked[0] + (ranked[1] if len(ranked) > 1 else 0)
    return float(top) / float(len(families))


def pair_family_views(
    families: Sequence[str],
    digests: Sequence[str],
    *,
    min_n: int = VIEW_MIN_N,
) -> dict[str, Tuple[int, ...]]:
    """Digest-balanced two-family batches. Every eligible pair, not a leftover set."""

    if len(families) != len(digests):
        raise ProtocolError("pair family views require aligned families and digests")
    buckets: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        buckets.setdefault(family, []).append(index)
    views: dict[str, Tuple[int, ...]] = {}
    names = sorted(buckets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_rows = sorted(buckets[left], key=lambda index: digests[index])
            right_rows = sorted(buckets[right], key=lambda index: digests[index])
            kept = min(len(left_rows), len(right_rows))
            if kept < min_n:
                continue
            views[f"pair:{left}+{right}"] = tuple(left_rows[:kept] + right_rows[:kept])
    return views


def top3_family_fraction(families: Sequence[str]) -> float:
    if not families:
        return 0.0
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    ranked = sorted(counts.values(), reverse=True)
    top = sum(ranked[:3])
    return float(top) / float(len(families))


def triple_family_views(
    families: Sequence[str],
    digests: Sequence[str],
    *,
    min_n: int = VIEW_MIN_N,
) -> dict[str, Tuple[int, ...]]:
    """Digest-balanced three-family batches. Every eligible triple, not a leftover set."""

    if len(families) != len(digests):
        raise ProtocolError("triple family views require aligned families and digests")
    buckets: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        buckets.setdefault(family, []).append(index)
    views: dict[str, Tuple[int, ...]] = {}
    names = sorted(buckets)
    for left_index, left in enumerate(names):
        for mid_index, mid in enumerate(names[left_index + 1 :], left_index + 1):
            for right in names[mid_index + 1 :]:
                rows = [
                    sorted(buckets[name], key=lambda index: digests[index])
                    for name in (left, mid, right)
                ]
                kept = min(len(row) for row in rows)
                if kept < min_n:
                    continue
                views[f"triple:{left}+{mid}+{right}"] = tuple(
                    rows[0][:kept] + rows[1][:kept] + rows[2][:kept]
                )
    return views


def family_combination_views(
    families: Sequence[str],
    digests: Sequence[str],
    *,
    min_size: int = 3,
    max_size: Optional[int] = None,
    min_n: int = VIEW_MIN_N,
) -> dict[str, Tuple[int, ...]]:
    """Digest-balanced batches for every eligible family combination."""

    if len(families) != len(digests):
        raise ProtocolError(
            "family combination views require aligned families and digests"
        )
    buckets: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        buckets.setdefault(family, []).append(index)
    names = sorted(buckets)
    upper = len(names) if max_size is None else min(int(max_size), len(names))
    if min_size < 1 or upper < min_size:
        return {}
    views: dict[str, Tuple[int, ...]] = {}
    for size in range(int(min_size), upper + 1):
        for group in combinations(names, size):
            rows = [
                sorted(buckets[name], key=lambda index: digests[index])
                for name in group
            ]
            kept = min(len(row) for row in rows)
            if kept < min_n:
                continue
            views[f"combination:{size}:{'+'.join(group)}"] = tuple(
                index for row in rows for index in row[:kept]
            )
    return views


def conditioned_fast_cap(fraction: float) -> float:
    if fraction <= 0.05:
        return 1.13
    if fraction <= 0.10:
        return 1.12
    return SHIPPED_FAST_CAP


def conditioned_balanced_cap(fraction: float) -> float:
    if fraction <= 0.05:
        return 1.52
    if fraction <= 0.10:
        return 1.48
    return SHIPPED_BALANCED_CAP


def tvball_worst(pooled_delta: float, family_deltas: Sequence[float]) -> float:
    if not family_deltas:
        return float(pooled_delta)
    return float(
        pooled_delta + TVBALL_EPSILON * (min(family_deltas) - max(family_deltas))
    )


def weighted_quality(qualities: Mapping[str, float]) -> float:
    return float(
        TIER_WEIGHTS["fast"] * float(qualities["fast"])
        + TIER_WEIGHTS["balanced"] * float(qualities["balanced"])
        + TIER_WEIGHTS["premium"] * float(qualities["premium"])
    )


def score_models(
    scores: np.ndarray,
    costs: np.ndarray,
    indexes: Sequence[int],
    models: Sequence[str],
) -> dict[str, Any]:
    if len(indexes) != len(models):
        raise ProtocolError("score_models length mismatch")
    if not indexes:
        raise ProtocolError("score_models empty index")
    chosen_scores = [
        float(scores[index, MODEL_INDEX[model]])
        for index, model in zip(indexes, models)
    ]
    chosen_costs = [
        float(costs[index, MODEL_INDEX[model]])
        for index, model in zip(indexes, models)
    ]
    light = math.fsum(float(costs[index, 0]) for index in indexes)
    if light <= 0.0:
        raise ProtocolError("light denominator is not positive")
    counts = {model: 0 for model in MODEL_IDS}
    for model in models:
        counts[model] += 1
    return {
        "actual_ratio": json_float(math.fsum(chosen_costs) / light),
        "counts": counts,
        "n": int(len(indexes)),
        "quality": json_float(math.fsum(chosen_scores) / float(len(indexes))),
    }


@dataclass(frozen=True)
class SplitReplica:
    """One public split with serving-path caches."""

    label: str
    inputs: InputBatch
    outcomes: OutcomeBatch
    scores: np.ndarray
    costs: np.ndarray
    families: Tuple[str, ...]
    digests: Tuple[str, ...]
    fb_predictions: Tuple[Tuple[float, Tuple[float, float]], ...]
    premium_rows: Tuple[Tuple[float, Tuple[float, float, float]], ...]
    premium_quality: Tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.families)

    @property
    def residual_frac(self) -> float:
        return residual_fraction(self.families)

    def uplift(self) -> np.ndarray:
        return np.asarray([row[0] for row in self.fb_predictions], dtype=np.float64)

    def realized_delta31(self) -> np.ndarray:
        return self.scores[:, 1] - self.scores[:, 0]

    def realized_deltak(self) -> np.ndarray:
        return self.scores[:, 2] - self.scores[:, 1]


@dataclass(frozen=True)
class ServingReplica:
    """Frozen shipped artifact plus the serving allocation entry points."""

    policy: RoutingPolicy
    brake: budget_brake_router.BrakeArtifact
    runaway_fraction: float
    max_upgrade_fraction: float
    shipped_caps: Mapping[str, float]
    shipped_brake_ratio: float
    shipped_count_cap: int

    @classmethod
    def load(cls) -> "ServingReplica":
        policy = budget_brake_router.load_bundled_policy()
        brake = budget_brake_router.load_bundled_artifact()
        value = brake.family_guard.value
        caps = {
            tier: float(value["predicted_caps"][tier])
            for tier in ("fast", "balanced", "premium")
        }
        if abs(caps["fast"] - SHIPPED_FAST_CAP) > 1e-12:
            raise ProtocolError("shipped Fast cap drifted")
        if abs(caps["balanced"] - SHIPPED_BALANCED_CAP) > 1e-12:
            raise ProtocolError("shipped Balanced cap drifted")
        if abs(caps["premium"] - SHIPPED_PREMIUM_CAP) > 1e-12:
            raise ProtocolError("shipped Premium cap drifted")
        brake_ratio = float(brake.budget_brake["brake_ratio"])
        if abs(brake_ratio - SHIPPED_BRAKE_RATIO) > 1e-12:
            raise ProtocolError("shipped brake_ratio drifted")
        return cls(
            policy=policy,
            brake=brake,
            runaway_fraction=float(value["runaway_fraction"]),
            max_upgrade_fraction=float(value["max_upgrade_fraction"]),
            shipped_caps=caps,
            shipped_brake_ratio=brake_ratio,
            shipped_count_cap=int(brake.budget_brake["count_cap"]),
        )

    def build_split(
        self, label: str, inputs: InputBatch, outcomes: OutcomeBatch
    ) -> SplitReplica:
        arrays = public_arrays(inputs, outcomes, self.policy)
        families = tuple(
            family_guard_router.prompt_family(episode) for episode in inputs.episodes
        )
        digests = tuple(
            budget_brake_router.content_digest(episode) for episode in inputs.episodes
        )
        fb_predictions = tuple(
            family_guard_router.guarded_prediction(episode, self.policy, self.brake.family_guard)
            for episode in inputs.episodes
        )
        premium_rows = tuple(
            budget_brake_router.premium_prediction_row(episode, self.policy, self.brake)
            for episode in inputs.episodes
        )
        premium_quality = tuple(
            budget_brake_router.predict_quality(episode, self.brake)
            for episode in inputs.episodes
        )
        return SplitReplica(
            label=label,
            inputs=inputs,
            outcomes=outcomes,
            scores=np.asarray(arrays.scores, dtype=np.float64),
            costs=np.asarray(arrays.costs, dtype=np.float64),
            families=families,
            digests=digests,
            fb_predictions=fb_predictions,
            premium_rows=premium_rows,
            premium_quality=premium_quality,
        )

    def allocate_fast_balanced(
        self,
        split: SplitReplica,
        indexes: Sequence[int],
        *,
        cap: float,
    ) -> Tuple[str, ...]:
        predictions = tuple(split.fb_predictions[index] for index in indexes)
        selected, _ratio = select_fast_balanced(
            predictions,
            cap=float(cap),
            runaway_fraction=self.runaway_fraction,
            max_upgrade_fraction=self.max_upgrade_fraction,
        )
        return selected

    def allocate_premium(
        self,
        split: SplitReplica,
        indexes: Sequence[int],
        *,
        brake_ratio: Optional[float] = None,
        guard_parent: bool = False,
        residual_multiplier: Optional[float] = None,
        denylist_extra: Sequence[str] = (),
        guard_brake_k1: bool = False,
    ) -> Tuple[str, ...]:
        batch = subset_inputs(split.inputs, indexes)
        raw_rows = tuple(split.premium_rows[index] for index in indexes)
        if guard_parent:
            parent_rows = tuple(
                (
                    row[0],
                    budget_brake_router.guard_premium_parent_costs(
                        split.inputs.episodes[index],
                        row[1],
                        self.brake,
                        residual_multiplier=residual_multiplier,
                    ),
                )
                for index, row in zip(indexes, raw_rows)
            )
        else:
            parent_rows = raw_rows
        parent, _ratio = _select_premium_configured(
            batch,
            parent_rows,
            float(self.shipped_caps["premium"]),
            self.brake.family_guard.base,
        )
        block = dict(self.brake.budget_brake)
        if brake_ratio is not None:
            block["brake_ratio"] = float(brake_ratio)
        if denylist_extra:
            block["denylist_families"] = list(block["denylist_families"]) + list(
                denylist_extra
            )
        if guard_brake_k1:
            brake_costs = tuple(
                budget_brake_router.guard_premium_brake_costs(
                    split.inputs.episodes[index],
                    row[1],
                    self.brake,
                    residual_multiplier=residual_multiplier,
                )
                for index, row in zip(indexes, raw_rows)
            )
        else:
            brake_costs = tuple(row[1] for row in raw_rows)
        return budget_brake_router.promote_premium_brake(
            parent,
            tuple(split.premium_quality[index] for index in indexes),
            tuple(split.families[index] for index in indexes),
            brake_costs,
            tuple(split.digests[index] for index in indexes),
            block,
        )

    def allocate_tier(
        self,
        split: SplitReplica,
        tier: str,
        indexes: Sequence[int],
        *,
        caps: Mapping[str, float],
        brake_ratio: Optional[float] = None,
        guard_parent: bool = False,
        residual_multiplier: Optional[float] = None,
        denylist_extra: Sequence[str] = (),
        guard_brake_k1: bool = False,
    ) -> Tuple[str, ...]:
        if tier == "premium":
            return self.allocate_premium(
                split,
                indexes,
                brake_ratio=brake_ratio,
                guard_parent=guard_parent,
                residual_multiplier=residual_multiplier,
                denylist_extra=denylist_extra,
                guard_brake_k1=guard_brake_k1,
            )
        return self.allocate_fast_balanced(
            split, indexes, cap=float(caps[tier])
        )

    def allocate_all(
        self,
        split: SplitReplica,
        indexes: Sequence[int],
        *,
        caps: Mapping[str, float],
        brake_ratio: Optional[float] = None,
        guard_parent: bool = False,
        residual_multiplier: Optional[float] = None,
        denylist_extra: Sequence[str] = (),
        guard_brake_k1: bool = False,
    ) -> dict[str, Tuple[str, ...]]:
        return {
            tier: self.allocate_tier(
                split,
                tier,
                indexes,
                caps=caps,
                brake_ratio=brake_ratio,
                guard_parent=guard_parent,
                residual_multiplier=residual_multiplier,
                denylist_extra=denylist_extra,
                guard_brake_k1=guard_brake_k1,
            )
            for tier in TIERS
        }

    def runtime_models(self, split: SplitReplica, tier: str) -> Tuple[str, ...]:
        plan = budget_brake_router.make_submission(
            split.inputs, self.policy, self.brake, tier
        )
        return tuple(decision.model_id for decision in plan.submission.decisions)

    def official(
        self, split: SplitReplica, selections: Mapping[str, Sequence[str]]
    ) -> Mapping[str, Any]:
        return official_score(split.inputs, split.outcomes, self.policy, selections)


def demote_residual_upgrades(
    families: Sequence[str],
    models: Sequence[str],
    quality: Sequence[float],
    digests: Sequence[str],
    upgrade_cap: float,
) -> Tuple[str, ...]:
    """Keep at most ``upgrade_cap`` of residual rows off Light.

    Demotes residual K1 first, then low predicted quality. Non-residual
    rows are untouched. ``upgrade_cap`` is a fraction in (0, 1].
    """

    if not (
        len(families) == len(models) == len(quality) == len(digests)
    ):
        raise ProtocolError("residual upgrade demotion arrays must align")
    cap = float(upgrade_cap)
    if not math.isfinite(cap) or cap <= 0.0 or cap > 1.0:
        raise ProtocolError("residual upgrade cap is outside (0, 1]")
    residual = [
        index
        for index, family in enumerate(families)
        if family == RESIDUAL_FAMILY
    ]
    if not residual:
        return tuple(models)
    allowed = int(math.floor(len(residual) * cap + 1e-15))
    upgraded = [
        index
        for index in residual
        if models[index] != LIGHT
    ]
    if len(upgraded) <= allowed:
        return tuple(models)
    upgraded.sort(
        key=lambda index: (
            0 if models[index] == K1 else 1,
            float(quality[index]),
            digests[index],
        )
    )
    selected = list(models)
    drop = len(upgraded) - allowed
    for index in upgraded[:drop]:
        selected[index] = LIGHT
    return tuple(selected)


def digest_views(n: int, digests: Sequence[str]) -> dict[str, Tuple[int, ...]]:
    """Content-only composition views. No outcome or ID feature is used."""

    order = sorted(range(n), key=lambda index: digests[index])
    views: dict[str, Tuple[int, ...]] = {
        "full": tuple(range(n)),
        "digest-even": tuple(order[index] for index in range(0, n, 2)),
        "digest-odd": tuple(order[index] for index in range(1, n, 2)),
    }
    if n >= SMALL_VIEW_N:
        views["digest-head-200"] = tuple(order[:SMALL_VIEW_N])
        views["digest-tail-200"] = tuple(order[-SMALL_VIEW_N:])
    return views


def family_views(families: Sequence[str]) -> dict[str, Tuple[int, ...]]:
    buckets: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        buckets.setdefault(family, []).append(index)
    return {
        f"family:{name}": tuple(indexes)
        for name, indexes in sorted(buckets.items())
        if len(indexes) >= VIEW_MIN_N
    }


def composition_views(split: SplitReplica) -> dict[str, Tuple[int, ...]]:
    views = digest_views(split.n, split.digests)
    views.update(family_views(split.families))
    return views


def model_counts(models: Sequence[str]) -> dict[str, int]:
    counts = {model: 0 for model in MODEL_IDS}
    for model in models:
        counts[model] += 1
    return counts


def official_tier_block(official: Mapping[str, Any], tier: str) -> dict[str, Any]:
    row = official["tiers"][tier]
    return {
        "budget_passed": bool(row["budget_passed"]),
        "budget_ratio": float(row["budget_ratio"]),
        "model_counts": {key: int(value) for key, value in row["model_counts"].items()},
        "near_budget": bool(row["near_budget"]),
        "quality_score": float(row["quality_score"]),
        "tier_score": float(row["tier_score"]),
    }


__all__ = (
    "AX31",
    "INFLATION",
    "K1",
    "LIGHT",
    "MODEL_INDEX",
    "NEAR_FRAC",
    "OFFICIAL_CAPS",
    "PINNED_DEV_FINAL_SCORE",
    "PIN_TOLERANCE",
    "ProtocolError",
    "RESIDUAL_FAMILY",
    "SHIPPED_BALANCED_CAP",
    "SHIPPED_BRAKE_RATIO",
    "SHIPPED_FAST_CAP",
    "SHIPPED_PREMIUM_CAP",
    "SMALL_VIEW_N",
    "ServingReplica",
    "SplitReplica",
    "TIER_WEIGHTS",
    "TIERS",
    "TVBALL_EPSILON",
    "VIEW_MIN_N",
    "composition_views",
    "conditioned_balanced_cap",
    "conditioned_fast_cap",
    "demote_residual_upgrades",
    "digest_views",
    "family_combination_views",
    "family_views",
    "json_float",
    "max_family_fraction",
    "model_counts",
    "pair_family_views",
    "top2_family_fraction",
    "top3_family_fraction",
    "triple_family_views",
    "official_tier_block",
    "pearson",
    "residual_fraction",
    "score_models",
    "tvball_worst",
    "weighted_quality",
)
