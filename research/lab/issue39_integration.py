# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Issue #39 remaining gates: reconstructed stress, worst ratio, Dev score.

Coverage already selected ``family_partial_q95``.  This module overlays those
frozen serving scales on the 128+ path, keeps the small-batch route as it is,
and compares against current serving on the E30 catalogues.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router import distributional_router as serving
from ossp_router.heuristic import episode_text
from ossp_router.protocol import MODEL_IDS, TIERS, InputBatch, load_bundled_policy
from research.lab.cap_certification import build_stress_views
from research.lab.generalization_followups import (
    COST_INFLATION,
    REQUIRED_MARGIN_FRACTION,
    SMALL_BATCH_UNIQUE_CUTOFF,
)
from research.lab.prefix_certificates import (
    DIRICHLET_ALPHA,
    FAMDOM_DUP_CAP,
    MIN_VIEW_N,
    View,
    _dirichlet_counts,
    _famdom_sizes,
    _sample_capped,
    family_folds,
)
from research.lab.public_pool import PublicPool


WIDE_SEEDS: Tuple[int, ...] = (2026082204, 2026083104)
WIDE_DIRICHLET_DRAWS = 1_000
WIDE_DIRICHLET_N = 880
WIDE_HALF_DRAWS = 20
WIDE_SMALL_DRAWS = 200
WIDE_FAMDOM_DRAWS = 200
WIDE_SMALL_RED_N = 100
WIDE_SMALL_BINDING_N = 300
WEIGHTED_LOSS_MAX = 0.0005
TIER_WEIGHTS: Mapping[str, float] = {
    "fast": 0.4,
    "balanced": 0.3,
    "premium": 0.3,
}
FROZEN_OFFICIAL_WEIGHTED = 0.7157670454545454
FROZEN_MAX_INFLATED: Mapping[str, float] = {
    "fast": 1.2279638846683032,
    "balanced": 1.9714489261211643,
    "premium": 3.6945688135994135,
}

_WORKER_CACHE: Optional["ServingPredictionCache"] = None
_WORKER_COSTS: Optional[np.ndarray] = None
_WORKER_SCORES: Optional[np.ndarray] = None
_WORKER_VIEWS: Optional[list[np.ndarray]] = None
_WORKER_ARTIFACT = None
_WORKER_POLICY = None
_WORKER_LIMITS: Optional[dict[str, float]] = None


@dataclass(frozen=True)
class ServingPredictionCache:
    """Per-episode serving predictions after inference-time stabilization."""

    structural: Tuple[Tuple[float, ...], ...]
    quality: Tuple[Tuple[float, ...], ...]
    raw_mean: Tuple[Tuple[float, ...], ...]
    raw_upper: Tuple[Tuple[float, ...], ...]
    families: Tuple[str, ...]
    family_ids: Tuple[int, ...]
    tie_keys: Tuple[str, ...]


def build_wide_catalogue(
    families: Sequence[str],
    digests: Sequence[str],
    *,
    seed: int,
) -> Tuple[View, ...]:
    """Dev-adjusted 3,435-view catalogue used for the six wide catalogs."""

    fam = np.asarray(list(families))
    names = tuple(sorted(dict.fromkeys(fam.tolist())))
    count = int(fam.size)
    universe = np.arange(count, dtype=np.int64)
    views: list[View] = []

    def accept(view: View) -> None:
        if int(view.index.size) < MIN_VIEW_N:
            return
        views.append(view)

    accept(View("official", "official-full", universe.copy(), "oof"))
    order = sorted(range(count), key=lambda index: digests[index])
    accept(
        View(
            "digest",
            "digest-even",
            np.asarray(order[0::2], dtype=np.int64),
            "oof",
        )
    )
    accept(
        View(
            "digest",
            "digest-odd",
            np.asarray(order[1::2], dtype=np.int64),
            "oof",
        )
    )
    accept(
        View(
            "digest",
            "digest-head-200",
            np.asarray(order[:200], dtype=np.int64),
            "oof",
        )
    )
    accept(
        View(
            "digest",
            "digest-tail-200",
            np.asarray(order[-200:], dtype=np.int64),
            "oof",
        )
    )
    for name, index in family_folds(families):
        accept(
            View("family", f"family:{name}", np.asarray(index, dtype=np.int64), "oof")
        )

    fam_rng = np.random.default_rng(int(seed))
    for name in names:
        focus = np.flatnonzero(fam == name)
        rest = np.flatnonzero(fam != name)
        n_focus, n_other, _batch, fallback = _famdom_sizes(
            int(focus.size), int(rest.size)
        )
        for draw in range(WIDE_FAMDOM_DRAWS):
            chosen_focus, tag_f = _sample_capped(fam_rng, focus, n_focus)
            chosen_other, tag_o = _sample_capped(fam_rng, rest, n_other)
            tag = fallback
            if tag_f is not None or tag_o is not None:
                tag = "+".join(
                    item
                    for item in (fallback, tag_f, tag_o)
                    if item and item != "none"
                )
            chosen = np.concatenate([chosen_focus, chosen_other])
            accept(
                View(
                    "famdom",
                    f"famdom-{name}-{draw:03d}",
                    chosen,
                    "oof",
                    tag,
                )
            )

    dir_rng = np.random.default_rng(int(seed) + 1)
    pools = {name: np.flatnonzero(fam == name) for name in names}
    dirichlet_n = min(WIDE_DIRICHLET_N, count)
    for draw in range(WIDE_DIRICHLET_DRAWS):
        counts = _dirichlet_counts(dir_rng, len(names), dirichlet_n, DIRICHLET_ALPHA)
        parts: list[np.ndarray] = []
        leftover = 0
        capacity: list[int] = []
        for name, take_count in zip(names, counts):
            taken, _tag = _sample_capped(dir_rng, pools[name], int(take_count))
            parts.append(taken)
            leftover += max(0, int(take_count) - int(taken.size))
            capacity.append(
                max(0, FAMDOM_DUP_CAP * int(pools[name].size) - int(taken.size))
            )
        if leftover > 0:
            for name, cap_left in zip(names, capacity):
                if leftover <= 0 or cap_left <= 0:
                    continue
                extra, _tag = _sample_capped(
                    dir_rng, pools[name], min(leftover, cap_left)
                )
                parts.append(extra)
                leftover -= int(extra.size)
        chosen = (
            np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        )
        accept(View("dirichlet", f"dirichlet-{draw:04d}", chosen, "oof"))

    half_rng = np.random.default_rng(int(seed) + 2)
    half_n = min(count // 2, count)
    for draw in range(WIDE_HALF_DRAWS):
        chosen = half_rng.choice(universe, size=int(half_n), replace=False)
        accept(View("half", f"half-{draw:02d}", chosen, "oof"))

    small_rng = np.random.default_rng(int(seed) + 3)
    for size in (WIDE_SMALL_RED_N, WIDE_SMALL_BINDING_N):
        take = min(int(size), count)
        for draw in range(WIDE_SMALL_DRAWS):
            chosen = small_rng.choice(universe, size=int(take), replace=False)
            accept(View("small", f"small-{size}-{draw:03d}", chosen, "oof"))
    return tuple(views)


def remap_views(
    views: Sequence[View], global_indexes: Sequence[int]
) -> Tuple[View, ...]:
    lookup = np.asarray(list(global_indexes), dtype=np.int64)
    remapped = []
    for view in views:
        remapped.append(
            View(
                view.kind,
                view.name,
                lookup[np.asarray(view.index, dtype=np.int64)],
                view.pred_source,
                view.fallback,
            )
        )
    return tuple(remapped)


def precompute_serving_cache(
    episodes: Sequence[Any],
    artifact: serving.DistributionalArtifact,
) -> ServingPredictionCache:
    vocabulary = {
        term: index for index, term in enumerate(artifact.vocabulary)
    }
    family_lookup = {name: index for index, name in enumerate(serving.FAMILY_NAMES)}
    structural_rows = []
    quality_rows = []
    mean_rows = []
    upper_rows = []
    families = []
    family_ids = []
    tie_keys = []
    for episode in episodes:
        stabilized = serving.stabilize_episode(episode)
        structural = serving.structural_features(stabilized)
        features = structural + serving._lexical_feature_row(
            episode_text(stabilized), vocabulary
        )
        quality = tuple(
            serving.predict_head(artifact.quality_heads[model_id], features)
            for model_id in MODEL_IDS
        )
        raw_mean = tuple(
            serving.predict_head(artifact.cost_mean_heads[model_id], features)
            for model_id in MODEL_IDS
        )
        raw_q50 = tuple(
            serving.predict_head(artifact.cost_q50_heads[model_id], features)
            for model_id in MODEL_IDS
        )
        raw_q90 = tuple(
            serving.predict_head(artifact.cost_q90_heads[model_id], features)
            for model_id in MODEL_IDS
        )
        family = serving.prompt_family(stabilized)
        structural_rows.append(structural)
        quality_rows.append(quality)
        mean_rows.append(raw_mean)
        upper_rows.append(
            tuple(max(left, right) for left, right in zip(raw_q50, raw_q90))
        )
        families.append(family)
        family_ids.append(family_lookup[family])
        tie_keys.append(serving._content_key(stabilized))
    return ServingPredictionCache(
        tuple(structural_rows),
        tuple(quality_rows),
        tuple(mean_rows),
        tuple(upper_rows),
        tuple(families),
        tuple(family_ids),
        tuple(tie_keys),
    )


def route_from_cache(
    cache: ServingPredictionCache,
    indexes: Sequence[int],
    *,
    artifact: serving.DistributionalArtifact,
    policy: Any,
    tier: str,
    q95_on_normal: bool,
) -> Tuple[int, ...]:
    """Reproduce ``_route_canonical`` on precomputed rows."""

    selected = [int(index) for index in indexes]
    structural = [cache.structural[index] for index in selected]
    quality = [cache.quality[index] for index in selected]
    raw_mean = [cache.raw_mean[index] for index in selected]
    raw_upper = [cache.raw_upper[index] for index in selected]
    families = [cache.families[index] for index in selected]
    family_ids = [cache.family_ids[index] for index in selected]
    tie_keys = [cache.tie_keys[index] for index in selected]
    unique_count = len(set(tie_keys))
    family_counts = {name: 0 for name in serving.FAMILY_NAMES}
    for family in families:
        family_counts[family] += 1
    row_count = len(selected)
    proportions = [
        family_counts[name] / row_count for name in serving.FAMILY_NAMES
    ]
    tv = 0.5 * math.fsum(
        abs(value - reference)
        for value, reference in zip(
            proportions, artifact.family_calibration.reference_proportions
        )
    )
    budget_multiplier = float(policy.tiers[tier].budget_multiplier)
    config = artifact.tier_config[tier]
    fallback = min(
        config.base_fraction,
        max(
            1.0 / budget_multiplier,
            config.base_fraction * (1.0 - config.composition_penalty * tv),
        ),
    )
    if serving.small_batch_route_enabled(unique_count):
        mean, q90 = serving._apply_family_scales(
            raw_mean,
            raw_upper,
            family_ids,
            serving.SMALL_BATCH_MEAN_SCALES,
            serving.SMALL_BATCH_UPPER_SCALES,
        )
        charges, light = serving._small_batch_surfaces(
            mean, q90, family_ids, config
        )
        return serving._allocate(
            quality,
            charges,
            light,
            tie_keys,
            budget_multiplier,
            serving._small_batch_target_fraction(
                config, budget_multiplier, unique_count, tv
            ),
            False,
        )
    if unique_count < artifact.gates.min_content_groups:
        return tuple(0 for _ in selected)
    if q95_on_normal:
        mean_scales = serving.SMALL_BATCH_MEAN_SCALES
        upper_scales = serving.SMALL_BATCH_UPPER_SCALES
    else:
        mean_scales = artifact.family_calibration.mean_scales
        upper_scales = artifact.family_calibration.q90_scales
    mean, q90 = serving._apply_family_scales(
        raw_mean,
        raw_upper,
        family_ids,
        mean_scales,
        upper_scales,
    )
    if tier == "premium":
        allow_k1 = (
            unique_count >= artifact.gates.premium_k1_min_groups
            and tv <= artifact.gates.premium_k1_max_tv
        )
        target = fallback if allow_k1 else config.base_fraction
    else:
        risk_features = serving._batch_features(
            quality,
            mean,
            q90,
            structural,
            families,
            tie_keys,
            artifact.family_calibration,
        )
        risk_fraction = serving.predict_head(
            artifact.risk_heads[tier], risk_features
        )
        target = min(
            config.base_fraction,
            max(fallback, risk_fraction - config.risk_reserve),
        )
        allow_k1 = (
            tier == "balanced"
            and unique_count >= artifact.gates.balanced_k1_min_groups
        )
    charges = serving._tier_charges(mean, q90, config)
    return serving._allocate(
        quality,
        charges,
        [values[0] for values in mean],
        tie_keys,
        budget_multiplier,
        target,
        allow_k1,
    )


def _realized(
    indexes: np.ndarray,
    actions: Sequence[int],
    costs: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    chosen = np.asarray(actions, dtype=np.int64)
    rows = np.asarray(indexes, dtype=np.int64)
    quality = float(np.mean(scores[rows, chosen]))
    light = float(np.sum(costs[rows, 0], dtype=np.float64))
    spent = float(np.sum(costs[rows, chosen], dtype=np.float64))
    ratio = spent / max(light, sys.float_info.min)
    return quality, ratio


def official_dev_metrics(
    cache: ServingPredictionCache,
    dev_indexes: np.ndarray,
    costs: np.ndarray,
    scores: np.ndarray,
    *,
    artifact: serving.DistributionalArtifact,
    policy: Any,
    q95_on_normal: bool,
) -> dict[str, Any]:
    weighted = 0.0
    tiers = {}
    for tier in TIERS:
        actions = route_from_cache(
            cache,
            [int(index) for index in dev_indexes],
            artifact=artifact,
            policy=policy,
            tier=tier,
            q95_on_normal=q95_on_normal,
        )
        quality, ratio = _realized(dev_indexes, actions, costs, scores)
        inflated = ratio * COST_INFLATION
        limit = float(policy.tiers[tier].budget_multiplier) * (
            1.0 - REQUIRED_MARGIN_FRACTION
        )
        counts = {
            MODEL_IDS[column]: int(sum(1 for action in actions if action == column))
            for column in range(3)
        }
        weighted += TIER_WEIGHTS[tier] * quality
        tiers[tier] = {
            "quality": quality,
            "ratio": ratio,
            "inflated_ratio": inflated,
            "limit": limit,
            "safe": inflated <= limit + 1e-12,
            "model_counts": counts,
        }
    return {
        "weighted_quality": float(weighted),
        "tiers": tiers,
        "all_safe": all(bool(row["safe"]) for row in tiers.values()),
    }


def verify_cache_matches_serving(
    pool: PublicPool,
    indexes: np.ndarray,
    cache: ServingPredictionCache,
    artifact: serving.DistributionalArtifact,
) -> dict[str, Any]:
    episodes = tuple(pool.episodes[int(index)] for index in indexes)
    inputs = InputBatch(
        schema_version=pool.inputs.schema_version,
        challenge_id=pool.inputs.challenge_id,
        split="dev",
        episodes=episodes,
    )
    mismatches = {}
    for tier in TIERS:
        submission = serving.make_submission(inputs, pool.policy, artifact, tier)
        expected = [
            MODEL_IDS.index(decision.model_id) for decision in submission.decisions
        ]
        actual = list(
            route_from_cache(
                cache,
                [int(index) for index in indexes],
                artifact=artifact,
                policy=pool.policy,
                tier=tier,
                q95_on_normal=False,
            )
        )
        if expected != actual:
            mismatches[tier] = int(
                sum(left != right for left, right in zip(expected, actual))
            )
    return {
        "matched": not mismatches,
        "mismatches": mismatches,
        "n": int(indexes.size),
    }


def _bind_worker_state(
    cache: ServingPredictionCache,
    costs: np.ndarray,
    scores: np.ndarray,
    views: Sequence[np.ndarray],
    artifact: serving.DistributionalArtifact,
    policy: Any,
    limits: Mapping[str, float],
) -> None:
    global _WORKER_CACHE, _WORKER_COSTS, _WORKER_SCORES
    global _WORKER_VIEWS, _WORKER_ARTIFACT, _WORKER_POLICY, _WORKER_LIMITS
    _WORKER_CACHE = cache
    _WORKER_COSTS = costs
    _WORKER_SCORES = scores
    _WORKER_VIEWS = list(views)
    _WORKER_ARTIFACT = artifact
    _WORKER_POLICY = policy
    _WORKER_LIMITS = dict(limits)


def _evaluate_view_index(view_index: int) -> dict[str, Any]:
    cache = _WORKER_CACHE
    views = _WORKER_VIEWS
    assert cache is not None and views is not None
    indexes = views[view_index]
    unique_count = len({cache.tie_keys[int(index)] for index in indexes})
    current_rows = {}
    candidate_rows = {}
    for tier in TIERS:
        current_actions = route_from_cache(
            cache,
            indexes,
            artifact=_WORKER_ARTIFACT,
            policy=_WORKER_POLICY,
            tier=tier,
            q95_on_normal=False,
        )
        if unique_count < SMALL_BATCH_UNIQUE_CUTOFF:
            candidate_actions = current_actions
        else:
            candidate_actions = route_from_cache(
                cache,
                indexes,
                artifact=_WORKER_ARTIFACT,
                policy=_WORKER_POLICY,
                tier=tier,
                q95_on_normal=True,
            )
        current_quality, current_ratio = _realized(
            indexes, current_actions, _WORKER_COSTS, _WORKER_SCORES
        )
        candidate_quality, candidate_ratio = _realized(
            indexes, candidate_actions, _WORKER_COSTS, _WORKER_SCORES
        )
        current_inflated = current_ratio * COST_INFLATION
        candidate_inflated = candidate_ratio * COST_INFLATION
        limit = _WORKER_LIMITS[tier]
        current_rows[tier] = {
            "inflated": current_inflated,
            "quality": current_quality,
            "violation": current_inflated > limit + 1e-12,
        }
        candidate_rows[tier] = {
            "inflated": candidate_inflated,
            "quality": candidate_quality,
            "violation": candidate_inflated > limit + 1e-12,
        }
    return {
        "n": int(np.asarray(indexes).size),
        "unique_count": int(unique_count),
        "normal_batch": unique_count >= SMALL_BATCH_UNIQUE_CUTOFF,
        "current": current_rows,
        "candidate": candidate_rows,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def worst(arm: str) -> dict[str, float]:
        return {
            tier: max(float(row[arm][tier]["inflated"]) for row in rows)
            for tier in TIERS
        }

    def violations(arm: str) -> dict[str, Any]:
        by_tier = {
            tier: int(sum(1 for row in rows if row[arm][tier]["violation"]))
            for tier in TIERS
        }
        view_violations = int(
            sum(
                1
                for row in rows
                if any(row[arm][tier]["violation"] for tier in TIERS)
            )
        )
        return {
            "by_tier": by_tier,
            "view_violations": view_violations,
            "tier_evaluations": int(sum(by_tier.values())),
        }

    normal = [row for row in rows if row["normal_batch"]]
    small = [row for row in rows if not row["normal_batch"]]
    return {
        "n_views": int(len(rows)),
        "n_normal_batch": int(len(normal)),
        "n_small_batch": int(len(small)),
        "current": {
            "max_inflated": worst("current") if rows else {tier: 0.0 for tier in TIERS},
            "violations": violations("current") if rows else {
                "by_tier": {tier: 0 for tier in TIERS},
                "view_violations": 0,
                "tier_evaluations": 0,
            },
        },
        "candidate": {
            "max_inflated": worst("candidate") if rows else {tier: 0.0 for tier in TIERS},
            "violations": violations("candidate") if rows else {
                "by_tier": {tier: 0 for tier in TIERS},
                "view_violations": 0,
                "tier_evaluations": 0,
            },
        },
        "normal_batch": {
            "current_max_inflated": worst("current") if normal else {
                tier: 0.0 for tier in TIERS
            },
            "candidate_max_inflated": worst("candidate") if normal else {
                tier: 0.0 for tier in TIERS
            },
        },
    }


def _evaluate_views(
    views: Sequence[View],
    cache: ServingPredictionCache,
    costs: np.ndarray,
    scores: np.ndarray,
    *,
    artifact: serving.DistributionalArtifact,
    policy: Any,
    workers: int,
    label: str,
) -> dict[str, Any]:
    indexes = [np.asarray(view.index, dtype=np.int64) for view in views]
    limits = {
        tier: float(policy.tiers[tier].budget_multiplier)
        * (1.0 - REQUIRED_MARGIN_FRACTION)
        for tier in TIERS
    }
    _bind_worker_state(cache, costs, scores, indexes, artifact, policy, limits)
    total = len(indexes)
    rows: list[dict[str, Any]] = []
    if workers <= 1 or total < 8:
        for index in range(total):
            rows.append(_evaluate_view_index(index))
            if (index + 1) % 500 == 0 or index + 1 == total:
                print(f"  {label} scored {index + 1}/{total}", flush=True)
    else:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=context,
        ) as pool:
            for done, row in enumerate(
                pool.map(_evaluate_view_index, range(total), chunksize=16),
                start=1,
            ):
                rows.append(row)
                if done % 500 == 0 or done == total:
                    print(f"  {label} scored {done}/{total}", flush=True)
    return _summarize_rows(rows)


def episode_digests(episodes: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(
        hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest()
        for episode in episodes
    )


def run_issue_39_integration(
    pool: PublicPool,
    train_indexes: np.ndarray,
    dev_indexes: np.ndarray,
    *,
    workers: Optional[int] = None,
) -> dict[str, Any]:
    """Run reconstructed E30 stress plus the official Dev score gate."""

    worker_count = int(workers) if workers is not None else min(16, os.cpu_count() or 1)
    artifact = serving.load_bundled_artifact()
    policy = pool.policy if pool.policy is not None else load_bundled_policy()
    print(
        f"issue #39 precomputing serving rows n={len(pool.episodes)} "
        f"workers={worker_count}",
        flush=True,
    )
    cache = precompute_serving_cache(pool.episodes, artifact)
    replica = verify_cache_matches_serving(pool, dev_indexes, cache, artifact)
    if not replica["matched"]:
        raise RuntimeError(
            f"issue #39 cache path drifted from serving: {replica['mismatches']}"
        )
    print("issue #39 cache path matches serving on official Dev", flush=True)

    current_dev = official_dev_metrics(
        cache,
        dev_indexes,
        pool.costs,
        pool.scores,
        artifact=artifact,
        policy=policy,
        q95_on_normal=False,
    )
    candidate_dev = official_dev_metrics(
        cache,
        dev_indexes,
        pool.costs,
        pool.scores,
        artifact=artifact,
        policy=policy,
        q95_on_normal=True,
    )
    weighted_loss = float(
        current_dev["weighted_quality"] - candidate_dev["weighted_quality"]
    )
    print(
        "issue #39 official Dev "
        f"current={current_dev['weighted_quality']:.12f} "
        f"q95={candidate_dev['weighted_quality']:.12f} "
        f"loss={weighted_loss:.6f}",
        flush=True,
    )

    train_families = tuple(pool.families[int(index)] for index in train_indexes)
    dev_families = tuple(pool.families[int(index)] for index in dev_indexes)
    train_views, train_catalogue = build_stress_views(train_families)
    dev_views, dev_catalogue = build_stress_views(dev_families)
    primary_views = remap_views(train_views, train_indexes) + remap_views(
        dev_views, dev_indexes
    )
    print(
        f"issue #39 primary views={len(primary_views)} "
        f"(train={train_catalogue['n_views']} dev={dev_catalogue['n_views']})",
        flush=True,
    )
    primary = _evaluate_views(
        primary_views,
        cache,
        pool.costs,
        pool.scores,
        artifact=artifact,
        policy=policy,
        workers=worker_count,
        label="primary",
    )

    wide_views: list[View] = []
    wide_catalogs = []
    public_indexes = np.arange(len(pool.episodes), dtype=np.int64)
    splits = (
        ("train", train_indexes, train_families),
        ("dev", dev_indexes, dev_families),
        (
            "public",
            public_indexes,
            tuple(pool.families),
        ),
    )
    for seed in WIDE_SEEDS:
        for split_name, split_indexes, split_families in splits:
            split_episodes = tuple(
                pool.episodes[int(index)] for index in split_indexes
            )
            local_views = build_wide_catalogue(
                split_families,
                episode_digests(split_episodes),
                seed=seed,
            )
            mapped = remap_views(local_views, split_indexes)
            wide_catalogs.append(
                {
                    "seed": int(seed),
                    "split": split_name,
                    "n_views": int(len(mapped)),
                }
            )
            wide_views.extend(mapped)
    print(f"issue #39 wide views={len(wide_views)} catalogs={len(wide_catalogs)}", flush=True)
    wide = _evaluate_views(
        wide_views,
        cache,
        pool.costs,
        pool.scores,
        artifact=artifact,
        policy=policy,
        workers=worker_count,
        label="wide",
    )

    def not_worse(candidate: Mapping[str, float], current: Mapping[str, float]) -> bool:
        return all(
            float(candidate[tier]) <= float(current[tier]) + 1e-12 for tier in TIERS
        )

    current_worst = {
        tier: max(
            float(primary["current"]["max_inflated"][tier]),
            float(wide["current"]["max_inflated"][tier]),
            float(current_dev["tiers"][tier]["inflated_ratio"]),
        )
        for tier in TIERS
    }
    candidate_worst = {
        tier: max(
            float(primary["candidate"]["max_inflated"][tier]),
            float(wide["candidate"]["max_inflated"][tier]),
            float(candidate_dev["tiers"][tier]["inflated_ratio"]),
        )
        for tier in TIERS
    }
    normal_current_worst = {
        tier: max(
            float(primary["normal_batch"]["current_max_inflated"][tier]),
            float(wide["normal_batch"]["current_max_inflated"][tier]),
            float(current_dev["tiers"][tier]["inflated_ratio"]),
        )
        for tier in TIERS
    }
    normal_candidate_worst = {
        tier: max(
            float(primary["normal_batch"]["candidate_max_inflated"][tier]),
            float(wide["normal_batch"]["candidate_max_inflated"][tier]),
            float(candidate_dev["tiers"][tier]["inflated_ratio"]),
        )
        for tier in TIERS
    }
    candidate_violations = int(
        primary["candidate"]["violations"]["view_violations"]
        + wide["candidate"]["violations"]["view_violations"]
    )
    stress_safe = candidate_violations == 0 and bool(candidate_dev["all_safe"])
    worst_ratio_ok = not_worse(candidate_worst, current_worst)
    normal_worst_ok = not_worse(normal_candidate_worst, normal_current_worst)
    normal_within_frozen = all(
        float(normal_candidate_worst[tier]) <= float(FROZEN_MAX_INFLATED[tier]) + 1e-12
        for tier in TIERS
    )
    score_ok = weighted_loss <= WEIGHTED_LOSS_MAX + 1e-12
    replica_ok = bool(replica["matched"])
    passed = bool(
        replica_ok
        and stress_safe
        and worst_ratio_ok
        and score_ok
    )
    failing = []
    if not replica_ok:
        failing.append("serving_replica")
    if not stress_safe:
        failing.append("stress_margin")
    if not worst_ratio_ok:
        failing.append("worst_ratio")
    if not score_ok:
        failing.append("weighted_score")
    return {
        "method": "family_partial_q95",
        "overlay": (
            "128+ path uses Train-frozen serving family_partial_q95 scales; "
            "small-batch Light-lower and power shrink stay off the 128+ path"
        ),
        "workers": int(worker_count),
        "serving_replica": replica,
        "catalogues": {
            "primary_views": int(len(primary_views)),
            "primary_expected": 11_680,
            "wide_views": int(len(wide_views)),
            "wide_expected": 20_610,
            "wide_catalogs": wide_catalogs,
            "reconstructed": bool(
                len(primary_views) == 11_680 and len(wide_views) == 20_610
            ),
        },
        "official_dev": {
            "current": {
                "weighted_quality": current_dev["weighted_quality"],
                "tiers": dict(current_dev["tiers"]),
                "all_safe": current_dev["all_safe"],
            },
            "candidate": {
                "weighted_quality": candidate_dev["weighted_quality"],
                "tiers": dict(candidate_dev["tiers"]),
                "all_safe": candidate_dev["all_safe"],
            },
            "weighted_loss": weighted_loss,
            "loss_limit": WEIGHTED_LOSS_MAX,
            "frozen_weighted": FROZEN_OFFICIAL_WEIGHTED,
        },
        "primary": primary,
        "wide": wide,
        "max_inflated": {
            "current": current_worst,
            "candidate": candidate_worst,
            "frozen": dict(FROZEN_MAX_INFLATED),
            "normal_batch_current": normal_current_worst,
            "normal_batch_candidate": normal_candidate_worst,
        },
        "gates": {
            "coverage_already_passed": True,
            "serving_replica": replica_ok,
            "stress_no_margin_violation": stress_safe,
            "worst_ratio_not_above_current": worst_ratio_ok,
            "normal_batch_worst_not_above_current": normal_worst_ok,
            "normal_batch_within_frozen_max": normal_within_frozen,
            "weighted_loss_at_most_0_0005": score_ok,
        },
        "failing_gates": failing,
        "serve_q95_on_normal_batches": passed,
        "passed": passed,
    }
