# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Run issues #37-#40 without using Dev to choose candidates.

Five grouped Train folds choose quality, cost, surface-robustness, and
small-batch candidates.  Only candidates that clear their Train gate are
evaluated once on Dev.  The script writes build output only and never changes
the serving artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np

from ossp_router import distributional_router as serving_router
from ossp_router.protocol import MODEL_IDS, TIERS, InputBatch
from research.lab.distributional_knapsack import (
    DistributionalPredictions,
    content_tie_keys,
    fit_distributional_models,
    fit_family_calibration,
    predict_distributional,
)
from research.lab.issue39_integration import run_issue_39_integration
from research.lab.generalization_followups import (
    BASE_COMMIT,
    COST_INFLATION,
    EXPERIMENT,
    REQUIRED_MARGIN_FRACTION,
    SMALL_BATCH_LOWER_FRACTIONS,
    SMALL_BATCH_POWERS,
    SMALL_BATCH_UNIQUE_CUTOFF,
    SURFACE_VARIANTS,
    TIER_WEIGHTS,
    action_flip_rate,
    apply_cost_calibration,
    apply_light_lower_calibration,
    blended_quality,
    canonicalized_episodes,
    cost_coverage,
    cross_calibrated_costs,
    cross_light_lower_credits,
    evaluate_small_batches,
    fit_cost_calibration,
    fit_light_lower_calibration,
    fit_serving_quality_models,
    fit_absolute_quality_models,
    fit_stacked_uplifts,
    fit_uplift_models,
    fixed_count_evaluation,
    grouped_bootstrap_quality_delta,
    make_small_batch_views,
    predict_uplifts,
    predict_absolute_quality,
    predict_stacked_uplifts,
    select_vocabulary_for_scores,
    stable_seed,
    stable_surface_episodes,
    transformed_episodes,
)
from research.lab.public_pool import ROOT, PublicPool, load_public_pool


OUT = ROOT / "build" / "run-generalization-followups"
CACHE_VERSION = "generalization-followups-cache-v1"
QUALITY_BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.0)
UPLIFT_SPECS: Mapping[str, Mapping[str, Any]] = {
    "direct_squared": {"loss": "squared_error", "vocabulary": "base", "dropout": 0.0},
    "direct_huber": {"loss": "huber", "vocabulary": "base", "dropout": 0.0},
    "direct_shrink512": {
        "loss": "squared_error",
        "vocabulary": "shrink512",
        "dropout": 0.0,
    },
    "direct_structural": {
        "loss": "squared_error",
        "vocabulary": "structural",
        "dropout": 0.0,
    },
    "direct_dropout": {
        "loss": "squared_error",
        "vocabulary": "base",
        "dropout": 0.25,
    },
}
COST_METHODS = (
    "aggregate_total",
    "global_q90",
    "family_partial_q90",
    "family_partial_q95",
)
STACK_SPECS: Mapping[str, Mapping[str, Any]] = {
    "stack_global_a10": {"alpha": 10.0, "family_interactions": False},
    "stack_family_a10": {"alpha": 10.0, "family_interactions": True},
    "stack_family_a100": {"alpha": 100.0, "family_interactions": True},
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _indices(pool: PublicPool) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(
        [label == "train" for label in pool.split_labels], dtype=bool
    )
    return np.flatnonzero(train), np.flatnonzero(~train)


def _fit_uplift_family(
    episodes: Sequence[Any],
    scores: np.ndarray,
    base_vocabulary: Sequence[str],
    *,
    random_state: int,
) -> dict[str, Any]:
    vocabularies = {
        "base": tuple(base_vocabulary),
        "shrink512": select_vocabulary_for_scores(episodes, scores, size=512),
        "structural": (),
    }
    fits = {}
    for offset, (name, spec) in enumerate(UPLIFT_SPECS.items()):
        fits[name] = fit_uplift_models(
            episodes,
            scores,
            vocabularies[str(spec["vocabulary"])],
            loss=str(spec["loss"]),
            lexical_dropout=float(spec["dropout"]),
            random_state=int(random_state + 7919 * offset),
        )
    return fits


def _predict_bundle(
    base_fit: Any,
    uplift_fits: Mapping[str, Any],
    episodes: Sequence[Any],
) -> dict[str, Any]:
    raw = predict_distributional(base_fit, episodes)
    return {
        "base_quality": raw.quality_mean,
        "cost_mean": raw.cost_mean,
        "cost_q90": raw.cost_q90,
        "uplifts": {
            name: predict_uplifts(fit, episodes)
            for name, fit in uplift_fits.items()
        },
    }


def _fit_fold(
    pool: PublicPool,
    train_indexes: np.ndarray,
    fold: int,
    cache_path: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not refresh:
        cached = joblib.load(cache_path)
        if cached.get("cache_version") == CACHE_VERSION:
            print(f"fold {fold}: loaded cache", flush=True)
            return cached
    train_folds = np.asarray(pool.folds, dtype=np.int64)[train_indexes]
    held_local = np.flatnonzero(train_folds == int(fold))
    fit_local = np.flatnonzero(train_folds != int(fold))
    held_global = train_indexes[held_local]
    fit_global = train_indexes[fit_local]
    fit_episodes = tuple(pool.episodes[int(index)] for index in fit_global)
    held_episodes = tuple(pool.episodes[int(index)] for index in held_global)
    started = time.perf_counter()
    print(
        f"fold {fold}: fit n={len(fit_episodes)} held={len(held_episodes)}",
        flush=True,
    )
    base_fit = fit_distributional_models(
        fit_episodes,
        pool.scores[fit_global],
        pool.costs[fit_global],
        random_state=stable_seed(f"fold-{fold}-base"),
    )
    uplift_fits = _fit_uplift_family(
        fit_episodes,
        pool.scores[fit_global],
        base_fit.vocabulary,
        random_state=stable_seed(f"fold-{fold}-uplift"),
    )
    original = _predict_bundle(base_fit, uplift_fits, held_episodes)
    surfaces = {}
    for variant in SURFACE_VARIANTS:
        surfaces[variant] = _predict_bundle(
            base_fit,
            uplift_fits,
            transformed_episodes(held_episodes, variant),
        )
    payload = {
        "cache_version": CACHE_VERSION,
        "fold": int(fold),
        "held_local": held_local,
        "original": original,
        "surfaces": surfaces,
        "elapsed_s": float(time.perf_counter() - started),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print(f"fold {fold}: wrote cache in {payload['elapsed_s']:.1f}s", flush=True)
    return payload


def _assemble_oof(pool: PublicPool, train_indexes: np.ndarray, folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = int(train_indexes.size)
    result = {
        "base_quality": np.empty((count, 3), dtype=np.float64),
        "cost_mean": np.empty((count, 3), dtype=np.float64),
        "cost_q90": np.empty((count, 3), dtype=np.float64),
        "uplifts": {
            name: np.empty((count, 2), dtype=np.float64) for name in UPLIFT_SPECS
        },
        "surfaces": {
            variant: {
                "base_quality": np.empty((count, 3), dtype=np.float64),
                "cost_mean": np.empty((count, 3), dtype=np.float64),
                "cost_q90": np.empty((count, 3), dtype=np.float64),
                "uplifts": {
                    name: np.empty((count, 2), dtype=np.float64)
                    for name in UPLIFT_SPECS
                },
            }
            for variant in SURFACE_VARIANTS
        },
    }
    seen = np.zeros(count, dtype=bool)
    for payload in folds:
        indexes = np.asarray(payload["held_local"], dtype=np.int64)
        if np.any(seen[indexes]):
            raise RuntimeError("OOF fold overlap")
        seen[indexes] = True
        for key in ("base_quality", "cost_mean", "cost_q90"):
            result[key][indexes] = payload["original"][key]
        for name in UPLIFT_SPECS:
            result["uplifts"][name][indexes] = payload["original"]["uplifts"][name]
        for variant in SURFACE_VARIANTS:
            for key in ("base_quality", "cost_mean", "cost_q90"):
                result["surfaces"][variant][key][indexes] = payload["surfaces"][variant][key]
            for name in UPLIFT_SPECS:
                result["surfaces"][variant]["uplifts"][name][indexes] = payload[
                    "surfaces"
                ][variant]["uplifts"][name]
    if not bool(np.all(seen)):
        raise RuntimeError("OOF predictions are incomplete")
    return result


def _fit_full(
    pool: PublicPool,
    train_indexes: np.ndarray,
    dev_indexes: np.ndarray,
    cache_path: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not refresh:
        cached = joblib.load(cache_path)
        if cached.get("cache_version") == CACHE_VERSION:
            print("full Train fit: loaded cache", flush=True)
            return cached
    train_episodes = tuple(pool.episodes[int(index)] for index in train_indexes)
    dev_episodes = tuple(pool.episodes[int(index)] for index in dev_indexes)
    started = time.perf_counter()
    print("full Train fit: fitting base and uplift heads", flush=True)
    base_fit = fit_distributional_models(
        train_episodes,
        pool.scores[train_indexes],
        pool.costs[train_indexes],
        random_state=stable_seed("full-base"),
    )
    uplift_fits = _fit_uplift_family(
        train_episodes,
        pool.scores[train_indexes],
        base_fit.vocabulary,
        random_state=stable_seed("full-uplift"),
    )
    payload = {
        "cache_version": CACHE_VERSION,
        "train_in_sample": _predict_bundle(base_fit, uplift_fits, train_episodes),
        "dev": _predict_bundle(base_fit, uplift_fits, dev_episodes),
        "dev_surfaces": {
            variant: _predict_bundle(
                base_fit,
                uplift_fits,
                transformed_episodes(dev_episodes, variant),
            )
            for variant in SURFACE_VARIANTS
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print(f"full Train fit: wrote cache in {payload['elapsed_s']:.1f}s", flush=True)
    return payload


def _fit_canonical_fold(
    pool: PublicPool,
    train_indexes: np.ndarray,
    fold: int,
    cache_path: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not refresh:
        cached = joblib.load(cache_path)
        if cached.get("cache_version") == CACHE_VERSION:
            print(f"canonical fold {fold}: loaded cache", flush=True)
            return cached
    train_folds = np.asarray(pool.folds, dtype=np.int64)[train_indexes]
    held_local = np.flatnonzero(train_folds == int(fold))
    fit_local = np.flatnonzero(train_folds != int(fold))
    held_global = train_indexes[held_local]
    fit_global = train_indexes[fit_local]
    fit_episodes = canonicalized_episodes(
        tuple(pool.episodes[int(index)] for index in fit_global)
    )
    held_episodes = canonicalized_episodes(
        tuple(pool.episodes[int(index)] for index in held_global)
    )
    started = time.perf_counter()
    print(f"canonical fold {fold}: fitting quality heads", flush=True)
    fitted = fit_absolute_quality_models(
        fit_episodes,
        pool.scores[fit_global],
        random_state=stable_seed(f"canonical-fold-{fold}"),
    )
    payload = {
        "cache_version": CACHE_VERSION,
        "held_local": held_local,
        "quality": predict_absolute_quality(fitted, held_episodes),
        "elapsed_s": float(time.perf_counter() - started),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print(
        f"canonical fold {fold}: wrote cache in {payload['elapsed_s']:.1f}s",
        flush=True,
    )
    return payload


def _assemble_canonical_oof(
    count: int, payloads: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    quality = np.empty((int(count), 3), dtype=np.float64)
    seen = np.zeros(int(count), dtype=bool)
    for payload in payloads:
        indexes = np.asarray(payload["held_local"], dtype=np.int64)
        if np.any(seen[indexes]):
            raise RuntimeError("canonical OOF fold overlap")
        seen[indexes] = True
        quality[indexes] = payload["quality"]
    if not bool(np.all(seen)):
        raise RuntimeError("canonical OOF predictions are incomplete")
    return quality


def _fit_canonical_full(
    pool: PublicPool,
    train_indexes: np.ndarray,
    dev_indexes: np.ndarray,
    cache_path: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not refresh:
        cached = joblib.load(cache_path)
        if cached.get("cache_version") == CACHE_VERSION:
            print("canonical full Train fit: loaded cache", flush=True)
            return cached
    train_episodes = canonicalized_episodes(
        tuple(pool.episodes[int(index)] for index in train_indexes)
    )
    dev_episodes = canonicalized_episodes(
        tuple(pool.episodes[int(index)] for index in dev_indexes)
    )
    started = time.perf_counter()
    print("canonical full Train fit: fitting quality heads", flush=True)
    fitted = fit_absolute_quality_models(
        train_episodes,
        pool.scores[train_indexes],
        random_state=stable_seed("canonical-full"),
    )
    payload = {
        "cache_version": CACHE_VERSION,
        "dev_quality": predict_absolute_quality(fitted, dev_episodes),
        "elapsed_s": float(time.perf_counter() - started),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print(
        f"canonical full Train fit: wrote cache in {payload['elapsed_s']:.1f}s",
        flush=True,
    )
    return payload


def _fit_inference_canonical_fold(
    pool: PublicPool,
    train_indexes: np.ndarray,
    fold: int,
    cache_path: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not refresh:
        cached = joblib.load(cache_path)
        if (
            cached.get("cache_version") == CACHE_VERSION
            and "stable_original_quality" in cached
        ):
            print(f"inference canonical fold {fold}: loaded cache", flush=True)
            return cached
    train_folds = np.asarray(pool.folds, dtype=np.int64)[train_indexes]
    held_local = np.flatnonzero(train_folds == int(fold))
    fit_local = np.flatnonzero(train_folds != int(fold))
    held_global = train_indexes[held_local]
    fit_global = train_indexes[fit_local]
    fit_episodes = tuple(pool.episodes[int(index)] for index in fit_global)
    held_episodes = tuple(pool.episodes[int(index)] for index in held_global)
    started = time.perf_counter()
    print(f"inference canonical fold {fold}: fitting quality heads", flush=True)
    fitted = fit_serving_quality_models(
        fit_episodes,
        pool.scores[fit_global],
        pool.costs[fit_global],
        random_state=stable_seed(f"fold-{fold}-base"),
    )
    payload = {
        "cache_version": CACHE_VERSION,
        "held_local": held_local,
        "original_quality": predict_absolute_quality(fitted, held_episodes),
        "canonical_quality": predict_absolute_quality(
            fitted, canonicalized_episodes(held_episodes)
        ),
        "stable_original_quality": predict_absolute_quality(
            fitted, stable_surface_episodes(held_episodes)
        ),
        "stable_surface_quality": {
            variant: predict_absolute_quality(
                fitted,
                stable_surface_episodes(
                    transformed_episodes(held_episodes, variant)
                ),
            )
            for variant in SURFACE_VARIANTS
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print(
        f"inference canonical fold {fold}: wrote cache in "
        f"{payload['elapsed_s']:.1f}s",
        flush=True,
    )
    return payload


def _assemble_inference_canonical_oof(
    count: int, payloads: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    original = np.empty((int(count), 3), dtype=np.float64)
    canonical = np.empty((int(count), 3), dtype=np.float64)
    stable_original = np.empty((int(count), 3), dtype=np.float64)
    stable_surfaces = {
        variant: np.empty((int(count), 3), dtype=np.float64)
        for variant in SURFACE_VARIANTS
    }
    seen = np.zeros(int(count), dtype=bool)
    for payload in payloads:
        indexes = np.asarray(payload["held_local"], dtype=np.int64)
        if np.any(seen[indexes]):
            raise RuntimeError("inference canonical OOF fold overlap")
        seen[indexes] = True
        original[indexes] = payload["original_quality"]
        canonical[indexes] = payload["canonical_quality"]
        stable_original[indexes] = payload["stable_original_quality"]
        for variant in SURFACE_VARIANTS:
            stable_surfaces[variant][indexes] = payload["stable_surface_quality"][
                variant
            ]
    if not bool(np.all(seen)):
        raise RuntimeError("inference canonical OOF predictions are incomplete")
    return original, canonical, stable_original, stable_surfaces


def _fit_inference_canonical_full(
    pool: PublicPool,
    train_indexes: np.ndarray,
    dev_indexes: np.ndarray,
    cache_path: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not refresh:
        cached = joblib.load(cache_path)
        if (
            cached.get("cache_version") == CACHE_VERSION
            and "stable_original_dev_quality" in cached
        ):
            print("inference canonical full Train fit: loaded cache", flush=True)
            return cached
    train_episodes = tuple(pool.episodes[int(index)] for index in train_indexes)
    dev_episodes = tuple(pool.episodes[int(index)] for index in dev_indexes)
    started = time.perf_counter()
    print("inference canonical full Train fit: fitting quality heads", flush=True)
    fitted = fit_serving_quality_models(
        train_episodes,
        pool.scores[train_indexes],
        pool.costs[train_indexes],
        random_state=stable_seed("full-base"),
    )
    payload = {
        "cache_version": CACHE_VERSION,
        "original_dev_quality": predict_absolute_quality(fitted, dev_episodes),
        "canonical_dev_quality": predict_absolute_quality(
            fitted, canonicalized_episodes(dev_episodes)
        ),
        "stable_original_dev_quality": predict_absolute_quality(
            fitted, stable_surface_episodes(dev_episodes)
        ),
        "stable_surface_dev_quality": {
            variant: predict_absolute_quality(
                fitted,
                stable_surface_episodes(
                    transformed_episodes(dev_episodes, variant)
                ),
            )
            for variant in SURFACE_VARIANTS
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print(
        "inference canonical full Train fit: wrote cache in "
        f"{payload['elapsed_s']:.1f}s",
        flush=True,
    )
    return payload


def _quality_candidate(
    base: np.ndarray,
    uplifts: Mapping[str, np.ndarray],
    name: str,
    weight: float,
) -> np.ndarray:
    signal = np.asarray(uplifts[name], dtype=np.float64)
    if signal.shape == np.asarray(base).shape:
        if float(weight) != 1.0:
            raise ValueError("absolute quality candidates do not support blending")
        return signal
    return blended_quality(base, signal, float(weight))


def _quality_experiment(
    oof: Mapping[str, Any],
    full: Mapping[str, Any],
    train_scores: np.ndarray,
    train_costs: np.ndarray,
    train_groups: Sequence[str],
    train_families: Sequence[str],
    train_folds: Sequence[int],
    dev_scores: np.ndarray,
    dev_costs: np.ndarray,
    dev_groups: Sequence[str],
    dev_families: Sequence[str],
) -> dict[str, Any]:
    baseline = fixed_count_evaluation(
        oof["base_quality"], train_scores, train_costs, train_groups
    )
    candidates = []
    for source in ("direct_squared", "direct_huber"):
        for weight in QUALITY_BLEND_WEIGHTS:
            quality = _quality_candidate(
                oof["base_quality"], oof["uplifts"], source, weight
            )
            evaluated = fixed_count_evaluation(
                quality, train_scores, train_costs, train_groups
            )
            interval = grouped_bootstrap_quality_delta(
                baseline["actions"],
                evaluated["actions"],
                train_scores,
                train_groups,
                seed=stable_seed(f"quality-{source}-{weight}"),
            )
            delta = float(evaluated["weighted_quality"] - baseline["weighted_quality"])
            candidates.append(
                {
                    "source": source,
                    "weight": weight,
                    "weighted_quality": evaluated["weighted_quality"],
                    "delta": delta,
                    "bootstrap": interval,
                    "train_gate": delta >= 0.001 and interval["lower_95"] > 0.0,
                    "tiers": evaluated["tiers"],
                }
            )
    fold_ids = np.asarray(train_folds, dtype=np.int64)
    family_array = np.asarray(tuple(str(value) for value in train_families), dtype=object)
    group_array = np.asarray(tuple(str(value) for value in train_groups), dtype=object)
    stack_sources = (
        "direct_squared",
        "direct_huber",
        "direct_structural",
    )
    for stack_name, spec in STACK_SPECS.items():
        stacked = np.empty((len(train_scores), 2), dtype=np.float64)
        for fold in sorted(set(int(value) for value in fold_ids)):
            held = fold_ids == fold
            fitted = ~held
            model = fit_stacked_uplifts(
                oof["base_quality"][fitted],
                {
                    name: oof["uplifts"][name][fitted]
                    for name in stack_sources
                },
                train_scores[fitted],
                family_array[fitted],
                group_array[fitted],
                sources=stack_sources,
                alpha=float(spec["alpha"]),
                family_interactions=bool(spec["family_interactions"]),
            )
            stacked[held] = predict_stacked_uplifts(
                model,
                oof["base_quality"][held],
                {name: oof["uplifts"][name][held] for name in stack_sources},
                family_array[held],
            )
        final_model = fit_stacked_uplifts(
            oof["base_quality"],
            {name: oof["uplifts"][name] for name in stack_sources},
            train_scores,
            train_families,
            train_groups,
            sources=stack_sources,
            alpha=float(spec["alpha"]),
            family_interactions=bool(spec["family_interactions"]),
        )
        dev_stacked = predict_stacked_uplifts(
            final_model,
            full["dev"]["base_quality"],
            {name: full["dev"]["uplifts"][name] for name in stack_sources},
            dev_families,
        )
        oof["uplifts"][stack_name] = stacked
        full["dev"]["uplifts"][stack_name] = dev_stacked
        quality = blended_quality(oof["base_quality"], stacked, 1.0)
        evaluated = fixed_count_evaluation(
            quality, train_scores, train_costs, train_groups
        )
        interval = grouped_bootstrap_quality_delta(
            baseline["actions"],
            evaluated["actions"],
            train_scores,
            train_groups,
            seed=stable_seed(f"quality-{stack_name}"),
        )
        delta = float(evaluated["weighted_quality"] - baseline["weighted_quality"])
        candidates.append(
            {
                "source": stack_name,
                "weight": 1.0,
                "weighted_quality": evaluated["weighted_quality"],
                "delta": delta,
                "bootstrap": interval,
                "train_gate": delta >= 0.001 and interval["lower_95"] > 0.0,
                "tiers": evaluated["tiers"],
                "stack": dict(spec),
            }
        )
    passed = [row for row in candidates if row["train_gate"]]
    selected = max(passed, key=lambda row: row["weighted_quality"]) if passed else None
    dev = None
    if selected is not None:
        baseline_dev = fixed_count_evaluation(
            full["dev"]["base_quality"], dev_scores, dev_costs, dev_groups
        )
        quality_dev = _quality_candidate(
            full["dev"]["base_quality"],
            full["dev"]["uplifts"],
            str(selected["source"]),
            float(selected["weight"]),
        )
        candidate_dev = fixed_count_evaluation(
            quality_dev, dev_scores, dev_costs, dev_groups
        )
        delta = float(
            candidate_dev["weighted_quality"] - baseline_dev["weighted_quality"]
        )
        dev = {
            "baseline": {
                "weighted_quality": baseline_dev["weighted_quality"],
                "tiers": baseline_dev["tiers"],
            },
            "candidate": {
                "weighted_quality": candidate_dev["weighted_quality"],
                "tiers": candidate_dev["tiers"],
            },
            "delta": delta,
            "gate": delta >= 0.0,
        }
    return {
        "issue": 38,
        "baseline_train_oof": {
            "weighted_quality": baseline["weighted_quality"],
            "tiers": baseline["tiers"],
        },
        "candidates_train_oof": candidates,
        "selected": None
        if selected is None
        else {"source": selected["source"], "weight": selected["weight"]},
        "dev_evaluated": dev is not None,
        "dev": dev,
        "passed": bool(dev is not None and dev["gate"]),
    }


def _surface_metrics(
    original_quality: np.ndarray,
    surface_quality: Mapping[str, np.ndarray],
    scores: np.ndarray,
    costs: np.ndarray,
    group_keys: Sequence[str],
) -> dict[str, Any]:
    original = fixed_count_evaluation(original_quality, scores, costs, group_keys)
    variants = {}
    flip_rates = []
    transformed_quality = []
    worst_ratios = {tier: float(original["tiers"][tier]["ratio"]) for tier in TIERS}
    for variant in SURFACE_VARIANTS:
        evaluated = fixed_count_evaluation(
            surface_quality[variant], scores, costs, group_keys
        )
        flip = action_flip_rate(original["actions"], evaluated["actions"])
        flip_rates.append(flip)
        transformed_quality.append(float(evaluated["weighted_quality"]))
        for tier in TIERS:
            worst_ratios[tier] = max(
                worst_ratios[tier], float(evaluated["tiers"][tier]["ratio"])
            )
        variants[variant] = {
            "flip_rate": flip,
            "weighted_quality": evaluated["weighted_quality"],
            "tiers": evaluated["tiers"],
        }
    return {
        "actions": original["actions"],
        "original_weighted_quality": original["weighted_quality"],
        "original_tiers": original["tiers"],
        "mean_flip_rate": float(np.mean(flip_rates)),
        "mean_transformed_weighted_quality": float(np.mean(transformed_quality)),
        "worst_ratio": worst_ratios,
        "variants": variants,
    }


def _surface_experiment(
    oof: Mapping[str, Any],
    full: Mapping[str, Any],
    canonical_oof: np.ndarray,
    canonical_dev: np.ndarray,
    inference_canonical_oof: np.ndarray,
    inference_canonical_dev: np.ndarray,
    stable_oof: np.ndarray,
    stable_oof_surfaces: Mapping[str, np.ndarray],
    stable_dev: np.ndarray,
    stable_dev_surfaces: Mapping[str, np.ndarray],
    train_scores: np.ndarray,
    train_costs: np.ndarray,
    train_groups: Sequence[str],
    dev_scores: np.ndarray,
    dev_costs: np.ndarray,
    dev_groups: Sequence[str],
) -> dict[str, Any]:
    baseline_surface = {
        variant: oof["surfaces"][variant]["base_quality"]
        for variant in SURFACE_VARIANTS
    }
    baseline = _surface_metrics(
        oof["base_quality"],
        baseline_surface,
        train_scores,
        train_costs,
        train_groups,
    )
    canonical_candidates = {
        "canonical_absolute": (canonical_oof, canonical_dev),
        "canonical_inference": (
            inference_canonical_oof,
            inference_canonical_dev,
        ),
    }
    for name, (train_quality, dev_quality) in canonical_candidates.items():
        oof["uplifts"][name] = train_quality
        full["dev"]["uplifts"][name] = dev_quality
        for variant in SURFACE_VARIANTS:
            oof["surfaces"][variant]["uplifts"][name] = train_quality
            full["dev_surfaces"][variant]["uplifts"][name] = dev_quality
    oof["uplifts"]["stable_inference"] = stable_oof
    full["dev"]["uplifts"]["stable_inference"] = stable_dev
    for variant in SURFACE_VARIANTS:
        oof["surfaces"][variant]["uplifts"]["stable_inference"] = (
            stable_oof_surfaces[variant]
        )
        full["dev_surfaces"][variant]["uplifts"]["stable_inference"] = (
            stable_dev_surfaces[variant]
        )
    rows = []
    surface_specs = [
        (source, weight)
        for source in (
            "direct_shrink512",
            "direct_structural",
            "direct_dropout",
        )
        for weight in QUALITY_BLEND_WEIGHTS
    ]
    surface_specs.append(("canonical_absolute", 1.0))
    surface_specs.append(("canonical_inference", 1.0))
    surface_specs.append(("stable_inference", 1.0))
    for source, weight in surface_specs:
        original = _quality_candidate(
            oof["base_quality"], oof["uplifts"], source, weight
        )
        surfaces = {
            variant: _quality_candidate(
                oof["surfaces"][variant]["base_quality"],
                oof["surfaces"][variant]["uplifts"],
                source,
                weight,
            )
            for variant in SURFACE_VARIANTS
        }
        metrics = _surface_metrics(
            original, surfaces, train_scores, train_costs, train_groups
        )
        if baseline["mean_flip_rate"] > 0.0:
            reduction = (
                1.0
                - metrics["mean_flip_rate"] / baseline["mean_flip_rate"]
            )
        else:
            reduction = 0.0
        original_loss = (
            baseline["original_weighted_quality"]
            - metrics["original_weighted_quality"]
        )
        train_gate = (
            reduction >= 0.25
            and original_loss <= 0.0005
            and metrics["mean_transformed_weighted_quality"]
            >= baseline["mean_transformed_weighted_quality"]
        )
        rows.append(
            {
                "source": source,
                "weight": weight,
                "flip_reduction": reduction,
                "original_quality_loss": original_loss,
                "train_gate": train_gate,
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key != "actions"
                },
            }
        )
    prior_vetoed_sources = {
        "direct_dropout",
        "canonical_absolute",
        "canonical_inference",
    }
    passed = [
        row
        for row in rows
        if row["train_gate"] and row["source"] not in prior_vetoed_sources
    ]
    # Stability is this experiment's primary endpoint.  Quality is already a
    # hard gate, so use it only to break ties between equally stable options.
    selected = (
        max(
            passed,
            key=lambda row: (
                row["flip_reduction"],
                row["metrics"]["mean_transformed_weighted_quality"],
            ),
        )
        if passed
        else None
    )
    dev = None
    if selected is not None:
        baseline_dev = _surface_metrics(
            full["dev"]["base_quality"],
            {
                variant: full["dev_surfaces"][variant]["base_quality"]
                for variant in SURFACE_VARIANTS
            },
            dev_scores,
            dev_costs,
            dev_groups,
        )
        source = str(selected["source"])
        weight = float(selected["weight"])
        candidate_dev = _surface_metrics(
            _quality_candidate(
                full["dev"]["base_quality"], full["dev"]["uplifts"], source, weight
            ),
            {
                variant: _quality_candidate(
                    full["dev_surfaces"][variant]["base_quality"],
                    full["dev_surfaces"][variant]["uplifts"],
                    source,
                    weight,
                )
                for variant in SURFACE_VARIANTS
            },
            dev_scores,
            dev_costs,
            dev_groups,
        )
        reduction = (
            1.0 - candidate_dev["mean_flip_rate"] / baseline_dev["mean_flip_rate"]
            if baseline_dev["mean_flip_rate"] > 0.0
            else 0.0
        )
        original_loss = (
            baseline_dev["original_weighted_quality"]
            - candidate_dev["original_weighted_quality"]
        )
        dev_gate = (
            reduction >= 0.25
            and original_loss <= 0.0005
            and candidate_dev["mean_transformed_weighted_quality"]
            >= baseline_dev["mean_transformed_weighted_quality"]
        )
        dev = {
            "baseline": {key: value for key, value in baseline_dev.items() if key != "actions"},
            "candidate": {
                key: value for key, value in candidate_dev.items() if key != "actions"
            },
            "flip_reduction": reduction,
            "original_quality_loss": original_loss,
            "gate": dev_gate,
        }
    return {
        "issue": 40,
        "baseline_train_oof": {
            key: value for key, value in baseline.items() if key != "actions"
        },
        "candidates_train_oof": rows,
        "prior_vetoed_sources": sorted(prior_vetoed_sources),
        "selected": None
        if selected is None
        else {"source": selected["source"], "weight": selected["weight"]},
        "dev_evaluated": dev is not None,
        "dev": dev,
        "passed": bool(dev is not None and dev["gate"]),
        "runtime_gate": "requires serving-artifact candidate before promotion",
    }


def _runtime_surface_audit(
    pool: PublicPool,
    dev_indexes: np.ndarray,
    cache_path: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    """Run the conservative normalizer through the bundled serving router."""

    runtime_cache_version = "generalization-runtime-surface-v2"
    if cache_path.is_file() and not refresh:
        cached = joblib.load(cache_path)
        if cached.get("cache_version") == runtime_cache_version:
            print("runtime surface audit: loaded cache", flush=True)
            return cached["report"]
    episodes = tuple(pool.episodes[int(index)] for index in dev_indexes)
    scores = np.asarray(pool.scores[dev_indexes], dtype=np.float64)
    costs = np.asarray(pool.costs[dev_indexes], dtype=np.float64)
    rows = np.arange(len(episodes), dtype=np.int64)
    artifact = serving_router.load_bundled_artifact()

    def route(batch_episodes: Sequence[Any]) -> tuple[dict[str, np.ndarray], float]:
        inputs = InputBatch(
            schema_version=pool.inputs.schema_version,
            challenge_id=pool.inputs.challenge_id,
            split="dev",
            episodes=tuple(batch_episodes),
        )
        started = time.perf_counter()
        actions = {}
        for tier in TIERS:
            submission = serving_router.make_submission(
                inputs,
                pool.policy,
                artifact,
                tier,
            )
            actions[tier] = np.asarray(
                [MODEL_IDS.index(decision.model_id) for decision in submission.decisions],
                dtype=np.int8,
            )
        return actions, float(time.perf_counter() - started)

    def metrics(actions: Mapping[str, np.ndarray]) -> dict[str, Any]:
        weighted = 0.0
        tier_rows = {}
        for tier in TIERS:
            selected = np.asarray(actions[tier], dtype=np.int64)
            quality = float(np.mean(scores[rows, selected]))
            ratio = float(
                np.sum(costs[rows, selected], dtype=np.float64)
                / np.sum(costs[:, 0], dtype=np.float64)
            )
            inflated = ratio * COST_INFLATION
            limit = (
                float(pool.policy.tiers[tier].budget_multiplier)
                * (1.0 - REQUIRED_MARGIN_FRACTION)
            )
            weighted += TIER_WEIGHTS[tier] * quality
            tier_rows[tier] = {
                "quality": quality,
                "ratio": ratio,
                "inflated_ratio": inflated,
                "limit": limit,
                "safe": inflated <= limit + 1e-12,
                "model_counts": {
                    MODEL_IDS[column]: int(np.count_nonzero(selected == column))
                    for column in range(3)
                },
            }
        return {
            "weighted_quality": float(weighted),
            "tiers": tier_rows,
            "all_safe": all(bool(row["safe"]) for row in tier_rows.values()),
        }

    baseline_actions, baseline_runtime = route(episodes)
    normalization_started = time.perf_counter()
    stable_episodes = stable_surface_episodes(episodes)
    normalization_s = float(time.perf_counter() - normalization_started)
    candidate_actions, candidate_route_runtime = route(stable_episodes)
    candidate_runtime = normalization_s + candidate_route_runtime
    baseline = metrics(baseline_actions)
    candidate = metrics(candidate_actions)
    original_action_flip = action_flip_rate(baseline_actions, candidate_actions)
    baseline_variants = {}
    candidate_variants = {}
    baseline_flips = []
    candidate_flips = []
    for variant in SURFACE_VARIANTS:
        transformed = transformed_episodes(episodes, variant)
        baseline_variant_actions, _elapsed = route(transformed)
        stable_transformed = stable_surface_episodes(transformed)
        if stable_transformed == stable_episodes:
            candidate_variant_actions = candidate_actions
        else:
            candidate_variant_actions, _elapsed = route(stable_transformed)
        baseline_flip = action_flip_rate(baseline_actions, baseline_variant_actions)
        candidate_flip = action_flip_rate(candidate_actions, candidate_variant_actions)
        baseline_flips.append(baseline_flip)
        candidate_flips.append(candidate_flip)
        baseline_variants[variant] = {
            **metrics(baseline_variant_actions),
            "flip_rate": baseline_flip,
        }
        candidate_variants[variant] = {
            **metrics(candidate_variant_actions),
            "flip_rate": candidate_flip,
        }
    baseline_mean_flip = float(np.mean(baseline_flips))
    candidate_mean_flip = float(np.mean(candidate_flips))
    flip_reduction = (
        1.0 - candidate_mean_flip / baseline_mean_flip
        if baseline_mean_flip > 0.0
        else 0.0
    )
    baseline_transformed_quality = float(
        np.mean([row["weighted_quality"] for row in baseline_variants.values()])
    )
    candidate_transformed_quality = float(
        np.mean([row["weighted_quality"] for row in candidate_variants.values()])
    )
    original_loss = float(
        baseline["weighted_quality"] - candidate["weighted_quality"]
    )
    latency_ratio = candidate_runtime / max(baseline_runtime, 1e-12)
    artifact_path = (
        ROOT / "src" / "ossp_router" / "resources" / serving_router.ARTIFACT_RESOURCE
    )
    report = {
        "source": "stable_inference",
        "baseline": baseline,
        "candidate": candidate,
        "original_action_flip_rate": original_action_flip,
        "baseline_variants": baseline_variants,
        "candidate_variants": candidate_variants,
        "baseline_mean_flip_rate": baseline_mean_flip,
        "candidate_mean_flip_rate": candidate_mean_flip,
        "flip_reduction": flip_reduction,
        "baseline_mean_transformed_quality": baseline_transformed_quality,
        "candidate_mean_transformed_quality": candidate_transformed_quality,
        "original_quality_loss": original_loss,
        "artifact_bytes": int(artifact_path.stat().st_size),
        "artifact_size_ratio": 1.0,
        "baseline_runtime_s": baseline_runtime,
        "candidate_runtime_s": candidate_runtime,
        "runtime_ratio": latency_ratio,
    }
    report["gate"] = (
        bool(candidate["all_safe"])
        and all(
            bool(row["all_safe"]) for row in candidate_variants.values()
        )
        and flip_reduction >= 0.25
        and candidate_transformed_quality >= baseline_transformed_quality
        and original_loss <= 0.0005
        and report["artifact_size_ratio"] <= 1.25
        and latency_ratio <= 1.25
    )
    payload = {
        "cache_version": runtime_cache_version,
        "report": report,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print("runtime surface audit: wrote cache", flush=True)
    return report


def _raw_predictions(bundle: Mapping[str, Any]) -> DistributionalPredictions:
    return DistributionalPredictions(
        np.asarray(bundle["base_quality"], dtype=np.float64),
        np.asarray(bundle["cost_mean"], dtype=np.float64),
        np.asarray(bundle["cost_q90"], dtype=np.float64),
    )


def _cost_experiment(
    oof: Mapping[str, Any],
    full: Mapping[str, Any],
    train_costs: np.ndarray,
    train_families: Sequence[str],
    train_folds: Sequence[int],
    dev_costs: np.ndarray,
    dev_families: Sequence[str],
) -> tuple[dict[str, Any], Optional[str], Optional[DistributionalPredictions], Any]:
    raw_oof = _raw_predictions(oof)
    train_rows = []
    cross_predictions = {}
    for method in COST_METHODS:
        calibrated = cross_calibrated_costs(
            raw_oof,
            train_costs,
            train_families,
            train_folds,
            method=method,
        )
        cross_predictions[method] = calibrated
        coverage = cost_coverage(calibrated, train_costs, train_families)
        train_rows.append({"method": method, "coverage": coverage})
    eligible = [
        row
        for row in train_rows
        if row["method"] != "aggregate_total" and row["coverage"]["all_cells_passed"]
    ]
    selected_row = (
        min(eligible, key=lambda row: row["coverage"]["mean_upper_to_actual"])
        if eligible
        else None
    )
    selected_method = None if selected_row is None else str(selected_row["method"])
    dev = None
    selected_dev = None
    selected_calibration = None
    raw_dev = _raw_predictions(full["dev"])
    in_sample_train = _raw_predictions(full["train_in_sample"])
    current_calibration = fit_family_calibration(
        in_sample_train, train_costs, train_families
    )
    current_dev = apply_cost_calibration(raw_dev, dev_families, current_calibration)
    current_coverage = cost_coverage(current_dev, dev_costs, dev_families)
    if selected_method is not None:
        selected_calibration = fit_cost_calibration(
            raw_oof,
            train_costs,
            train_families,
            method=selected_method,
        )
        selected_dev = apply_cost_calibration(
            raw_dev, dev_families, selected_calibration
        )
        candidate_coverage = cost_coverage(
            selected_dev, dev_costs, dev_families
        )
        dev = {
            "current_aggregate": current_coverage,
            "candidate": candidate_coverage,
            "coverage_gate": bool(candidate_coverage["all_cells_passed"]),
        }
        dev["gate"] = False
    report = {
        "issue": 39,
        "candidates_train_cross_calibrated": train_rows,
        "selected": selected_method,
        "dev_evaluated": dev is not None,
        "dev": dev,
        "coverage_passed": bool(
            dev is not None and dev["coverage_gate"]
        ),
        "integration_gate": (
            "pending: normal-batch stress, worst-ratio, and weighted-score replay"
        ),
        "passed": False,
    }
    return report, selected_method, selected_dev, selected_calibration


def _selected_quality(
    base: np.ndarray,
    uplifts: Mapping[str, np.ndarray],
    quality_report: Mapping[str, Any],
    surface_report: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    if surface_report["passed"]:
        selected = surface_report["selected"]
        return (
            _quality_candidate(base, uplifts, selected["source"], selected["weight"]),
            f"surface:{selected['source']}@{selected['weight']}",
        )
    if quality_report["passed"]:
        selected = quality_report["selected"]
        return (
            _quality_candidate(base, uplifts, selected["source"], selected["weight"]),
            f"quality:{selected['source']}@{selected['weight']}",
        )
    return np.asarray(base, dtype=np.float64), "current_absolute_quality"


def _small_batch_experiment(
    pool: PublicPool,
    train_indexes: np.ndarray,
    dev_indexes: np.ndarray,
    oof: Mapping[str, Any],
    full: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    surface_report: Mapping[str, Any],
    cost_method: Optional[str],
    selected_dev_cost: Optional[DistributionalPredictions],
    selected_calibration: Any,
) -> dict[str, Any]:
    train_scores = pool.scores[train_indexes]
    train_costs = pool.costs[train_indexes]
    train_families = tuple(pool.families[int(index)] for index in train_indexes)
    train_ties = content_tie_keys(
        tuple(pool.episodes[int(index)] for index in train_indexes)
    )
    train_quality, quality_name = _selected_quality(
        oof["base_quality"], oof["uplifts"], quality_report, surface_report
    )
    raw_oof = _raw_predictions(oof)
    chosen_method = cost_method or "aggregate_total"
    train_pred = cross_calibrated_costs(
        DistributionalPredictions(
            train_quality, raw_oof.cost_mean, raw_oof.cost_q90
        ),
        train_costs,
        train_families,
        np.asarray(pool.folds, dtype=np.int64)[train_indexes],
        method=chosen_method,
    )
    calibration = fit_cost_calibration(
        raw_oof,
        train_costs,
        train_families,
        method=chosen_method,
    )
    budget_multipliers = {
        tier: float(pool.policy.tiers[tier].budget_multiplier) for tier in TIERS
    }
    train_views = make_small_batch_views(
        train_families,
        train_ties,
        train_pred,
        draws_per_kind=12,
        seed=stable_seed("small-batch-train"),
    )
    aggressive = evaluate_small_batches(
        train_pred,
        train_scores,
        train_costs,
        train_families,
        train_ties,
        train_views,
        calibration.reference_proportions,
        budget_multipliers,
        power=0.0,
        bootstrap_seed=stable_seed("small-train-aggressive"),
    )
    aggressive_summary = {
        key: value for key, value in aggressive.items() if key != "rows"
    }
    gate_discriminates = int(aggressive_summary["margin_violations"]) > 0
    train_candidates = []
    train_fold_ids = np.asarray(pool.folds, dtype=np.int64)[train_indexes]
    for lower_fraction in SMALL_BATCH_LOWER_FRACTIONS:
        lower_credit = cross_light_lower_credits(
            train_pred,
            train_costs,
            train_families,
            train_fold_ids,
            fraction=lower_fraction,
        )
        for full_upper in (False, True):
            envelope = "full_upper" if full_upper else "tier_tail"
            for power in SMALL_BATCH_POWERS:
                evaluated = evaluate_small_batches(
                    train_pred,
                    train_scores,
                    train_costs,
                    train_families,
                    train_ties,
                    train_views,
                    calibration.reference_proportions,
                    budget_multipliers,
                    power=power,
                    light_lower_credit=lower_credit,
                    full_upper=full_upper,
                    bootstrap_seed=stable_seed(
                        f"small-train-{lower_fraction}-{envelope}-{power}"
                    ),
                )
                summary = {
                    key: value for key, value in evaluated.items() if key != "rows"
                }
                summary.update(
                    {
                        "lower_fraction": lower_fraction,
                        "upper_envelope": envelope,
                        "full_upper": full_upper,
                    }
                )
                summary["train_gate"] = (
                    summary["margin_violations"] == 0
                    and summary["mean_weighted_quality_delta"] >= 0.001
                    and summary["quality_delta_lower_95"] > 0.0
                )
                train_candidates.append(summary)
    eligible = [
        row
        for row in train_candidates
        if row["train_gate"] and gate_discriminates
    ]
    # Safety and a positive grouped confidence bound are hard gates.  Once a
    # candidate clears both, the issue's objective is to recover most quality.
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["mean_weighted_quality_delta"]),
                float(row["quality_delta_lower_95"]),
            ),
        )
        if eligible
        else None
    )
    dev = None
    if selected is not None:
        dev_scores = pool.scores[dev_indexes]
        dev_costs = pool.costs[dev_indexes]
        dev_families = tuple(pool.families[int(index)] for index in dev_indexes)
        dev_ties = content_tie_keys(
            tuple(pool.episodes[int(index)] for index in dev_indexes)
        )
        dev_quality, _name = _selected_quality(
            full["dev"]["base_quality"],
            full["dev"]["uplifts"],
            quality_report,
            surface_report,
        )
        if selected_dev_cost is not None and selected_calibration is not None:
            dev_pred = DistributionalPredictions(
                dev_quality,
                selected_dev_cost.cost_mean,
                selected_dev_cost.cost_q90,
            )
            reference = selected_calibration.reference_proportions
        else:
            raw_dev = _raw_predictions(full["dev"])
            fallback_calibration = fit_cost_calibration(
                raw_oof,
                train_costs,
                train_families,
                method="aggregate_total",
            )
            fallback_dev = apply_cost_calibration(
                raw_dev, dev_families, fallback_calibration
            )
            dev_pred = DistributionalPredictions(
                dev_quality, fallback_dev.cost_mean, fallback_dev.cost_q90
            )
            reference = fallback_calibration.reference_proportions
        lower_calibration = fit_light_lower_calibration(
            train_pred,
            train_costs,
            train_families,
            fraction=float(selected["lower_fraction"]),
        )
        dev_lower_credit = apply_light_lower_calibration(
            dev_pred,
            dev_families,
            lower_calibration,
        )
        dev_views = make_small_batch_views(
            dev_families,
            dev_ties,
            dev_pred,
            draws_per_kind=12,
            seed=stable_seed("small-batch-dev"),
        )
        evaluated = evaluate_small_batches(
            dev_pred,
            dev_scores,
            dev_costs,
            dev_families,
            dev_ties,
            dev_views,
            reference,
            budget_multipliers,
            power=float(selected["power"]),
            light_lower_credit=dev_lower_credit,
            full_upper=bool(selected["full_upper"]),
            bootstrap_seed=stable_seed("small-dev-two-sided-final"),
        )
        dev = {key: value for key, value in evaluated.items() if key != "rows"}
        dev.update(
            {
                "lower_fraction": selected["lower_fraction"],
                "upper_envelope": selected["upper_envelope"],
            }
        )
        dev["gate"] = (
            dev["margin_violations"] == 0
            and dev["mean_weighted_quality_delta"] >= 0.001
            and dev["quality_delta_lower_95"] > 0.0
        )
    return {
        "issue": 37,
        "quality_signal": quality_name,
        "cost_calibration": chosen_method,
        "experiment_phase": "two-sided-envelope-after-denominator-veto",
        "activation": {
            "minimum_unique_content_groups": 1,
            "maximum_unique_content_groups": SMALL_BATCH_UNIQUE_CUTOFF - 1,
            "normal_batch_path_unchanged": True,
        },
        "train_candidates": train_candidates,
        "aggressive_control": aggressive_summary,
        "aggressive_control_failed": gate_discriminates,
        "selected_power": None if selected is None else selected["power"],
        "selected": None
        if selected is None
        else {
            "power": selected["power"],
            "lower_fraction": selected["lower_fraction"],
            "upper_envelope": selected["upper_envelope"],
        },
        "dev_evaluated": dev is not None,
        "dev": dev,
        "passed": bool(dev is not None and dev["gate"]),
    }


def _audit_document(
    pool: PublicPool,
    train_indexes: np.ndarray,
    oof: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    surface_report: Mapping[str, Any],
) -> dict[str, Any]:
    quality, name = _selected_quality(
        oof["base_quality"], oof["uplifts"], quality_report, surface_report
    )
    rows = []
    for local, global_index in enumerate(train_indexes):
        rows.append(
            {
                "group_key": pool.group_keys[int(global_index)],
                "fold": int(pool.folds[int(global_index)]),
                "family": pool.families[int(global_index)],
                "base_quality": oof["base_quality"][local].tolist(),
                "selected_quality": quality[local].tolist(),
                "cost_mean": oof["cost_mean"][local].tolist(),
                "cost_q90": oof["cost_q90"][local].tolist(),
            }
        )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "scope": "Train grouped out-of-fold predictions; no prompt or episode ID",
        "quality_signal": name,
        "rows": rows,
    }


def run(pool: PublicPool, cache_dir: Path, *, refresh_cache: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    train_indexes, dev_indexes = _indices(pool)
    fold_values = sorted(
        set(int(value) for value in np.asarray(pool.folds)[train_indexes])
    )
    fold_payloads = [
        _fit_fold(
            pool,
            train_indexes,
            fold,
            cache_dir / f"fold-{fold}.joblib",
            refresh=refresh_cache,
        )
        for fold in fold_values
    ]
    oof = _assemble_oof(pool, train_indexes, fold_payloads)
    canonical_payloads = [
        _fit_canonical_fold(
            pool,
            train_indexes,
            fold,
            cache_dir / f"canonical-fold-{fold}.joblib",
            refresh=refresh_cache,
        )
        for fold in fold_values
    ]
    canonical_oof = _assemble_canonical_oof(
        int(train_indexes.size), canonical_payloads
    )
    inference_canonical_payloads = [
        _fit_inference_canonical_fold(
            pool,
            train_indexes,
            fold,
            cache_dir / f"inference-canonical-fold-{fold}.joblib",
            refresh=refresh_cache,
        )
        for fold in fold_values
    ]
    (
        inference_original_oof,
        inference_canonical_oof,
        stable_oof,
        stable_oof_surfaces,
    ) = _assemble_inference_canonical_oof(
        int(train_indexes.size), inference_canonical_payloads
    )
    if not np.allclose(
        inference_original_oof,
        oof["base_quality"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("serving quality refit did not reproduce OOF predictions")
    full = _fit_full(
        pool,
        train_indexes,
        dev_indexes,
        cache_dir / "full-train.joblib",
        refresh=refresh_cache,
    )
    canonical_full = _fit_canonical_full(
        pool,
        train_indexes,
        dev_indexes,
        cache_dir / "canonical-full-train.joblib",
        refresh=refresh_cache,
    )
    inference_canonical_full = _fit_inference_canonical_full(
        pool,
        train_indexes,
        dev_indexes,
        cache_dir / "inference-canonical-full-train.joblib",
        refresh=refresh_cache,
    )
    if not np.allclose(
        inference_canonical_full["original_dev_quality"],
        full["dev"]["base_quality"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("serving quality refit did not reproduce Dev predictions")
    train_scores = pool.scores[train_indexes]
    train_costs = pool.costs[train_indexes]
    dev_scores = pool.scores[dev_indexes]
    dev_costs = pool.costs[dev_indexes]
    train_groups = tuple(pool.group_keys[int(index)] for index in train_indexes)
    dev_groups = tuple(pool.group_keys[int(index)] for index in dev_indexes)
    train_families = tuple(pool.families[int(index)] for index in train_indexes)
    dev_families = tuple(pool.families[int(index)] for index in dev_indexes)
    train_folds = np.asarray(pool.folds, dtype=np.int64)[train_indexes]
    print("evaluating issue #38 quality objective", flush=True)
    quality_report = _quality_experiment(
        oof,
        full,
        train_scores,
        train_costs,
        train_groups,
        train_families,
        train_folds,
        dev_scores,
        dev_costs,
        dev_groups,
        dev_families,
    )
    print("evaluating issue #40 surface robustness", flush=True)
    surface_report = _surface_experiment(
        oof,
        full,
        canonical_oof,
        canonical_full["dev_quality"],
        inference_canonical_oof,
        inference_canonical_full["canonical_dev_quality"],
        stable_oof,
        stable_oof_surfaces,
        inference_canonical_full["stable_original_dev_quality"],
        inference_canonical_full["stable_surface_dev_quality"],
        train_scores,
        train_costs,
        train_groups,
        dev_scores,
        dev_costs,
        dev_groups,
    )
    runtime_surface = None
    if (
        surface_report["passed"]
        and surface_report["selected"] is not None
        and surface_report["selected"]["source"] == "stable_inference"
    ):
        print("evaluating issue #40 bundled runtime", flush=True)
        runtime_surface = _runtime_surface_audit(
            pool,
            dev_indexes,
            cache_dir / "runtime-surface-audit.joblib",
            refresh=refresh_cache,
        )
        surface_report["runtime_audit"] = runtime_surface
        surface_report["passed"] = bool(runtime_surface["gate"])
        surface_report["runtime_gate"] = (
            "passed" if runtime_surface["gate"] else "failed"
        )
    print("evaluating issue #39 cost calibration", flush=True)
    cost_report, cost_method, selected_dev_cost, selected_calibration = _cost_experiment(
        oof,
        full,
        train_costs,
        train_families,
        train_folds,
        dev_costs,
        dev_families,
    )
    print("evaluating issue #37 small batches", flush=True)
    small_report = _small_batch_experiment(
        pool,
        train_indexes,
        dev_indexes,
        oof,
        full,
        quality_report,
        surface_report,
        cost_method,
        selected_dev_cost,
        selected_calibration,
    )
    reports = {
        "37": small_report,
        "38": quality_report,
        "39": cost_report,
        "40": surface_report,
    }
    passed = [int(number) for number, block in reports.items() if block["passed"]]
    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "base_commit": BASE_COMMIT,
        "selection_policy": (
            "grouped Train OOF chooses candidates; Dev is evaluated once only "
            "after the issue-specific Train gate passes"
        ),
        "identity": pool.identity,
        "folds": fold_values,
        "fit_runtime_s": {
            "folds": [payload["elapsed_s"] for payload in fold_payloads],
            "full_train": full["elapsed_s"],
        },
        "issues": reports,
        "passed_issues": passed,
        "scoped_candidate_ready": bool(37 in passed and 40 in passed),
        "serving_artifact_changed": False,
        "promotion_ready": bool(38 in passed or 40 in passed) and 39 in passed,
        "elapsed_s": float(time.perf_counter() - started),
    }
    audit = _audit_document(
        pool, train_indexes, oof, quality_report, surface_report
    )
    return report, audit


def _merge_issue_39(
    report: dict[str, Any], integration: Mapping[str, Any]
) -> dict[str, Any]:
    block = dict(report["issues"]["39"])
    block["integration"] = {
        key: value
        for key, value in integration.items()
        if key not in {"passed"}
    }
    failing = list(integration.get("failing_gates") or [])
    block["integration_gate"] = (
        "passed" if integration["passed"] else "failed: " + ",".join(failing)
    )
    if block.get("dev") is not None:
        block["dev"] = dict(block["dev"])
        block["dev"]["gate"] = bool(
            block["dev"].get("coverage_gate") and integration["passed"]
        )
        block["dev"]["weighted_loss"] = integration["official_dev"]["weighted_loss"]
        block["dev"]["serve_q95_on_normal_batches"] = integration[
            "serve_q95_on_normal_batches"
        ]
    block["passed"] = bool(block.get("coverage_passed") and integration["passed"])
    updated = dict(report)
    issues = dict(updated["issues"])
    issues["39"] = block
    updated["issues"] = issues
    passed = [int(number) for number, item in issues.items() if item["passed"]]
    updated["passed_issues"] = passed
    updated["promotion_ready"] = bool(38 in passed or 40 in passed) and 39 in passed
    return updated


def _print_summary(report: Mapping[str, Any]) -> None:
    print("\ngeneralization follow-up summary", flush=True)
    for issue in (37, 38, 39, 40):
        block = report["issues"][str(issue)]
        print(
            f"  #{issue}: passed={block['passed']} "
            f"dev_evaluated={block['dev_evaluated']}",
            flush=True,
        )
        if issue == 37:
            print(
                f"    power={block['selected_power']} dev={block['dev']}",
                flush=True,
            )
        elif issue in (38, 40):
            print(
                f"    selected={block['selected']} dev={block['dev']}",
                flush=True,
            )
        else:
            print(
                f"    selected={block['selected']} "
                f"integration={block.get('integration_gate')} "
                f"dev={block['dev']}",
                flush=True,
            )
    print(
        f"passed={report['passed_issues']} "
        f"promotion_ready={report['promotion_ready']} "
        f"elapsed_s={report['elapsed_s']:.1f}",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "report.json")
    parser.add_argument("--audit-output", type=Path, default=OUT / "episode-audit.json")
    parser.add_argument("--cache-dir", type=Path, default=OUT / "cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--issue-39-integration",
        action="store_true",
        help="run reconstructed E30 stress and Dev score gates for issue #39",
    )
    parser.add_argument("--issue-39-workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.refresh_cache and args.cache_dir.is_dir():
        shutil.rmtree(args.cache_dir)
    pool = load_public_pool()
    if args.issue_39_integration:
        if not args.output.is_file():
            raise SystemExit(f"missing coverage report: {args.output}")
        report = json.loads(args.output.read_text(encoding="utf-8"))
        train_indexes, dev_indexes = _indices(pool)
        workers = None if int(args.issue_39_workers) <= 0 else int(args.issue_39_workers)
        print("evaluating issue #39 remaining gates", flush=True)
        integration = run_issue_39_integration(
            pool,
            train_indexes,
            dev_indexes,
            workers=workers,
        )
        report = _merge_issue_39(report, integration)
        integration_path = args.output.parent / "issue-39-integration.json"
        _write_json(integration_path, integration)
        _write_json(args.output, report)
        _print_summary(report)
        print(
            "issue #39 integration "
            f"passed={integration['passed']} "
            f"serve_q95={integration['serve_q95_on_normal_batches']} "
            f"failing={integration['failing_gates']}",
            flush=True,
        )
        print(f"wrote {integration_path}", flush=True)
        print(f"wrote {args.output}", flush=True)
        return 0
    report, audit = run(pool, args.cache_dir, refresh_cache=args.refresh_cache)
    _write_json(args.output, report)
    _write_json(args.audit_output, audit)
    _print_summary(report)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
