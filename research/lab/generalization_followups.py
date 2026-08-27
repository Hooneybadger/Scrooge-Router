# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared, label-isolated helpers for the post-distributional experiments.

The module keeps the four follow-up questions separate from the serving
artifact.  Candidate selection uses grouped Train predictions whose rows were
not used to fit their heads.  Dev is consumed only by the final runner.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.protocol import MODEL_IDS, TIERS, Episode, Message
from research.lab.distributional_knapsack import (
    DEFAULT_TIER_CONFIG,
    FAMILY_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    DistributionalPredictions,
    FamilyCalibration,
    _gbr,
    allocate_priority_queue,
    episode_text,
    feature_matrix,
    risk_cost_surfaces,
    select_vocabulary,
)


BASE_COMMIT = "90398dd8a04c0c21f5dbdc128b5ce894b58117cc"
EXPERIMENT = "generalization-followups-v1"
COST_INFLATION = 1.054
REQUIRED_MARGIN_FRACTION = 0.01
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 20260901
SMALL_BATCH_SIZES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 96, 127)
SMALL_BATCH_UNIQUE_CUTOFF = 128
SMALL_BATCH_POWERS: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
SMALL_BATCH_LOWER_FRACTIONS: Tuple[float, ...] = (0.0, 0.01, 0.025, 0.05)
SURFACE_VARIANTS: Tuple[str, ...] = (
    "line_endings",
    "trailing_space",
    "choice_labels",
    "unicode_nfc",
)
TIER_WEIGHTS: Mapping[str, float] = {
    "fast": 0.4,
    "balanced": 0.3,
    "premium": 0.3,
}
# The fixed-count diagnostic copies the submitted Dev allocation size.  It
# changes only ranking, which isolates quality-head generalization from the
# batch-risk and cost paths.
FIXED_ACTION_FRACTIONS: Mapping[str, Tuple[float, float]] = {
    "fast": (357.0 / 880.0, 0.0),
    "balanced": ((696.0 + 18.0) / 880.0, 18.0 / 880.0),
    "premium": ((646.0 + 127.0) / 880.0, 127.0 / 880.0),
}


@dataclass(frozen=True)
class UpliftFit:
    """Two adjacent model-upgrade heads on one frozen feature layout."""

    vocabulary: Tuple[str, ...]
    models: Tuple[Any, Any]
    loss: str
    lexical_dropout: float


@dataclass(frozen=True)
class SmallBatchView:
    """One predeclared small-batch slice."""

    kind: str
    size: int
    indexes: Tuple[int, ...]


@dataclass(frozen=True)
class LinearUpliftStack:
    """Small linear stack over independently fitted adjacent-gain heads."""

    sources: Tuple[str, ...]
    family_interactions: bool
    feature_mean: Tuple[np.ndarray, np.ndarray]
    feature_scale: Tuple[np.ndarray, np.ndarray]
    coefficients: Tuple[np.ndarray, np.ndarray]
    intercepts: Tuple[float, float]


@dataclass(frozen=True)
class AbsoluteQualityFit:
    """Three absolute quality heads on one explicit vocabulary."""

    vocabulary: Tuple[str, ...]
    models: Tuple[Any, Any, Any]


@dataclass(frozen=True)
class LightLowerCalibration:
    """Family-aware lower envelope for the all-Light batch denominator."""

    family_names: Tuple[str, ...]
    reference_proportions: Tuple[float, ...]
    fraction: float
    scales: np.ndarray


def _quality_targets(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(MODEL_IDS):
        raise ValueError("quality scores must be an n-by-3 matrix")
    return np.column_stack((matrix[:, 1] - matrix[:, 0], matrix[:, 2] - matrix[:, 1]))


def _drop_lexical_features(
    features: np.ndarray, *, rate: float, random_state: int
) -> np.ndarray:
    if not 0.0 <= rate < 1.0:
        raise ValueError("lexical dropout must be in [0, 1)")
    matrix = np.asarray(features, dtype=np.float32).copy()
    if rate == 0.0 or matrix.shape[1] == len(STRUCTURAL_FEATURE_NAMES):
        return matrix
    rng = np.random.default_rng(int(random_state))
    lexical = matrix[:, len(STRUCTURAL_FEATURE_NAMES) :]
    lexical[rng.random(lexical.shape) < rate] = 0.0
    return matrix


def fit_uplift_models(
    episodes: Sequence[Episode],
    scores: np.ndarray,
    vocabulary: Sequence[str],
    *,
    loss: str = "squared_error",
    lexical_dropout: float = 0.0,
    random_state: int = 20260901,
) -> UpliftFit:
    """Fit Light→AX31 and AX31→K1 score changes directly."""

    targets = _quality_targets(scores)
    vocab = tuple(str(value) for value in vocabulary)
    features = feature_matrix(episodes, vocab)
    features = _drop_lexical_features(
        features, rate=lexical_dropout, random_state=random_state
    )
    models = []
    for column in range(2):
        model = _gbr(
            loss=loss,
            random_state=int(random_state + 1009 * column),
        ).fit(features, targets[:, column])
        models.append(model)
    return UpliftFit(vocab, (models[0], models[1]), loss, float(lexical_dropout))


def predict_uplifts(fit: UpliftFit, episodes: Sequence[Episode]) -> np.ndarray:
    """Predict the two adjacent score changes."""

    features = feature_matrix(episodes, fit.vocabulary)
    result = np.column_stack([model.predict(features) for model in fit.models])
    if result.shape != (len(episodes), 2):
        raise RuntimeError("uplift prediction shape drifted")
    return np.asarray(result, dtype=np.float64)


def _stack_step_features(
    absolute_quality: np.ndarray,
    uplift_predictions: Mapping[str, np.ndarray],
    families: Sequence[str],
    sources: Sequence[str],
    step: int,
    *,
    family_interactions: bool,
) -> np.ndarray:
    absolute = np.asarray(absolute_quality, dtype=np.float64)
    base = absolute[:, step + 1] - absolute[:, step]
    signals = [base]
    for name in sources:
        values = np.asarray(uplift_predictions[name], dtype=np.float64)
        if values.shape != (absolute.shape[0], 2):
            raise ValueError("stacked uplift prediction shape drifted")
        signals.append(values[:, step])
    signal_matrix = np.column_stack(signals)
    columns = [
        signal_matrix,
        np.mean(signal_matrix, axis=1, keepdims=True),
        np.std(signal_matrix, axis=1, keepdims=True),
    ]
    labels = tuple(str(value) for value in families)
    if len(labels) != absolute.shape[0]:
        raise ValueError("stacked uplift families do not align")
    encoded = np.zeros((absolute.shape[0], len(FAMILY_NAMES)), dtype=np.float64)
    lookup = {name: index for index, name in enumerate(FAMILY_NAMES)}
    for row, name in enumerate(labels):
        encoded[row, lookup[name]] = 1.0
    columns.append(encoded)
    if family_interactions:
        # Interactions adjust only disagreement with the current absolute
        # heads. They cannot create a family-only score shortcut.
        for column in range(1, signal_matrix.shape[1]):
            disagreement = (signal_matrix[:, column] - base)[:, None]
            columns.append(encoded * disagreement)
    return np.column_stack(columns)


def fit_stacked_uplifts(
    absolute_quality: np.ndarray,
    uplift_predictions: Mapping[str, np.ndarray],
    scores: np.ndarray,
    families: Sequence[str],
    group_keys: Sequence[str],
    *,
    sources: Sequence[str] = (
        "direct_squared",
        "direct_huber",
        "direct_structural",
    ),
    alpha: float = 10.0,
    family_interactions: bool = True,
) -> LinearUpliftStack:
    """Fit a group-weighted ridge stack on held-out primary predictions."""

    try:
        from sklearn.linear_model import Ridge
    except ImportError as exc:  # pragma: no cover - research dependency guard
        raise RuntimeError("stacked uplift fitting requires research dependencies") from exc
    if alpha <= 0.0:
        raise ValueError("stacked uplift alpha must be positive")
    actual = _quality_targets(scores)
    keys = tuple(str(value) for value in group_keys)
    if len(keys) != actual.shape[0]:
        raise ValueError("stacked uplift groups do not align")
    counts: dict[str, int] = defaultdict(int)
    for key in keys:
        counts[key] += 1
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    means = []
    scales = []
    coefficients = []
    intercepts = []
    source_names = tuple(str(value) for value in sources)
    for step in range(2):
        features = _stack_step_features(
            absolute_quality,
            uplift_predictions,
            families,
            source_names,
            step,
            family_interactions=family_interactions,
        )
        mean = np.average(features, axis=0, weights=weights)
        variance = np.average((features - mean) ** 2, axis=0, weights=weights)
        scale = np.sqrt(np.maximum(variance, 1e-12))
        standardized = (features - mean) / scale
        model = Ridge(alpha=float(alpha), fit_intercept=True).fit(
            standardized, actual[:, step], sample_weight=weights
        )
        means.append(mean)
        scales.append(scale)
        coefficients.append(np.asarray(model.coef_, dtype=np.float64))
        intercepts.append(float(model.intercept_))
    return LinearUpliftStack(
        source_names,
        bool(family_interactions),
        (means[0], means[1]),
        (scales[0], scales[1]),
        (coefficients[0], coefficients[1]),
        (intercepts[0], intercepts[1]),
    )


def predict_stacked_uplifts(
    fit: LinearUpliftStack,
    absolute_quality: np.ndarray,
    uplift_predictions: Mapping[str, np.ndarray],
    families: Sequence[str],
) -> np.ndarray:
    """Apply a fitted linear uplift stack."""

    predictions = []
    for step in range(2):
        features = _stack_step_features(
            absolute_quality,
            uplift_predictions,
            families,
            fit.sources,
            step,
            family_interactions=fit.family_interactions,
        )
        standardized = (features - fit.feature_mean[step]) / fit.feature_scale[step]
        predictions.append(
            fit.intercepts[step] + standardized @ fit.coefficients[step]
        )
    return np.column_stack(predictions)


def fit_absolute_quality_models(
    episodes: Sequence[Episode],
    scores: np.ndarray,
    *,
    vocabulary_size: int = 1_024,
    random_state: int = 20260901,
) -> AbsoluteQualityFit:
    """Fit the current absolute objective without the unrelated cost heads."""

    vocabulary = select_vocabulary_for_scores(
        episodes, scores, size=int(vocabulary_size)
    )
    features = feature_matrix(episodes, vocabulary)
    matrix = np.asarray(scores, dtype=np.float64)
    models = []
    for column in range(3):
        models.append(
            _gbr(
                loss="squared_error",
                random_state=int(random_state + 1009 * column),
            ).fit(features, matrix[:, column])
        )
    return AbsoluteQualityFit(
        vocabulary, (models[0], models[1], models[2])
    )


def fit_serving_quality_models(
    episodes: Sequence[Episode],
    scores: np.ndarray,
    costs: np.ndarray,
    *,
    vocabulary_size: int = 1_024,
    random_state: int = 20260901,
) -> AbsoluteQualityFit:
    """Refit the serving quality heads with the serving vocabulary objective."""

    score_matrix = np.asarray(scores, dtype=np.float64)
    cost_matrix = np.asarray(costs, dtype=np.float64)
    if score_matrix.shape != cost_matrix.shape or score_matrix.shape[1] != 3:
        raise ValueError("serving quality scores and costs must align n-by-3")
    targets = np.column_stack(
        (
            score_matrix[:, 1] - score_matrix[:, 0],
            score_matrix[:, 2] - score_matrix[:, 1],
            np.log1p(cost_matrix[:, 0]),
            np.log1p(cost_matrix[:, 1]),
            np.log1p(cost_matrix[:, 2]),
        )
    )
    vocabulary = select_vocabulary(
        tuple(episode_text(episode) for episode in episodes),
        targets,
        size=int(vocabulary_size),
    )
    features = feature_matrix(episodes, vocabulary)
    models = []
    for column in range(3):
        models.append(
            _gbr(
                loss="squared_error",
                random_state=int(random_state + 1009 * column),
            ).fit(features, score_matrix[:, column])
        )
    return AbsoluteQualityFit(
        tuple(vocabulary),
        (models[0], models[1], models[2]),
    )


def predict_absolute_quality(
    fit: AbsoluteQualityFit, episodes: Sequence[Episode]
) -> np.ndarray:
    features = feature_matrix(episodes, fit.vocabulary)
    return np.clip(
        np.column_stack([model.predict(features) for model in fit.models]),
        0.0,
        1.0,
    )


def blended_quality(
    absolute_quality: np.ndarray, direct_uplifts: np.ndarray, weight: float
) -> np.ndarray:
    """Blend independently predicted and directly predicted adjacent gains."""

    absolute = np.asarray(absolute_quality, dtype=np.float64)
    direct = np.asarray(direct_uplifts, dtype=np.float64)
    if absolute.ndim != 2 or absolute.shape[1] != 3:
        raise ValueError("absolute quality must be an n-by-3 matrix")
    if direct.shape != (absolute.shape[0], 2):
        raise ValueError("direct uplift matrix does not align")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("uplift blend weight must be in [0, 1]")
    adjacent = np.column_stack(
        (absolute[:, 1] - absolute[:, 0], absolute[:, 2] - absolute[:, 1])
    )
    uplift = (1.0 - float(weight)) * adjacent + float(weight) * direct
    result = np.empty_like(absolute)
    result[:, 0] = absolute[:, 0]
    result[:, 1] = result[:, 0] + uplift[:, 0]
    result[:, 2] = result[:, 1] + uplift[:, 1]
    return np.clip(result, 0.0, 1.0)


def _ranked_indexes(values: np.ndarray, keys: Sequence[str]) -> list[int]:
    return sorted(
        range(len(values)),
        key=lambda index: (-float(values[index]), str(keys[index]), index),
    )


def fixed_count_actions(
    quality: np.ndarray,
    tier: str,
    tie_keys: Sequence[str],
    *,
    fractions: Mapping[str, Tuple[float, float]] = FIXED_ACTION_FRACTIONS,
) -> np.ndarray:
    """Choose a fixed number of adjacent upgrades using quality rank only."""

    matrix = np.asarray(quality, dtype=np.float64)
    if tier not in TIERS or tier not in fractions:
        raise ValueError(f"unknown fixed-count tier: {tier!r}")
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("fixed-count quality must be an n-by-3 matrix")
    keys = tuple(str(value) for value in tie_keys)
    if len(keys) != matrix.shape[0]:
        raise ValueError("fixed-count tie keys do not align")
    upgrade_fraction, k1_fraction = fractions[tier]
    n_upgrade = min(matrix.shape[0], int(round(matrix.shape[0] * upgrade_fraction)))
    n_k1 = min(n_upgrade, int(round(matrix.shape[0] * k1_fraction)))
    first = _ranked_indexes(matrix[:, 1] - matrix[:, 0], keys)[:n_upgrade]
    selected = np.zeros(matrix.shape[0], dtype=np.int8)
    selected[first] = 1
    if n_k1:
        second_values = matrix[:, 2] - matrix[:, 1]
        second = sorted(
            first,
            key=lambda index: (-float(second_values[index]), keys[index], index),
        )[:n_k1]
        selected[second] = 2
    return selected


def fixed_count_evaluation(
    quality: np.ndarray,
    scores: np.ndarray,
    costs: np.ndarray,
    tie_keys: Sequence[str],
) -> dict[str, Any]:
    """Evaluate all tiers with frozen action counts."""

    actual_scores = np.asarray(scores, dtype=np.float64)
    actual_costs = np.asarray(costs, dtype=np.float64)
    tiers: dict[str, Any] = {}
    actions: dict[str, np.ndarray] = {}
    weighted = 0.0
    for tier in TIERS:
        chosen = fixed_count_actions(quality, tier, tie_keys)
        actions[tier] = chosen
        rows = np.arange(chosen.size, dtype=np.int64)
        q = float(np.mean(actual_scores[rows, chosen]))
        ratio = float(actual_costs[rows, chosen].sum() / actual_costs[:, 0].sum())
        weighted += TIER_WEIGHTS[tier] * q
        tiers[tier] = {
            "quality": q,
            "ratio": ratio,
            "model_counts": {
                model_id: int(np.count_nonzero(chosen == column))
                for column, model_id in enumerate(MODEL_IDS)
            },
        }
    return {"actions": actions, "tiers": tiers, "weighted_quality": float(weighted)}


def grouped_bootstrap_quality_delta(
    baseline_actions: Mapping[str, np.ndarray],
    candidate_actions: Mapping[str, np.ndarray],
    scores: np.ndarray,
    group_keys: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap the weighted per-row score delta by content group."""

    actual = np.asarray(scores, dtype=np.float64)
    count = actual.shape[0]
    row_delta = np.zeros(count, dtype=np.float64)
    rows = np.arange(count, dtype=np.int64)
    for tier in TIERS:
        base = np.asarray(baseline_actions[tier], dtype=np.int64)
        candidate = np.asarray(candidate_actions[tier], dtype=np.int64)
        if base.shape != (count,) or candidate.shape != (count,):
            raise ValueError("bootstrap action vectors do not align")
        row_delta += TIER_WEIGHTS[tier] * (
            actual[rows, candidate] - actual[rows, base]
        )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        grouped[str(key)].append(index)
    blocks = tuple(np.asarray(indexes, dtype=np.int64) for indexes in grouped.values())
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        numerator = 0.0
        denominator = 0
        for block_index in picked:
            block = blocks[int(block_index)]
            numerator += float(row_delta[block].sum())
            denominator += int(block.size)
        samples[draw] = numerator / max(1, denominator)
    return {
        "mean": float(np.mean(row_delta)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _finite_quantile(values: np.ndarray, fraction: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return 1.0
    rank = int(math.ceil((array.size + 1) * float(fraction))) - 1
    return float(array[min(max(rank, 0), array.size - 1)])


def _finite_lower_quantile(values: np.ndarray, fraction: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return 1.0
    rank = int(math.floor((array.size + 1) * float(fraction))) - 1
    return float(array[min(max(rank, 0), array.size - 1)])


def fit_light_lower_calibration(
    predictions: DistributionalPredictions,
    actual_costs: np.ndarray,
    families: Sequence[str],
    *,
    fraction: float,
    family_names: Sequence[str] = FAMILY_NAMES,
    minimum_family_size: int = 50,
) -> LightLowerCalibration:
    """Fit a lower bound for the all-Light cost used as a batch denominator."""

    if not 0.0 <= fraction <= 0.5:
        raise ValueError("Light lower-tail fraction must be in [0, 0.5]")
    actual = np.asarray(actual_costs, dtype=np.float64)
    mean = np.asarray(predictions.cost_mean, dtype=np.float64)
    if actual.shape != mean.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError("Light lower calibration matrices do not align")
    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    if labels.shape != (actual.shape[0],):
        raise ValueError("Light lower calibration families do not align")
    names = tuple(str(value) for value in family_names)
    unknown = set(labels.tolist()) - set(names)
    if unknown:
        raise ValueError(f"Light lower calibration saw unknown families: {unknown}")
    ratios = actual[:, 0] / np.maximum(mean[:, 0], 1e-12)
    global_scale = _finite_lower_quantile(ratios, fraction)
    scales = np.empty(len(names), dtype=np.float64)
    reference = []
    for family_index, name in enumerate(names):
        selected = labels == name
        size = int(np.count_nonzero(selected))
        reference.append(float(np.mean(selected)))
        if size >= int(minimum_family_size):
            scale = _finite_lower_quantile(ratios[selected], fraction)
        else:
            scale = global_scale
        scales[family_index] = float(np.clip(scale, 1e-12, 1.0))
    return LightLowerCalibration(
        names,
        tuple(reference),
        float(fraction),
        scales,
    )


def apply_light_lower_calibration(
    predictions: DistributionalPredictions,
    families: Sequence[str],
    calibration: LightLowerCalibration,
) -> np.ndarray:
    """Return per-row conservative all-Light denominator credits."""

    labels = tuple(str(value) for value in families)
    if len(labels) != predictions.cost_mean.shape[0]:
        raise ValueError("Light lower calibration rows do not align")
    lookup = {name: index for index, name in enumerate(calibration.family_names)}
    try:
        encoded = np.asarray([lookup[name] for name in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown Light lower calibration family: {exc.args[0]}") from exc
    return np.maximum(
        predictions.cost_mean[:, 0] * calibration.scales[encoded],
        np.finfo(np.float64).tiny,
    )


def cross_light_lower_credits(
    predictions: DistributionalPredictions,
    actual_costs: np.ndarray,
    families: Sequence[str],
    folds: Sequence[int],
    *,
    fraction: float,
) -> np.ndarray:
    """Cross-fit Light denominator bounds without using a row's held fold."""

    fold_ids = np.asarray(folds, dtype=np.int64)
    if fold_ids.shape != (predictions.cost_mean.shape[0],):
        raise ValueError("Light lower calibration folds do not align")
    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    credits = np.empty(predictions.cost_mean.shape[0], dtype=np.float64)
    for fold in sorted(set(int(value) for value in fold_ids)):
        held = fold_ids == fold
        fitted = ~held
        calibration = fit_light_lower_calibration(
            DistributionalPredictions(
                predictions.quality_mean[fitted],
                predictions.cost_mean[fitted],
                predictions.cost_q90[fitted],
            ),
            np.asarray(actual_costs)[fitted],
            labels[fitted],
            fraction=fraction,
        )
        credits[held] = apply_light_lower_calibration(
            DistributionalPredictions(
                predictions.quality_mean[held],
                predictions.cost_mean[held],
                predictions.cost_q90[held],
            ),
            labels[held],
            calibration,
        )
    return credits


def fit_cost_calibration(
    predictions: DistributionalPredictions,
    actual_costs: np.ndarray,
    families: Sequence[str],
    *,
    method: str,
    family_names: Sequence[str] = FAMILY_NAMES,
    minimum_family_size: int = 50,
) -> FamilyCalibration:
    """Fit one predeclared cost calibration from held-out predictions."""

    actual = np.asarray(actual_costs, dtype=np.float64)
    mean = np.asarray(predictions.cost_mean, dtype=np.float64)
    upper = np.asarray(predictions.cost_q90, dtype=np.float64)
    if actual.shape != mean.shape or actual.shape != upper.shape or actual.shape[1] != 3:
        raise ValueError("cost calibration matrices do not align")
    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    if labels.shape != (actual.shape[0],):
        raise ValueError("cost calibration families do not align")
    names = tuple(str(value) for value in family_names)
    reference = tuple(float(np.mean(labels == name)) for name in names)
    global_mean = np.sum(actual, axis=0) / np.maximum(np.sum(mean, axis=0), 1e-12)
    ratios = actual / np.maximum(upper, 1e-12)
    global_q90 = np.asarray(
        [_finite_quantile(ratios[:, column], 0.90) for column in range(3)]
    )
    global_q95 = np.asarray(
        [_finite_quantile(ratios[:, column], 0.95) for column in range(3)]
    )
    mean_scales = np.empty((len(names), 3), dtype=np.float64)
    upper_scales = np.empty((len(names), 3), dtype=np.float64)
    for family_index, name in enumerate(names):
        selected = labels == name
        size = int(np.count_nonzero(selected))
        if size:
            local_mean = np.sum(actual[selected], axis=0) / np.maximum(
                np.sum(mean[selected], axis=0), 1e-12
            )
        else:
            local_mean = global_mean
        weight = size / float(size + minimum_family_size)
        mean_scales[family_index] = np.exp(
            weight * np.log(np.maximum(local_mean, 1e-12))
            + (1.0 - weight) * np.log(np.maximum(global_mean, 1e-12))
        )
        if method == "aggregate_total":
            if size:
                upper_scales[family_index] = np.sum(actual[selected], axis=0) / np.maximum(
                    np.sum(upper[selected], axis=0), 1e-12
                )
            else:
                upper_scales[family_index] = global_mean
        elif method == "global_q90":
            upper_scales[family_index] = global_q90
        elif method in {"family_partial_q90", "family_partial_q95"}:
            target = 0.90 if method.endswith("q90") else 0.95
            global_scale = global_q90 if target == 0.90 else global_q95
            if size:
                local = np.asarray(
                    [
                        _finite_quantile(ratios[selected, column], target)
                        for column in range(3)
                    ]
                )
            else:
                local = global_scale
            pooled = np.exp(
                weight * np.log(np.maximum(local, 1e-12))
                + (1.0 - weight) * np.log(np.maximum(global_scale, 1e-12))
            )
            upper_scales[family_index] = np.maximum(pooled, 1.0)
        else:
            raise ValueError(f"unknown cost calibration method: {method!r}")
    return FamilyCalibration(names, reference, mean_scales, upper_scales)


def apply_cost_calibration(
    predictions: DistributionalPredictions,
    families: Sequence[str],
    calibration: FamilyCalibration,
) -> DistributionalPredictions:
    """Apply calibration while retaining the supplied quality matrix."""

    labels = tuple(str(value) for value in families)
    lookup = {name: index for index, name in enumerate(calibration.family_names)}
    encoded = np.asarray([lookup[name] for name in labels], dtype=np.int64)
    mean = predictions.cost_mean * calibration.mean_scales[encoded]
    upper = predictions.cost_q90 * calibration.q90_scales[encoded]
    return DistributionalPredictions(
        np.asarray(predictions.quality_mean, dtype=np.float64),
        np.maximum(mean, np.finfo(np.float64).tiny),
        np.maximum(upper, mean),
    )


def cross_calibrated_costs(
    predictions: DistributionalPredictions,
    actual_costs: np.ndarray,
    families: Sequence[str],
    folds: Sequence[int],
    *,
    method: str,
) -> DistributionalPredictions:
    """Calibrate each fold using only the other folds' held-out predictions."""

    fold_ids = np.asarray(folds, dtype=np.int64)
    if fold_ids.shape != (predictions.cost_mean.shape[0],):
        raise ValueError("cost calibration folds do not align")
    out_mean = np.empty_like(predictions.cost_mean, dtype=np.float64)
    out_upper = np.empty_like(predictions.cost_q90, dtype=np.float64)
    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    for fold in sorted(set(int(value) for value in fold_ids)):
        held = fold_ids == fold
        fitted = ~held
        calibration = fit_cost_calibration(
            DistributionalPredictions(
                predictions.quality_mean[fitted],
                predictions.cost_mean[fitted],
                predictions.cost_q90[fitted],
            ),
            np.asarray(actual_costs)[fitted],
            labels[fitted],
            method=method,
        )
        calibrated = apply_cost_calibration(
            DistributionalPredictions(
                predictions.quality_mean[held],
                predictions.cost_mean[held],
                predictions.cost_q90[held],
            ),
            labels[held],
            calibration,
        )
        out_mean[held] = calibrated.cost_mean
        out_upper[held] = calibrated.cost_q90
    return DistributionalPredictions(predictions.quality_mean, out_mean, out_upper)


def cost_coverage(
    predictions: DistributionalPredictions,
    actual_costs: np.ndarray,
    families: Sequence[str],
    *,
    minimum_cell_size: int = 50,
) -> dict[str, Any]:
    """Report global and sufficiently populated model-by-family coverage."""

    actual = np.asarray(actual_costs, dtype=np.float64)
    upper = np.asarray(predictions.cost_q90, dtype=np.float64)
    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    global_rows = {}
    for column, model_id in enumerate(MODEL_IDS):
        global_rows[model_id] = float(np.mean(upper[:, column] >= actual[:, column]))
    cells = []
    for name in FAMILY_NAMES:
        selected = labels == name
        size = int(np.count_nonzero(selected))
        if size < minimum_cell_size:
            continue
        for column, model_id in enumerate(MODEL_IDS):
            coverage = float(np.mean(upper[selected, column] >= actual[selected, column]))
            cells.append(
                {
                    "family": name,
                    "model_id": model_id,
                    "n": size,
                    "coverage": coverage,
                    "passed": coverage >= 0.90,
                }
            )
    return {
        "global": global_rows,
        "cells": cells,
        "minimum_cell_coverage": min(
            (float(row["coverage"]) for row in cells), default=1.0
        ),
        "all_cells_passed": all(bool(row["passed"]) for row in cells),
        "mean_upper_to_actual": float(
            np.mean(upper / np.maximum(actual, np.finfo(np.float64).tiny))
        ),
    }


_CHOICE_DOT = re.compile(r"(?m)^(\s*)([A-Ea-e1-5])[.)]\s+")
_CHOICE_PAREN = re.compile(r"(?m)^(\s*)\(([A-Ea-e1-5])\)\s+")


def stable_surface_text(text: str) -> str:
    """Normalize encoding and whitespace while preserving choice notation."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))


def canonical_surface_text(text: str) -> str:
    """Map presentation-only variants to one feature-extraction form."""

    normalized = stable_surface_text(text)
    normalized = _CHOICE_PAREN.sub(r"\1(\2) ", normalized)
    return _CHOICE_DOT.sub(r"\1(\2) ", normalized)


def transform_surface_text(text: str, variant: str) -> str:
    """Apply one deterministic, meaning-preserving surface transformation."""

    if variant == "unicode_nfc":
        return unicodedata.normalize("NFC", text)
    if variant == "line_endings":
        return text.replace("\r\n", "\n").replace("\n", "\r\n")
    if variant == "trailing_space":
        return "\n".join(f"{line} " if line else line for line in text.split("\n"))
    if variant == "choice_labels":
        changed = _CHOICE_PAREN.sub(r"\1\2. ", text)
        if changed != text:
            return changed
        return _CHOICE_DOT.sub(r"\1(\2) ", text)
    raise ValueError(f"unknown surface transformation: {variant!r}")


def transform_episode(episode: Episode, variant: str) -> Episode:
    """Transform prompt content without changing its ID or message roles."""

    if episode.prompt is not None:
        return Episode(
            episode_id=episode.episode_id,
            prompt=transform_surface_text(episode.prompt, variant),
        )
    if episode.messages is None:
        raise ValueError("episode has neither prompt nor messages")
    return Episode(
        episode_id=episode.episode_id,
        messages=tuple(
            Message(message.role, transform_surface_text(message.content, variant))
            for message in episode.messages
        ),
    )


def canonicalize_episode(episode: Episode) -> Episode:
    """Canonicalize presentation without changing IDs or message roles."""

    if episode.prompt is not None:
        return Episode(
            episode_id=episode.episode_id,
            prompt=canonical_surface_text(episode.prompt),
        )
    if episode.messages is None:
        raise ValueError("episode has neither prompt nor messages")
    return Episode(
        episode_id=episode.episode_id,
        messages=tuple(
            Message(message.role, canonical_surface_text(message.content))
            for message in episode.messages
        ),
    )


def canonicalized_episodes(episodes: Sequence[Episode]) -> Tuple[Episode, ...]:
    return tuple(canonicalize_episode(episode) for episode in episodes)


def stable_surface_episode(episode: Episode) -> Episode:
    """Apply the conservative surface normalizer to one episode."""

    if episode.prompt is not None:
        return Episode(
            episode_id=episode.episode_id,
            prompt=stable_surface_text(episode.prompt),
        )
    if episode.messages is None:
        raise ValueError("episode has neither prompt nor messages")
    return Episode(
        episode_id=episode.episode_id,
        messages=tuple(
            Message(message.role, stable_surface_text(message.content))
            for message in episode.messages
        ),
    )


def stable_surface_episodes(
    episodes: Sequence[Episode],
) -> Tuple[Episode, ...]:
    return tuple(stable_surface_episode(episode) for episode in episodes)


def transformed_episodes(
    episodes: Sequence[Episode], variant: str
) -> Tuple[Episode, ...]:
    return tuple(transform_episode(episode, variant) for episode in episodes)


def action_flip_rate(
    original: Mapping[str, np.ndarray], transformed: Mapping[str, np.ndarray]
) -> float:
    """Return the fraction of tier-row decisions changed by a transformation."""

    changed = 0
    total = 0
    for tier in TIERS:
        left = np.asarray(original[tier], dtype=np.int8)
        right = np.asarray(transformed[tier], dtype=np.int8)
        if left.shape != right.shape:
            raise ValueError("surface action vectors do not align")
        changed += int(np.count_nonzero(left != right))
        total += int(left.size)
    return changed / max(1, total)


def select_vocabulary_for_scores(
    episodes: Sequence[Episode],
    scores: np.ndarray,
    *,
    size: int,
) -> Tuple[str, ...]:
    """Select a vocabulary using adjacent score changes only."""

    if size == 0:
        return ()
    texts = []
    for episode in episodes:
        if episode.prompt is not None:
            texts.append(episode.prompt)
        elif episode.messages is not None:
            texts.append("\n".join(message.content for message in episode.messages))
        else:
            raise ValueError("episode has neither prompt nor messages")
    return select_vocabulary(texts, _quality_targets(scores), size=size)


def _representatives(keys: Sequence[str]) -> Tuple[int, ...]:
    seen: dict[str, int] = {}
    for index, key in enumerate(keys):
        seen.setdefault(str(key), index)
    return tuple(seen[key] for key in sorted(seen))


def make_small_batch_views(
    families: Sequence[str],
    tie_keys: Sequence[str],
    predictions: DistributionalPredictions,
    *,
    sizes: Sequence[int] = SMALL_BATCH_SIZES,
    draws_per_kind: int = 12,
    seed: int,
) -> Tuple[SmallBatchView, ...]:
    """Create sealed uniform, concentrated-family, and predicted-tail views."""

    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    representatives = np.asarray(_representatives(tie_keys), dtype=np.int64)
    if representatives.size < max(sizes):
        raise ValueError("not enough unique content for small-batch catalog")
    rng = np.random.default_rng(int(seed))
    by_family = {
        name: representatives[labels[representatives] == name]
        for name in FAMILY_NAMES
    }
    light = np.maximum(predictions.cost_mean[:, 0], 1e-12)
    tail_score = np.max(predictions.cost_q90 / light[:, None], axis=1)
    views: list[SmallBatchView] = []
    for raw_size in sizes:
        size = int(raw_size)
        for _draw in range(int(draws_per_kind)):
            uniform = tuple(
                int(value) for value in rng.choice(representatives, size=size, replace=False)
            )
            views.append(SmallBatchView("uniform", size, uniform))
            eligible = [values for values in by_family.values() if values.size >= size]
            source = eligible[int(rng.integers(0, len(eligible)))] if eligible else representatives
            concentrated = tuple(
                int(value) for value in rng.choice(source, size=size, replace=False)
            )
            views.append(SmallBatchView("single_family", size, concentrated))
            pool_size = min(representatives.size, max(size, size * 3))
            tail_pool = representatives[
                np.argsort(-tail_score[representatives], kind="stable")[:pool_size]
            ]
            tail = tuple(
                int(value) for value in rng.choice(tail_pool, size=size, replace=False)
            )
            views.append(SmallBatchView("predicted_tail", size, tail))
            duplicate_count = max(1, min(size, int(math.ceil(math.sqrt(size)))))
            duplicate_pool_size = min(
                representatives.size,
                max(duplicate_count, duplicate_count * 3),
            )
            duplicate_pool = representatives[
                np.argsort(-tail_score[representatives], kind="stable")[
                    :duplicate_pool_size
                ]
            ]
            duplicate_sources = rng.choice(
                duplicate_pool,
                size=duplicate_count,
                replace=False,
            )
            duplicate_tail = np.resize(duplicate_sources, size)
            rng.shuffle(duplicate_tail)
            views.append(
                SmallBatchView(
                    "duplicate_tail",
                    size,
                    tuple(int(value) for value in duplicate_tail),
                )
            )
    return tuple(views)


def _composition_tv(families: Sequence[str], reference: Sequence[float]) -> float:
    lookup = {name: index for index, name in enumerate(FAMILY_NAMES)}
    encoded = np.asarray([lookup[str(value)] for value in families], dtype=np.int64)
    proportions = np.bincount(encoded, minlength=len(FAMILY_NAMES)) / len(encoded)
    return float(0.5 * np.sum(np.abs(proportions - np.asarray(reference))))


def small_batch_target_fraction(
    tier: str,
    budget_multiplier: float,
    unique_count: int,
    family_tv: float,
    *,
    power: float,
) -> float:
    """Shrink the ordinary composition cap continuously toward all-Light."""

    config = DEFAULT_TIER_CONFIG[tier]
    lower = 1.0 / float(budget_multiplier)
    fallback = float(
        np.clip(
            config.base_fraction * (1.0 - config.composition_penalty * family_tv),
            lower,
            config.base_fraction,
        )
    )
    if power == 0.0:
        return 1.0
    share = min(
        1.0,
        max(0.0, unique_count / float(SMALL_BATCH_UNIQUE_CUTOFF)),
    ) ** float(power)
    return float(lower + (fallback - lower) * share)


def small_batch_route_enabled(unique_count: int) -> bool:
    """Return whether the two-sided small-batch route owns this batch."""

    return 0 < int(unique_count) < SMALL_BATCH_UNIQUE_CUTOFF


def small_batch_cost_surfaces(
    predictions: DistributionalPredictions,
    tier: str,
    *,
    light_lower_credit: Optional[np.ndarray] = None,
    full_upper: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ordinary or two-sided interval costs for a small batch."""

    charges, light = risk_cost_surfaces(predictions, tier)
    if light_lower_credit is None:
        if full_upper:
            raise ValueError("full upper costs require a Light lower envelope")
        return charges, light
    lower = np.asarray(light_lower_credit, dtype=np.float64)
    if lower.shape != (predictions.cost_mean.shape[0],):
        raise ValueError("small-batch Light lower credits do not align")
    if full_upper:
        guarded = np.asarray(predictions.cost_q90, dtype=np.float64).copy()
    else:
        guarded = np.asarray(charges, dtype=np.float64).copy()
    guarded[:, 0] = lower
    guarded[:, 1:] = np.maximum(guarded[:, 1:], lower[:, None])
    return guarded, lower


def evaluate_small_batches(
    predictions: DistributionalPredictions,
    scores: np.ndarray,
    costs: np.ndarray,
    families: Sequence[str],
    tie_keys: Sequence[str],
    views: Sequence[SmallBatchView],
    reference_proportions: Sequence[float],
    budget_multipliers: Mapping[str, float],
    *,
    power: float,
    light_lower_credit: Optional[np.ndarray] = None,
    full_upper: bool = False,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate one small-batch schedule against the all-Light fallback."""

    actual_scores = np.asarray(scores, dtype=np.float64)
    actual_costs = np.asarray(costs, dtype=np.float64)
    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    keys = np.asarray(tuple(str(value) for value in tie_keys), dtype=object)
    lower_credit = None
    if light_lower_credit is not None:
        lower_credit = np.asarray(light_lower_credit, dtype=np.float64)
        if lower_credit.shape != (actual_scores.shape[0],):
            raise ValueError("small-batch Light lower credits do not align")
    weighted_deltas = []
    violations = []
    rows = []
    for view_index, view in enumerate(views):
        indexes = np.asarray(view.indexes, dtype=np.int64)
        view_families = tuple(str(value) for value in labels[indexes])
        view_keys = tuple(str(value) for value in keys[indexes])
        unique_count = len(set(view_keys))
        tv = _composition_tv(view_families, reference_proportions)
        view_delta = 0.0
        tier_rows = {}
        for tier in TIERS:
            budget = float(budget_multipliers[tier])
            sliced = DistributionalPredictions(
                predictions.quality_mean[indexes],
                predictions.cost_mean[indexes],
                predictions.cost_q90[indexes],
            )
            sliced_lower = None if lower_credit is None else lower_credit[indexes]
            charges, light = small_batch_cost_surfaces(
                sliced,
                tier,
                light_lower_credit=sliced_lower,
                full_upper=full_upper,
            )
            target = small_batch_target_fraction(
                tier, budget, unique_count, tv, power=power
            )
            chosen = allocate_priority_queue(
                sliced.quality_mean,
                charges,
                light,
                budget_multiplier=budget,
                target_fraction=target,
                tie_keys=view_keys,
                allow_k1=False,
            )
            local_rows = np.arange(chosen.size, dtype=np.int64)
            ratio = float(
                actual_costs[indexes][local_rows, chosen].sum()
                / actual_costs[indexes, 0].sum()
            )
            quality = float(np.mean(actual_scores[indexes][local_rows, chosen]))
            light_quality = float(np.mean(actual_scores[indexes, 0]))
            delta = quality - light_quality
            view_delta += TIER_WEIGHTS[tier] * delta
            threshold = budget * (1.0 - REQUIRED_MARGIN_FRACTION)
            violated = ratio * COST_INFLATION > threshold + 1e-12
            if violated:
                violations.append(f"{view_index}:{view.kind}:{view.size}:{tier}")
            tier_rows[tier] = {
                "quality": quality,
                "quality_delta": delta,
                "ratio": ratio,
                "inflated_ratio": ratio * COST_INFLATION,
                "target_fraction": target,
                "violated": violated,
                "upgrades": int(np.count_nonzero(chosen)),
            }
        weighted_deltas.append(view_delta)
        rows.append(
            {
                "kind": view.kind,
                "size": view.size,
                "weighted_quality_delta": view_delta,
                "tiers": tier_rows,
            }
        )
    values = np.asarray(weighted_deltas, dtype=np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        sample = rng.integers(0, values.size, size=values.size)
        draws[index] = float(np.mean(values[sample]))
    worst = {
        tier: max(float(row["tiers"][tier]["inflated_ratio"]) for row in rows)
        for tier in TIERS
    }

    def breakdown(field: str) -> dict[str, Any]:
        values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            values[str(row[field])].append(row)
        result = {}
        for name, selected_rows in values.items():
            result[name] = {
                "n_views": len(selected_rows),
                "mean_weighted_quality_delta": float(
                    np.mean(
                        [
                            float(row["weighted_quality_delta"])
                            for row in selected_rows
                        ]
                    )
                ),
                "margin_violations": int(
                    sum(
                        bool(row["tiers"][tier]["violated"])
                        for row in selected_rows
                        for tier in TIERS
                    )
                ),
                "max_inflated_ratio": {
                    tier: max(
                        float(row["tiers"][tier]["inflated_ratio"])
                        for row in selected_rows
                    )
                    for tier in TIERS
                },
                "mean_upgrades": {
                    tier: float(
                        np.mean(
                            [
                                int(row["tiers"][tier]["upgrades"])
                                for row in selected_rows
                            ]
                        )
                    )
                    for tier in TIERS
                },
            }
        return result

    return {
        "power": power,
        "n_views": len(rows),
        "mean_weighted_quality_delta": float(np.mean(values)),
        "quality_delta_lower_95": float(np.quantile(draws, 0.025)),
        "quality_delta_upper_95": float(np.quantile(draws, 0.975)),
        "margin_violations": len(violations),
        "violation_examples": violations[:20],
        "max_inflated_ratio": worst,
        "by_size": breakdown("size"),
        "by_kind": breakdown("kind"),
        "rows": rows,
    }


def stable_seed(label: str, base: int = 20260901) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int(base + int.from_bytes(digest[:4], "big") % 1_000_000)


__all__ = (
    "BASE_COMMIT",
    "BOOTSTRAP_DRAWS",
    "COST_INFLATION",
    "EXPERIMENT",
    "FIXED_ACTION_FRACTIONS",
    "LightLowerCalibration",
    "LinearUpliftStack",
    "AbsoluteQualityFit",
    "REQUIRED_MARGIN_FRACTION",
    "SMALL_BATCH_POWERS",
    "SMALL_BATCH_LOWER_FRACTIONS",
    "SMALL_BATCH_SIZES",
    "SMALL_BATCH_UNIQUE_CUTOFF",
    "SURFACE_VARIANTS",
    "SmallBatchView",
    "UpliftFit",
    "action_flip_rate",
    "apply_cost_calibration",
    "apply_light_lower_calibration",
    "blended_quality",
    "canonical_surface_text",
    "canonicalize_episode",
    "canonicalized_episodes",
    "cost_coverage",
    "cross_calibrated_costs",
    "cross_light_lower_credits",
    "evaluate_small_batches",
    "fit_cost_calibration",
    "fit_light_lower_calibration",
    "fit_serving_quality_models",
    "fit_absolute_quality_models",
    "fit_stacked_uplifts",
    "fit_uplift_models",
    "fixed_count_actions",
    "fixed_count_evaluation",
    "grouped_bootstrap_quality_delta",
    "make_small_batch_views",
    "predict_uplifts",
    "predict_stacked_uplifts",
    "predict_absolute_quality",
    "select_vocabulary_for_scores",
    "small_batch_target_fraction",
    "small_batch_route_enabled",
    "small_batch_cost_surfaces",
    "stable_seed",
    "stable_surface_episode",
    "stable_surface_episodes",
    "stable_surface_text",
    "transform_episode",
    "transform_surface_text",
    "transformed_episodes",
)
