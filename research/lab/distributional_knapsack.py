# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributional quality/cost routing with chance-bounded allocation.

This module is a new model and allocation family.  It learns an
explicit prompt lexicon, fits model-wise boosted trees for quality and cost
quantiles, and spends through a canonical concave-prefix marginal queue.  It
owns its feature, score, risk, and allocation paths end to end.

The module is the research fit. Serving reads the compiled standard-library
artifact in ``ossp_router.distributional_router``.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ossp_router.cost_calibrated_router import prompt_family
from ossp_router.heuristic import episode_text
from ossp_router.protocol import MODEL_IDS, TIERS, Episode


EXPERIMENT_ID = "distributional-knapsack-v1"
FEATURE_VERSION = "explicit-lexicon-v1"
MODEL_COLUMN = {model_id: index for index, model_id in enumerate(MODEL_IDS)}

DEFAULT_VOCABULARY_SIZE = 1_024
DEFAULT_MIN_DOCUMENT_FREQUENCY = 4
DEFAULT_MAX_DOCUMENT_FRACTION = 0.90
DEFAULT_DP_BUCKETS = 2_048
DEFAULT_STABLE_BATCH_SIZE = 880
NUMBER_TOKEN = "<number>"
HEX_TOKEN = "<hex>"

FAMILY_NAMES: Tuple[str, ...] = (
    "english_multiple_choice",
    "korean_multiple_choice",
    "korean_reasoning",
    "latex_math",
    "long_context",
    "other",
    "python_program",
    "rule_reasoning",
    "symbolic_math",
    "word_problem",
)


@dataclass(frozen=True)
class TierRoutingConfig:
    """Frozen distributional chance envelope for one serving tier."""

    base_fraction: float
    composition_penalty: float
    risk_reserve: float
    ax31_tail_weight: float
    k1_tail_weight: float


DEFAULT_TIER_CONFIG: Mapping[str, TierRoutingConfig] = {
    "fast": TierRoutingConfig(0.92, 1.20, 0.020, 0.20, 1.00),
    "balanced": TierRoutingConfig(0.88, 2.10, 0.030, 0.30, 1.00),
    "premium": TierRoutingConfig(0.94, 2.80, 0.000, 0.00, 1.25),
}
MIN_CONTENT_GROUPS = 128
BALANCED_K1_MIN_GROUPS = 350
PREMIUM_K1_MIN_GROUPS = 600
PREMIUM_K1_MAX_TV = 0.10

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?|[_A-Za-z][_A-Za-z0-9]*", re.UNICODE)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)*|\d*\.\d+)$")
_HEX = re.compile(r"^(?:0x)?[0-9a-f]{12,}$", re.IGNORECASE)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_CHOICE = re.compile(r"(?:^|\n)\s*(?:\(?[A-Ea-e1-5]\)|[A-Ea-e1-5][.])\s+", re.MULTILINE)
_CODE_LINE = re.compile(
    r"(?:^|\n)\s*(?:def |class |from |import |if |for |while |SELECT |function )",
    re.MULTILINE,
)
_MATH = re.compile(r"\\(?:frac|sum|int|sqrt|boxed)|\$\$|[=+*/^]", re.IGNORECASE)


STRUCTURAL_FEATURE_NAMES: Tuple[str, ...] = (
    "log_chars",
    "log_utf8_bytes",
    "log_words",
    "log_unique_words",
    "log_lines",
    "log_messages",
    "mean_word_length",
    "max_word_length_log",
    "mean_line_length_log",
    "max_line_length_log",
    "letter_fraction",
    "upper_fraction",
    "digit_fraction",
    "space_fraction",
    "punctuation_fraction",
    "symbol_fraction",
    "non_ascii_fraction",
    "hangul_fraction",
    "newline_fraction",
    "quote_fraction",
    "bracket_fraction",
    "operator_fraction",
    "choice_count_log",
    "code_line_count_log",
    "math_marker_count_log",
    "url_count_log",
    "question_count_log",
    "colon_count_log",
    "semicolon_count_log",
    "markdown_heading_count_log",
    "list_item_count_log",
    "number_token_fraction",
)


def _canonical_token(token: str) -> str:
    folded = token.casefold()
    if _NUMBER.fullmatch(folded):
        return NUMBER_TOKEN
    if _HEX.fullmatch(folded):
        return HEX_TOKEN
    return folded


def word_tokens(text: str) -> Tuple[str, ...]:
    """Return bounded, Unicode-aware lexical tokens without hashing."""

    # Both ends are retained for long prompts so suffix questions and leading
    # instructions remain represented while inference work stays bounded.
    bounded = text if len(text) <= 24_000 else text[:16_000] + text[-8_000:]
    return tuple(_canonical_token(match.group(0)) for match in _WORD.finditer(bounded))


def lexical_terms(text: str) -> frozenset[str]:
    """Binary word, adjacent-word, and affix terms for the learned lexicon."""

    tokens = word_tokens(text)
    terms: set[str] = set()
    previous: str | None = None
    for token in tokens:
        terms.add(f"w:{token}")
        if len(token) >= 5:
            terms.add(f"p:{token[:4]}")
            terms.add(f"s:{token[-4:]}")
        if previous is not None:
            terms.add(f"b:{previous}\x1f{token}")
        previous = token
    return frozenset(terms)


def structural_features(episode: Episode) -> Tuple[float, ...]:
    """Extract the non-hashed distributional structural row."""

    text = episode_text(episode)
    characters = len(text)
    safe_characters = max(1, characters)
    tokens = word_tokens(text)
    token_lengths = [len(token) for token in tokens]
    lines = text.splitlines() or [text]
    line_lengths = [len(line) for line in lines]
    category = Counter(unicodedata.category(character)[0] for character in text)
    letters = category["L"]
    punctuation = category["P"]
    symbols = category["S"]
    digits = sum(character.isdecimal() for character in text)
    spaces = sum(character.isspace() for character in text)
    non_ascii = sum(ord(character) >= 128 for character in text)
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in text)
    upper = sum(character.isupper() for character in text)
    quotes = sum(character in "'\"`“”‘’" for character in text)
    brackets = sum(character in "()[]{}<>" for character in text)
    operators = sum(character in "+-*/=^%|&!" for character in text)
    number_tokens = sum(token == NUMBER_TOKEN for token in tokens)
    message_count = len(episode.messages) if episode.messages is not None else 1
    dense = (
        math.log1p(characters),
        math.log1p(len(text.encode("utf-8"))),
        math.log1p(len(tokens)),
        math.log1p(len(set(tokens))),
        math.log1p(len(lines)),
        math.log1p(message_count),
        math.fsum(token_lengths) / max(1, len(token_lengths)),
        math.log1p(max(token_lengths, default=0)),
        math.log1p(math.fsum(line_lengths) / max(1, len(line_lengths))),
        math.log1p(max(line_lengths, default=0)),
        letters / safe_characters,
        upper / safe_characters,
        digits / safe_characters,
        spaces / safe_characters,
        punctuation / safe_characters,
        symbols / safe_characters,
        non_ascii / safe_characters,
        hangul / safe_characters,
        text.count("\n") / safe_characters,
        quotes / safe_characters,
        brackets / safe_characters,
        operators / safe_characters,
        math.log1p(len(_CHOICE.findall(text))),
        math.log1p(len(_CODE_LINE.findall(text))),
        math.log1p(len(_MATH.findall(text))),
        math.log1p(len(_URL.findall(text))),
        math.log1p(text.count("?")),
        math.log1p(text.count(":")),
        math.log1p(text.count(";")),
        math.log1p(sum(line.lstrip().startswith("#") for line in lines)),
        math.log1p(
            sum(
                line.lstrip().startswith(("- ", "* ", "+ "))
                or bool(re.match(r"\s*\d+[.)]\s", line))
                for line in lines
            )
        ),
        number_tokens / max(1, len(tokens)),
    )
    if len(dense) != len(STRUCTURAL_FEATURE_NAMES):
        raise RuntimeError("distributional structural feature width drifted")
    return dense


def _standardized_targets(targets: np.ndarray) -> np.ndarray:
    matrix = np.asarray(targets, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (matrix - mean) / scale


def select_vocabulary(
    texts: Sequence[str],
    targets: np.ndarray,
    *,
    size: int = DEFAULT_VOCABULARY_SIZE,
    min_document_frequency: int = DEFAULT_MIN_DOCUMENT_FREQUENCY,
    max_document_fraction: float = DEFAULT_MAX_DOCUMENT_FRACTION,
) -> Tuple[str, ...]:
    """Select an explicit supervised lexicon from training rows only.

    Terms are ranked by a shrinkage-adjusted multivariate mean shift.  Unlike
    feature hashing, every deployed coordinate has a stable, inspectable term.
    """

    if len(texts) != len(targets):
        raise ValueError("texts and vocabulary targets must align")
    if size <= 0 or min_document_frequency <= 0:
        raise ValueError("invalid distributional vocabulary limits")
    normalized = _standardized_targets(targets)
    global_mean = normalized.mean(axis=0)
    counts: Counter[str] = Counter()
    sums: dict[str, np.ndarray] = {}
    for text, row in zip(texts, normalized):
        for term in lexical_terms(text):
            counts[term] += 1
            if term in sums:
                sums[term] += row
            else:
                sums[term] = row.copy()
    maximum = int(math.floor(len(texts) * max_document_fraction))
    ranked: list[tuple[float, int, str]] = []
    for term, count in counts.items():
        if count < min_document_frequency or count > maximum:
            continue
        shift = sums[term] / count - global_mean
        shrinkage = count / (count + 24.0)
        association = float(np.dot(shift, shift)) * shrinkage
        ranked.append((association, count, term))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return tuple(term for _score, _count, term in ranked[:size])


def feature_matrix(
    episodes: Sequence[Episode], vocabulary: Sequence[str]
) -> np.ndarray:
    index = {term: column for column, term in enumerate(vocabulary)}
    width = len(STRUCTURAL_FEATURE_NAMES) + len(vocabulary)
    matrix = np.zeros((len(episodes), width), dtype=np.float32)
    offset = len(STRUCTURAL_FEATURE_NAMES)
    for row, episode in enumerate(episodes):
        matrix[row, :offset] = structural_features(episode)
        for term in lexical_terms(episode_text(episode)):
            column = index.get(term)
            if column is not None:
                matrix[row, offset + column] = 1.0
    return matrix


@dataclass(frozen=True)
class DistributionalPredictions:
    quality_mean: np.ndarray
    cost_mean: np.ndarray
    cost_q90: np.ndarray


@dataclass(frozen=True)
class FitBundle:
    vocabulary: Tuple[str, ...]
    quality_models: Tuple[Any, ...]
    cost_mean_models: Tuple[Any, ...]
    cost_q50_models: Tuple[Any, ...]
    cost_q90_models: Tuple[Any, ...]


@dataclass(frozen=True)
class FamilyCalibration:
    """Full-public aggregate calibration and reference composition."""

    family_names: Tuple[str, ...]
    reference_proportions: Tuple[float, ...]
    mean_scales: np.ndarray
    q90_scales: np.ndarray


def _gbr(
    *,
    loss: str,
    alpha: float = 0.9,
    learning_rate: float = 0.045,
    n_estimators: int = 120,
    random_state: int,
) -> Any:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - research dependency guard
        raise RuntimeError("distributional fitting requires research/requirements.txt") from exc
    parameters: dict[str, Any] = {
        "learning_rate": float(learning_rate),
        "loss": loss,
        "max_depth": 3,
        "min_samples_leaf": 18,
        "n_estimators": int(n_estimators),
        "random_state": int(random_state),
        "subsample": 0.85,
    }
    if loss == "quantile":
        parameters["alpha"] = float(alpha)
    return GradientBoostingRegressor(**parameters)


def fit_distributional_models(
    episodes: Sequence[Episode],
    scores: np.ndarray,
    costs: np.ndarray,
    *,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    random_state: int = 20260830,
) -> FitBundle:
    """Fit all distributional heads on the supplied training rows."""

    score_matrix = np.asarray(scores, dtype=np.float64)
    cost_matrix = np.asarray(costs, dtype=np.float64)
    if score_matrix.shape != cost_matrix.shape or score_matrix.shape[1] != 3:
        raise ValueError("distributional scores/costs must be aligned n-by-3 matrices")
    vocabulary_targets = np.column_stack(
        (
            score_matrix[:, 1] - score_matrix[:, 0],
            score_matrix[:, 2] - score_matrix[:, 1],
            np.log1p(cost_matrix[:, 0]),
            np.log1p(cost_matrix[:, 1]),
            np.log1p(cost_matrix[:, 2]),
        )
    )
    texts = tuple(episode_text(episode) for episode in episodes)
    vocabulary = select_vocabulary(
        texts, vocabulary_targets, size=vocabulary_size
    )
    features = feature_matrix(episodes, vocabulary)
    quality_models = []
    mean_models = []
    q50_models = []
    q90_models = []
    log_costs = np.log1p(np.maximum(cost_matrix, 0.0))
    for column in range(3):
        seed = int(random_state + 1009 * column)
        quality_models.append(
            _gbr(loss="squared_error", random_state=seed).fit(
                features, score_matrix[:, column]
            )
        )
        mean_models.append(
            _gbr(
                loss="squared_error",
                learning_rate=0.04,
                n_estimators=160,
                random_state=seed + 1,
            ).fit(
                features, cost_matrix[:, column]
            )
        )
        q50_models.append(
            _gbr(loss="quantile", alpha=0.50, random_state=seed + 2).fit(
                features, log_costs[:, column]
            )
        )
        q90_models.append(
            _gbr(loss="quantile", alpha=0.90, random_state=seed + 3).fit(
                features, log_costs[:, column]
            )
        )
    return FitBundle(
        vocabulary=vocabulary,
        quality_models=tuple(quality_models),
        cost_mean_models=tuple(mean_models),
        cost_q50_models=tuple(q50_models),
        cost_q90_models=tuple(q90_models),
    )


def predict_distributional(
    fit: FitBundle, episodes: Sequence[Episode]
) -> DistributionalPredictions:
    features = feature_matrix(episodes, fit.vocabulary)

    def predict(models: Sequence[Any]) -> np.ndarray:
        return np.column_stack([model.predict(features) for model in models])

    quality = np.clip(predict(fit.quality_models), 0.0, 1.0)
    mean = np.maximum(predict(fit.cost_mean_models), np.finfo(np.float64).tiny)
    q50 = np.expm1(np.maximum(predict(fit.cost_q50_models), 0.0))
    q90 = np.expm1(np.maximum(predict(fit.cost_q90_models), 0.0))
    return DistributionalPredictions(quality, mean, np.maximum(q50, q90))


def fit_family_calibration(
    predictions: DistributionalPredictions,
    actual_costs: np.ndarray,
    families: Sequence[str],
    *,
    family_names: Sequence[str] = FAMILY_NAMES,
) -> FamilyCalibration:
    """Calibrate each cost head to each public family in aggregate."""

    actual = np.asarray(actual_costs, dtype=np.float64)
    if actual.shape != predictions.cost_mean.shape or actual.shape[1] != 3:
        raise ValueError("distributional calibration costs do not align")
    labels = np.asarray(tuple(str(value) for value in families), dtype=object)
    if labels.shape != (actual.shape[0],):
        raise ValueError("distributional calibration families do not align")
    names = tuple(str(value) for value in family_names)
    if set(labels.tolist()) - set(names):
        raise ValueError("distributional calibration saw an unknown family")
    mean_scales = np.ones((len(names), 3), dtype=np.float64)
    q90_scales = np.ones((len(names), 3), dtype=np.float64)
    reference = []
    for family_index, name in enumerate(names):
        selected = labels == name
        if not np.any(selected):
            raise ValueError(f"distributional calibration family is empty: {name}")
        reference.append(float(np.mean(selected)))
        for model_index in range(3):
            total = float(np.sum(actual[selected, model_index], dtype=np.float64))
            predicted_mean = float(
                np.sum(
                    predictions.cost_mean[selected, model_index], dtype=np.float64
                )
            )
            predicted_q90 = float(
                np.sum(predictions.cost_q90[selected, model_index], dtype=np.float64)
            )
            if min(total, predicted_mean, predicted_q90) <= 0.0:
                raise ValueError("distributional calibration totals must be positive")
            mean_scales[family_index, model_index] = total / predicted_mean
            q90_scales[family_index, model_index] = total / predicted_q90
    return FamilyCalibration(names, tuple(reference), mean_scales, q90_scales)


def apply_family_calibration(
    predictions: DistributionalPredictions,
    families: Sequence[str],
    calibration: FamilyCalibration,
) -> DistributionalPredictions:
    """Apply the frozen family aggregate scales to item cost predictions."""

    labels = tuple(str(value) for value in families)
    if len(labels) != predictions.quality_mean.shape[0]:
        raise ValueError("distributional prediction families do not align")
    lookup = {name: index for index, name in enumerate(calibration.family_names)}
    try:
        encoded = np.asarray([lookup[name] for name in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown distributional family: {exc.args[0]}") from exc
    mean = predictions.cost_mean * calibration.mean_scales[encoded]
    q90 = predictions.cost_q90 * calibration.q90_scales[encoded]
    return DistributionalPredictions(predictions.quality_mean, mean, q90)


def risk_cost_surfaces(
    predictions: DistributionalPredictions,
    tier: str,
    *,
    configs: Mapping[str, TierRoutingConfig] = DEFAULT_TIER_CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return tier-specific distributional charges and Light credits."""

    if tier not in configs:
        raise ValueError(f"unknown distributional tier: {tier!r}")
    config = configs[tier]
    mean = np.asarray(predictions.cost_mean, dtype=np.float64)
    tail_gap = np.maximum(predictions.cost_q90 - mean, 0.0)
    charge = mean.copy()
    charge[:, 1] += config.ax31_tail_weight * tail_gap[:, 1]
    charge[:, 2] += config.k1_tail_weight * tail_gap[:, 2]
    light_credit = np.maximum(mean[:, 0], np.finfo(np.float64).tiny)
    charge = np.maximum(charge, light_credit[:, None])
    return charge, light_credit


def _batch_risk_feature_names() -> Tuple[str, ...]:
    names = [
        "log_batch_size",
        "log_unique_content",
        "unique_content_fraction",
        "largest_family_fraction",
        "second_family_fraction",
        "family_concentration",
        "family_entropy",
        "family_total_variation",
    ]
    names.extend(f"family_fraction:{name}" for name in FAMILY_NAMES)
    for model_id in MODEL_IDS:
        names.extend(
            (
                f"cost_mean_total_ratio:{model_id}",
                f"cost_q90_total_ratio:{model_id}",
                f"item_cost_ratio_mean:{model_id}",
                f"item_cost_ratio_std:{model_id}",
                f"item_cost_ratio_q10:{model_id}",
                f"item_cost_ratio_q50:{model_id}",
                f"item_cost_ratio_q90:{model_id}",
                f"item_cost_ratio_q99:{model_id}",
                f"quality_mean:{model_id}",
                f"quality_std:{model_id}",
                f"quality_q10:{model_id}",
                f"quality_q50:{model_id}",
                f"quality_q90:{model_id}",
            )
        )
    for model_id in MODEL_IDS[1:]:
        names.extend(
            (
                f"uplift_mean:{model_id}",
                f"uplift_std:{model_id}",
                f"uplift_q10:{model_id}",
                f"uplift_q50:{model_id}",
                f"uplift_q90:{model_id}",
                f"density_mean:{model_id}",
                f"density_q10:{model_id}",
                f"density_q50:{model_id}",
                f"density_q90:{model_id}",
            )
        )
    for name in STRUCTURAL_FEATURE_NAMES:
        names.extend((f"structural_mean:{name}", f"structural_std:{name}"))
    return tuple(names)


BATCH_RISK_FEATURE_NAMES = _batch_risk_feature_names()


def batch_risk_features(
    predictions: DistributionalPredictions,
    structural: np.ndarray,
    families: Sequence[str],
    tie_keys: Sequence[str],
    calibration: FamilyCalibration,
) -> np.ndarray:
    """Summarize one observable batch for the finite-sample risk head."""

    quality = np.asarray(predictions.quality_mean, dtype=np.float64)
    mean = np.asarray(predictions.cost_mean, dtype=np.float64)
    q90 = np.asarray(predictions.cost_q90, dtype=np.float64)
    # Item features are quantized to the same float32 contract used by the
    # boosted heads. Batch moments accumulate those frozen values in float64
    # so the standard-library serving replica can reproduce them directly.
    dense = np.asarray(structural, dtype=np.float32)
    if quality.ndim != 2 or quality.shape != mean.shape or quality.shape != q90.shape:
        raise ValueError("distributional batch prediction matrices do not align")
    if quality.shape[1] != 3 or dense.shape != (
        quality.shape[0],
        len(STRUCTURAL_FEATURE_NAMES),
    ):
        raise ValueError("distributional batch feature matrices do not align")
    if quality.shape[0] == 0:
        raise ValueError("empty distributional batch")
    labels = tuple(str(value) for value in families)
    stable_keys = tuple(str(value) for value in tie_keys)
    if len(labels) != quality.shape[0] or len(stable_keys) != quality.shape[0]:
        raise ValueError("distributional batch labels do not align")
    if tuple(calibration.family_names) != FAMILY_NAMES:
        raise ValueError("distributional family order drifted")
    lookup = {name: index for index, name in enumerate(FAMILY_NAMES)}
    try:
        encoded = np.asarray([lookup[name] for name in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown distributional family: {exc.args[0]}") from exc

    count = quality.shape[0]
    unique_count = len(set(stable_keys))
    proportions = np.bincount(encoded, minlength=len(FAMILY_NAMES)).astype(
        np.float64
    ) / count
    positive = proportions[proportions > 0.0]
    ranked = np.sort(proportions)
    reference = np.asarray(calibration.reference_proportions, dtype=np.float64)
    if reference.shape != (len(FAMILY_NAMES),):
        raise ValueError("distributional reference composition drifted")
    row = [
        math.log1p(count),
        math.log1p(unique_count),
        unique_count / count,
        float(ranked[-1]),
        float(ranked[-2]),
        float(np.sum(proportions * proportions, dtype=np.float64)),
        float(-np.sum(positive * np.log(positive), dtype=np.float64)),
        float(0.5 * np.sum(np.abs(proportions - reference), dtype=np.float64)),
    ]
    row.extend(float(value) for value in proportions)
    generic_charge = mean.copy()
    generic_charge[:, 2] = np.maximum(mean[:, 2], q90[:, 2])
    light = np.maximum(mean[:, 0], np.finfo(np.float64).tiny)
    light_total = float(np.sum(light, dtype=np.float64))
    for model_index in range(3):
        ratio = generic_charge[:, model_index] / light
        ratio_quantiles = np.quantile(ratio, (0.10, 0.50, 0.90, 0.99))
        model_quality = quality[:, model_index]
        quality_quantiles = np.quantile(model_quality, (0.10, 0.50, 0.90))
        row.extend(
            (
                float(np.sum(mean[:, model_index], dtype=np.float64) / light_total),
                float(np.sum(q90[:, model_index], dtype=np.float64) / light_total),
                float(np.mean(ratio)),
                float(np.std(ratio)),
                *(float(value) for value in ratio_quantiles),
                float(np.mean(model_quality)),
                float(np.std(model_quality)),
                *(float(value) for value in quality_quantiles),
            )
        )
    for model_index in (1, 2):
        uplift = quality[:, model_index] - quality[:, 0]
        increment = np.maximum(
            generic_charge[:, model_index] - generic_charge[:, 0], 1e-12
        )
        density = uplift / increment
        row.extend(
            (
                float(np.mean(uplift)),
                float(np.std(uplift)),
                *(float(value) for value in np.quantile(uplift, (0.10, 0.50, 0.90))),
                float(np.mean(density)),
                *(float(value) for value in np.quantile(density, (0.10, 0.50, 0.90))),
            )
        )
    for column in range(dense.shape[1]):
        values = np.asarray(dense[:, column], dtype=np.float64)
        row.extend((float(np.mean(values)), float(np.std(values))))
    result = np.asarray(row, dtype=np.float64)
    if result.shape != (len(BATCH_RISK_FEATURE_NAMES),):
        raise RuntimeError("distributional batch-risk feature width drifted")
    return result


def fit_batch_risk_model(
    features: np.ndarray,
    safe_target_fractions: np.ndarray,
    *,
    quantile: float = 0.005,
    random_state: int = 20260830,
) -> Any:
    """Fit the low-quantile finite-sample budget head."""

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - research dependency guard
        raise RuntimeError("distributional fitting requires research/requirements.txt") from exc
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(safe_target_fractions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(BATCH_RISK_FEATURE_NAMES):
        raise ValueError("distributional risk training features have the wrong width")
    if targets.shape != (matrix.shape[0],):
        raise ValueError("distributional risk training targets do not align")
    if not 0.0 < quantile < 0.5:
        raise ValueError("distributional risk quantile must be below the median")
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=0.7,
        random_state=int(random_state),
    ).fit(matrix, targets)


def allocate_knapsack(
    quality: np.ndarray,
    charges: np.ndarray,
    light_credit: np.ndarray,
    *,
    budget_multiplier: float,
    target_fraction: float,
    buckets: int = DEFAULT_DP_BUCKETS,
    allow_k1: bool = True,
) -> np.ndarray:
    """Solve a deterministic discretized multiple-choice upgrade knapsack.

    Light is the zero-weight baseline.  Incremental chance charges are rounded
    upward, so discretization cannot spend beyond the fitted risk envelope.
    """

    score = np.asarray(quality, dtype=np.float64)
    cost = np.asarray(charges, dtype=np.float64)
    denominator = np.asarray(light_credit, dtype=np.float64)
    if score.ndim != 2 or score.shape != cost.shape or score.shape[1] != 3:
        raise ValueError("distributional knapsack matrices must be aligned n-by-3")
    if denominator.shape != (score.shape[0],):
        raise ValueError("distributional light credits do not align")
    if buckets <= 0 or not (0.0 < target_fraction <= 1.0):
        raise ValueError("invalid distributional knapsack envelope")
    base = float(np.sum(cost[:, 0], dtype=np.float64))
    cap = (
        float(budget_multiplier)
        * float(target_fraction)
        * float(np.sum(denominator, dtype=np.float64))
    )
    extra = cap - base
    if extra <= 0.0:
        return np.zeros(score.shape[0], dtype=np.int8)
    unit = extra / buckets
    allowed = 3 if allow_k1 else 2
    increment = np.maximum(cost[:, :allowed] - cost[:, [0]], 0.0)
    weights = np.ceil(increment / unit - 1e-12).astype(np.int64)
    weights[:, 0] = 0
    gains = score[:, :allowed] - score[:, [0]]

    negative = -np.inf
    state = np.full(buckets + 1, negative, dtype=np.float64)
    state[0] = 0.0
    parent = np.zeros((score.shape[0], buckets + 1), dtype=np.int8)
    for row in range(score.shape[0]):
        next_state = state.copy()
        chosen = np.zeros(buckets + 1, dtype=np.int8)
        for action in range(1, allowed):
            weight = int(weights[row, action])
            if weight > buckets:
                continue
            candidate = state[: buckets + 1 - weight] + gains[row, action]
            target = next_state[weight:]
            better = candidate > target + 1e-15
            if np.any(better):
                target[better] = candidate[better]
                chosen[weight:][better] = action
        state = next_state
        parent[row] = chosen

    selected = np.zeros(score.shape[0], dtype=np.int8)
    budget = int(np.argmax(state))
    for row in range(score.shape[0] - 1, -1, -1):
        action = int(parent[row, budget])
        selected[row] = action
        budget -= int(weights[row, action])
    return selected


def allocate_priority_queue(
    quality: np.ndarray,
    charges: np.ndarray,
    light_credit: np.ndarray,
    *,
    budget_multiplier: float,
    target_fraction: float,
    tie_keys: Sequence[str] | None = None,
    risk_groups: Sequence[Sequence[str]] = (),
    allow_k1: bool = True,
    _group_duplicates: bool = True,
) -> np.ndarray:
    """Allocate risk-charged upgrades with a deterministic marginal queue.

    Each row is reduced to its upper concave quality/cost chain before its
    marginal transitions enter the queue.  The queue therefore emits one
    canonical, non-increasing-density stream.  Allocation stops at the first
    atomic transition that does not fit; smaller budgets are prefixes of
    larger budgets instead of unrelated remainder-filling solutions.

    Every queue entry remains an explicit state transition (Light→AX31,
    Light→K1, or AX31→K1).  This is neither a global Lagrange-price search
    nor a post-hoc fill stage.
    """

    score = np.asarray(quality, dtype=np.float64)
    cost = np.asarray(charges, dtype=np.float64)
    denominator = np.asarray(light_credit, dtype=np.float64)
    if score.ndim != 2 or score.shape != cost.shape or score.shape[1] != 3:
        raise ValueError("distributional priority matrices must be aligned n-by-3")
    if denominator.shape != (score.shape[0],):
        raise ValueError("distributional priority light credits do not align")
    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("invalid distributional priority envelope")
    if tie_keys is None:
        stable_keys = tuple(f"{index:012d}" for index in range(score.shape[0]))
    else:
        stable_keys = tuple(str(value) for value in tie_keys)
        if len(stable_keys) != score.shape[0]:
            raise ValueError("distributional priority tie keys do not align")

    if _group_duplicates and len(set(stable_keys)) != len(stable_keys):
        ordered_array, representatives, inverse, counts = np.unique(
            np.asarray(stable_keys, dtype=object),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        ordered = tuple(str(value) for value in ordered_array)
        representatives = np.asarray(representatives, dtype=np.int64)
        multiplicity = np.asarray(counts, dtype=np.float64)
        expanded = representatives[np.asarray(inverse, dtype=np.int64)]
        if not (
            np.allclose(score, score[expanded], rtol=0, atol=1e-12)
            and np.allclose(cost, cost[expanded], rtol=0, atol=1e-12)
            and np.allclose(denominator, denominator[expanded], rtol=0, atol=1e-12)
        ):
            raise ValueError("identical distributional content produced different predictions")
        collapsed_groups = []
        for raw_groups in risk_groups:
            labels = np.asarray(tuple(str(value) for value in raw_groups), dtype=object)
            if not np.array_equal(labels, labels[expanded]):
                raise ValueError("identical distributional content crossed risk groups")
            collapsed_groups.append(tuple(str(labels[row]) for row in representatives))
        grouped = allocate_priority_queue(
            score[representatives] * multiplicity[:, None],
            cost[representatives] * multiplicity[:, None],
            denominator[representatives] * multiplicity,
            budget_multiplier=budget_multiplier,
            target_fraction=target_fraction,
            tie_keys=ordered,
            risk_groups=tuple(collapsed_groups),
            allow_k1=allow_k1,
            _group_duplicates=False,
        )
        return np.asarray(grouped, dtype=np.int8)[inverse]

    group_layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for raw_groups in risk_groups:
        labels = tuple(str(value) for value in raw_groups)
        if len(labels) != score.shape[0]:
            raise ValueError("distributional priority risk groups do not align")
        names = {name: index for index, name in enumerate(sorted(set(labels)))}
        encoded = np.asarray([names[name] for name in labels], dtype=np.int64)
        totals = np.bincount(
            encoded, weights=cost[:, 0], minlength=len(names)
        ).astype(np.float64)
        credits = np.bincount(
            encoded, weights=denominator, minlength=len(names)
        ).astype(np.float64)
        caps = float(budget_multiplier) * float(target_fraction) * credits
        group_layers.append((encoded, totals, caps))

    selected = np.zeros(score.shape[0], dtype=np.int8)
    current_total = float(np.sum(cost[:, 0], dtype=np.float64))
    cap = (
        float(budget_multiplier)
        * float(target_fraction)
        * float(np.sum(denominator, dtype=np.float64))
    )
    if current_total >= cap:
        return selected

    # Build the upper concave chain for every multiple-choice item.  Points
    # dominated in both charge and quality never become queue transitions.
    paths: list[tuple[int, ...]] = []
    for row in range(score.shape[0]):
        candidates = [(0.0, 0.0, 0)]
        for action in range(1, 3 if allow_k1 else 2):
            candidates.append(
                (
                    max(0.0, float(cost[row, action] - cost[row, 0])),
                    float(score[row, action] - score[row, 0]),
                    action,
                )
            )
        candidates.sort(key=lambda point: (point[0], -point[1], point[2]))

        points: list[tuple[float, float, int]] = [(0.0, 0.0, 0)]
        for increment, gain, action in candidates:
            if action == 0 or gain <= points[-1][1] + 1e-15:
                continue
            if increment <= points[-1][0] + 1e-15:
                # A beneficial zero-charge action is a real transition from
                # Light, not a replacement for the mandatory baseline.
                if points[-1][2] == 0:
                    points.append((points[-1][0], gain, action))
                elif gain > points[-1][1] + 1e-15:
                    points[-1] = (points[-1][0], gain, action)
                continue
            points.append((increment, gain, action))

        hull: list[tuple[float, float, int]] = []
        for point in points:
            while len(hull) >= 2:
                left, middle = hull[-2], hull[-1]
                left_slope = (middle[1] - left[1]) / max(
                    middle[0] - left[0], 1e-15
                )
                right_slope = (point[1] - middle[1]) / max(
                    point[0] - middle[0], 1e-15
                )
                if right_slope <= left_slope + 1e-15:
                    break
                hull.pop()
            hull.append(point)
        paths.append(tuple(point[2] for point in hull))

    # (-density, -gain, incremental charge, content key, row, path step,
    #  source action, target action)
    queue: list[tuple[float, float, float, str, int, int, int, int]] = []

    def offer(row: int, step: int) -> None:
        path = paths[row]
        if step + 1 >= len(path):
            return
        source, target = path[step], path[step + 1]
        gain = float(score[row, target] - score[row, source])
        increment = max(0.0, float(cost[row, target] - cost[row, source]))
        if gain <= 0.0:
            return
        density = gain / max(increment, 1e-15)
        heapq.heappush(
            queue,
            (
                -density,
                -gain,
                increment,
                stable_keys[row],
                row,
                step,
                source,
                target,
            ),
        )

    for row in range(score.shape[0]):
        offer(row, 0)

    while queue:
        (
            _density,
            _gain,
            increment,
            _key,
            row,
            step,
            source,
            target,
        ) = heapq.heappop(queue)
        if int(selected[row]) != source:
            continue
        if current_total + increment > cap + 1e-12:
            break
        if any(
            totals[encoded[row]] + increment > caps[encoded[row]] + 1e-12
            for encoded, totals, caps in group_layers
        ):
            break
        selected[row] = target
        current_total += increment
        for encoded, totals, _caps in group_layers:
            totals[encoded[row]] += increment
        offer(row, step + 1)
    return selected


@dataclass(frozen=True)
class RoutingResult:
    selected: np.ndarray
    target_fraction: float
    fallback_fraction: float
    family_total_variation: float
    unique_content_groups: int
    k1_enabled: bool
    risk_head_fraction: float | None


def content_tie_keys(episodes: Sequence[Episode]) -> Tuple[str, ...]:
    """Stable atomic-group keys; episode IDs and row positions are excluded."""

    return tuple(
        hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest()
        for episode in episodes
    )


def episode_families(episodes: Sequence[Episode]) -> Tuple[str, ...]:
    return tuple(prompt_family(episode) for episode in episodes)


def _composition_tv(
    families: Sequence[str], calibration: FamilyCalibration
) -> float:
    lookup = {name: index for index, name in enumerate(calibration.family_names)}
    try:
        encoded = np.asarray([lookup[str(name)] for name in families], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown distributional family: {exc.args[0]}") from exc
    proportions = np.bincount(
        encoded, minlength=len(calibration.family_names)
    ).astype(np.float64) / len(encoded)
    reference = np.asarray(calibration.reference_proportions, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(proportions - reference), dtype=np.float64))


def _risk_head_prediction(model: Any, row: np.ndarray) -> float:
    prediction = np.asarray(model.predict(row.reshape(1, -1)), dtype=np.float64)
    if prediction.shape != (1,) or not np.isfinite(prediction[0]):
        raise ValueError("distributional risk head returned an invalid prediction")
    return float(prediction[0])


def route_predictions(
    predictions: DistributionalPredictions,
    tier: str,
    *,
    budget_multiplier: float,
    structural: np.ndarray,
    families: Sequence[str],
    tie_keys: Sequence[str],
    calibration: FamilyCalibration,
    risk_models: Mapping[str, Any],
    configs: Mapping[str, TierRoutingConfig] = DEFAULT_TIER_CONFIG,
) -> RoutingResult:
    """Run the frozen distributional batch-risk state machine and canonical allocator."""

    if tier not in TIERS or tier not in configs:
        raise ValueError(f"unknown distributional tier: {tier!r}")
    if budget_multiplier <= 1.0:
        raise ValueError("distributional budget multiplier must exceed the Light baseline")
    count = predictions.quality_mean.shape[0]
    if count == 0 or len(families) != count or len(tie_keys) != count:
        raise ValueError("distributional route batch does not align")
    config = configs[tier]
    unique_count = len(set(str(value) for value in tie_keys))
    tv = _composition_tv(families, calibration)
    lower = 1.0 / float(budget_multiplier)
    fallback = float(
        np.clip(
            config.base_fraction * (1.0 - config.composition_penalty * tv),
            lower,
            config.base_fraction,
        )
    )
    if unique_count < MIN_CONTENT_GROUPS:
        return RoutingResult(
            selected=np.zeros(count, dtype=np.int8),
            target_fraction=lower,
            fallback_fraction=fallback,
            family_total_variation=tv,
            unique_content_groups=unique_count,
            k1_enabled=False,
            risk_head_fraction=None,
        )

    risk_fraction: float | None = None
    if tier == "premium":
        k1_enabled = (
            unique_count >= PREMIUM_K1_MIN_GROUPS and tv <= PREMIUM_K1_MAX_TV
        )
        target = fallback if k1_enabled else config.base_fraction
    else:
        model = risk_models.get(tier)
        if model is None:
            raise ValueError(f"distributional {tier} risk head is missing")
        risk_row = batch_risk_features(
            predictions, structural, families, tie_keys, calibration
        )
        risk_fraction = _risk_head_prediction(model, risk_row)
        target = min(
            config.base_fraction,
            max(fallback, risk_fraction - config.risk_reserve),
        )
        k1_enabled = tier == "balanced" and unique_count >= BALANCED_K1_MIN_GROUPS
    charges, light_credit = risk_cost_surfaces(predictions, tier, configs=configs)
    selected = allocate_priority_queue(
        predictions.quality_mean,
        charges,
        light_credit,
        budget_multiplier=budget_multiplier,
        target_fraction=float(target),
        tie_keys=tie_keys,
        allow_k1=k1_enabled,
    )
    return RoutingResult(
        selected=selected,
        target_fraction=float(target),
        fallback_fraction=fallback,
        family_total_variation=tv,
        unique_content_groups=unique_count,
        k1_enabled=k1_enabled,
        risk_head_fraction=risk_fraction,
    )


def selected_metrics(
    selected: np.ndarray, actual_scores: np.ndarray, actual_costs: np.ndarray
) -> dict[str, Any]:
    columns = np.asarray(selected, dtype=np.int64)
    rows = np.arange(columns.size, dtype=np.int64)
    ratio = float(
        np.sum(actual_costs[rows, columns], dtype=np.float64)
        / np.sum(actual_costs[:, 0], dtype=np.float64)
    )
    quality = float(np.mean(actual_scores[rows, columns], dtype=np.float64))
    return {
        "model_counts": {
            model_id: int(np.count_nonzero(columns == column))
            for column, model_id in enumerate(MODEL_IDS)
        },
        "quality": quality,
        "ratio": ratio,
    }


def prediction_targets(scores: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Public helper for deterministic vocabulary tests and diagnostics."""

    return np.column_stack(
        (
            scores[:, 1] - scores[:, 0],
            scores[:, 2] - scores[:, 1],
            np.log1p(costs[:, 0]),
            np.log1p(costs[:, 1]),
            np.log1p(costs[:, 2]),
        )
    )


__all__ = (
    "BALANCED_K1_MIN_GROUPS",
    "BATCH_RISK_FEATURE_NAMES",
    "DEFAULT_DP_BUCKETS",
    "DEFAULT_STABLE_BATCH_SIZE",
    "DEFAULT_TIER_CONFIG",
    "DEFAULT_VOCABULARY_SIZE",
    "DistributionalPredictions",
    "EXPERIMENT_ID",
    "FAMILY_NAMES",
    "FEATURE_VERSION",
    "FamilyCalibration",
    "FitBundle",
    "MIN_CONTENT_GROUPS",
    "PREMIUM_K1_MAX_TV",
    "PREMIUM_K1_MIN_GROUPS",
    "RoutingResult",
    "STRUCTURAL_FEATURE_NAMES",
    "TierRoutingConfig",
    "allocate_knapsack",
    "allocate_priority_queue",
    "apply_family_calibration",
    "batch_risk_features",
    "content_tie_keys",
    "episode_families",
    "feature_matrix",
    "fit_batch_risk_model",
    "fit_distributional_models",
    "fit_family_calibration",
    "lexical_terms",
    "predict_distributional",
    "prediction_targets",
    "risk_cost_surfaces",
    "route_predictions",
    "select_vocabulary",
    "selected_metrics",
    "structural_features",
    "word_tokens",
)
