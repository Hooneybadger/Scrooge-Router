# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the quality head study quality heads: Q_A and Q_K, Train-OOF selection, no policy, no image.

the quality head study attacks sign discrimination, not uplift-magnitude regression. It imports
the the modeling foundation foundation and does not fork it. Exact public costs only; the the cost certificate layer
cost layer is never read. Dev paths are never constructed.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.lab.prompt_features import (
    FEATURE_VERSION,
    STRUCTURAL_FEATURE_NAMES,
    feature_signature,
)
from ossp_router.protocol import MODEL_IDS, SCORE_DECIMAL_PLACES, TIERS
from research.lab.modeling import (
    BOOTSTRAP_SEED,
    FEATURE_VERSION as MODELING_FEATURE_VERSION,
    FOLD_SEED,
    FOLDS,
    HASH_BINS,
    INTERCEPT_POLICY,
    TIER_WEIGHTS,
    TrainBundle,
    family_folds,
    feature_matrix,
    group_folds,
    load_train,
    official_score,
    oof_predict,
    paired_group_bootstrap,
    quantile_higher,
    reject_dev_reference,
    ridge_fit,
    ridge_predict,
    sort_mapping,
    weighted_final,
)


EXPERIMENT = "the quality head study"
REPORT_TYPE = "scrooge-quality_head-quality-heads-v1"
SCHEMA_VERSION = 1
DECISION_PASS = "record-quality_head-quality-heads"
DECISION_FAIL = "record-quality_head-close-train-gates"
EXPECTED_INPUTS_SHA256 = (
    "029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4"
)
EXPECTED_OUTCOMES_SHA256 = (
    "97a5a787086b3e1d9fa9c7945518543540e527ea248df4a4760de581b612a4ba"
)

TARGET_FORMS: Tuple[str, ...] = ("direct_signed", "two_head_difference")
ALPHAS: Tuple[float, ...] = (30.0, 100.0, 300.0)
BINS: Tuple[int, ...] = (256, 512)
QA_REFERENCE_CAPS: Tuple[str, ...] = ("1.05", "1.15")
QK_TOTAL_RATIO_CAPS: Tuple[str, ...] = ("2.00", "3.40")
QA_SELECT_CAP = "1.15"
FOLD_SEED_QUALITY = 2026082202
BOOTSTRAP_SEED_QUALITY = 2026082203
BOOTSTRAP_DRAWS = 1000
RHO_MAX = 0.7986
HARM_AVOIDANCE_PRIZE = 0.045029
GATE_WEIGHTED_OOF_GAIN = 0.005
GATE_FOLD_WINS = 4
GATE_LOFO_WORST_FAMILY = -0.005
KOREAN_REASONING_N_FLAG = 40

PARENT_F_PINS = {
    "balanced": {
        "ax31_count": 1320,
        "axk1_think_count": 0,
        "quality": "0.665340909091",
        "realized_cost_ratio": "1.375850604373",
    },
    "fast": {
        "ax31_count": 186,
        "axk1_think_count": 0,
        "quality": "0.609090909091",
        "realized_cost_ratio": "1.019525788415",
    },
    "premium": {
        "ax31_count": 1651,
        "axk1_think_count": 0,
        "quality": "0.68125",
        "realized_cost_ratio": "1.983960802953",
    },
    "weighted": "0.647613636364",
}

SELECTION_CRITERION = (
    "Score each config by exact-cost greedy allocation quality: rank episodes "
    "by predicted uplift divided by exact incremental cost, upgrade while the "
    "exact realized ratio stays within a reference cap, and take the resulting "
    "Train OOF quality with the OFFICIAL Decimal scorer. "
    "Q_A: reference caps 1.05 and 1.15; the config score is the mean of the two "
    "OOF qualities. "
    "Q_K: on top of the 2-action allocation produced by the winning Q_A config "
    "at cap 1.15, reference incremental K1 caps such that total ratio is 2.00 "
    "and 3.40; config score is the mean of the two. "
    "Episodes with predicted uplift <= 0 are NEVER upgraded. "
    "Tie-break order: higher fold-win count, then fewer features (256 before 512), "
    "then lower alpha."
)

_COST_CERT_MARKERS = ("research.lab.cost_certificates", "cost-certificates", "cost_certificates")
_LIGHT = 0
_AX31 = 1
_K1 = 2
_RATIO_ATOL = 1e-15


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def _json_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(np.float64(value))
    if not np.isfinite(number):
        return None
    return number


def content_tie_keys(texts: Sequence[str]) -> Tuple[str, ...]:
    """Charter §6: allocation ties break on content digest only."""

    return tuple(
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
    )


def assert_no_cost_layer() -> None:
    """the quality head study selection is independent of the the cost certificate layer cost model.

    Exact public costs only. A the cost certificate layer module in sys.modules is a contract
    failure — we never import, open, hash, or glob a cost-layer artifact.
    """

    for name in list(sys.modules):
        lowered = name.replace("\\", "/").lower()
        if any(marker in lowered for marker in _COST_CERT_MARKERS):
            raise RuntimeError(
                "the quality head study forbids reading a the cost certificate layer cost-layer artifact; "
                f"found module {name!r}"
            )


def _score_text(value: Decimal) -> str:
    quantum = Decimal(1).scaleb(-SCORE_DECIMAL_PLACES)
    with localcontext() as context:
        context.prec = 160
        context.rounding = ROUND_HALF_EVEN
        quantized = value.quantize(quantum)
    text = format(quantized, "f")
    if quantized == 0:
        return "0"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


class DecimalQuality:
    """Official Decimal quality (mean of selected Outcome.score)."""

    def __init__(self, bundle: TrainBundle) -> None:
        index = {
            (outcome.episode_id, outcome.model_id): outcome.score
            for outcome in bundle.outcomes.outcomes
        }
        self._ids = tuple(episode.episode_id for episode in bundle.inputs.episodes)
        self._scores = index
        self._n = Decimal(len(self._ids))

    def quality(self, model_ids: Sequence[str]) -> Decimal:
        if len(model_ids) != len(self._ids):
            raise ValueError("model_ids must align with episodes")
        total = sum(
            (self._scores[(episode_id, model_id)] for episode_id, model_id in zip(self._ids, model_ids)),
            Decimal("0"),
        )
        return total / self._n

    def quality_text(self, model_ids: Sequence[str]) -> str:
        return _score_text(self.quality(model_ids))

    def quality_float(self, model_ids: Sequence[str]) -> float:
        return _json_float(self.quality(model_ids))


def _rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    index = 0
    while index < arr.size:
        end = index + 1
        while end < arr.size and arr[order[end]] == arr[order[index]]:
            end += 1
        average = 0.5 * float(index + 1 + end)
        ranks[order[index:end]] = average
        index = end
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    x_raw = np.asarray(left, dtype=np.float64).reshape(-1)
    y_raw = np.asarray(right, dtype=np.float64).reshape(-1)
    if x_raw.size != y_raw.size or x_raw.size == 0:
        return None
    x_c = x_raw - x_raw.mean()
    y_c = y_raw - y_raw.mean()
    denom = float(np.sqrt(np.dot(x_c, x_c) * np.dot(y_c, y_c)))
    if denom == 0.0:
        return None
    return _json_float(np.dot(x_c, y_c) / denom)


def spearman(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    return pearson(_rankdata(left), _rankdata(right))


def _coef_lists(vectors: Sequence[np.ndarray]) -> list[list[float]]:
    return [[_json_float(value) for value in np.asarray(vector, dtype=np.float64)] for vector in vectors]


def _predict_from_coefs(
    coefs: Sequence[np.ndarray], features: np.ndarray, target_form: str
) -> np.ndarray:
    if target_form == "direct_signed":
        if len(coefs) != 1:
            raise ValueError("direct_signed expects one coefficient vector")
        return ridge_predict(coefs[0], features)
    if target_form != "two_head_difference":
        raise ValueError(f"unknown target_form: {target_form!r}")
    if len(coefs) != 2:
        raise ValueError("two_head_difference expects two coefficient vectors")
    return ridge_predict(coefs[1], features) - ridge_predict(coefs[0], features)


@dataclass(frozen=True)
class QualityHeads:
    """Deployable Q_A / Q_K pair. predict() goes through g_features only."""

    feature_version: str
    feature_signature_qa: str
    feature_signature_qk: str
    bins_qa: int
    bins_qk: int
    alpha_qa: float
    alpha_qk: float
    target_form_qa: str
    target_form_qk: str
    coef_qa: Tuple[np.ndarray, ...]
    coef_qk: Tuple[np.ndarray, ...]

    def predict(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        features_qa = feature_matrix(texts, bins=int(self.bins_qa))
        if int(self.bins_qk) == int(self.bins_qa):
            features_qk = features_qa
        else:
            features_qk = feature_matrix(texts, bins=int(self.bins_qk))
        pred_qa = _predict_from_coefs(self.coef_qa, features_qa, self.target_form_qa)
        pred_qk = _predict_from_coefs(self.coef_qk, features_qk, self.target_form_qk)
        return pred_qa, pred_qk

    def to_dict(self) -> dict[str, Any]:
        return sort_mapping(
            {
                "alpha_qa": _json_float(self.alpha_qa),
                "alpha_qk": _json_float(self.alpha_qk),
                "bins_qa": int(self.bins_qa),
                "bins_qk": int(self.bins_qk),
                "coef_qa": _coef_lists(self.coef_qa),
                "coef_qk": _coef_lists(self.coef_qk),
                "feature_signature_qa": self.feature_signature_qa,
                "feature_signature_qk": self.feature_signature_qk,
                "feature_version": self.feature_version,
                "target_form_qa": self.target_form_qa,
                "target_form_qk": self.target_form_qk,
            }
        )

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "QualityHeads":
        bins_qa = int(payload["bins_qa"])
        bins_qk = int(payload["bins_qk"])
        if bins_qa not in HASH_BINS or bins_qk not in HASH_BINS:
            raise ValueError("serialized bins are outside the the modeling foundation closed list")
        expected_qa = feature_signature(bins_qa)
        expected_qk = feature_signature(bins_qk)
        if payload["feature_signature_qa"] != expected_qa:
            raise ValueError("Q_A feature signature mismatch")
        if payload["feature_signature_qk"] != expected_qk:
            raise ValueError("Q_K feature signature mismatch")
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError("feature_version mismatch")
        coef_qa = tuple(
            np.asarray(vector, dtype=np.float64) for vector in payload["coef_qa"]
        )
        coef_qk = tuple(
            np.asarray(vector, dtype=np.float64) for vector in payload["coef_qk"]
        )
        expected_width_qa = 1 + len(STRUCTURAL_FEATURE_NAMES) + bins_qa
        expected_width_qk = 1 + len(STRUCTURAL_FEATURE_NAMES) + bins_qk
        if any(vector.shape != (expected_width_qa,) for vector in coef_qa):
            raise ValueError("Q_A coefficient width does not match bins")
        if any(vector.shape != (expected_width_qk,) for vector in coef_qk):
            raise ValueError("Q_K coefficient width does not match bins")
        return QualityHeads(
            feature_version=str(payload["feature_version"]),
            feature_signature_qa=str(payload["feature_signature_qa"]),
            feature_signature_qk=str(payload["feature_signature_qk"]),
            bins_qa=bins_qa,
            bins_qk=bins_qk,
            alpha_qa=float(payload["alpha_qa"]),
            alpha_qk=float(payload["alpha_qk"]),
            target_form_qa=str(payload["target_form_qa"]),
            target_form_qk=str(payload["target_form_qk"]),
            coef_qa=coef_qa,
            coef_qk=coef_qk,
        )


def _realized_ratio(costs: np.ndarray, light_total: float) -> float:
    if light_total <= 0.0:
        raise ValueError("light_total must be positive")
    return float(np.asarray(costs, dtype=np.float64).sum() / float(light_total))


def greedy_upgrade_mask(
    pred_uplift: np.ndarray,
    increment: np.ndarray,
    current_costs: np.ndarray,
    light_total: float,
    cap: float,
    *,
    eligible: np.ndarray,
    tie_keys: Sequence[str],
) -> np.ndarray:
    """Prefix density greedy. Non-positive predicted uplift is never upgraded."""

    pred = np.asarray(pred_uplift, dtype=np.float64).reshape(-1)
    inc = np.asarray(increment, dtype=np.float64).reshape(-1)
    current = np.asarray(current_costs, dtype=np.float64).reshape(-1)
    allow = np.asarray(eligible, dtype=bool).reshape(-1)
    if not (pred.size == inc.size == current.size == allow.size == len(tie_keys)):
        raise ValueError("greedy_upgrade_mask requires aligned inputs")
    chosen = np.zeros(pred.size, dtype=bool)
    free = allow & (pred > 0.0) & (inc <= 0.0)
    chosen[free] = True
    cost_sum = float(current.sum() + inc[free].sum())
    paid = allow & (pred > 0.0) & (inc > 0.0)
    density = np.full(pred.size, -np.inf, dtype=np.float64)
    density[paid] = pred[paid] / inc[paid]
    ranked = sorted(
        (int(index) for index in np.flatnonzero(paid)),
        key=lambda index: (-density[index], tie_keys[index]),
    )
    light = float(light_total)
    cap_f = float(cap)
    for index in ranked:
        trial = cost_sum + float(inc[index])
        if trial / light <= cap_f + _RATIO_ATOL:
            chosen[index] = True
            cost_sum = trial
        else:
            break
    if _realized_ratio(current + inc * chosen.astype(np.float64), light) > cap_f + _RATIO_ATOL:
        raise RuntimeError("greedy allocation exceeded the reference cap")
    return chosen


def models_two_action(upgrade_a: np.ndarray) -> Tuple[str, ...]:
    return tuple("ax31" if flag else "ax31-light" for flag in upgrade_a)


def models_three_action(upgrade_a: np.ndarray, upgrade_k: np.ndarray) -> Tuple[str, ...]:
    chosen = []
    for ax31, k1 in zip(upgrade_a, upgrade_k):
        if k1:
            if not ax31:
                raise RuntimeError("K1 upgrade requires an AX31 base selection")
            chosen.append("axk1-think")
        elif ax31:
            chosen.append("ax31")
        else:
            chosen.append("ax31-light")
    return tuple(chosen)


def allocate_two_action(
    pred_qa: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    cap: float,
    tie_keys: Sequence[str],
) -> np.ndarray:
    increment = costs[:, _AX31] - costs[:, _LIGHT]
    return greedy_upgrade_mask(
        pred_qa,
        increment,
        costs[:, _LIGHT],
        light_total,
        cap,
        eligible=np.ones(pred_qa.shape[0], dtype=bool),
        tie_keys=tie_keys,
    )


def allocate_k1_on_top(
    pred_qk: np.ndarray,
    upgrade_a: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    total_cap: float,
    tie_keys: Sequence[str],
) -> np.ndarray:
    increment = costs[:, _K1] - costs[:, _AX31]
    current = np.where(upgrade_a, costs[:, _AX31], costs[:, _LIGHT])
    return greedy_upgrade_mask(
        pred_qk,
        increment,
        current,
        light_total,
        total_cap,
        eligible=np.asarray(upgrade_a, dtype=bool),
        tie_keys=tie_keys,
    )


def allocate_matched(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    cap: float,
    tie_keys: Sequence[str],
) -> Tuple[str, ...]:
    upgrade_a = allocate_two_action(pred_qa, costs, light_total, cap, tie_keys)
    upgrade_k = allocate_k1_on_top(
        pred_qk, upgrade_a, costs, light_total, cap, tie_keys
    )
    return models_three_action(upgrade_a, upgrade_k)


def fit_head(
    features: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    *,
    target_form: str,
    alpha: float,
) -> Tuple[np.ndarray, ...]:
    if target_form == "direct_signed":
        return (ridge_fit(features, y_right - y_left, alpha=alpha),)
    if target_form != "two_head_difference":
        raise ValueError(f"unknown target_form: {target_form!r}")
    return (
        ridge_fit(features, y_left, alpha=alpha),
        ridge_fit(features, y_right, alpha=alpha),
    )


def predict_head(
    coefs: Sequence[np.ndarray], features: np.ndarray, target_form: str
) -> np.ndarray:
    return _predict_from_coefs(coefs, features, target_form)


def oof_head_predict(
    features: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    folds: Sequence[int],
    *,
    target_form: str,
    alpha: float,
) -> np.ndarray:
    """OOF predictions for one head. Reuses the modeling foundation oof_predict / ridge_fit."""

    if target_form == "direct_signed":
        return oof_predict(features, y_right - y_left, folds, alpha=alpha)
    if target_form != "two_head_difference":
        raise ValueError(f"unknown target_form: {target_form!r}")
    pred_left = oof_predict(features, y_left, folds, alpha=alpha)
    pred_right = oof_predict(features, y_right, folds, alpha=alpha)
    return pred_right - pred_left


def _grid() -> Tuple[dict[str, Any], ...]:
    rows = []
    for target_form in TARGET_FORMS:
        for alpha in ALPHAS:
            for bins in BINS:
                rows.append(
                    {
                        "alpha": float(alpha),
                        "bins": int(bins),
                        "target_form": target_form,
                    }
                )
    return tuple(rows)


def _argmax_pattern(score_l: float, score_a: float, score_k: float) -> str:
    top = max(score_l, score_a, score_k)
    winners = []
    if score_l == top:
        winners.append("L")
    if score_a == top:
        winners.append("A")
    if score_k == top:
        winners.append("K")
    return "=".join(winners)


def _comparison_block(
    score_left: np.ndarray,
    score_right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    greater = score_right > score_left
    equal = score_right == score_left
    less = score_right < score_left
    keep_strict_harm_on_left = np.where(less, score_left, score_right)
    keep_nonpositive_on_left = np.where(score_right <= score_left, score_left, score_right)

    def _cell(mask: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(np.count_nonzero(mask)),
            f"sum_score_{left_name}": _json_float(score_left[mask].sum()),
            f"sum_score_{right_name}": _json_float(score_right[mask].sum()),
            "sum_uplift": _json_float((score_right - score_left)[mask].sum()),
        }

    return {
        f"{right_name}_eq_{left_name}": _cell(equal),
        f"{right_name}_gt_{left_name}": _cell(greater),
        f"{right_name}_lt_{left_name}": _cell(less),
        "harm_avoidance_ceiling_keep_lt_on_left": {
            "n_left": int(np.count_nonzero(less)),
            "n_right": int(np.count_nonzero(~less)),
            "quality": _json_float(keep_strict_harm_on_left.mean()),
        },
        "harm_avoidance_ceiling_keep_le_on_left": {
            "n_left": int(np.count_nonzero(score_right <= score_left)),
            "n_right": int(np.count_nonzero(score_right > score_left)),
            "quality": _json_float(keep_nonpositive_on_left.mean()),
        },
        "mean_uplift": _json_float((score_right - score_left).mean()),
        "n": int(score_left.size),
    }


def descriptive_structure(
    scores: np.ndarray, families: Sequence[str]
) -> dict[str, Any]:
    score_l = scores[:, _LIGHT]
    score_a = scores[:, _AX31]
    score_k = scores[:, _K1]
    patterns = [_argmax_pattern(float(left), float(ax31), float(k1)) for left, ax31, k1 in zip(score_l, score_a, score_k)]
    three_way = {
        name: int(count)
        for name, count in sorted(Counter(patterns).items(), key=lambda item: item[0])
    }
    family_names = tuple(sorted(dict.fromkeys(families)))
    per_family = {}
    family_mean_uplift_a = []
    for name in family_names:
        mask = np.asarray([family == name for family in families])
        block = {
            "a_vs_l": _comparison_block(
                score_l[mask], score_a[mask], left_name="L", right_name="A"
            ),
            "k_vs_a": _comparison_block(
                score_a[mask], score_k[mask], left_name="A", right_name="K"
            ),
            "n": int(np.count_nonzero(mask)),
            "three_way": {
                key: int(count)
                for key, count in sorted(
                    Counter(item for item, chosen in zip(patterns, mask) if chosen).items()
                )
            },
        }
        per_family[name] = block
        family_mean_uplift_a.append((float((score_a[mask] - score_l[mask]).mean()), name))
    family_mean_uplift_a.sort()
    worst_three = [name for _uplift, name in family_mean_uplift_a[:3]]
    return {
        "a_vs_l": _comparison_block(score_l, score_a, left_name="L", right_name="A"),
        "k_vs_a": _comparison_block(score_a, score_k, left_name="A", right_name="K"),
        "per_family": per_family,
        "three_way": three_way,
        "worst_families_by_mean_uplift_a": worst_three,
        "worst_family_blocks": {name: per_family[name] for name in worst_three},
    }


def _precompute_oof(
    matrices: Mapping[int, np.ndarray],
    folds: Sequence[int],
    y_left: np.ndarray,
    y_right: np.ndarray,
) -> dict[Tuple[str, float, int], np.ndarray]:
    """Share feature matrices and folds across the pre-registered 12-config grid."""

    predicted = {}
    for spec in _grid():
        key = (spec["target_form"], spec["alpha"], spec["bins"])
        predicted[key] = oof_head_predict(
            matrices[spec["bins"]],
            y_left,
            y_right,
            folds,
            target_form=spec["target_form"],
            alpha=spec["alpha"],
        )
    return predicted


def _fold_quality(
    scores: np.ndarray, model_ids: Sequence[str], fold_ids: np.ndarray, fold: int
) -> float:
    mask = fold_ids == fold
    columns = np.asarray([MODEL_IDS.index(model_id) for model_id in model_ids], dtype=np.int64)
    rows = np.arange(scores.shape[0], dtype=np.int64)
    return _json_float(scores[rows, columns][mask].mean())


def _parent_fold_quality(
    scores: np.ndarray,
    parent_models: Sequence[str],
    fold_ids: np.ndarray,
    fold: int,
) -> float:
    return _fold_quality(scores, parent_models, fold_ids, fold)


def _config_record(
    *,
    head: str,
    spec: Mapping[str, Any],
    selection_score: float,
    per_cap: Sequence[Mapping[str, Any]],
    fold_win_count: int,
    selected: bool,
) -> dict[str, Any]:
    return {
        "alpha": _json_float(spec["alpha"]),
        "bins": int(spec["bins"]),
        "fold_win_count": int(fold_win_count),
        "head": head,
        "per_cap": list(per_cap),
        "selected": bool(selected),
        "selection_score": _json_float(selection_score),
        "target_form": spec["target_form"],
    }


def _select_winner(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        form_rank = 0 if row["target_form"] == "direct_signed" else 1
        return (
            -float(row["selection_score"]),
            -int(row["fold_win_count"]),
            int(row["bins"]),
            float(row["alpha"]),
            form_rank,
        )

    return min(rows, key=key)


def _correlation_block(pred: np.ndarray, target: np.ndarray, *, unequal_mask: np.ndarray) -> dict[str, Any]:
    sign_target = np.sign(target)
    pred_pos = pred > 0.0
    actual_pos = target > 0.0
    unequal_n = int(np.count_nonzero(unequal_mask))
    if unequal_n:
        sign_accuracy = _json_float(np.mean(pred_pos[unequal_mask] == actual_pos[unequal_mask]))
    else:
        sign_accuracy = None
    return {
        "n_unequal": unequal_n,
        "pearson_sign": _json_optional_float(pearson(pred, sign_target)),
        "pearson_target": _json_optional_float(pearson(pred, target)),
        "rho_max": _json_float(RHO_MAX),
        "sign_accuracy_unequal": sign_accuracy,
        "spearman_sign": _json_optional_float(spearman(pred, sign_target)),
        "spearman_target": _json_optional_float(spearman(pred, target)),
    }


def drive_parent_f(bundle: TrainBundle) -> dict[str, Any]:
    """Reproduce frozen the feasibility ladder selections on Train. Full-fit, therefore advantaged."""

    from ossp_router.feasibility_ladder import load_bundled_artifact, make_submission

    artifact = load_bundled_artifact()
    submissions = {}
    model_ids = {}
    for tier in TIERS:
        plan = make_submission(bundle.inputs, bundle.policy, artifact, tier)
        submissions[tier] = plan.submission
        model_ids[tier] = tuple(
            decision.model_id for decision in plan.submission.decisions
        )
    report = official_score(bundle.inputs, bundle.outcomes, bundle.policy, submissions)
    per_tier = {}
    qualities = {}
    for tier in TIERS:
        block = report["tiers"][tier]
        qualities[tier] = float(block["quality_score"])
        counts = {
            model_id: int(block["model_counts"].get(model_id, 0)) for model_id in MODEL_IDS
        }
        per_tier[tier] = {
            "ax31_count": counts["ax31"],
            "axk1_think_count": counts["axk1-think"],
            "budget_passed": bool(block["budget_passed"]),
            "model_counts": counts,
            "quality": block["quality_score"],
            "realized_cost_ratio": block["budget_ratio"],
            "tier_score": block["tier_score"],
        }
    return {
        "advantaged_reference": True,
        "driven": True,
        "model_ids": model_ids,
        "note": (
            "Parent-F is full-fit on Train and therefore an ADVANTAGED "
            "reference, which makes later OOF comparisons conservative."
        ),
        "official_weighted_final": report["final_score"],
        "per_tier": per_tier,
        "weighted_final_from_quality": _json_float(
            weighted_final(qualities["fast"], qualities["balanced"], qualities["premium"])
        ),
    }


def _episode_columns(model_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([MODEL_IDS.index(model_id) for model_id in model_ids], dtype=np.int64)


def _episode_scores(scores: np.ndarray, model_ids: Sequence[str]) -> np.ndarray:
    rows = np.arange(scores.shape[0], dtype=np.int64)
    return scores[rows, _episode_columns(model_ids)]


def _weighted_episode_scores(
    scores: np.ndarray, models_by_tier: Mapping[str, Sequence[str]]
) -> np.ndarray:
    return (
        TIER_WEIGHTS["fast"] * _episode_scores(scores, models_by_tier["fast"])
        + TIER_WEIGHTS["balanced"] * _episode_scores(scores, models_by_tier["balanced"])
        + TIER_WEIGHTS["premium"] * _episode_scores(scores, models_by_tier["premium"])
    )


def _count_models(model_ids: Sequence[str]) -> dict[str, int]:
    counts = {model_id: 0 for model_id in MODEL_IDS}
    for model_id in model_ids:
        counts[model_id] += 1
    return counts


def _allocation_view(
    bundle: TrainBundle,
    model_ids: Sequence[str],
    *,
    official: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    scorer = official or official_score(
        bundle.inputs,
        bundle.outcomes,
        bundle.policy,
        {tier: model_ids for tier in TIERS},
    )
    tier = scorer["tiers"]["fast"]
    return {
        "model_counts": {
            model_id: int(tier["model_counts"].get(model_id, 0)) for model_id in MODEL_IDS
        },
        "quality": tier["quality_score"],
        "realized_cost_ratio": _json_float(
            _realized_ratio(
                bundle.costs[
                    np.arange(bundle.costs.shape[0]), _episode_columns(model_ids)
                ],
                bundle.light_total,
            )
        ),
    }


def _matched_models(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    parent_f: Mapping[str, Any],
    tie_keys: Sequence[str],
) -> dict[str, Tuple[str, ...]]:
    models = {}
    for tier in TIERS:
        cap = float(parent_f["per_tier"][tier]["realized_cost_ratio"])
        models[tier] = allocate_matched(
            pred_qa, pred_qk, costs, light_total, cap, tie_keys
        )
    return models


def _lofo_predictions(
    matrices: Mapping[int, np.ndarray],
    families: Sequence[str],
    y_left: np.ndarray,
    y_right: np.ndarray,
    *,
    target_form: str,
    alpha: float,
    bins: int,
) -> np.ndarray:
    features = matrices[int(bins)]
    predicted = np.empty(features.shape[0], dtype=np.float64)
    for name, held in family_folds(families):
        train = np.ones(features.shape[0], dtype=bool)
        train[held] = False
        coefs = fit_head(
            features[train],
            y_left[train],
            y_right[train],
            target_form=target_form,
            alpha=alpha,
        )
        predicted[held] = predict_head(coefs, features[held], target_form)
    return predicted


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "alphas": [float(alpha) for alpha in ALPHAS],
            "bins": list(BINS),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED_QUALITY,
            "cost_layer_used": False,
            "feature_signature": {
                str(bins): feature_signature(bins) for bins in BINS
            },
            "feature_version": FEATURE_VERSION,
            "fold_seed": FOLD_SEED_QUALITY,
            "folds": FOLDS,
            "gate_thresholds": {
                "fold_wins": GATE_FOLD_WINS,
                "lofo_overall_no_regression": 0.0,
                "lofo_worst_family_gain": GATE_LOFO_WORST_FAMILY,
                "paired_bootstrap_q2_5_gt": 0.0,
                "weighted_oof_gain": GATE_WEIGHTED_OOF_GAIN,
            },
            "harm_avoidance_prize": HARM_AVOIDANCE_PRIZE,
            "intercept_policy": INTERCEPT_POLICY,
            "parent_f_pins": PARENT_F_PINS,
            "parent_f_is_advantaged_full_fit": True,
            "qa_reference_caps": list(QA_REFERENCE_CAPS),
            "qk_total_ratio_caps": list(QK_TOTAL_RATIO_CAPS),
            "rho_max": RHO_MAX,
            "selection_criterion": SELECTION_CRITERION,
            "target_forms": list(TARGET_FORMS),
            "tie_break": (
                "higher fold-win count, then fewer features (256 before 512), "
                "then lower alpha; residual determinism: direct_signed before "
                "two_head_difference"
            ),
        }
    )


def assemble_report(
    *,
    identity: Mapping[str, Any],
    locked: Mapping[str, Any],
    observed: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    decision: str,
) -> dict[str, Any]:
    if decision not in (DECISION_PASS, DECISION_FAIL):
        raise RuntimeError(f"invalid the quality head study decision: {decision!r}")
    report = {
        "cost_layer_used": False,
        "decision": decision,
        "dev_opened": False,
        "diagnostic": diagnostic,
        "experiment": EXPERIMENT,
        "identity": identity,
        "locked": locked,
        "observed": observed,
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
    }
    if report["dev_opened"] is not False:
        raise RuntimeError("the quality head study report must assert dev_opened is false")
    if report["cost_layer_used"] is not False:
        raise RuntimeError("the quality head study report must assert cost_layer_used is false")
    return sort_mapping(report)


def measure(bundle: TrainBundle) -> Mapping[str, Any]:
    assert_no_cost_layer()
    reject_dev_reference(bundle.identity.get("policy_source") or "bundled")
    if FEATURE_VERSION != MODELING_FEATURE_VERSION:
        raise RuntimeError("the quality head study feature version drifted from the modeling foundation")
    if bundle.identity["train_inputs_sha256"] != EXPECTED_INPUTS_SHA256:
        raise ValueError("train-inputs-hash-mismatch")
    if bundle.identity["train_outcomes_sha256"] != EXPECTED_OUTCOMES_SHA256:
        raise ValueError("train-outcomes-hash-mismatch")

    quantile_higher(np.asarray([0.0, 1.0], dtype=np.float64), 0.5)

    fold_ids = np.asarray(
        group_folds(bundle.episodes, folds=FOLDS, seed=FOLD_SEED_QUALITY),
        dtype=np.int64,
    )
    if int(FOLD_SEED_QUALITY) != int(FOLD_SEED):
        raise RuntimeError("the quality head study fold seed must match the the modeling foundation pin 2026082202")
    if int(BOOTSTRAP_SEED_QUALITY) != int(BOOTSTRAP_SEED):
        raise RuntimeError("the quality head study bootstrap seed must match the the modeling foundation pin 2026082203")

    matrices = {bins: feature_matrix(bundle.texts, bins=bins) for bins in BINS}
    tie_keys = content_tie_keys(bundle.texts)
    scores = bundle.scores
    costs = bundle.costs
    score_l = scores[:, _LIGHT]
    score_a = scores[:, _AX31]
    score_k = scores[:, _K1]
    uplift_a = score_a - score_l
    uplift_k = score_k - score_a
    decimal_quality = DecimalQuality(bundle)
    structure = descriptive_structure(scores, bundle.families)

    parent_f = drive_parent_f(bundle)
    parent_models = parent_f["model_ids"]
    parent_weighted_episodes = _weighted_episode_scores(scores, parent_models)

    qa_oof = _precompute_oof(matrices, fold_ids, score_l, score_a)
    qa_rows = []
    qa_allocations = {}
    for spec in _grid():
        pred = qa_oof[(spec["target_form"], spec["alpha"], spec["bins"])]
        per_cap = []
        qualities = []
        fold_wins = 0
        fold_score_sum = np.zeros(FOLDS, dtype=np.float64)
        for cap_text in QA_REFERENCE_CAPS:
            upgrade = allocate_two_action(
                pred, costs, bundle.light_total, float(Decimal(cap_text)), tie_keys
            )
            model_ids = models_two_action(upgrade)
            qa_allocations[(spec["target_form"], spec["alpha"], spec["bins"], cap_text)] = (
                upgrade,
                model_ids,
            )
            quality = decimal_quality.quality_float(model_ids)
            qualities.append(quality)
            realized = _realized_ratio(
                np.where(upgrade, costs[:, _AX31], costs[:, _LIGHT]),
                bundle.light_total,
            )
            if realized > float(Decimal(cap_text)) + _RATIO_ATOL:
                raise RuntimeError("Q_A greedy exceeded its reference cap")
            per_cap.append(
                {
                    "cap": cap_text,
                    "n_upgraded": int(np.count_nonzero(upgrade)),
                    "quality": _json_float(quality),
                    "quality_official": decimal_quality.quality_text(model_ids),
                    "realized_cost_ratio": _json_float(realized),
                }
            )
            for fold in range(FOLDS):
                fold_score_sum[fold] += _fold_quality(scores, model_ids, fold_ids, fold)
        for fold in range(FOLDS):
            ours = fold_score_sum[fold] / float(len(QA_REFERENCE_CAPS))
            parent = _parent_fold_quality(scores, parent_models["fast"], fold_ids, fold)
            if ours > parent:
                fold_wins += 1
        qa_rows.append(
            _config_record(
                head="Q_A",
                spec=spec,
                selection_score=float(sum(qualities) / len(qualities)),
                per_cap=per_cap,
                fold_win_count=fold_wins,
                selected=False,
            )
        )
    qa_winner = dict(_select_winner(qa_rows))
    qa_winner["selected"] = True
    for row in qa_rows:
        if (
            row["target_form"] == qa_winner["target_form"]
            and row["alpha"] == qa_winner["alpha"]
            and row["bins"] == qa_winner["bins"]
        ):
            row["selected"] = True

    qa_key = (qa_winner["target_form"], qa_winner["alpha"], qa_winner["bins"], QA_SELECT_CAP)
    upgrade_a_base, models_a_base = qa_allocations[qa_key]
    pred_qa_selected = qa_oof[
        (qa_winner["target_form"], qa_winner["alpha"], qa_winner["bins"])
    ]

    qk_oof = _precompute_oof(matrices, fold_ids, score_a, score_k)
    qk_rows = []
    for spec in _grid():
        pred = qk_oof[(spec["target_form"], spec["alpha"], spec["bins"])]
        per_cap = []
        qualities = []
        fold_wins = 0
        fold_score_sum = np.zeros(FOLDS, dtype=np.float64)
        for cap_text in QK_TOTAL_RATIO_CAPS:
            upgrade_k = allocate_k1_on_top(
                pred,
                upgrade_a_base,
                costs,
                bundle.light_total,
                float(Decimal(cap_text)),
                tie_keys,
            )
            model_ids = models_three_action(upgrade_a_base, upgrade_k)
            quality = decimal_quality.quality_float(model_ids)
            qualities.append(quality)
            selected_costs = np.where(
                upgrade_k,
                costs[:, _K1],
                np.where(upgrade_a_base, costs[:, _AX31], costs[:, _LIGHT]),
            )
            realized = _realized_ratio(selected_costs, bundle.light_total)
            if realized > float(Decimal(cap_text)) + _RATIO_ATOL:
                raise RuntimeError("Q_K greedy exceeded its reference cap")
            per_cap.append(
                {
                    "cap": cap_text,
                    "n_upgraded": int(np.count_nonzero(upgrade_k)),
                    "quality": _json_float(quality),
                    "quality_official": decimal_quality.quality_text(model_ids),
                    "realized_cost_ratio": _json_float(realized),
                }
            )
            for fold in range(FOLDS):
                fold_score_sum[fold] += _fold_quality(scores, model_ids, fold_ids, fold)
        for fold in range(FOLDS):
            ours = fold_score_sum[fold] / float(len(QK_TOTAL_RATIO_CAPS))
            parent = _parent_fold_quality(scores, parent_models["premium"], fold_ids, fold)
            if ours > parent:
                fold_wins += 1
        qk_rows.append(
            _config_record(
                head="Q_K",
                spec=spec,
                selection_score=float(sum(qualities) / len(qualities)),
                per_cap=per_cap,
                fold_win_count=fold_wins,
                selected=False,
            )
        )
    qk_winner = dict(_select_winner(qk_rows))
    qk_winner["selected"] = True
    for row in qk_rows:
        if (
            row["target_form"] == qk_winner["target_form"]
            and row["alpha"] == qk_winner["alpha"]
            and row["bins"] == qk_winner["bins"]
        ):
            row["selected"] = True

    pred_qk_selected = qk_oof[
        (qk_winner["target_form"], qk_winner["alpha"], qk_winner["bins"])
    ]

    matched = _matched_models(
        pred_qa_selected,
        pred_qk_selected,
        costs,
        bundle.light_total,
        parent_f,
        tie_keys,
    )
    oof_official = official_score(
        bundle.inputs, bundle.outcomes, bundle.policy, matched
    )
    our_weighted = float(oof_official["final_score"])
    parent_weighted = float(parent_f["official_weighted_final"])
    oof_gain = our_weighted - parent_weighted
    our_weighted_episodes = _weighted_episode_scores(scores, matched)
    gain_episodes = our_weighted_episodes - parent_weighted_episodes

    per_fold = []
    fold_wins_gate = 0
    for fold in range(FOLDS):
        mask = fold_ids == fold
        ours = _json_float(our_weighted_episodes[mask].mean())
        parent = _json_float(parent_weighted_episodes[mask].mean())
        gain = ours - parent
        win = gain > 0.0
        if win:
            fold_wins_gate += 1
        per_fold.append(
            {
                "fold": fold,
                "gain": _json_float(gain),
                "n": int(np.count_nonzero(mask)),
                "ours": ours,
                "parent": parent,
                "win": bool(win),
            }
        )

    bootstrap = paired_group_bootstrap(
        gain_episodes,
        bundle.group_keys,
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED_QUALITY,
    )

    pred_qa_lofo = _lofo_predictions(
        matrices,
        bundle.families,
        score_l,
        score_a,
        target_form=qa_winner["target_form"],
        alpha=qa_winner["alpha"],
        bins=qa_winner["bins"],
    )
    pred_qk_lofo = _lofo_predictions(
        matrices,
        bundle.families,
        score_a,
        score_k,
        target_form=qk_winner["target_form"],
        alpha=qk_winner["alpha"],
        bins=qk_winner["bins"],
    )
    lofo_matched = _matched_models(
        pred_qa_lofo,
        pred_qk_lofo,
        costs,
        bundle.light_total,
        parent_f,
        tie_keys,
    )
    lofo_official = official_score(
        bundle.inputs, bundle.outcomes, bundle.policy, lofo_matched
    )
    lofo_weighted = float(lofo_official["final_score"])
    lofo_overall_gain = lofo_weighted - parent_weighted
    lofo_episodes = _weighted_episode_scores(scores, lofo_matched)

    per_family = []
    for name, held in family_folds(bundle.families):
        ours = _json_float(lofo_episodes[held].mean())
        parent = _json_float(parent_weighted_episodes[held].mean())
        gain = ours - parent
        per_family.append(
            {
                "family": name,
                "gain": _json_float(gain),
                "high_variance": bool(name == "korean_reasoning"),
                "n": int(held.size),
                "ours": ours,
                "parent": parent,
            }
        )
    worst_family = min(per_family, key=lambda row: (row["gain"], row["family"]))

    gates = {
        "1_weighted_oof_gain": {
            "ours": _json_float(our_weighted),
            "ours_official": oof_official["final_score"],
            "parent": _json_float(parent_weighted),
            "parent_official": parent_f["official_weighted_final"],
            "pass": bool(oof_gain >= GATE_WEIGHTED_OOF_GAIN),
            "threshold": GATE_WEIGHTED_OOF_GAIN,
            "value": _json_float(oof_gain),
        },
        "2_fold_wins": {
            "n_folds": FOLDS,
            "pass": bool(fold_wins_gate >= GATE_FOLD_WINS),
            "threshold": GATE_FOLD_WINS,
            "value": fold_wins_gate,
            "wins": f"{fold_wins_gate}/{FOLDS}",
        },
        "3_paired_group_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            **{key: _json_float(bootstrap[key]) for key in ("mean", "q2_5", "q50", "q97_5")},
            "pass": bool(bootstrap["q2_5"] > 0.0),
            "seed": BOOTSTRAP_SEED_QUALITY,
            "threshold": 0.0,
        },
        "4_lofo_overall_no_regression": {
            "ours": _json_float(lofo_weighted),
            "ours_official": lofo_official["final_score"],
            "parent": _json_float(parent_weighted),
            "pass": bool(lofo_overall_gain >= 0.0),
            "threshold": 0.0,
            "value": _json_float(lofo_overall_gain),
        },
        "5_lofo_worst_family": {
            "family": worst_family["family"],
            "high_variance_note": (
                "korean_reasoning has only n=40 on Train, so gate 5 is "
                "high-variance; family sizes are reported alongside the gains."
            ),
            "n": int(worst_family["n"]),
            "pass": bool(worst_family["gain"] >= GATE_LOFO_WORST_FAMILY),
            "threshold": GATE_LOFO_WORST_FAMILY,
            "value": _json_float(worst_family["gain"]),
        },
    }
    all_pass = all(item["pass"] for item in gates.values())
    decision = DECISION_PASS if all_pass else DECISION_FAIL

    unlimited_a = allocate_two_action(
        pred_qa_selected,
        costs,
        bundle.light_total,
        float("inf"),
        tie_keys,
    )
    unlimited_models = models_two_action(unlimited_a)
    unlimited_quality = decimal_quality.quality_float(unlimited_models)
    all_ax31_quality = float(score_a.mean())
    prize_captured = (unlimited_quality - all_ax31_quality) / HARM_AVOIDANCE_PRIZE

    qa_corr = _correlation_block(pred_qa_selected, uplift_a, unequal_mask=uplift_a != 0.0)
    qk_corr = _correlation_block(pred_qk_selected, uplift_k, unequal_mask=uplift_k != 0.0)

    family_qk = []
    for name, held in family_folds(bundle.families):
        family_qk.append(
            {
                "family": name,
                "frac_actual_positive": _json_float(np.mean(uplift_k[held] > 0.0)),
                "frac_pred_positive": _json_float(np.mean(pred_qk_selected[held] > 0.0)),
                "mean_actual": _json_float(uplift_k[held].mean()),
                "mean_pred": _json_float(pred_qk_selected[held].mean()),
                "n": int(held.size),
            }
        )
    rule = next(item for item in family_qk if item["family"] == "rule_reasoning")

    coef_qa = fit_head(
        matrices[qa_winner["bins"]],
        score_l,
        score_a,
        target_form=qa_winner["target_form"],
        alpha=qa_winner["alpha"],
    )
    coef_qk = fit_head(
        matrices[qk_winner["bins"]],
        score_a,
        score_k,
        target_form=qk_winner["target_form"],
        alpha=qk_winner["alpha"],
    )
    heads = QualityHeads(
        feature_version=FEATURE_VERSION,
        feature_signature_qa=feature_signature(int(qa_winner["bins"])),
        feature_signature_qk=feature_signature(int(qk_winner["bins"])),
        bins_qa=int(qa_winner["bins"]),
        bins_qk=int(qk_winner["bins"]),
        alpha_qa=float(qa_winner["alpha"]),
        alpha_qk=float(qk_winner["alpha"]),
        target_form_qa=str(qa_winner["target_form"]),
        target_form_qk=str(qk_winner["target_form"]),
        coef_qa=coef_qa,
        coef_qk=coef_qk,
    )

    oof_per_tier = {}
    for tier in TIERS:
        block = oof_official["tiers"][tier]
        oof_per_tier[tier] = {
            "model_counts": {
                model_id: int(block["model_counts"].get(model_id, 0))
                for model_id in MODEL_IDS
            },
            "quality": block["quality_score"],
            "realized_cost_ratio": block["budget_ratio"],
            "tier_score": block["tier_score"],
        }

    observed = {
        "descriptive": structure,
        "gates": gates,
        "lofo_per_family": per_family,
        "matched_cost_oof": {
            "note": (
                "Matched-cost uses Parent-F's Train realized ratios as exact "
                "public-cost caps. Parent-F is full-fit on Train and therefore "
                "ADVANTAGED; this comparison is conservative."
            ),
            "official_weighted_final": oof_official["final_score"],
            "per_tier": oof_per_tier,
            "weighted_final_from_quality": _json_float(our_weighted),
        },
        "parent_f_frozen_ladder": {
            "advantaged_reference": True,
            "driven": True,
            "note": parent_f["note"],
            "official_weighted_final": parent_f["official_weighted_final"],
            "per_tier": parent_f["per_tier"],
            "weighted_final_from_quality": parent_f["weighted_final_from_quality"],
        },
        "per_fold": per_fold,
        "prize_fraction": {
            "all_ax31_quality": _json_float(all_ax31_quality),
            "denominator": HARM_AVOIDANCE_PRIZE,
            "n_upgraded_unlimited_pred_positive": int(np.count_nonzero(unlimited_a)),
            "unlimited_pred_positive_quality": _json_float(unlimited_quality),
            "value": _json_float(prize_captured),
        },
        "qa_base_at_1.15": {
            "model_counts": _count_models(models_a_base),
            "n_upgraded": int(np.count_nonzero(upgrade_a_base)),
            "quality_official": decimal_quality.quality_text(models_a_base),
        },
        "selected_qa": {
            "alpha": _json_float(qa_winner["alpha"]),
            "bins": int(qa_winner["bins"]),
            "fold_win_count": int(qa_winner["fold_win_count"]),
            "selection_score": _json_float(qa_winner["selection_score"]),
            "target_form": qa_winner["target_form"],
        },
        "selected_qk": {
            "alpha": _json_float(qk_winner["alpha"]),
            "bins": int(qk_winner["bins"]),
            "fold_win_count": int(qk_winner["fold_win_count"]),
            "selection_score": _json_float(qk_winner["selection_score"]),
            "target_form": qk_winner["target_form"],
        },
    }
    diagnostic = {
        "config_table": list(qa_rows) + list(qk_rows),
        "correlations": {
            "Q_A": qa_corr,
            "Q_K": qk_corr,
            "rho_max": RHO_MAX,
        },
        "qk_per_family": family_qk,
        "qk_rule_reasoning": {
            "identified_as_harmful": bool(rule["mean_pred"] < 0.0),
            **rule,
        },
        "qk_usable_signal": {
            "mean_actual_uplift": _json_float(uplift_k.mean()),
            "oof_pearson": qk_corr["pearson_target"],
            "oof_spearman": qk_corr["spearman_target"],
            "sign_accuracy_k_ne_a": qk_corr["sign_accuracy_unequal"],
        },
    }
    return {
        "decision": decision,
        "diagnostic": diagnostic,
        "heads": heads,
        "identity": dict(bundle.identity),
        "observed": observed,
    }


__all__ = (
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED_QUALITY",
    "DECISION_FAIL",
    "DECISION_PASS",
    "DecimalQuality",
    "EXPERIMENT",
    "FOLD_SEED_QUALITY",
    "HARM_AVOIDANCE_PRIZE",
    "QualityHeads",
    "RHO_MAX",
    "SELECTION_CRITERION",
    "allocate_k1_on_top",
    "allocate_matched",
    "allocate_two_action",
    "assemble_report",
    "assert_no_cost_layer",
    "content_tie_keys",
    "descriptive_structure",
    "drive_parent_f",
    "fit_head",
    "greedy_upgrade_mask",
    "load_train",
    "locked_record",
    "measure",
    "models_three_action",
    "models_two_action",
    "oof_head_predict",
    "pearson",
    "predict_head",
    "spearman",
)
