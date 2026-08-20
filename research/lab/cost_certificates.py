# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the cost certificate layer cost certification layer: calibrated increments, family multipliers, K1 certificate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.lab.prompt_features import FEATURE_VERSION, feature_signature
from research.lab.modeling import (
    FACTOR_CLIP,
    FOLD_SEED,
    FOLDS,
    HASH_BINS,
    INTERCEPT_POLICY,
    N_BINS,
    OFFICIAL_CAPS,
    OPERATING_TARGETS,
    RankRecal,
    STRESS_BACKSTOP,
    TIER_WEIGHTS,
    TrainBundle,
    family_folds,
    feature_matrix,
    group_folds,
    load_train,
    official_score,
    oof_predict,
    pav_nonincreasing,
    quantile_higher,
    rank_recalibration,
    reject_dev_reference,
    ridge_fit,
    ridge_predict,
    sort_mapping,
)


EXPERIMENT = "the cost certificate layer"
REPORT_TYPE = "scrooge-cost_cert-cost-layer-v1"
SCHEMA_VERSION = 1
DECISION_PASS = "record-cost_cert-cost-layer"
FOLD_SEED_COST_CERT = FOLD_SEED  # 2026082202 — identical prompt-group folds to the modeling foundation
COVERAGE_SEED = 2026082204
VARIANTS: Tuple[str, ...] = ("per_model_log", "direct_log1p_inc")
ALPHAS: Tuple[float, ...] = (30.0, 100.0, 300.0)
BINS_GRID: Tuple[int, ...] = (256, 512)
CONFIGS: Tuple[Tuple[str, float, int], ...] = tuple(
    (variant, alpha, bins)
    for variant in VARIANTS
    for alpha in ALPHAS
    for bins in BINS_GRID
)
BOUND_QUANTILES: Tuple[float, ...] = (0.50, 0.75, 0.90)
OPERATING_P = 0.75
SMALLBATCH_C: Tuple[float, ...] = (0.0, 1.0, 2.0)
N_REF = 880
COVERAGE_DRAWS = 200
COVERAGE_SIZES: Tuple[int, ...] = (100, 300, 880)
FAMILY_DOMINANT_SHARE = 0.75
COVERAGE_TARGET = 0.99
RECAL_CLIP = FACTOR_CLIP
RECAL_BINS = N_BINS
LOG_COST_FLOOR = 1e-18
PRED_INC_FLOOR = 1e-18
LADDER_EPS = 1e-18
NEAR_BUDGET = {"fast": 1.1875, "balanced": 1.90, "premium": 3.80}
# Charter §1 2-action oracle saturation (not re-derived). Slack for K1 is
# operating_target − this floor. Fast (1.15) therefore has zero K1 slack.
TWO_ACTION_SATURATION_RATIO = 1.2169
DENYLIST_RATIO_FACTOR = 4.0
DENYLIST_CRITERION = (
    "A family is placed on the K1 denylist if and only if Train evidence "
    "shows at least one of: (D1) the family's mean uplift_K = "
    "mean(score_K - score_A) is strictly negative (K1 is net-harmful); "
    "(D2) the family's all-K1 cost ratio is at least four times the "
    "global all-K1 cost ratio (K1 cost bomb). Both quantities are "
    "computed on the full Train split from public cost and score "
    "arrays; neither uses a fitted cost head, a quality head, Dev, "
    "or any tuned threshold beyond the pre-registered factor 4."
)
SELECTION_CRITERION = (
    "Choose ONE cost configuration by, in strict priority order: "
    "(1) post-recalibration OOF aggregate ratio closest to 1.0 for inc_A; "
    "(2) highest held-out coverage from §1e at the winning c; "
    "(3) lowest worst-fold F in §1d; "
    "(4) fewer features (256 before 512), then lower alpha. "
    "Quality arrays must NEVER enter this criterion."
)
LADDER_CLAMP_RULE = (
    "After Duan smearing, enforce 0 < pred_L <= pred_A <= pred_K by the "
    "monotone raise-only projection pred_L' = max(pred_L, 1e-18), "
    "pred_A' = max(pred_A, pred_L'), pred_K' = max(pred_K, pred_A'). "
    "A cheaper tier is never lowered to meet a more expensive one; "
    "increments are then pred_A'-pred_L' and pred_K'-pred_A'. "
    "Variant direct_log1p_inc additionally clips expm1*smear increments "
    "at 0 before reconstructing the ladder from pred_L."
)
DENSITY_RULE = (
    "Certificate selection ranks eligible items by descending exact-cost "
    "density (score_K - score_A) / max(pred_inc_K, 1e-18), tie-broken by "
    "ascending episode index. Exact-cost uplift is allowed because the cost certificate layer "
    "is a cost experiment; the quality arrays never enter config selection."
)
U_QUANTILE_NAMES: Tuple[str, ...] = (
    "min",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q99",
    "max",
)
U_QUANTILE_PROBS: Tuple[float, ...] = (
    0.0,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.0,
)


def json_float(value: Any) -> float:
    return float(np.float64(value))


def json_floats(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [json_float(item) for item in array]


def config_key(variant: str, alpha: float, bins: int) -> str:
    return f"{variant}|alpha={alpha:g}|bins={bins}"


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "alphas": [json_float(alpha) for alpha in ALPHAS],
            "bins": list(BINS_GRID),
            "bound_quantiles": [json_float(prob) for prob in BOUND_QUANTILES],
            "coverage_draws": COVERAGE_DRAWS,
            "coverage_seed": COVERAGE_SEED,
            "coverage_sizes": list(COVERAGE_SIZES),
            "coverage_target": json_float(COVERAGE_TARGET),
            "denylist_criterion": DENYLIST_CRITERION,
            "denylist_ratio_factor": json_float(DENYLIST_RATIO_FACTOR),
            "density_rule": DENSITY_RULE,
            "family_dominant_share": json_float(FAMILY_DOMINANT_SHARE),
            "feature_version": FEATURE_VERSION,
            "fold_seed": FOLD_SEED_COST_CERT,
            "folds": FOLDS,
            "grid": [
                {
                    "alpha": json_float(alpha),
                    "bins": int(bins),
                    "variant": variant,
                }
                for variant, alpha, bins in CONFIGS
            ],
            "hash_bins_allowed": list(HASH_BINS),
            "intercept_policy": INTERCEPT_POLICY,
            "ladder_clamp_rule": LADDER_CLAMP_RULE,
            "n_ref": N_REF,
            "near_budget": dict(NEAR_BUDGET),
            "official_caps": dict(OFFICIAL_CAPS),
            "operating_p": json_float(OPERATING_P),
            "operating_targets": dict(OPERATING_TARGETS),
            "recal_clip": [json_float(RECAL_CLIP[0]), json_float(RECAL_CLIP[1])],
            "recal_n_bins": RECAL_BINS,
            "selection_criterion": SELECTION_CRITERION,
            "smallbatch_c": [json_float(item) for item in SMALLBATCH_C],
            "stress_backstop": json_float(STRESS_BACKSTOP),
            "two_action_saturation_ratio": json_float(TWO_ACTION_SATURATION_RATIO),
            "variants": list(VARIANTS),
        }
    )


def duan_smearing_factor(residuals: np.ndarray) -> float:
    """Duan (1983) smearing: mean(exp(residual)) in the log (or log1p) space."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Duan smearing requires a non-empty residual vector")
    return json_float(np.mean(np.exp(values)))


def clamp_price_ladder(
    pred_l: np.ndarray, pred_a: np.ndarray, pred_k: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raise-only monotone projection onto 0 < L <= A <= K. See LADDER_CLAMP_RULE."""

    light = np.maximum(np.asarray(pred_l, dtype=np.float64).reshape(-1), LADDER_EPS)
    ax31 = np.maximum(np.asarray(pred_a, dtype=np.float64).reshape(-1), light)
    k1 = np.maximum(np.asarray(pred_k, dtype=np.float64).reshape(-1), ax31)
    return light, ax31, k1


def floor_inc(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), PRED_INC_FLOOR)


def actual_increments(costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(costs, dtype=np.float64)
    return matrix[:, 1] - matrix[:, 0], matrix[:, 2] - matrix[:, 1]


@dataclass(frozen=True)
class FittedHeads:
    """Full-fit (or subset-fit) incremental cost heads plus Duan smears."""

    variant: str
    alpha: float
    coefs: Mapping[str, np.ndarray]
    smears: Mapping[str, float]
    n_clipped_inc_a: int
    n_clipped_inc_k: int


def fit_heads(X: np.ndarray, costs: np.ndarray, *, variant: str, alpha: float) -> FittedHeads:
    """Fit one pre-registered variant on a design matrix and cost columns."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown cost variant {variant!r}")
    features = np.asarray(X, dtype=np.float64)
    matrix = np.maximum(np.asarray(costs, dtype=np.float64), LOG_COST_FLOOR)
    inc_a_raw = costs[:, 1] - costs[:, 0]
    inc_k_raw = costs[:, 2] - costs[:, 1]
    n_clip_a = int(np.sum(np.asarray(inc_a_raw, dtype=np.float64) < 0.0))
    n_clip_k = int(np.sum(np.asarray(inc_k_raw, dtype=np.float64) < 0.0))
    coefs: dict[str, np.ndarray] = {}
    smears: dict[str, float] = {}
    if variant == "per_model_log":
        for column, name in enumerate(("L", "A", "K")):
            target = np.log(matrix[:, column])
            coef = ridge_fit(features, target, alpha=float(alpha))
            fitted = ridge_predict(coef, features)
            coefs[name] = np.asarray(coef, dtype=np.float64)
            smears[name] = duan_smearing_factor(target - fitted)
    else:
        target_l = np.log(matrix[:, 0])
        coef_l = ridge_fit(features, target_l, alpha=float(alpha))
        coefs["L"] = np.asarray(coef_l, dtype=np.float64)
        smears["L"] = duan_smearing_factor(target_l - ridge_predict(coef_l, features))
        for raw, name in ((inc_a_raw, "inc_A"), (inc_k_raw, "inc_K")):
            target = np.log1p(np.clip(np.asarray(raw, dtype=np.float64), 0.0, None))
            coef = ridge_fit(features, target, alpha=float(alpha))
            fitted = ridge_predict(coef, features)
            coefs[name] = np.asarray(coef, dtype=np.float64)
            smears[name] = duan_smearing_factor(target - fitted)
    return FittedHeads(
        variant=variant,
        alpha=float(alpha),
        coefs=coefs,
        smears=smears,
        n_clipped_inc_a=n_clip_a,
        n_clipped_inc_k=n_clip_k,
    )


def predict_heads(
    X: np.ndarray, heads: FittedHeads
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pred_L, pred_A, pred_K, inc_A, inc_K after smearing and ladder clamp."""

    features = np.asarray(X, dtype=np.float64)
    if heads.variant == "per_model_log":
        pred_l = np.exp(ridge_predict(heads.coefs["L"], features)) * heads.smears["L"]
        pred_a = np.exp(ridge_predict(heads.coefs["A"], features)) * heads.smears["A"]
        pred_k = np.exp(ridge_predict(heads.coefs["K"], features)) * heads.smears["K"]
        pred_l, pred_a, pred_k = clamp_price_ladder(pred_l, pred_a, pred_k)
        return pred_l, pred_a, pred_k, pred_a - pred_l, pred_k - pred_a
    pred_l = np.exp(ridge_predict(heads.coefs["L"], features)) * heads.smears["L"]
    inc_a = np.expm1(ridge_predict(heads.coefs["inc_A"], features)) * heads.smears["inc_A"]
    inc_k = np.expm1(ridge_predict(heads.coefs["inc_K"], features)) * heads.smears["inc_K"]
    inc_a = np.maximum(inc_a, 0.0)
    inc_k = np.maximum(inc_k, 0.0)
    pred_l = np.maximum(np.asarray(pred_l, dtype=np.float64).reshape(-1), LADDER_EPS)
    pred_a = pred_l + inc_a
    pred_k = pred_a + inc_k
    return pred_l, pred_a, pred_k, inc_a, inc_k


def oof_incremental_costs(
    X: np.ndarray,
    costs: np.ndarray,
    folds: Sequence[int],
    *,
    variant: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FittedHeads]:
    """Prompt-group OOF predictions with per-fold Duan smearing.

    The returned FittedHeads is a dummy-shaped container holding the
    *full-data clip counts* plus empty coefs; callers that need a
    deployable transform should call ``fit_heads`` on the full matrix.
    The last element is unused by most callers — kept so tests can
    inspect clip counts without a second pass. We also return a
    representative FittedHeads built on the last fold only? No: clip
    counts are computed on the full cost matrix and stored on a
    lightweight heads object fitted on fold 0's complement so smear
    arithmetic stays testable. Prefer ``n_clipped_*`` via a dedicated
    count instead.
    """

    features = np.asarray(X, dtype=np.float64)
    matrix = np.asarray(costs, dtype=np.float64)
    fold_ids = np.asarray(list(folds))
    n_rows = features.shape[0]
    pred_l = np.empty(n_rows, dtype=np.float64)
    pred_a = np.empty(n_rows, dtype=np.float64)
    pred_k = np.empty(n_rows, dtype=np.float64)
    inc_a = np.empty(n_rows, dtype=np.float64)
    inc_k = np.empty(n_rows, dtype=np.float64)
    last_heads: Optional[FittedHeads] = None
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        test = fold_ids == fold
        heads = fit_heads(features[train], matrix[train], variant=variant, alpha=alpha)
        last_heads = heads
        p_l, p_a, p_k, i_a, i_k = predict_heads(features[test], heads)
        pred_l[test] = p_l
        pred_a[test] = p_a
        pred_k[test] = p_k
        inc_a[test] = i_a
        inc_k[test] = i_k
    if last_heads is None:
        raise ValueError("oof_incremental_costs requires at least one fold")
    clip_heads = FittedHeads(
        variant=variant,
        alpha=float(alpha),
        coefs=last_heads.coefs,
        smears=last_heads.smears,
        n_clipped_inc_a=int(np.sum((matrix[:, 1] - matrix[:, 0]) < 0.0)),
        n_clipped_inc_k=int(np.sum((matrix[:, 2] - matrix[:, 1]) < 0.0)),
    )
    return pred_l, pred_a, pred_k, inc_a, inc_k, clip_heads


def fit_recal(pred_inc: np.ndarray, actual_inc: np.ndarray) -> RankRecal:
    return rank_recalibration(
        floor_inc(pred_inc),
        np.asarray(actual_inc, dtype=np.float64).reshape(-1),
        n_bins=RECAL_BINS,
        clip=RECAL_CLIP,
    )


def apply_recal(recal: RankRecal, pred_inc: np.ndarray) -> np.ndarray:
    return recal.apply(floor_inc(pred_inc))


def recal_report(
    recal: RankRecal, pred_inc: np.ndarray, actual_inc: np.ndarray
) -> dict[str, Any]:
    applied = apply_recal(recal, pred_inc)
    actual = np.asarray(actual_inc, dtype=np.float64).reshape(-1)
    pred_sum = float(applied.sum())
    actual_sum = float(actual.sum())
    lo = float(RECAL_CLIP[0])
    hi = float(RECAL_CLIP[1])
    n_lo = int(np.sum(recal.clipped_factors <= lo + 1e-15))
    n_hi = int(np.sum(recal.clipped_factors >= hi - 1e-15))
    return {
        "aggregate_ratio": json_float(actual_sum / pred_sum) if pred_sum > 0.0 else None,
        "clipped_factors": json_floats(recal.clipped_factors),
        "edges": json_floats(recal.edges),
        "n_bins_hit_clip_high": n_hi,
        "n_bins_hit_clip_low": n_lo,
        "n_bins_hit_clip": n_lo + n_hi,
        "pav_factors": json_floats(recal.pav_factors),
        "raw_factors": json_floats(recal.raw_factors),
        "sum_actual": json_float(actual_sum),
        "sum_recal_pred": json_float(pred_sum),
    }


def transform_divergence(
    oof_recal: RankRecal,
    full_recal: RankRecal,
    oof_pred: np.ndarray,
) -> dict[str, Any]:
    """Quantify the A6 OOF-vs-full-fit recalibration gap (do not hide it)."""

    factor_delta = np.asarray(oof_recal.clipped_factors, dtype=np.float64) - np.asarray(
        full_recal.clipped_factors, dtype=np.float64
    )
    oof_applied = apply_recal(oof_recal, oof_pred)
    full_on_oof = apply_recal(full_recal, oof_pred)
    applied_delta = oof_applied - full_on_oof
    return {
        "applied_on_oof_pred_max_abs": json_float(np.max(np.abs(applied_delta))),
        "applied_on_oof_pred_mean_abs": json_float(np.mean(np.abs(applied_delta))),
        "clipped_factor_max_abs": json_float(np.max(np.abs(factor_delta))),
        "clipped_factor_mean_abs": json_float(np.mean(np.abs(factor_delta))),
        "note": (
            "Charter §6/A6: the estimator that chose safety is the OOF "
            "recalibration; the deployed transform is the same procedure "
            "baked on full-fit predictions. This block is the measured gap."
        ),
    }


def describe_distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        name: json_float(quantile_higher(array, prob))
        for name, prob in zip(U_QUANTILE_NAMES, U_QUANTILE_PROBS)
    }


def ratio_or_none(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0.0:
        return None
    return json_float(numerator / denominator)


def family_ratio(
    actual: np.ndarray, pred: np.ndarray, mask: np.ndarray
) -> Optional[float]:
    return ratio_or_none(float(actual[mask].sum()), float(pred[mask].sum()))


def k1_denylist(
    families: Sequence[str], scores: np.ndarray, costs: np.ndarray
) -> dict[str, Any]:
    """Apply DENYLIST_CRITERION mechanically. Does not enter config selection."""

    names = tuple(sorted(dict.fromkeys(families)))
    fam = np.asarray(list(families))
    score = np.asarray(scores, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64)
    light_total = float(cost[:, 0].sum())
    k1_total = float(cost[:, 2].sum())
    global_k1_ratio = k1_total / light_total if light_total > 0.0 else 0.0
    threshold = DENYLIST_RATIO_FACTOR * global_k1_ratio
    denied: list[str] = []
    evidence: dict[str, Any] = {}
    for name in names:
        mask = fam == name
        mean_uplift_k = float((score[mask, 2] - score[mask, 1]).mean())
        family_light = float(cost[mask, 0].sum())
        family_k1 = float(cost[mask, 2].sum())
        family_k1_ratio = family_k1 / family_light if family_light > 0.0 else 0.0
        d1 = bool(mean_uplift_k < 0.0)
        d2 = bool(family_k1_ratio >= threshold)
        flagged = bool(d1 or d2)
        if flagged:
            denied.append(name)
        evidence[name] = {
            "d1_mean_uplift_k_negative": d1,
            "d2_k1_cost_bomb": d2,
            "family_all_k1_ratio": json_float(family_k1_ratio),
            "flagged": flagged,
            "mean_uplift_k": json_float(mean_uplift_k),
            "n": int(mask.sum()),
        }
    return {
        "criterion": DENYLIST_CRITERION,
        "denied": denied,
        "evidence": evidence,
        "global_all_k1_ratio": json_float(global_k1_ratio),
        "ratio_threshold": json_float(threshold),
    }


def n_k1_allowed(slack: float, b_cert: float, n: int) -> int:
    """Full-batch certificate arithmetic: floor(slack * n / B_cert)."""

    if b_cert <= 0.0 or slack <= 0.0 or int(n) <= 0:
        return 0
    return int(np.floor(float(slack) * float(n) / float(b_cert)))


def n_k1_allowed_for_batch(
    slack: float,
    b_cert: float,
    light_batch: float,
    light_ref: float,
    n_ref: int,
) -> int:
    """Currency-correct form that reduces to ``n_k1_allowed`` on the Train batch.

    item_bound_ratio = (B_cert * light_ref / n_ref) / light_batch
    n_k1_allowed = floor(slack / item_bound_ratio)
    """

    if b_cert <= 0.0 or slack <= 0.0 or light_batch <= 0.0 or light_ref <= 0.0:
        return 0
    return int(np.floor(float(slack) * float(light_batch) * float(n_ref) / (float(b_cert) * float(light_ref))))


def smallbatch_factor(n: int, c: float, n_ref: int = N_REF) -> float:
    if int(n) <= 0:
        raise ValueError("smallbatch_factor requires n >= 1")
    extra = max(0.0, (1.0 / math.sqrt(float(n))) - (1.0 / math.sqrt(float(n_ref))))
    return 1.0 + float(c) * extra


def composition_safety(
    family_weights: Mapping[str, float],
    multipliers: Mapping[str, float],
    *,
    n: int,
    c: float,
    n_ref: int = N_REF,
    unseen_boost: float = 1.0,
) -> float:
    """s(batch) = (sum_f w_f * m_f_lofo) * smallbatch(n). Unseen families get unseen_boost."""

    total_w = 0.0
    mixed = 0.0
    for name in sorted(family_weights):
        weight = float(family_weights[name])
        if weight <= 0.0:
            continue
        multiplier = float(multipliers[name]) if name in multipliers else float(unseen_boost)
        mixed += weight * multiplier
        total_w += weight
    if total_w <= 0.0:
        mixed = 1.0
    else:
        mixed /= total_w
    return mixed * smallbatch_factor(int(n), float(c), n_ref=int(n_ref))


def predicted_light_weights(
    families: Sequence[str], pred_l: np.ndarray
) -> dict[str, float]:
    fam = np.asarray(list(families))
    light = np.asarray(pred_l, dtype=np.float64).reshape(-1)
    total = float(light.sum())
    weights: dict[str, float] = {}
    for name in sorted(dict.fromkeys(fam.tolist())):
        share = float(light[fam == name].sum())
        weights[name] = share / total if total > 0.0 else 0.0
    return weights


def slacks_from_operating() -> dict[str, float]:
    return {
        tier: max(0.0, float(target) - TWO_ACTION_SATURATION_RATIO)
        for tier, target in OPERATING_TARGETS.items()
    }


def _rank_eligible(
    eligible: np.ndarray, density: np.ndarray, limit: int
) -> np.ndarray:
    index = np.flatnonzero(eligible)
    if index.size == 0 or int(limit) <= 0:
        return np.zeros(0, dtype=np.int64)
    dens = np.asarray(density, dtype=np.float64)
    order = np.lexsort((index, -dens[index]))
    return index[order][: int(limit)]


def _inflation_f(
    actual_u: np.ndarray, eligible: np.ndarray, b_raw: float
) -> Optional[float]:
    chosen = np.asarray(eligible, dtype=bool)
    n_sel = int(chosen.sum())
    if n_sel == 0 or b_raw <= 0.0:
        return None
    return json_float(float(actual_u[chosen].sum()) / (float(n_sel) * float(b_raw)))


@dataclass(frozen=True)
class CostLayer:
    """Fitted, serializable the cost certificate layer cost certification layer."""

    feature_version: str
    feature_signature: str
    bins: int
    alpha: float
    variant: str
    ridge_coefficients: Mapping[str, Tuple[float, ...]]
    smearing_factors: Mapping[str, float]
    ladder_clamp_rule: str
    recal_a_edges: Tuple[float, ...]
    recal_a_factors: Tuple[float, ...]
    recal_k_edges: Tuple[float, ...]
    recal_k_factors: Tuple[float, ...]
    family_multipliers_lofo_a: Mapping[str, float]
    family_multipliers_lofo_k: Mapping[str, float]
    unseen_boost: float
    unseen_boost_a: float
    unseen_boost_k: float
    smallbatch_c: float
    n_ref: int
    b_raw: Mapping[str, float]
    b_cert: Mapping[str, float]
    selected_p: float
    k1_denylist: Tuple[str, ...]
    operating_targets: Mapping[str, float]
    official_caps: Mapping[str, float]
    stress_backstop: float
    near_budget: Mapping[str, float]
    two_action_saturation_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return sort_mapping(
            {
                "alpha": json_float(self.alpha),
                "b_cert": {key: json_float(value) for key, value in self.b_cert.items()},
                "b_raw": {key: json_float(value) for key, value in self.b_raw.items()},
                "bins": int(self.bins),
                "family_multipliers_lofo_a": {
                    key: json_float(value)
                    for key, value in self.family_multipliers_lofo_a.items()
                },
                "family_multipliers_lofo_k": {
                    key: json_float(value)
                    for key, value in self.family_multipliers_lofo_k.items()
                },
                "feature_signature": self.feature_signature,
                "feature_version": self.feature_version,
                "k1_denylist": list(self.k1_denylist),
                "ladder_clamp_rule": self.ladder_clamp_rule,
                "n_ref": int(self.n_ref),
                "near_budget": {
                    key: json_float(value) for key, value in self.near_budget.items()
                },
                "official_caps": {
                    key: json_float(value) for key, value in self.official_caps.items()
                },
                "operating_targets": {
                    key: json_float(value) for key, value in self.operating_targets.items()
                },
                "recal_a_edges": [json_float(value) for value in self.recal_a_edges],
                "recal_a_factors": [json_float(value) for value in self.recal_a_factors],
                "recal_k_edges": [json_float(value) for value in self.recal_k_edges],
                "recal_k_factors": [json_float(value) for value in self.recal_k_factors],
                "ridge_coefficients": {
                    name: [json_float(value) for value in coef]
                    for name, coef in self.ridge_coefficients.items()
                },
                "selected_p": json_float(self.selected_p),
                "smallbatch_c": json_float(self.smallbatch_c),
                "smearing_factors": {
                    name: json_float(value) for name, value in self.smearing_factors.items()
                },
                "stress_backstop": json_float(self.stress_backstop),
                "two_action_saturation_ratio": json_float(self.two_action_saturation_ratio),
                "unseen_boost": json_float(self.unseen_boost),
                "unseen_boost_a": json_float(self.unseen_boost_a),
                "unseen_boost_k": json_float(self.unseen_boost_k),
                "variant": self.variant,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CostLayer":
        data = dict(payload)
        return cls(
            feature_version=str(data["feature_version"]),
            feature_signature=str(data["feature_signature"]),
            bins=int(data["bins"]),
            alpha=float(data["alpha"]),
            variant=str(data["variant"]),
            ridge_coefficients={
                str(name): tuple(float(value) for value in coef)
                for name, coef in data["ridge_coefficients"].items()
            },
            smearing_factors={
                str(name): float(value) for name, value in data["smearing_factors"].items()
            },
            ladder_clamp_rule=str(data["ladder_clamp_rule"]),
            recal_a_edges=tuple(float(value) for value in data["recal_a_edges"]),
            recal_a_factors=tuple(float(value) for value in data["recal_a_factors"]),
            recal_k_edges=tuple(float(value) for value in data["recal_k_edges"]),
            recal_k_factors=tuple(float(value) for value in data["recal_k_factors"]),
            family_multipliers_lofo_a={
                str(name): float(value)
                for name, value in data["family_multipliers_lofo_a"].items()
            },
            family_multipliers_lofo_k={
                str(name): float(value)
                for name, value in data["family_multipliers_lofo_k"].items()
            },
            unseen_boost=float(data["unseen_boost"]),
            unseen_boost_a=float(data["unseen_boost_a"]),
            unseen_boost_k=float(data["unseen_boost_k"]),
            smallbatch_c=float(data["smallbatch_c"]),
            n_ref=int(data["n_ref"]),
            b_raw={str(key): float(value) for key, value in data["b_raw"].items()},
            b_cert={str(key): float(value) for key, value in data["b_cert"].items()},
            selected_p=float(data["selected_p"]),
            k1_denylist=tuple(str(name) for name in data["k1_denylist"]),
            operating_targets={
                str(key): float(value) for key, value in data["operating_targets"].items()
            },
            official_caps={
                str(key): float(value) for key, value in data["official_caps"].items()
            },
            stress_backstop=float(data["stress_backstop"]),
            near_budget={str(key): float(value) for key, value in data["near_budget"].items()},
            two_action_saturation_ratio=float(data["two_action_saturation_ratio"]),
        )

    def _heads(self) -> FittedHeads:
        return FittedHeads(
            variant=self.variant,
            alpha=self.alpha,
            coefs={
                name: np.asarray(coef, dtype=np.float64)
                for name, coef in self.ridge_coefficients.items()
            },
            smears=dict(self.smearing_factors),
            n_clipped_inc_a=0,
            n_clipped_inc_k=0,
        )

    def _recal_a(self) -> RankRecal:
        factors = np.asarray(self.recal_a_factors, dtype=np.float64)
        return RankRecal(
            edges=np.asarray(self.recal_a_edges, dtype=np.float64),
            raw_factors=factors,
            pav_factors=factors,
            clipped_factors=factors,
        )

    def _recal_k(self) -> RankRecal:
        factors = np.asarray(self.recal_k_factors, dtype=np.float64)
        return RankRecal(
            edges=np.asarray(self.recal_k_edges, dtype=np.float64),
            raw_factors=factors,
            pav_factors=factors,
            clipped_factors=factors,
        )

    def predict(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Deployable path: feature_matrix → heads → clamp → rank isotonic.

        ``feature_matrix`` calls ``research.lab.prompt_features.feature_row`` so a
        later router can reproduce this bit-for-bit.
        """

        features = feature_matrix(texts, bins=int(self.bins))
        pred_l, pred_a, pred_k, inc_a, inc_k = predict_heads(features, self._heads())
        inc_a = apply_recal(self._recal_a(), inc_a)
        inc_k = apply_recal(self._recal_k(), inc_k)
        pred_a = pred_l + inc_a
        pred_k = pred_a + inc_k
        return {
            "inc_A": inc_a,
            "inc_K": inc_k,
            "pred_A": pred_a,
            "pred_K": pred_k,
            "pred_L": pred_l,
        }


def _unseen_boost(m_f: Mapping[str, Optional[float]], m_lofo: Mapping[str, Optional[float]]) -> Optional[float]:
    ratios = []
    for name in sorted(m_f):
        base = m_f[name]
        lofo = m_lofo[name]
        if base is None or lofo is None or base <= 0.0:
            continue
        ratios.append(float(lofo) / float(base))
    if not ratios:
        return None
    return json_float(max(ratios))


def _lofo_recalibrated(
    X: np.ndarray,
    costs: np.ndarray,
    families: Sequence[str],
    *,
    variant: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fam = np.asarray(list(families))
    pred_l = np.empty(fam.shape[0], dtype=np.float64)
    inc_a = np.empty(fam.shape[0], dtype=np.float64)
    inc_k = np.empty(fam.shape[0], dtype=np.float64)
    actual_a, actual_k = actual_increments(costs)
    for name, held_index in family_folds(families):
        held = np.zeros(fam.shape[0], dtype=bool)
        held[held_index] = True
        train = ~held
        heads = fit_heads(X[train], costs[train], variant=variant, alpha=alpha)
        p_l_tr, _a_tr, _k_tr, i_a_tr, i_k_tr = predict_heads(X[train], heads)
        p_l_te, _a_te, _k_te, i_a_te, i_k_te = predict_heads(X[held], heads)
        del p_l_tr
        rec_a = fit_recal(i_a_tr, actual_a[train])
        rec_k = fit_recal(i_k_tr, actual_k[train])
        pred_l[held] = p_l_te
        inc_a[held] = apply_recal(rec_a, i_a_te)
        inc_k[held] = apply_recal(rec_k, i_k_te)
    return pred_l, inc_a, inc_k


def _coverage_table(
    families: Sequence[str],
    pred_l: np.ndarray,
    pred_inc_a: np.ndarray,
    actual_l: np.ndarray,
    actual_a: np.ndarray,
    multipliers: Mapping[str, float],
    unseen_boost: float,
    *,
    seed: int,
) -> dict[str, Any]:
    fam = np.asarray(list(families))
    names = tuple(sorted(dict.fromkeys(fam.tolist())))
    pred_l = np.asarray(pred_l, dtype=np.float64)
    pred_inc = np.asarray(pred_inc_a, dtype=np.float64)
    actual_l = np.asarray(actual_l, dtype=np.float64)
    actual_a = np.asarray(actual_a, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    per_c: dict[str, Any] = {}
    for c_value in SMALLBATCH_C:
        per_family: dict[str, Any] = {}
        binding = True
        min_cov = 1.0
        for name in names:
            focus = np.flatnonzero(fam == name)
            other = np.flatnonzero(fam != name)
            n_ok = 0
            n_total = 0
            for size in COVERAGE_SIZES:
                n_focus = int((3 * int(size)) // 4)
                n_other = int(size) - n_focus
                for _draw in range(COVERAGE_DRAWS):
                    chosen_focus = rng.choice(focus, size=n_focus, replace=True)
                    chosen_other = rng.choice(other, size=n_other, replace=True)
                    chosen = np.concatenate([chosen_focus, chosen_other])
                    batch_fam = fam[chosen]
                    batch_pred_l = pred_l[chosen]
                    light_actual = float(actual_l[chosen].sum())
                    pred_spent = float((pred_l[chosen] + pred_inc[chosen]).sum())
                    actual_spent = float(actual_a[chosen].sum())
                    pred_ratio = pred_spent / light_actual if light_actual > 0.0 else 0.0
                    actual_ratio = actual_spent / light_actual if light_actual > 0.0 else 0.0
                    weights = predicted_light_weights(batch_fam.tolist(), batch_pred_l)
                    safety = composition_safety(
                        weights,
                        multipliers,
                        n=int(size),
                        c=float(c_value),
                        n_ref=N_REF,
                        unseen_boost=float(unseen_boost),
                    )
                    n_total += 1
                    if safety * pred_ratio >= actual_ratio:
                        n_ok += 1
            coverage = float(n_ok) / float(n_total) if n_total else 0.0
            per_family[name] = {
                "coverage": json_float(coverage),
                "n_designs": n_total,
                "n_covered": n_ok,
            }
            min_cov = min(min_cov, coverage)
            if coverage < COVERAGE_TARGET:
                binding = False
        per_c[f"{c_value:g}"] = {
            "c": json_float(c_value),
            "meets_99_on_every_family": binding,
            "min_family_coverage": json_float(min_cov),
            "per_family": per_family,
        }
    winning: Optional[float] = None
    for c_value in SMALLBATCH_C:
        if per_c[f"{c_value:g}"]["meets_99_on_every_family"]:
            winning = float(c_value)
            break
    return {
        "per_c": per_c,
        "reference_allocation": "all-AX31",
        "winning_c": json_float(winning) if winning is not None else None,
        "winning_c_found": winning is not None,
    }


@dataclass(frozen=True)
class ConfigMetrics:
    """Quality-free selection row. Scores must never be added to this type."""

    variant: str
    alpha: float
    bins: int
    oof_ratio_a: float
    coverage_key: Tuple[int, float]
    worst_fold_f: float


def select_cost_config(rows: Sequence[ConfigMetrics]) -> ConfigMetrics:
    """Apply SELECTION_CRITERION mechanically. No quality field exists here."""

    if not rows:
        raise ValueError("select_cost_config requires at least one config")
    # Assert the selection surface cannot carry a quality array.
    forbidden = {"quality", "score", "scores", "uplift", "q_a", "q_k"}
    for row in rows:
        extra = set(row.__dataclass_fields__) & forbidden
        if extra:
            raise RuntimeError(f"quality entered cost selection via {sorted(extra)}")
    ordered = sorted(
        rows,
        key=lambda row: (
            abs(float(row.oof_ratio_a) - 1.0),
            -row.coverage_key[0],
            -row.coverage_key[1],
            float(row.worst_fold_f),
            int(row.bins),
            float(row.alpha),
            row.variant,
        ),
    )
    return ordered[0]


def _certificate_block(
    *,
    actual_u_inc: np.ndarray,
    pred_u_inc: np.ndarray,
    pred_u_inc_lofo: np.ndarray,
    actual_inc_k: np.ndarray,
    exact_uplift_k: np.ndarray,
    pred_inc_k: np.ndarray,
    pred_inc_k_lofo: np.ndarray,
    folds: Sequence[int],
    families: Sequence[str],
    denied: Sequence[str],
    costs: np.ndarray,
    light_total: float,
) -> dict[str, Any]:
    fold_ids = np.asarray(list(folds))
    fam = np.asarray(list(families))
    denied_set = set(denied)
    denied_mask = np.array([name in denied_set for name in fam], dtype=bool)
    k_gt_a = np.asarray(exact_uplift_k, dtype=np.float64) > 0.0
    density_oof = np.asarray(exact_uplift_k, dtype=np.float64) / np.maximum(
        np.asarray(pred_inc_k, dtype=np.float64), PRED_INC_FLOOR
    )
    density_lofo = np.asarray(exact_uplift_k, dtype=np.float64) / np.maximum(
        np.asarray(pred_inc_k_lofo, dtype=np.float64), PRED_INC_FLOOR
    )
    slacks = slacks_from_operating()
    n_train = int(actual_u_inc.shape[0])
    per_p: dict[str, Any] = {}
    for prob in BOUND_QUANTILES:
        b_raw = json_float(quantile_higher(actual_u_inc, float(prob)))
        pred_ok = np.asarray(pred_u_inc, dtype=np.float64) <= b_raw
        pred_ok_lofo = np.asarray(pred_u_inc_lofo, dtype=np.float64) <= b_raw
        eligible = pred_ok & ~denied_mask
        eligible_lofo = pred_ok_lofo & ~denied_mask
        leak_mask = eligible & (actual_u_inc > b_raw)
        n_eligible = int(eligible.sum())
        n_cond = int(eligible.sum())
        leak_rate = float(leak_mask.sum()) / float(n_cond) if n_cond else 0.0
        excess = actual_u_inc[leak_mask] - b_raw
        mean_excess = json_float(excess.mean()) if excess.size else None
        fold_f: dict[str, Optional[float]] = {}
        fold_values = []
        for fold in range(FOLDS):
            mask = (fold_ids == fold) & eligible
            value = _inflation_f(actual_u_inc, mask, b_raw)
            fold_f[str(fold)] = value
            if value is not None:
                fold_values.append(value)
        lofo_f: dict[str, Optional[float]] = {}
        for name, index in family_folds(families):
            mask = np.zeros(n_train, dtype=bool)
            mask[index] = True
            value = _inflation_f(actual_u_inc, mask & eligible_lofo, b_raw)
            lofo_f[name] = value
        worst_fold = max(fold_values) if fold_values else 1.0
        f_safe = json_float(max(1.0, float(worst_fold)))
        b_cert = json_float(b_raw * f_safe)
        n_allowed = {
            tier: n_k1_allowed(slack, b_cert, n_train) for tier, slack in slacks.items()
        }
        n_allowed_batch = {
            tier: n_k1_allowed_for_batch(slack, b_cert, light_total, light_total, n_train)
            for tier, slack in slacks.items()
        }
        checks = []
        worst_ratio = 0.0
        n_pass = 0
        n_fail = 0
        for tier, slack in slacks.items():
            if slack <= 0.0:
                continue
            for fold in range(FOLDS):
                view = fold_ids == fold
                light_batch = float(costs[view, 0].sum())
                allowed = n_k1_allowed_for_batch(
                    slack, b_cert, light_batch, light_total, n_train
                )
                chosen = _rank_eligible(eligible & view, density_oof, allowed)
                added = (
                    float(actual_inc_k[chosen].sum()) / light_batch if light_batch > 0.0 else 0.0
                )
                rel = added / slack if slack > 0.0 else 0.0
                passed = bool(added <= slack + 1e-15)
                n_pass += int(passed)
                n_fail += int(not passed)
                worst_ratio = max(worst_ratio, rel)
                checks.append(
                    {
                        "actual_added": json_float(added),
                        "actual_added_over_slack": json_float(rel),
                        "n_eligible": int((eligible & view).sum()),
                        "n_k1_allowed": int(allowed),
                        "n_selected": int(chosen.size),
                        "passed": passed,
                        "slack": json_float(slack),
                        "tier": tier,
                        "view": f"fold-{fold}",
                    }
                )
            for name, index in family_folds(families):
                view = np.zeros(n_train, dtype=bool)
                view[index] = True
                light_batch = float(costs[view, 0].sum())
                allowed = n_k1_allowed_for_batch(
                    slack, b_cert, light_batch, light_total, n_train
                )
                chosen = _rank_eligible(eligible_lofo & view, density_lofo, allowed)
                added = (
                    float(actual_inc_k[chosen].sum()) / light_batch if light_batch > 0.0 else 0.0
                )
                rel = added / slack if slack > 0.0 else 0.0
                passed = bool(added <= slack + 1e-15)
                n_pass += int(passed)
                n_fail += int(not passed)
                worst_ratio = max(worst_ratio, rel)
                checks.append(
                    {
                        "actual_added": json_float(added),
                        "actual_added_over_slack": json_float(rel),
                        "n_eligible": int((eligible_lofo & view).sum()),
                        "n_k1_allowed": int(allowed),
                        "n_selected": int(chosen.size),
                        "passed": passed,
                        "slack": json_float(slack),
                        "tier": tier,
                        "view": f"lofo-{name}",
                    }
                )
        arithmetic_ok = all(
            n_allowed[tier] * b_cert <= slacks[tier] * n_train + 1e-12
            for tier in slacks
            if slacks[tier] > 0.0
        )
        per_p[f"{prob:.2f}"] = {
            "B_cert": b_cert,
            "B_raw": b_raw,
            "F_per_fold": fold_f,
            "F_per_lofo_family": lofo_f,
            "F_safe": f_safe,
            "arithmetic_holds": arithmetic_ok,
            "certificate_holds": bool(n_fail == 0 and arithmetic_ok),
            "eligible_and_k_gt_a": int((eligible & k_gt_a).sum()),
            "eligible_pool": n_eligible,
            "eligible_pool_no_denylist": int(pred_ok.sum()),
            "k_gt_a_population": int(k_gt_a.sum()),
            "leak_conditional_mean_excess": mean_excess,
            "leak_rate": json_float(leak_rate),
            "n_fail": n_fail,
            "n_k1_allowed": n_allowed,
            "n_k1_allowed_batch_form": n_allowed_batch,
            "n_pass": n_pass,
            "overlap_with_k_gt_a": int((eligible & k_gt_a).sum()),
            "p": json_float(prob),
            "worst_actual_added_over_slack": json_float(worst_ratio),
            "worst_fold_F": json_float(worst_fold),
        }
        # checks are large; keep them only in a compact fail list plus counts
        fails = [item for item in checks if not item["passed"]]
        per_p[f"{prob:.2f}"]["failing_views"] = fails
    operating = per_p[f"{OPERATING_P:.2f}"]
    vacuous = bool(
        operating["n_k1_allowed"]["premium"] <= 0 or operating["eligible_pool"] <= 0
    )
    all_hold = all(block["certificate_holds"] for block in per_p.values())
    return {
        "all_p_hold": all_hold,
        "density_rule": DENSITY_RULE,
        "operating_p": json_float(OPERATING_P),
        "per_p": per_p,
        "slacks": {tier: json_float(value) for tier, value in slacks.items()},
        "vacuous": vacuous,
    }


def _evaluate_one_config(
    *,
    X: np.ndarray,
    costs: np.ndarray,
    scores: np.ndarray,
    families: Sequence[str],
    folds: Sequence[int],
    variant: str,
    alpha: float,
    bins: int,
    denylist: Mapping[str, Any],
    light_total: float,
) -> dict[str, Any]:
    actual_a, actual_k = actual_increments(costs)
    oof_l, oof_a_lvl, oof_k_lvl, oof_inc_a, oof_inc_k, clip_heads = oof_incremental_costs(
        X, costs, folds, variant=variant, alpha=alpha
    )
    del oof_a_lvl, oof_k_lvl
    oof_recal_a = fit_recal(oof_inc_a, actual_a)
    oof_recal_k = fit_recal(oof_inc_k, actual_k)
    oof_recal_inc_a = apply_recal(oof_recal_a, oof_inc_a)
    oof_recal_inc_k = apply_recal(oof_recal_k, oof_inc_k)
    report_a = recal_report(oof_recal_a, oof_inc_a, actual_a)
    report_k = recal_report(oof_recal_k, oof_inc_k, actual_k)

    full_heads = fit_heads(X, costs, variant=variant, alpha=alpha)
    full_l, _fa, _fk, full_inc_a, full_inc_k = predict_heads(X, full_heads)
    del _fa, _fk
    full_recal_a = fit_recal(full_inc_a, actual_a)
    full_recal_k = fit_recal(full_inc_k, actual_k)
    divergence = {
        "inc_A": transform_divergence(oof_recal_a, full_recal_a, oof_inc_a),
        "inc_K": transform_divergence(oof_recal_k, full_recal_k, oof_inc_k),
    }

    lofo_l, lofo_inc_a, lofo_inc_k = _lofo_recalibrated(
        X, costs, families, variant=variant, alpha=alpha
    )
    fam = np.asarray(list(families))
    m_a: dict[str, Optional[float]] = {}
    m_k: dict[str, Optional[float]] = {}
    m_a_lofo: dict[str, Optional[float]] = {}
    m_k_lofo: dict[str, Optional[float]] = {}
    below_one: dict[str, list[str]] = {"inc_A": [], "inc_K": [], "inc_A_lofo": [], "inc_K_lofo": []}
    for name in sorted(dict.fromkeys(fam.tolist())):
        mask = fam == name
        m_a[name] = family_ratio(actual_a, oof_recal_inc_a, mask)
        m_k[name] = family_ratio(actual_k, oof_recal_inc_k, mask)
        m_a_lofo[name] = family_ratio(actual_a, lofo_inc_a, mask)
        m_k_lofo[name] = family_ratio(actual_k, lofo_inc_k, mask)
        if m_a[name] is not None and m_a[name] < 1.0:
            below_one["inc_A"].append(name)
        if m_k[name] is not None and m_k[name] < 1.0:
            below_one["inc_K"].append(name)
        if m_a_lofo[name] is not None and m_a_lofo[name] < 1.0:
            below_one["inc_A_lofo"].append(name)
        if m_k_lofo[name] is not None and m_k_lofo[name] < 1.0:
            below_one["inc_K_lofo"].append(name)
    boost_a = _unseen_boost(m_a, m_a_lofo)
    boost_k = _unseen_boost(m_k, m_k_lofo)
    boost_candidates = [value for value in (boost_a, boost_k) if value is not None]
    unseen_boost = json_float(max(boost_candidates)) if boost_candidates else 1.0
    m_a_lofo_finite = {
        name: (1.0 if value is None else float(value)) for name, value in m_a_lofo.items()
    }
    coverage = _coverage_table(
        families,
        oof_l,
        oof_recal_inc_a,
        costs[:, 0],
        costs[:, 1],
        m_a_lofo_finite,
        unseen_boost if boost_a is None else float(boost_a),
        seed=COVERAGE_SEED,
    )

    n_train = int(costs.shape[0])
    mean_light = light_total / float(n_train)
    u = costs[:, 2] / mean_light
    u_inc = (costs[:, 2] - costs[:, 1]) / mean_light
    pred_u_inc = oof_recal_inc_k / mean_light
    pred_u_inc_lofo = lofo_inc_k / mean_light
    u_by_family = {
        name: {
            "u": describe_distribution(u[fam == name]),
            "u_inc": describe_distribution(u_inc[fam == name]),
            "n": int((fam == name).sum()),
        }
        for name in sorted(dict.fromkeys(fam.tolist()))
    }
    exact_uplift_k = scores[:, 2] - scores[:, 1]
    certificate = _certificate_block(
        actual_u_inc=u_inc,
        pred_u_inc=pred_u_inc,
        pred_u_inc_lofo=pred_u_inc_lofo,
        actual_inc_k=actual_k,
        exact_uplift_k=exact_uplift_k,
        pred_inc_k=oof_recal_inc_k,
        pred_inc_k_lofo=lofo_inc_k,
        folds=folds,
        families=families,
        denied=tuple(denylist["denied"]),
        costs=costs,
        light_total=light_total,
    )
    operating = certificate["per_p"][f"{OPERATING_P:.2f}"]
    winning_c = coverage["winning_c"]
    if winning_c is None:
        cov_block = coverage["per_c"]["2"]
        coverage_key = (0, float(cov_block["min_family_coverage"]))
        deployed_c = 2.0
    else:
        cov_block = coverage["per_c"][f"{winning_c:g}"]
        coverage_key = (1, float(cov_block["min_family_coverage"]))
        deployed_c = float(winning_c)

    layer = CostLayer(
        feature_version=FEATURE_VERSION,
        feature_signature=feature_signature(int(bins)),
        bins=int(bins),
        alpha=float(alpha),
        variant=variant,
        ridge_coefficients={
            name: tuple(json_float(value) for value in coef)
            for name, coef in full_heads.coefs.items()
        },
        smearing_factors={
            name: json_float(value) for name, value in full_heads.smears.items()
        },
        ladder_clamp_rule=LADDER_CLAMP_RULE,
        recal_a_edges=tuple(json_float(value) for value in full_recal_a.edges),
        recal_a_factors=tuple(json_float(value) for value in full_recal_a.clipped_factors),
        recal_k_edges=tuple(json_float(value) for value in full_recal_k.edges),
        recal_k_factors=tuple(json_float(value) for value in full_recal_k.clipped_factors),
        family_multipliers_lofo_a={
            name: json_float(1.0 if value is None else value)
            for name, value in m_a_lofo.items()
        },
        family_multipliers_lofo_k={
            name: json_float(1.0 if value is None else value)
            for name, value in m_k_lofo.items()
        },
        unseen_boost=unseen_boost,
        unseen_boost_a=json_float(boost_a if boost_a is not None else 1.0),
        unseen_boost_k=json_float(boost_k if boost_k is not None else 1.0),
        smallbatch_c=float(deployed_c),
        n_ref=N_REF,
        b_raw={
            key: json_float(block["B_raw"]) for key, block in certificate["per_p"].items()
        },
        b_cert={
            key: json_float(block["B_cert"]) for key, block in certificate["per_p"].items()
        },
        selected_p=float(OPERATING_P),
        k1_denylist=tuple(denylist["denied"]),
        operating_targets=dict(OPERATING_TARGETS),
        official_caps=dict(OFFICIAL_CAPS),
        stress_backstop=float(STRESS_BACKSTOP),
        near_budget=dict(NEAR_BUDGET),
        two_action_saturation_ratio=float(TWO_ACTION_SATURATION_RATIO),
    )
    metrics = ConfigMetrics(
        variant=variant,
        alpha=float(alpha),
        bins=int(bins),
        oof_ratio_a=float(report_a["aggregate_ratio"]),
        coverage_key=coverage_key,
        worst_fold_f=float(operating["worst_fold_F"]),
    )
    return {
        "certificate": certificate,
        "clip_counts": {
            "inc_A": int(clip_heads.n_clipped_inc_a),
            "inc_K": int(clip_heads.n_clipped_inc_k),
            "note": (
                "Negative raw increments clipped to 0 before log1p in "
                "direct_log1p_inc; per_model_log does not clip increments "
                "because it models log(cost_m) and reconstructs increments "
                "from the clamped ladder."
            ),
        },
        "coverage": coverage,
        "divergence_oof_vs_fullfit": divergence,
        "family_multipliers": {
            "below_one": below_one,
            "m_f_inc_A": {name: m_a[name] for name in sorted(m_a)},
            "m_f_inc_K": {name: m_k[name] for name in sorted(m_k)},
            "m_f_lofo_inc_A": {name: m_a_lofo[name] for name in sorted(m_a_lofo)},
            "m_f_lofo_inc_K": {name: m_k_lofo[name] for name in sorted(m_k_lofo)},
            "spread_inc_A": {
                "max": json_float(max(v for v in m_a.values() if v is not None)),
                "min": json_float(min(v for v in m_a.values() if v is not None)),
            },
            "spread_inc_K": {
                "max": json_float(max(v for v in m_k.values() if v is not None)),
                "min": json_float(min(v for v in m_k.values() if v is not None)),
            },
            "unseen_boost": unseen_boost,
            "unseen_boost_inc_A": boost_a,
            "unseen_boost_inc_K": boost_k,
        },
        "full_fit_pred_L_sum": json_float(full_l.sum()),
        "layer": layer,
        "lofo_pred_L_sum": json_float(lofo_l.sum()),
        "metrics": metrics,
        "n_clipped_inc_A": int(clip_heads.n_clipped_inc_a),
        "n_clipped_inc_K": int(clip_heads.n_clipped_inc_k),
        "oof_recal_inc_A": report_a,
        "oof_recal_inc_K": report_k,
        "u_distribution": {
            "global_u": describe_distribution(u),
            "global_u_inc": describe_distribution(u_inc),
            "per_family": u_by_family,
        },
        "winning_c": winning_c,
    }


def _gate_values(selected: Mapping[str, Any]) -> dict[str, Any]:
    slacks = slacks_from_operating()
    certificate = selected["certificate"]
    operating = certificate["per_p"][f"{OPERATING_P:.2f}"]
    implied = {}
    gate6 = True
    gate7 = True
    for tier, target in OPERATING_TARGETS.items():
        slack = slacks[tier]
        n_k1 = int(operating["n_k1_allowed"][tier])
        b_cert = float(operating["B_cert"])
        # the cost certificate layer certifies the K1 increment, not an AX31 allocation. Fast has
        # zero K1 slack (operating 1.15 sits below the 2-action saturation
        # 1.2169), so its implied K1-added ratio is 0. Balanced/Premium add
        # at most ``slack`` on top of the locked 2-action floor.
        k1_added_max = 0.0 if slack <= 0.0 else slack
        implied_max = (
            k1_added_max
            if slack <= 0.0
            else TWO_ACTION_SATURATION_RATIO + k1_added_max
        )
        cap = float(OFFICIAL_CAPS[tier])
        if slack <= 0.0:
            ok6 = True
            ok7 = True
        else:
            ok6 = implied_max <= float(target) + 1e-12
            ok7 = implied_max * float(STRESS_BACKSTOP) <= cap + 1e-12
        gate6 = gate6 and ok6
        gate7 = gate7 and ok7
        implied[tier] = {
            "B_cert": json_float(b_cert),
            "gate6_operating": ok6,
            "gate7_backstop": ok7,
            "implied_max_ratio": json_float(implied_max),
            "k1_added_max": json_float(k1_added_max),
            "n_k1_allowed": n_k1,
            "official_cap": json_float(cap),
            "operating_target": json_float(target),
            "slack": json_float(slack),
            "times_backstop": json_float(implied_max * float(STRESS_BACKSTOP)),
        }
    gate8 = bool(operating["arithmetic_holds"])
    gate8_empirical = bool(certificate["all_p_hold"])
    return {
        "gate6_operating_targets": gate6,
        "gate7_official_cap_backstop": gate7,
        "gate8_certificate_arithmetic": gate8,
        "gate8_certificate_empirical": gate8_empirical,
        "gate9_coverage_reported": True,
        "implied_ratio_by_tier": implied,
        "vacuous": bool(certificate["vacuous"]),
        "winning_c_found": bool(selected["coverage"]["winning_c_found"]),
    }


def decide(gates: Mapping[str, Any]) -> str:
    if not gates["gate6_operating_targets"]:
        return "record-cost_cert-close-operating-target"
    if not gates["gate7_official_cap_backstop"]:
        return "record-cost_cert-close-official-cap-backstop"
    if not gates["gate8_certificate_arithmetic"]:
        return "record-cost_cert-close-certificate-arithmetic"
    if not gates["gate8_certificate_empirical"]:
        return "record-cost_cert-close-certificate-empirical"
    if gates["vacuous"]:
        return "record-cost_cert-close-certificate-vacuous"
    return DECISION_PASS


def assemble_report(
    *,
    identity: Mapping[str, Any],
    locked: Mapping[str, Any],
    observed: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    decision: str,
) -> dict[str, Any]:
    report = {
        "decision": decision,
        "dev_opened": False,
        "diagnostic": diagnostic,
        "experiment": EXPERIMENT,
        "identity": identity,
        "locked": locked,
        "observed": observed,
        "quality_entered_cost_selection": False,
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
    }
    if report["dev_opened"] is not False:
        raise RuntimeError("the cost certificate layer report must assert dev_opened is false")
    if report["quality_entered_cost_selection"] is not False:
        raise RuntimeError("the cost certificate layer report must assert quality_entered_cost_selection is false")
    return sort_mapping(report)


def fit_and_evaluate(bundle: TrainBundle) -> dict[str, Any]:
    """Train-only the cost certificate layer fit. ``scores`` are used only for denylist + density rank."""

    folds = group_folds(bundle.episodes, folds=FOLDS, seed=FOLD_SEED_COST_CERT)
    denylist = k1_denylist(bundle.families, bundle.scores, bundle.costs)
    matrices = {int(bins): feature_matrix(bundle.texts, bins=int(bins)) for bins in BINS_GRID}
    evaluated: dict[str, Any] = {}
    metrics_rows: list[ConfigMetrics] = []
    table = []
    for variant, alpha, bins in CONFIGS:
        key = config_key(variant, alpha, bins)
        result = _evaluate_one_config(
            X=matrices[int(bins)],
            costs=bundle.costs,
            scores=bundle.scores,
            families=bundle.families,
            folds=folds,
            variant=variant,
            alpha=float(alpha),
            bins=int(bins),
            denylist=denylist,
            light_total=float(bundle.light_total),
        )
        evaluated[key] = result
        metrics_rows.append(result["metrics"])
        table.append(
            {
                "aggregate_ratio_inc_A": result["oof_recal_inc_A"]["aggregate_ratio"],
                "aggregate_ratio_inc_K": result["oof_recal_inc_K"]["aggregate_ratio"],
                "alpha": json_float(alpha),
                "bins": int(bins),
                "key": key,
                "min_family_coverage_at_deployed_c": (
                    result["coverage"]["per_c"][
                        "2" if result["winning_c"] is None else f"{result['winning_c']:g}"
                    ]["min_family_coverage"]
                ),
                "n_clipped_inc_A": result["n_clipped_inc_A"],
                "n_clipped_inc_K": result["n_clipped_inc_K"],
                "variant": variant,
                "winning_c": result["winning_c"],
                "worst_fold_F_operating_p": result["certificate"]["per_p"][
                    f"{OPERATING_P:.2f}"
                ]["worst_fold_F"],
            }
        )
    chosen = select_cost_config(metrics_rows)
    chosen_key = config_key(chosen.variant, chosen.alpha, chosen.bins)
    selected = evaluated[chosen_key]
    gates = _gate_values(selected)
    decision = decide(gates)
    # Drop the live CostLayer object from the per-config dump (it is serialized
    # separately) and strip non-JSON metrics objects.
    diagnostic_configs = {}
    for key, result in evaluated.items():
        payload = {
            name: value
            for name, value in result.items()
            if name not in {"layer", "metrics"}
        }
        diagnostic_configs[key] = payload
    observed = {
        "certificate": selected["certificate"],
        "coverage": selected["coverage"],
        "denylist": denylist,
        "dev_opened": False,
        "divergence_oof_vs_fullfit": selected["divergence_oof_vs_fullfit"],
        "family_multipliers": selected["family_multipliers"],
        "gates": gates,
        "n_clipped_inc_A": selected["n_clipped_inc_A"],
        "n_clipped_inc_K": selected["n_clipped_inc_K"],
        "oof_recal_inc_A": selected["oof_recal_inc_A"],
        "oof_recal_inc_K": selected["oof_recal_inc_K"],
        "quality_entered_cost_selection": False,
        "selected_config": {
            "alpha": json_float(chosen.alpha),
            "bins": int(chosen.bins),
            "key": chosen_key,
            "oof_ratio_inc_A": json_float(chosen.oof_ratio_a),
            "variant": chosen.variant,
            "winning_c": selected["winning_c"],
            "worst_fold_F": json_float(chosen.worst_fold_f),
        },
        "u_distribution": selected["u_distribution"],
    }
    diagnostic = {
        "config_table": table,
        "configs": diagnostic_configs,
        "imported_modeling_symbols": [
            "family_folds",
            "feature_matrix",
            "group_folds",
            "load_train",
            "official_score",
            "oof_predict",
            "pav_nonincreasing",
            "quantile_higher",
            "rank_recalibration",
            "ridge_fit",
            "ridge_predict",
        ],
        "oof_predict_imported": oof_predict is not None,
        "pav_nonincreasing_imported": pav_nonincreasing is not None,
        "official_score_imported": official_score is not None,
        "tier_weights": dict(TIER_WEIGHTS),
    }
    return {
        "decision": decision,
        "diagnostic": diagnostic,
        "layer": selected["layer"],
        "observed": observed,
    }


# Re-export for tests and the runner.
__all__ = (
    "ALPHAS",
    "BINS_GRID",
    "CONFIGS",
    "COVERAGE_SEED",
    "CostLayer",
    "ConfigMetrics",
    "DENYLIST_CRITERION",
    "EXPERIMENT",
    "FOLD_SEED_COST_CERT",
    "LADDER_CLAMP_RULE",
    "N_REF",
    "OPERATING_P",
    "SELECTION_CRITERION",
    "STRESS_BACKSTOP",
    "VARIANTS",
    "assemble_report",
    "clamp_price_ladder",
    "composition_safety",
    "config_key",
    "duan_smearing_factor",
    "fit_and_evaluate",
    "fit_heads",
    "fit_recal",
    "k1_denylist",
    "load_train",
    "locked_record",
    "n_k1_allowed",
    "n_k1_allowed_for_batch",
    "oof_incremental_costs",
    "predict_heads",
    "reject_dev_reference",
    "select_cost_config",
    "smallbatch_factor",
)
