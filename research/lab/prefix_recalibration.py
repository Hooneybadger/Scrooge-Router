# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the prefix recalibration layer H4 selected candidate: prefix-curve recalibration + dual certificate.

Reuses the prefix certificate layer view construction, prefix sweeps, kappa estimation, K1
guards, official-scorer path, gate evaluation, and policy serialization.
Changes only what charter §12.3 specifies. Q_A is never loaded. Dev is
never opened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.lab.prompt_features import FEATURE_VERSION, feature_signature
from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.modeling import (
    FACTOR_CLIP,
    FOLD_SEED,
    FOLDS,
    HASH_BINS,
    INTERCEPT_POLICY,
    N_BINS,
    OFFICIAL_CAPS,
    OPERATING_TARGETS,
    TIER_WEIGHTS,
    RankRecal,
    TrainBundle,
    family_folds,
    feature_matrix,
    group_folds,
    load_train,
    official_score,
    oof_predict,
    paired_group_bootstrap,
    pav_nonincreasing,
    quantile_higher,
    rank_recalibration,
    reject_dev_reference,
    ridge_predict,
    sort_mapping,
    weighted_final,
)
from research.lab.cost_certificates import (
    FittedHeads,
    actual_increments,
    fit_heads,
    floor_inc,
    oof_incremental_costs,
    predict_heads,
)
from research.lab.prefix_certificates import (
    F_GRID,
    F_GRID_ARRAY,
    FLOAT64_TABLE_NOTE,
    K1_DENYLIST,
    K1_DENSITY_EPS,
    PARENT_F_EXACT,
    PARENT_F_PINS,
    QUALITY_HEADS_PATH,
    View,
    _count_models,
    _episode_scores,
    _gather,
    _lofo_qk,
    _parent_assignments,
    _realized_ratio,
    _score_mean,
    apply_recal_baked,
    build_views,
    content_digests,
    design_matrix_g_features,
    family_of_text,
    json_float,
    json_floats,
    load_q_k_head,
    models_from_masks,
    order_k1,
    phi_view,
    prefix_k,
    prefix_mask,
    sort_pred_inc,
)

# Local shim over research.lab.prefix_certificates._lofo_cost: the prefix certificate layer/the cost certificate layer bake n_bins=10 and
# clip=(0.5, 6.0). H4 re-selects those. Required change (NOT applied):
# parameterize research.lab.prefix_certificates._lofo_cost / research.lab.cost_certificates.fit_recal with
# n_bins, clip, and an optional prefix-curve fitter. Workaround: cache
# raw LOFO head predictions and apply the selected recal locally.


EXPERIMENT = "the prefix recalibration layer"
REPORT_TYPE = "scrooge-prefix_recal-h4-selected-candidate-v1"
SCHEMA_VERSION = 1
DECISION_TWO = "record-prefix_recal-two-candidates"
FOLD_SEED_PREFIX_RECAL = FOLD_SEED  # 2026082202
BOOTSTRAP_SEED = 2026082203
BOOTSTRAP_DRAWS = 1000
LOCKED_VARIANT = "direct_log1p_inc"
LOCKED_ALPHA = 300.0
LOCKED_BINS = 512
RECAL_BINS_LEGACY = N_BINS
RECAL_CLIP_LEGACY = FACTOR_CLIP
RECAL_GRID_N_BINS: Tuple[int, ...] = (10, 20)
RECAL_GRID_CLIPS: Tuple[Tuple[float, float], ...] = ((0.5, 6.0), (0.25, 200.0))
RECAL_FIT_KINDS: Tuple[str, ...] = ("aggregate", "prefix")
LOCKED_INC_A = 1.029434
LOCKED_INC_K = 1.002749
LOCKED_RATIO_ATOL = 5e-7
SORT_RULE = "sortA_pred_inc"
K_ORDER = "orderK_qk"
Q_ELIG = 0.0
KAPPA_MIN_INCREMENT = 0.01
KAPPA_Q = 0.9975
K1_QUALITY_VACUOUS_MAX = 0.001
K1_ITEM_CAP_FRAC = 0.005
K1_COUNT_CAP_FRAC = 0.10
K1_MIN_N = 300
K1_FAMILY_MULT_CLIP = (1.0, 3.0)
K1_M_UNSEEN = 3.0
H2_6_EXCEEDANCE_MAX = 0.01
H2_5_THRESHOLD = -0.010
# the cost certificate layer locked clipped family multipliers for inc_K. Inherited, not recomputed.
LOCKED_FAMILY_MULT_INC_K: Mapping[str, float] = {
    "english_multiple_choice": 1.0,
    "korean_multiple_choice": 1.0,
    "korean_reasoning": 2.924070990122561,
    "latex_math": 3.0,
    "long_context": 1.0,
    "other": 1.144443861318176,
    "python_program": 1.16905263768859,
    "rule_reasoning": 1.0,
    "symbolic_math": 1.1969945163656923,
    "word_problem": 1.0,
}
TIER_PRED_BANDS: Mapping[str, Tuple[float, float]] = {
    "fast": (1.05, 1.25),
    "balanced": (1.15, 2.0),
    "premium": (1.80, float("inf")),
}
CANDIDATE_NAMES: Tuple[str, ...] = ("candidate_kappa_max", "candidate_kappa_q9975")
_LIGHT = "ax31-light"
_AX31 = "ax31"
_K1 = "axk1-think"
LEGAL_MODEL_IDS = MODEL_IDS

COST_HEAD_LOCK = (
    "Reuse the cost certificate layer's mechanically selected cost head: variant "
    "direct_log1p_inc, alpha=300.0, bins=512. Folds seed 2026082202. "
    "Sort rule sortA_pred_inc with content-digest final tiebreak. "
    "Assert the head reproduces OOF post-recalibration aggregate ratios "
    "inc_A = 1.029434 / inc_K = 1.002749 under the legacy "
    "(n_bins=10, clip=(0.5, 6.0)) aggregate-fit so the pipeline is "
    "unchanged. Recalibration is then re-selected on the H4 grid. "
    "Charter §10.4 / §12.3."
)
RECAL_SELECTION_CRITERION = (
    "Grid, pre-registered, exactly four combinations: n_bins in (10, 20) "
    "x clip in ((0.5, 6.0), (0.25, 200.0)). For each cell report both "
    "the standard aggregate-ratio fit and the prefix-curve fit. "
    "Selection criterion, pre-registered and mechanical: minimize "
    "max over binding views and over f of "
    "|predicted_ratio(f) - actual_ratio(f)|. Tie-break: smaller n_bins, "
    "then the tighter clip. quality_entered_calibration_selection: false. "
    "Charter §12.3 items 1 and 2."
)
PREFIX_FIT_SPEC = (
    "Choose the monotone non-increasing (PAV) per-bin factors that "
    "minimize the squared error of the CUMULATIVE prefix ratio curve "
    "predicted_ratio(f) against the actual actual_ratio(f) over the "
    "pinned 101-point f grid, aggregated over the OOF folds. Keep PAV "
    "non-increasing and keep the clip."
)
H4_HYPOTHESIS = (
    "Recalibrate to the prefix curve so the certificate is tight, and "
    "use a dual certificate so the vacuous interval cannot return. "
    "certified_ratio(f) = max( 1 + kappa_tier * (predicted_ratio(f) - 1), "
    "Phi_binding_q9975(f) ). "
    "f* = max { f : certified_ratio(f) <= limit_tier }. "
    "Phi_binding_q9975(f) is the q99.75 of the ACTUAL realized ratio at "
    "prefix fraction f across the binding view layer "
    "(research.lab.modeling.quantile_higher); enforce monotonicity by running "
    "maximum. Measure kappa only where predicted_ratio(f) - 1 >= 0.01; "
    "at runtime, where predicted_ratio(f) - 1 < 0.01, drop the kappa "
    "term and use the empirical term alone. Charter §12.3."
)
BINDING_LAYER_REASON = (
    "lofo-combined and n=100 are red-team because the official evaluation "
    "builds batches with the same harness as Train/Dev; a batch that is "
    "100% one family after that family was left out of training is a "
    "composition the harness cannot produce. This split is fixed by "
    "design, not by result. Charter §11.4."
)
BINDING_LAYER_SPEC = (
    "binding: oof-fold 5, lofo-{family} 10, famdom-{family}-{d} 2000 "
    "(seed 2026082204), dirichlet-{d} 1000 (seed 2026082205), half-{d} 20 "
    "(seed 2026082206), small-{n}-{d} for n in (300, 880) 200 each "
    "(seed 2026082207)."
)
REDTEAM_LAYER_SPEC = (
    "red-team: lofo-combined-{family} 10, small-100-{d} 200. "
    "Measured and reported, never used to set kappa."
)
K1_GUARD_SPEC = (
    "1. K1 only when the tier batch has n >= 300. "
    "2. Item cap: skip any episode whose conservative predicted inc_K "
    "exceeds 0.005 * predicted_light_total. Conservative means the "
    "recalibrated prediction multiplied by the the cost certificate layer clipped family "
    "multiplier clipped to [1.0, 3.0]. "
    "3. Certify the K1 increment with its own kappa_K, estimated like "
    "AX31 kappa on the K1 increment and the binding layer only. "
    "4. Denylist, fixed and inherited: "
    "('rule_reasoning', 'python_program', 'korean_reasoning'). "
    "5. Count cap m <= floor(0.10 * n). "
    "6. Eligibility also requires predicted Q_K > 0. Ordering: "
    "descending predicted Q_K (orderK_qk). "
    "New: Fast is hard-coded k1_enabled=false per charter §1 / §12.3 "
    "item 5. Apply the dual certificate to the K1 increment too "
    "(empirical Psi_binding_q9975(m) OR-ed with the kappa_K term, "
    "taking the max). If K1 remains quality-vacuous "
    "(quality_from_k1 <= 0.001), record k1_enabled=false for Balanced "
    "and Premium too and proceed — that is an acceptable outcome, not "
    "a failure, and a guard must not be weakened to force K1 on. "
    "If m* is 0 everywhere, record k1_enabled=false and proceed."
)
ADOPTION_RULE = (
    "Both candidate_kappa_max and candidate_kappa_q9975 are built and "
    "reported. The operator chooses. Default conservative is kappa_max; "
    "relaxation requires explicit approval. "
    "adoption_deferred_to_operator: true. Charter §11.5."
)
H4_GATE_SPEC = (
    "H2-1 weighted Train OOF gain > 0; H2-2 fold wins >= 4/5; "
    "H2-3 prompt-group paired bootstrap q2.5 > 0 (draws 1000, seed "
    "2026082203); H2-4 split reporting: report the LOFO quality "
    "component (per-tier realized quality with no budget zeroing "
    "applied) and the LOFO budget component (which views/tiers "
    "exceeded the official limit) as two separate numbers. Gate the "
    "quality component on no-regression. Count zeroing under the cost "
    "gates, never as quality regression. H2-5 LOFO worst-family quality "
    "gain >= -0.010 (n<50 families reported separately). "
    "H2-6 held-out exceedance <= 0.01: estimate the certificate (both "
    "terms) from 4 of 5 folds and verify on the 5th, and estimate "
    "excluding each family and verify on that family. Report the "
    "exceedance rate and the worst held-out realized ratio at f*. "
    "H3-cost: certified_ratio(f*) <= official limit, and every binding "
    "view's realized ratio <= official limit. Report the per-tier ruin "
    "count and frequency on binding and on red-team. "
    "H2-9 K1 certificate on every binding view; H2-10 Premium K1 "
    "count > 0 (failure => k1_enabled=false adoption, not rejection)."
)
FLOAT64_NOTE = (
    FLOAT64_TABLE_NOTE
    + " the prefix recalibration layer predicted_ratio(f) is the same cumsum construction on "
    "recalibrated predicted increments. Official Decimal is used only "
    "at selected OOF/LOFO operating points and H2-1..H2-5 gate numbers."
)

# Local shim: research.lab.prompt_features has no prompt_family. Required change
# (NOT applied): add family_of_text(text) to g_features mirroring
# research.lab.validation.prompt_family. Workaround: reuse the prefix certificate layer's text-only
# stand-in so Train and route() share one rule without forking g_features.


def view_layer(view: View) -> str:
    """Pre-registered binding / red-team split. Design property, not a result."""

    if view.kind == "lofo-combined":
        return "red-team"
    if view.kind == "small" and str(view.name).startswith("small-100-"):
        return "red-team"
    return "binding"


def family_multiplier_inc_k(family: str) -> float:
    raw = float(LOCKED_FAMILY_MULT_INC_K.get(str(family), K1_M_UNSEEN))
    lo, hi = K1_FAMILY_MULT_CLIP
    return float(min(float(hi), max(float(lo), raw)))


def conservative_inc_k(inc_k: np.ndarray, families: Sequence[str]) -> np.ndarray:
    predicted = np.asarray(inc_k, dtype=np.float64).reshape(-1)
    if predicted.size != len(families):
        raise ValueError("conservative_inc_k requires aligned families")
    scale = np.asarray(
        [family_multiplier_inc_k(name) for name in families], dtype=np.float64
    )
    return predicted * scale


def predicted_ratio_curve(
    pred_light: np.ndarray,
    pred_inc_a: np.ndarray,
    order: np.ndarray,
    grid: np.ndarray = F_GRID_ARRAY,
) -> np.ndarray:
    """predicted_ratio(f) = 1 + prefix-sum(recal inc_A) / predicted light.

    Equals 1.0 at f=0. Monotone non-decreasing when every predicted
    increment is non-negative, which the locked head enforces.
    """

    light = np.asarray(pred_light, dtype=np.float64).reshape(-1)
    increment = np.asarray(pred_inc_a, dtype=np.float64).reshape(-1)
    return phi_view(light, light + increment, order, grid)


def kappa_term_curve(pred_ratio: np.ndarray, kappa: float) -> np.ndarray:
    predicted = np.asarray(pred_ratio, dtype=np.float64).reshape(-1)
    return 1.0 + float(kappa) * (predicted - 1.0)


def certified_curve(pred_ratio: np.ndarray, kappa: float) -> np.ndarray:
    """Backward-compatible kappa-only curve. Dual certificate is preferred."""

    return kappa_term_curve(pred_ratio, kappa)


def certified_ratio_curve(
    pred_ratio: np.ndarray,
    kappa: float,
    phi: np.ndarray,
    *,
    min_increment: float = KAPPA_MIN_INCREMENT,
) -> np.ndarray:
    """Dual certificate. Charter §12.3 item 3 plus item 4 alignment.

    certified_ratio(f) = max(1 + kappa*(predicted_ratio(f)-1), Phi(f))
    where predicted_ratio(f)-1 >= min_increment; otherwise the kappa
    term is dropped and the empirical term is used alone.
    """

    predicted = np.asarray(pred_ratio, dtype=np.float64).reshape(-1)
    empirical = np.asarray(phi, dtype=np.float64).reshape(-1)
    if predicted.size != empirical.size:
        raise ValueError("certified_ratio_curve requires aligned curves")
    kappa_term = kappa_term_curve(predicted, kappa)
    use_kappa = (predicted - 1.0) >= float(min_increment)
    return np.where(use_kappa, np.maximum(kappa_term, empirical), empirical)


def select_f_star_kappa(
    pred_ratio: np.ndarray,
    kappa: float,
    limit: float,
    grid: np.ndarray = F_GRID_ARRAY,
) -> float:
    """Kappa-only f* (tests / diagnostics). Runtime uses select_f_star_certified."""

    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    certified = kappa_term_curve(pred_ratio, kappa)
    if certified.size != knots.size:
        raise ValueError("select_f_star_kappa requires an aligned grid")
    ok = (certified <= float(limit) + 1e-15) & (knots <= 1.0 + 1e-15)
    if not np.any(ok):
        return 0.0
    return json_float(knots[int(np.flatnonzero(ok)[-1])])


def select_f_star_certified(
    pred_ratio: np.ndarray,
    kappa: float,
    phi: np.ndarray,
    limit: float,
    grid: np.ndarray = F_GRID_ARRAY,
    *,
    min_increment: float = KAPPA_MIN_INCREMENT,
) -> float:
    """f* = max { f : certified_ratio(f) <= limit }. Charter §12.3 item 3."""

    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    certified = certified_ratio_curve(
        pred_ratio, kappa, phi, min_increment=min_increment
    )
    if certified.size != knots.size:
        raise ValueError("select_f_star_certified requires an aligned grid")
    ok = (certified <= float(limit) + 1e-15) & (knots <= 1.0 + 1e-15)
    if not np.any(ok):
        return 0.0
    return json_float(knots[int(np.flatnonzero(ok)[-1])])


def binding_term_at(
    pred_ratio: float,
    kappa: float,
    phi_value: float,
    *,
    min_increment: float = KAPPA_MIN_INCREMENT,
) -> dict[str, Any]:
    increment = float(pred_ratio) - 1.0
    kappa_used = bool(increment >= float(min_increment))
    kappa_term = 1.0 + float(kappa) * increment if kappa_used else None
    empirical = float(phi_value)
    if kappa_used and kappa_term is not None:
        certified = max(float(kappa_term), empirical)
        binds = "empirical" if empirical + 1e-15 >= float(kappa_term) else "kappa"
    else:
        certified = empirical
        binds = "empirical"
    return {
        "binds": binds,
        "certified_ratio": json_float(certified),
        "empirical_term": json_float(empirical),
        "kappa_term": None if kappa_term is None else json_float(kappa_term),
        "kappa_term_applied": bool(kappa_used),
    }


def running_maximum(values: np.ndarray) -> Tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    monotone = np.maximum.accumulate(raw)
    changed = int(np.sum(np.abs(monotone - raw) > 1e-15))
    max_lift = json_float(float(np.max(monotone - raw))) if raw.size else 0.0
    return monotone, {"max_lift": max_lift, "n_points_changed": changed}


def phi_from_actual_curves(curves: Sequence[np.ndarray]) -> Tuple[np.ndarray, dict[str, Any]]:
    if not curves:
        ones = np.ones(F_GRID_ARRAY.size, dtype=np.float64)
        return ones, {"max_lift": 0.0, "n_points_changed": 0, "n_views": 0}
    matrix = np.stack([np.asarray(curve, dtype=np.float64).reshape(-1) for curve in curves], axis=0)
    raw = np.empty(matrix.shape[1], dtype=np.float64)
    for col in range(int(matrix.shape[1])):
        raw[col] = float(quantile_higher(matrix[:, col], KAPPA_Q))
    monotone, change = running_maximum(raw)
    change["n_views"] = int(matrix.shape[0])
    change["raw"] = json_floats(raw)
    return monotone, change


def in_tier_band(pred_ratio: float, tier: str) -> bool:
    lo, hi = TIER_PRED_BANDS[tier]
    value = float(pred_ratio)
    return bool(lo - 1e-15 <= value <= hi + 1e-15)


def kappa_cells_from_curves(
    pred_ratio: np.ndarray,
    actual_ratio: np.ndarray,
    *,
    min_increment: float = KAPPA_MIN_INCREMENT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (kappa, rho, included) aligned to the grid.

    kappa = (actual-1)/(predicted-1). rho = actual/predicted.
    Cells with predicted-1 < min_increment are excluded (NaN).
    """

    predicted = np.asarray(pred_ratio, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_ratio, dtype=np.float64).reshape(-1)
    if predicted.size != actual.size:
        raise ValueError("kappa_cells_from_curves requires aligned curves")
    denom = predicted - 1.0
    included = denom >= float(min_increment)
    kappa = np.full(predicted.shape, np.nan, dtype=np.float64)
    rho = np.full(predicted.shape, np.nan, dtype=np.float64)
    if np.any(included):
        kappa[included] = (actual[included] - 1.0) / denom[included]
        safe = predicted[included]
        rho[included] = np.divide(
            actual[included], safe, out=np.full(safe.shape, np.nan), where=safe > 0.0
        )
    return kappa, rho, included


def summarize_values(values: np.ndarray) -> Optional[dict[str, float]]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "max": json_float(float(array.max())),
        "n": int(array.size),
        "q50": json_float(quantile_higher(array, 0.50)),
        "q90": json_float(quantile_higher(array, 0.90)),
        "q99": json_float(quantile_higher(array, 0.99)),
        "q9975": json_float(quantile_higher(array, KAPPA_Q)),
        "q999": json_float(quantile_higher(array, 0.999)),
    }


def k1_mask(
    families: Sequence[str],
    q_k: np.ndarray,
    conservative_k: np.ndarray,
    pred_l: np.ndarray,
    *,
    denylist: Sequence[str] = K1_DENYLIST,
    item_cap_frac: float = K1_ITEM_CAP_FRAC,
    min_n: int = K1_MIN_N,
) -> Tuple[np.ndarray, dict[str, int]]:
    """Six-guard eligibility. Count cap and kappa_K bind at selection time."""

    labels = tuple(families)
    pred_q = np.asarray(q_k, dtype=np.float64).reshape(-1)
    pred_inc = np.asarray(conservative_k, dtype=np.float64).reshape(-1)
    light = np.asarray(pred_l, dtype=np.float64).reshape(-1)
    n_rows = int(pred_q.size)
    binds = {
        "count_cap_bind": 0,
        "denylist_bind": 0,
        "item_cap_bind": 0,
        "kappa_k_cert_bind": 0,
        "n_lt_300_bind": 0,
        "qk_nonpositive_bind": 0,
    }
    if n_rows == 0:
        return np.zeros(0, dtype=bool), binds
    if n_rows < int(min_n):
        binds["n_lt_300_bind"] = int(n_rows)
        return np.zeros(n_rows, dtype=bool), binds
    denied = set(denylist)
    fam_ok = np.array([label not in denied for label in labels], dtype=bool)
    q_ok = pred_q > 0.0
    item_cap = float(item_cap_frac) * float(light.sum())
    cap_ok = pred_inc <= item_cap + 1e-15
    chosen = fam_ok & q_ok & cap_ok
    binds["denylist_bind"] = int(np.sum((~fam_ok) & q_ok))
    binds["item_cap_bind"] = int(np.sum(fam_ok & q_ok & ~cap_ok))
    binds["qk_nonpositive_bind"] = int(np.sum(fam_ok & ~q_ok))
    return chosen, binds


def predicted_k_increment(
    conservative_k: np.ndarray,
    pred_inc_a: np.ndarray,
    in_prefix: np.ndarray,
    ranked: np.ndarray,
    pred_light_total: float,
) -> np.ndarray:
    """Cumulative predicted K1 extra / predicted light, length = |ranked|."""

    order = np.asarray(ranked, dtype=np.int64).reshape(-1)
    if order.size == 0 or float(pred_light_total) <= 0.0:
        return np.zeros(int(order.size), dtype=np.float64)
    extra_k = np.asarray(conservative_k, dtype=np.float64).reshape(-1)[order]
    extra_a = np.asarray(pred_inc_a, dtype=np.float64).reshape(-1)[order]
    prefix = np.asarray(in_prefix, dtype=bool).reshape(-1)[order]
    extra = extra_k + np.where(prefix, 0.0, extra_a)
    return np.cumsum(extra) / float(pred_light_total)


def actual_k_increment(
    actual_light: np.ndarray,
    actual_ax31: np.ndarray,
    actual_k1: np.ndarray,
    in_prefix: np.ndarray,
    ranked: np.ndarray,
) -> np.ndarray:
    order = np.asarray(ranked, dtype=np.int64).reshape(-1)
    light_sum = float(np.asarray(actual_light, dtype=np.float64).sum())
    if order.size == 0 or light_sum <= 0.0:
        return np.zeros(int(order.size), dtype=np.float64)
    light = np.asarray(actual_light, dtype=np.float64).reshape(-1)
    ax31 = np.asarray(actual_ax31, dtype=np.float64).reshape(-1)
    k1 = np.asarray(actual_k1, dtype=np.float64).reshape(-1)
    prefix = np.asarray(in_prefix, dtype=bool).reshape(-1)
    extra = np.where(prefix[order], k1[order] - ax31[order], k1[order] - light[order])
    return np.cumsum(extra) / light_sum


def select_m_star(
    pred_k_inc: np.ndarray,
    *,
    certified_ax31: float,
    kappa_k: float,
    limit: float,
    count_cap: int,
    psi: Optional[np.ndarray] = None,
    min_increment: float = KAPPA_MIN_INCREMENT,
) -> int:
    """m* under the dual K1 certificate. Charter §12.3 item 5.

    certified_k_extra(m) = max(kappa_K * predicted_K_increment(m), Psi(m))
    when predicted_K_increment(m) >= min_increment; otherwise Psi(m) alone.
    m* = max { m : certified_ax31 + certified_k_extra(m) <= limit }.
    """

    extras = np.asarray(pred_k_inc, dtype=np.float64).reshape(-1)
    allowed = min(int(extras.size), int(count_cap))
    if allowed <= 0:
        return 0
    psi_arr = None if psi is None else np.asarray(psi, dtype=np.float64).reshape(-1)
    chosen = 0
    for count in range(1, allowed + 1):
        pred_extra = float(extras[count - 1])
        if psi_arr is not None and psi_arr.size:
            psi_val = float(psi_arr[min(count - 1, int(psi_arr.size) - 1)])
        else:
            psi_val = 0.0
        if pred_extra >= float(min_increment) and math.isfinite(float(kappa_k)):
            cert_extra = max(float(kappa_k) * pred_extra, psi_val)
        else:
            cert_extra = psi_val
        accounted = float(certified_ax31) + cert_extra
        if accounted <= float(limit) + 1e-15:
            chosen = count
        else:
            break
    return int(chosen)


def _recal_edges_and_bins(
    predicted: np.ndarray, *, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, list[float], list[float], list[float], list[float]]:
    """Equal-count bins and the rank recalibration study midpoint edges. Shared by both fit kinds."""

    bins = int(n_bins)
    order = np.argsort(predicted, kind="stable")
    groups = np.array_split(order, bins)
    if any(group.size == 0 for group in groups):
        raise ValueError("recal-bins-insufficient")
    raw_factors: list[float] = []
    weights: list[float] = []
    bin_pred_max: list[float] = []
    bin_pred_min: list[float] = []
    for group in groups:
        pred_sum = float(predicted[group].sum())
        if (not math.isfinite(pred_sum)) or pred_sum <= 0.0:
            raise ValueError("recal-bin-undefined")
        raw_factors.append(float("nan"))
        weights.append(pred_sum)
        bin_pred_max.append(float(predicted[group].max()))
        bin_pred_min.append(float(predicted[group].min()))
    edges: list[float] = []
    for index in range(bins - 1):
        left = bin_pred_max[index]
        right = bin_pred_min[index + 1]
        if left < right:
            edges.append(0.5 * (left + right))
        else:
            edges.append(left)
    edges_arr = np.asarray(edges, dtype=np.float64)
    bin_index = np.clip(np.digitize(predicted, edges_arr, right=True), 0, bins - 1)
    return (
        edges_arr,
        bin_index,
        raw_factors,
        weights,
        bin_pred_max,
        bin_pred_min,
    )


def fit_recal_params(
    pred_inc: np.ndarray,
    actual_inc: np.ndarray,
    *,
    n_bins: int,
    clip: Tuple[float, float],
) -> RankRecal:
    """Local shim over research.lab.cost_certificates.fit_recal (hardcoded n_bins/clip)."""

    return rank_recalibration(
        floor_inc(pred_inc),
        np.asarray(actual_inc, dtype=np.float64).reshape(-1),
        n_bins=int(n_bins),
        clip=(float(clip[0]), float(clip[1])),
    )


def fit_prefix_recal(
    pred_inc: np.ndarray,
    actual_inc: np.ndarray,
    pred_light: np.ndarray,
    actual_light: np.ndarray,
    actual_ax31: np.ndarray,
    fold_ids: np.ndarray,
    digests: Sequence[str],
    *,
    n_bins: int,
    clip: Tuple[float, float],
) -> RankRecal:
    """Prefix-curve PAV recalibration. Charter §12.3 item 2.

    Per-bin factors minimize squared error of the cumulative prefix
    ratio curve against actual_ratio(f) over the pinned 101-point grid,
    aggregated over the OOF folds. PAV stays non-increasing; clip stays.
    """

    predicted = floor_inc(pred_inc)
    actual = np.asarray(actual_inc, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape:
        raise ValueError("fit_prefix_recal requires aligned increments")
    bins = int(n_bins)
    order_all = np.argsort(predicted, kind="stable")
    groups = np.array_split(order_all, bins)
    if any(group.size == 0 for group in groups):
        raise ValueError("recal-bins-insufficient")
    agg_raw: list[float] = []
    weights: list[float] = []
    bin_pred_max: list[float] = []
    bin_pred_min: list[float] = []
    for group in groups:
        pred_sum = float(predicted[group].sum())
        act_sum = float(actual[group].sum())
        if (not math.isfinite(pred_sum)) or pred_sum <= 0.0 or not math.isfinite(act_sum):
            raise ValueError("recal-bin-undefined")
        agg_raw.append(act_sum / pred_sum)
        weights.append(pred_sum)
        bin_pred_max.append(float(predicted[group].max()))
        bin_pred_min.append(float(predicted[group].min()))
    edges: list[float] = []
    for index in range(bins - 1):
        left = bin_pred_max[index]
        right = bin_pred_min[index + 1]
        edges.append(0.5 * (left + right) if left < right else left)
    edges_arr = np.asarray(edges, dtype=np.float64)
    bin_index = np.clip(np.digitize(predicted, edges_arr, right=True), 0, bins - 1)

    p_light = np.asarray(pred_light, dtype=np.float64).reshape(-1)
    a_light = np.asarray(actual_light, dtype=np.float64).reshape(-1)
    a_ax31 = np.asarray(actual_ax31, dtype=np.float64).reshape(-1)
    folds = np.asarray(fold_ids, dtype=np.int64).reshape(-1)
    design_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for fold in range(int(FOLDS)):
        idx = np.flatnonzero(folds == fold)
        if idx.size == 0:
            continue
        local_pred = predicted[idx]
        local_light_pred = p_light[idx]
        pred_den = float(local_light_pred.sum())
        if pred_den <= 0.0:
            continue
        local_digests = tuple(digests[int(item)] for item in idx)
        local_order = sort_pred_inc(local_pred, local_digests)
        n_local = int(idx.size)
        ordered_bins = bin_index[idx][local_order]
        ordered_pred = local_pred[local_order]
        onehot = np.zeros((n_local, bins), dtype=np.float64)
        onehot[np.arange(n_local), ordered_bins] = ordered_pred
        cum_mass = np.cumsum(onehot, axis=0)
        actual_curve = phi_view(a_light[idx], a_ax31[idx], local_order)
        ks = np.clip(
            np.floor(F_GRID_ARRAY * float(n_local) + 1e-15).astype(np.int64), 0, n_local
        )
        design = np.zeros((F_GRID_ARRAY.size, bins), dtype=np.float64)
        nonempty = ks > 0
        design[nonempty] = cum_mass[ks[nonempty] - 1] / pred_den
        design_rows.append(design)
        target_rows.append(actual_curve - 1.0)
    if not design_rows:
        ls_factors = np.asarray(agg_raw, dtype=np.float64)
    else:
        design = np.vstack(design_rows)
        target = np.concatenate(target_rows)
        ls_factors, *_ = np.linalg.lstsq(design, target, rcond=None)
        if (not np.all(np.isfinite(ls_factors))) or ls_factors.size != bins:
            ls_factors = np.asarray(agg_raw, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    pav = pav_nonincreasing(np.asarray(ls_factors, dtype=np.float64), weight_arr)
    clipped = np.clip(pav, float(clip[0]), float(clip[1]))
    return RankRecal(
        edges=edges_arr,
        raw_factors=np.asarray(ls_factors, dtype=np.float64),
        pav_factors=np.asarray(pav, dtype=np.float64),
        clipped_factors=np.asarray(clipped, dtype=np.float64),
    )


def apply_recal_params(recal: RankRecal, pred_inc: np.ndarray) -> np.ndarray:
    return recal.apply(floor_inc(pred_inc))


def clip_binds(recal: RankRecal, clip: Tuple[float, float]) -> dict[str, Any]:
    lo = float(clip[0])
    hi = float(clip[1])
    factors = np.asarray(recal.clipped_factors, dtype=np.float64)
    raw = np.asarray(recal.raw_factors, dtype=np.float64)
    n_lo = int(np.sum(factors <= lo + 1e-15))
    n_hi = int(np.sum(factors >= hi - 1e-15))
    raw_hi = int(np.sum(np.isfinite(raw) & (raw > hi + 1e-15)))
    return {
        "clip_high_binds": bool(n_hi > 0),
        "clip_low_binds": bool(n_lo > 0),
        "n_bins_hit_clip_high": n_hi,
        "n_bins_hit_clip_low": n_lo,
        "n_raw_above_clip": raw_hi,
        "raw_max": json_float(float(np.nanmax(raw))) if raw.size else None,
    }


def tighter_clip_key(clip: Tuple[float, float]) -> float:
    """Smaller span is tighter. Used only as a pre-registered tie-break."""

    return float(clip[1]) - float(clip[0])


@dataclass(frozen=True)
class LofoRawPack:
    """Cached LOFO head predictions so recal re-selection does not refit."""

    held_index: np.ndarray
    train_index: np.ndarray
    pred_l_held: np.ndarray
    inc_a_held: np.ndarray
    inc_k_held: np.ndarray
    pred_l_train: np.ndarray
    inc_a_train: np.ndarray
    inc_k_train: np.ndarray


def cache_lofo_raw(
    features: np.ndarray, costs: np.ndarray, families: Sequence[str]
) -> dict[str, LofoRawPack]:
    """Local shim over research.lab.prefix_certificates._lofo_cost. Fits heads once per family."""

    n_rows = int(features.shape[0])
    cached: dict[str, LofoRawPack] = {}
    for name, held_index in family_folds(families):
        held = np.zeros(n_rows, dtype=bool)
        held[held_index] = True
        train = ~held
        heads = fit_heads(
            features[train], costs[train], variant=LOCKED_VARIANT, alpha=LOCKED_ALPHA
        )
        p_l_tr, _a_tr, _k_tr, i_a_tr, i_k_tr = predict_heads(features[train], heads)
        p_l_te, _a_te, _k_te, i_a_te, i_k_te = predict_heads(features[held], heads)
        cached[str(name)] = LofoRawPack(
            held_index=np.asarray(held_index, dtype=np.int64),
            train_index=np.flatnonzero(train),
            pred_l_held=p_l_te,
            inc_a_held=i_a_te,
            inc_k_held=i_k_te,
            pred_l_train=p_l_tr,
            inc_a_train=i_a_tr,
            inc_k_train=i_k_tr,
        )
    return cached


def apply_lofo_recal(
    cached: Mapping[str, LofoRawPack],
    costs: np.ndarray,
    families: Sequence[str],
    fold_ids: np.ndarray,
    digests: Sequence[str],
    *,
    n_bins: int,
    clip: Tuple[float, float],
    fit_kind: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(costs.shape[0])
    pred_l = np.empty(n_rows, dtype=np.float64)
    inc_a = np.empty(n_rows, dtype=np.float64)
    inc_k = np.empty(n_rows, dtype=np.float64)
    actual_a, actual_k = actual_increments(costs)
    for name, pack in cached.items():
        train = pack.train_index
        if fit_kind == "prefix":
            rec_a = fit_prefix_recal(
                pack.inc_a_train,
                actual_a[train],
                pack.pred_l_train,
                costs[train, 0],
                costs[train, 1],
                fold_ids[train],
                tuple(digests[int(item)] for item in train),
                n_bins=n_bins,
                clip=clip,
            )
        else:
            rec_a = fit_recal_params(
                pack.inc_a_train, actual_a[train], n_bins=n_bins, clip=clip
            )
        rec_k = fit_recal_params(
            pack.inc_k_train, actual_k[train], n_bins=n_bins, clip=clip
        )
        pred_l[pack.held_index] = pack.pred_l_held
        inc_a[pack.held_index] = apply_recal_params(rec_a, pack.inc_a_held)
        inc_k[pack.held_index] = apply_recal_params(rec_k, pack.inc_k_held)
    return pred_l, inc_a, inc_k


@dataclass(frozen=True)
class SelectedPolicy:
    """Deployable H4 allocator. route() uses g_features plus stdlib arithmetic."""

    feature_version: str
    feature_signature: str
    bins: int
    alpha: float
    variant: str
    ridge_coefficients: Mapping[str, Tuple[float, ...]]
    smearing_factors: Mapping[str, float]
    recal_a_edges: Tuple[float, ...]
    recal_a_factors: Tuple[float, ...]
    recal_k_edges: Tuple[float, ...]
    recal_k_factors: Tuple[float, ...]
    qk_bins: int
    qk_alpha: float
    qk_target_form: str
    qk_feature_signature: str
    qk_coefficients: Tuple[float, ...]
    kappa_tier: Mapping[str, float]
    kappa_k: float
    kappa_min_increment: float
    phi_binding_q9975: Tuple[float, ...]
    psi_binding_q9975: Tuple[float, ...]
    recal_n_bins: int
    recal_clip: Tuple[float, float]
    recal_fit: str
    sort_rule: str
    k_order: str
    q_elig: float
    k1_denylist: Tuple[str, ...]
    k1_item_cap_frac: float
    k1_count_cap_frac: float
    k1_min_n: int
    k1_density_eps: float
    family_multipliers_inc_k: Mapping[str, float]
    m_unseen: float
    family_mult_clip: Tuple[float, float]
    operating_targets: Mapping[str, float]
    official_caps: Mapping[str, float]
    k1_enabled: Mapping[str, bool]
    f_grid: Tuple[float, ...]
    intercept_policy: str
    candidate_name: str

    def to_dict(self) -> dict[str, Any]:
        return sort_mapping(
            {
                "alpha": json_float(self.alpha),
                "bins": int(self.bins),
                "candidate_name": self.candidate_name,
                "f_grid": json_floats(self.f_grid),
                "family_mult_clip": [
                    json_float(self.family_mult_clip[0]),
                    json_float(self.family_mult_clip[1]),
                ],
                "family_multipliers_inc_k": {
                    name: json_float(value)
                    for name, value in self.family_multipliers_inc_k.items()
                },
                "feature_signature": self.feature_signature,
                "feature_version": self.feature_version,
                "intercept_policy": self.intercept_policy,
                "k1_count_cap_frac": json_float(self.k1_count_cap_frac),
                "k1_density_eps": json_float(self.k1_density_eps),
                "k1_denylist": list(self.k1_denylist),
                "k1_enabled": {tier: bool(self.k1_enabled[tier]) for tier in TIERS},
                "k1_item_cap_frac": json_float(self.k1_item_cap_frac),
                "k1_min_n": int(self.k1_min_n),
                "k_order": self.k_order,
                "kappa_k": json_float(self.kappa_k),
                "kappa_min_increment": json_float(self.kappa_min_increment),
                "kappa_tier": {tier: json_float(self.kappa_tier[tier]) for tier in TIERS},
                "m_unseen": json_float(self.m_unseen),
                "phi_binding_q9975": json_floats(self.phi_binding_q9975),
                "psi_binding_q9975": json_floats(self.psi_binding_q9975),
                "official_caps": {tier: json_float(self.official_caps[tier]) for tier in TIERS},
                "operating_targets": {
                    tier: json_float(self.operating_targets[tier]) for tier in TIERS
                },
                "q_elig": json_float(self.q_elig),
                "qk_alpha": json_float(self.qk_alpha),
                "qk_bins": int(self.qk_bins),
                "qk_coefficients": json_floats(self.qk_coefficients),
                "qk_feature_signature": self.qk_feature_signature,
                "qk_target_form": self.qk_target_form,
                "recal_a_edges": json_floats(self.recal_a_edges),
                "recal_a_factors": json_floats(self.recal_a_factors),
                "recal_clip": [
                    json_float(self.recal_clip[0]),
                    json_float(self.recal_clip[1]),
                ],
                "recal_fit": self.recal_fit,
                "recal_k_edges": json_floats(self.recal_k_edges),
                "recal_k_factors": json_floats(self.recal_k_factors),
                "recal_n_bins": int(self.recal_n_bins),
                "ridge_coefficients": {
                    name: json_floats(values) for name, values in self.ridge_coefficients.items()
                },
                "smearing_factors": {
                    name: json_float(value) for name, value in self.smearing_factors.items()
                },
                "sort_rule": self.sort_rule,
                "variant": self.variant,
            }
        )

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "SelectedPolicy":
        bins = int(payload["bins"])
        if bins not in HASH_BINS:
            raise ValueError("serialized bins are outside the the modeling foundation closed list")
        if payload["feature_signature"] != feature_signature(bins):
            raise ValueError("feature signature mismatch")
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError("feature_version mismatch")
        qk_bins = int(payload["qk_bins"])
        if payload["qk_feature_signature"] != feature_signature(qk_bins):
            raise ValueError("Q_K feature signature mismatch")
        clip = payload["family_mult_clip"]
        return SelectedPolicy(
            feature_version=str(payload["feature_version"]),
            feature_signature=str(payload["feature_signature"]),
            bins=bins,
            alpha=float(payload["alpha"]),
            variant=str(payload["variant"]),
            ridge_coefficients={
                name: tuple(float(item) for item in payload["ridge_coefficients"][name])
                for name in payload["ridge_coefficients"]
            },
            smearing_factors={
                name: float(value) for name, value in payload["smearing_factors"].items()
            },
            recal_a_edges=tuple(float(item) for item in payload["recal_a_edges"]),
            recal_a_factors=tuple(float(item) for item in payload["recal_a_factors"]),
            recal_k_edges=tuple(float(item) for item in payload["recal_k_edges"]),
            recal_k_factors=tuple(float(item) for item in payload["recal_k_factors"]),
            qk_bins=qk_bins,
            qk_alpha=float(payload["qk_alpha"]),
            qk_target_form=str(payload["qk_target_form"]),
            qk_feature_signature=str(payload["qk_feature_signature"]),
            qk_coefficients=tuple(float(item) for item in payload["qk_coefficients"]),
            kappa_tier={tier: float(payload["kappa_tier"][tier]) for tier in TIERS},
            kappa_k=float(payload["kappa_k"]),
            kappa_min_increment=float(payload["kappa_min_increment"]),
            phi_binding_q9975=tuple(float(item) for item in payload["phi_binding_q9975"]),
            psi_binding_q9975=tuple(float(item) for item in payload["psi_binding_q9975"]),
            recal_n_bins=int(payload["recal_n_bins"]),
            recal_clip=(
                float(payload["recal_clip"][0]),
                float(payload["recal_clip"][1]),
            ),
            recal_fit=str(payload["recal_fit"]),
            sort_rule=str(payload["sort_rule"]),
            k_order=str(payload["k_order"]),
            q_elig=float(payload["q_elig"]),
            k1_denylist=tuple(str(item) for item in payload["k1_denylist"]),
            k1_item_cap_frac=float(payload["k1_item_cap_frac"]),
            k1_count_cap_frac=float(payload["k1_count_cap_frac"]),
            k1_min_n=int(payload["k1_min_n"]),
            k1_density_eps=float(payload["k1_density_eps"]),
            family_multipliers_inc_k={
                str(name): float(value)
                for name, value in payload["family_multipliers_inc_k"].items()
            },
            m_unseen=float(payload["m_unseen"]),
            family_mult_clip=(float(clip[0]), float(clip[1])),
            operating_targets={tier: float(payload["operating_targets"][tier]) for tier in TIERS},
            official_caps={tier: float(payload["official_caps"][tier]) for tier in TIERS},
            k1_enabled={tier: bool(payload["k1_enabled"][tier]) for tier in TIERS},
            f_grid=tuple(float(item) for item in payload["f_grid"]),
            intercept_policy=str(payload["intercept_policy"]),
            candidate_name=str(payload["candidate_name"]),
        )

    def predict_arrays(self, texts: Sequence[str]) -> dict[str, Any]:
        features = design_matrix_g_features(texts, bins=int(self.bins))
        if int(self.qk_bins) == int(self.bins):
            features_qk = features
        else:
            features_qk = design_matrix_g_features(texts, bins=int(self.qk_bins))
        heads = FittedHeads(
            variant=self.variant,
            alpha=float(self.alpha),
            coefs={
                name: np.asarray(values, dtype=np.float64)
                for name, values in self.ridge_coefficients.items()
            },
            smears={name: float(value) for name, value in self.smearing_factors.items()},
            n_clipped_inc_a=0,
            n_clipped_inc_k=0,
        )
        pred_l, _pred_a, _pred_k, inc_a, inc_k = predict_heads(features, heads)
        recal_a = apply_recal_baked(
            inc_a,
            np.asarray(self.recal_a_edges, dtype=np.float64),
            np.asarray(self.recal_a_factors, dtype=np.float64),
        )
        recal_k = apply_recal_baked(
            inc_k,
            np.asarray(self.recal_k_edges, dtype=np.float64),
            np.asarray(self.recal_k_factors, dtype=np.float64),
        )
        q_k = ridge_predict(np.asarray(self.qk_coefficients, dtype=np.float64), features_qk)
        families = tuple(family_of_text(text) for text in texts)
        digests = content_digests(texts)
        return {
            "digests": digests,
            "families": families,
            "inc_a": recal_a,
            "inc_k": recal_k,
            "pred_l": pred_l,
            "q_k": q_k,
        }

    def allocate(self, tier: str, texts: Sequence[str]) -> Tuple[str, ...]:
        predicted = self.predict_arrays(texts)
        return allocate_from_arrays(
            tier,
            predicted["inc_a"],
            predicted["inc_k"],
            predicted["pred_l"],
            predicted["q_k"],
            predicted["families"],
            predicted["digests"],
            self,
        )

    def route(self, tier: str, texts: Sequence[str]) -> Tuple[str, ...]:
        """Runtime path: feature extraction goes entirely through g_features."""

        return self.allocate(tier, texts)


def allocate_from_arrays(
    tier: str,
    inc_a: np.ndarray,
    inc_k: np.ndarray,
    pred_l: np.ndarray,
    q_k: np.ndarray,
    families: Sequence[str],
    digests: Sequence[str],
    policy: SelectedPolicy,
    *,
    force_invariant_fail: bool = False,
    limit_override: Optional[float] = None,
    trace: Optional[dict[str, Any]] = None,
) -> Tuple[str, ...]:
    """Runtime-adaptive prefix + certified K1, with self-cert and all-light."""

    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    pred_inc_a = np.asarray(inc_a, dtype=np.float64).reshape(-1)
    pred_inc_k = np.asarray(inc_k, dtype=np.float64).reshape(-1)
    light = np.asarray(pred_l, dtype=np.float64).reshape(-1)
    pred_q = np.asarray(q_k, dtype=np.float64).reshape(-1)
    n_rows = int(pred_inc_a.size)
    empty_trace = {
        "all_light_fallback": False,
        "binds_term": "empirical",
        "certified_ax31": 1.0,
        "certified_ratio": 1.0,
        "empirical_term": 1.0,
        "f_star": 0.0,
        "invariant_fail": False,
        "k_star": 0,
        "kappa_k_cert_bind": 0,
        "kappa_term": None,
        "kappa_term_applied": False,
        "m_star": 0,
        "pred_k_increment": 0.0,
        "pred_ratio": 1.0,
        "self_cert_all_light": False,
        "self_cert_shed": False,
    }
    if n_rows == 0:
        if trace is not None:
            trace.update(empty_trace)
        return tuple()
    if not (
        pred_inc_k.size
        == light.size
        == pred_q.size
        == len(families)
        == len(digests)
        == n_rows
    ):
        raise ValueError("allocate_from_arrays requires aligned inputs")

    def _all_light(*, invariant: bool, shed: bool) -> Tuple[str, ...]:
        if trace is not None:
            payload = dict(empty_trace)
            payload.update(
                {
                    "all_light_fallback": True,
                    "invariant_fail": bool(invariant),
                    "self_cert_all_light": bool(shed) and not invariant,
                }
            )
            trace.update(payload)
        return tuple(_LIGHT for _ in range(n_rows))

    finite = (
        np.all(np.isfinite(pred_inc_a))
        and np.all(np.isfinite(pred_inc_k))
        and np.all(np.isfinite(light))
        and np.all(np.isfinite(pred_q))
    )
    light_sum = float(light.sum())
    kappa = float(policy.kappa_tier[tier])
    if force_invariant_fail or not finite or light_sum <= 0.0 or not math.isfinite(kappa):
        return _all_light(invariant=True, shed=False)

    order = sort_pred_inc(pred_inc_a, digests)
    grid = np.asarray(policy.f_grid, dtype=np.float64)
    pred_ratio = predicted_ratio_curve(light, pred_inc_a, order, grid)
    if (not np.all(np.isfinite(pred_ratio))) or float(pred_ratio[0]) > 1.0 + 1e-12:
        return _all_light(invariant=True, shed=False)
    if np.any(np.diff(pred_ratio) < -1e-12):
        return _all_light(invariant=True, shed=False)

    limit = float(limit_override) if limit_override is not None else float(
        policy.operating_targets[tier]
    )
    phi = np.asarray(policy.phi_binding_q9975, dtype=np.float64)
    if phi.size != grid.size:
        return _all_light(invariant=True, shed=False)
    min_inc = float(policy.kappa_min_increment)
    f_star = select_f_star_certified(
        pred_ratio, kappa, phi, limit, grid, min_increment=min_inc
    )
    if f_star > 1.0:
        f_star = 1.0
    k_star = prefix_k(f_star, n_rows)
    upgrade_a = prefix_mask(order, k_star, n_rows)
    upgrade_k = np.zeros(n_rows, dtype=bool)
    col = int(round(float(f_star) * 100.0))
    col = min(max(col, 0), int(grid.size) - 1)
    pred_at = json_float(pred_ratio[col])
    term = binding_term_at(pred_at, kappa, float(phi[col]), min_increment=min_inc)
    certified_ax31 = float(term["certified_ratio"])
    if certified_ax31 > limit + 1e-15:
        return _all_light(invariant=True, shed=False)

    cons_k = conservative_inc_k(pred_inc_k, families)
    binds = {
        "count_cap_bind": 0,
        "denylist_bind": 0,
        "item_cap_bind": 0,
        "kappa_k_cert_bind": 0,
        "n_lt_300_bind": 0,
        "qk_nonpositive_bind": 0,
    }
    m_star = 0
    pred_k_inc_used = 0.0
    ranked = np.zeros(0, dtype=np.int64)
    k1_on = (
        tier != "fast"
        and bool(policy.k1_enabled.get(tier, False))
        and math.isfinite(float(policy.kappa_k))
    )
    psi = np.asarray(policy.psi_binding_q9975, dtype=np.float64)
    if k1_on:
        eligible, binds = k1_mask(
            families,
            pred_q,
            cons_k,
            light,
            denylist=policy.k1_denylist,
            item_cap_frac=float(policy.k1_item_cap_frac),
            min_n=int(policy.k1_min_n),
        )
        ranked = order_k1(
            eligible,
            pred_q,
            pred_inc_k,
            digests,
            rule=policy.k_order,
            eps=float(policy.k1_density_eps),
        )
        count_cap = int(math.floor(float(policy.k1_count_cap_frac) * float(n_rows)))
        if int(ranked.size) > count_cap:
            binds["count_cap_bind"] = int(ranked.size) - count_cap
        pred_k_curve = predicted_k_increment(
            cons_k, pred_inc_a, upgrade_a, ranked, light_sum
        )
        m_star = select_m_star(
            pred_k_curve,
            certified_ax31=certified_ax31,
            kappa_k=float(policy.kappa_k),
            limit=limit,
            count_cap=count_cap,
            psi=psi,
            min_increment=min_inc,
        )
        available = min(int(ranked.size), count_cap)
        if available > m_star:
            binds["kappa_k_cert_bind"] = int(available - m_star)
        if m_star > 0:
            upgrade_k[ranked[:m_star]] = True
            upgrade_a[ranked[:m_star]] = True
            pred_k_inc_used = json_float(pred_k_curve[m_star - 1])

    def _accounted(mask_a: np.ndarray, mask_k: np.ndarray) -> Tuple[float, int, int, float]:
        k_used = int(np.sum(mask_a | mask_k))
        m_used = int(np.sum(mask_k))
        frac = float(k_used) / float(n_rows) if n_rows else 0.0
        frac = min(1.0, max(0.0, frac))
        pred_used = predicted_ratio_curve(
            light, pred_inc_a, order, np.asarray([frac], dtype=np.float64)
        )
        phi_at = float(np.interp(frac, grid, phi))
        term_a = binding_term_at(
            float(pred_used[0]), kappa, phi_at, min_increment=min_inc
        )
        cert_a = float(term_a["certified_ratio"])
        extra = 0.0
        cert_extra = 0.0
        if m_used > 0:
            k_rank = order_k1(
                mask_k,
                pred_q,
                pred_inc_k,
                digests,
                rule=policy.k_order,
                eps=float(policy.k1_density_eps),
            )
            curve = predicted_k_increment(cons_k, pred_inc_a, mask_a, k_rank, light_sum)
            extra = float(curve[m_used - 1]) if curve.size >= m_used else 0.0
            if psi.size:
                psi_val = float(psi[min(m_used - 1, int(psi.size) - 1)])
            else:
                psi_val = 0.0
            if extra >= min_inc and math.isfinite(float(policy.kappa_k)):
                cert_extra = max(float(policy.kappa_k) * extra, psi_val)
            else:
                cert_extra = psi_val
        return cert_a + cert_extra, k_used, m_used, extra

    accounted, k_used, m_used, extra_used = _accounted(upgrade_a, upgrade_k)
    shed = False
    if accounted > limit + 1e-15 or k_used > k_star + m_star or m_used > m_star:
        shed = True
        k_ranked = order_k1(
            upgrade_k,
            pred_q,
            pred_inc_k,
            digests,
            rule=policy.k_order,
            eps=float(policy.k1_density_eps),
        )
        for index in k_ranked[::-1]:
            upgrade_k[int(index)] = False
            accounted, k_used, m_used, extra_used = _accounted(upgrade_a, upgrade_k)
            if accounted <= limit + 1e-15 and m_used <= m_star:
                break
        if accounted > limit + 1e-15 or k_used > k_star + m_used:
            for index in order[::-1]:
                if upgrade_k[int(index)] or not upgrade_a[int(index)]:
                    continue
                upgrade_a[int(index)] = False
                accounted, k_used, m_used, extra_used = _accounted(upgrade_a, upgrade_k)
                if accounted <= limit + 1e-15:
                    break
        if accounted > limit + 1e-15:
            return _all_light(invariant=False, shed=True)

    if trace is not None:
        trace.update(
            {
                "all_light_fallback": False,
                "binds": binds,
                "binds_term": term["binds"],
                "certified_ax31": json_float(certified_ax31),
                "certified_ratio": json_float(certified_ax31),
                "empirical_term": term["empirical_term"],
                "f_star": json_float(f_star),
                "invariant_fail": False,
                "k_star": int(k_star),
                "kappa_k_cert_bind": int(binds.get("kappa_k_cert_bind", 0)),
                "kappa_term": term["kappa_term"],
                "kappa_term_applied": bool(term["kappa_term_applied"]),
                "m_star": int(m_star),
                "pred_k_increment": json_float(extra_used if m_used else pred_k_inc_used),
                "pred_ratio": pred_at,
                "self_cert_all_light": False,
                "self_cert_shed": bool(shed),
            }
        )
    return models_from_masks(upgrade_a, upgrade_k)


def allocate(tier: str, texts: Sequence[str], policy: SelectedPolicy) -> Tuple[str, ...]:
    return policy.allocate(tier, texts)


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "adoption_rule": ADOPTION_RULE,
            "binding_layer_reason": BINDING_LAYER_REASON,
            "binding_layer_spec": BINDING_LAYER_SPEC,
            "bootstrap_draws": int(BOOTSTRAP_DRAWS),
            "bootstrap_seed": int(BOOTSTRAP_SEED),
            "cost_head_lock": COST_HEAD_LOCK,
            "f_grid": json_floats(F_GRID),
            "family_mult_clip": [
                json_float(K1_FAMILY_MULT_CLIP[0]),
                json_float(K1_FAMILY_MULT_CLIP[1]),
            ],
            "feature_signature": feature_signature(LOCKED_BINS),
            "feature_version": FEATURE_VERSION,
            "float64_table_note": FLOAT64_NOTE,
            "fold_seed": int(FOLD_SEED_PREFIX_RECAL),
            "folds": int(FOLDS),
            "h2_5_threshold": json_float(H2_5_THRESHOLD),
            "h2_6_exceedance_max": json_float(H2_6_EXCEEDANCE_MAX),
            "h4_gate_spec": H4_GATE_SPEC,
            "h4_hypothesis": H4_HYPOTHESIS,
            "hash_bins_allowed": list(HASH_BINS),
            "intercept_policy": INTERCEPT_POLICY,
            "k1_count_cap_frac": json_float(K1_COUNT_CAP_FRAC),
            "k1_denylist": list(K1_DENYLIST),
            "k1_guard_spec": K1_GUARD_SPEC,
            "k1_item_cap_frac": json_float(K1_ITEM_CAP_FRAC),
            "k1_min_n": int(K1_MIN_N),
            "k_order": K_ORDER,
            "kappa_min_increment": json_float(KAPPA_MIN_INCREMENT),
            "kappa_q": json_float(KAPPA_Q),
            "locked_alpha": json_float(LOCKED_ALPHA),
            "locked_bins": int(LOCKED_BINS),
            "locked_family_multipliers_inc_k": {
                name: json_float(value) for name, value in LOCKED_FAMILY_MULT_INC_K.items()
            },
            "locked_inc_A": json_float(LOCKED_INC_A),
            "locked_inc_K": json_float(LOCKED_INC_K),
            "locked_variant": LOCKED_VARIANT,
            "m_unseen": json_float(K1_M_UNSEEN),
            "official_caps": {tier: json_float(OFFICIAL_CAPS[tier]) for tier in TIERS},
            "operating_targets": {tier: json_float(OPERATING_TARGETS[tier]) for tier in TIERS},
            "parent_f_exact": PARENT_F_EXACT,
            "parent_f_pins": PARENT_F_PINS,
            "k1_quality_vacuous_max": json_float(K1_QUALITY_VACUOUS_MAX),
            "prefix_fit_spec": PREFIX_FIT_SPEC,
            "q_elig": json_float(Q_ELIG),
            "recal_clip_legacy": [
                json_float(RECAL_CLIP_LEGACY[0]),
                json_float(RECAL_CLIP_LEGACY[1]),
            ],
            "recal_grid_clips": [
                [json_float(clip[0]), json_float(clip[1])] for clip in RECAL_GRID_CLIPS
            ],
            "recal_grid_n_bins": list(RECAL_GRID_N_BINS),
            "recal_n_bins_legacy": int(RECAL_BINS_LEGACY),
            "recal_selection_criterion": RECAL_SELECTION_CRITERION,
            "redteam_layer_spec": REDTEAM_LAYER_SPEC,
            "sort_rule": SORT_RULE,
            "tier_pred_bands": {
                tier: {
                    "hi": None if not math.isfinite(hi) else json_float(hi),
                    "lo": json_float(lo),
                }
                for tier, (lo, hi) in TIER_PRED_BANDS.items()
            },
            "tier_weights": {tier: json_float(TIER_WEIGHTS[tier]) for tier in TIERS},
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
    report = {
        "adoption_deferred_to_operator": True,
        "decision": decision,
        "dev_opened": False,
        "diagnostic": diagnostic,
        "experiment": EXPERIMENT,
        "identity": identity,
        "locked": locked,
        "observed": observed,
        "q_a_used": False,
        "quality_entered_calibration_selection": False,
        "quality_entered_cost_selection": False,
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
    }
    if report["dev_opened"] is not False:
        raise RuntimeError("the prefix recalibration layer report must assert dev_opened is false")
    if report["q_a_used"] is not False:
        raise RuntimeError("the prefix recalibration layer report must assert q_a_used is false")
    if report["adoption_deferred_to_operator"] is not True:
        raise RuntimeError("the prefix recalibration layer must defer adoption to the operator")
    return sort_mapping(report)


def decide(*, measurement_ok: bool, reason: str = "") -> str:
    """Record both candidates whenever the measurement itself is valid."""

    if measurement_ok:
        return DECISION_TWO
    slug = str(reason or "measurement-failed").strip("-")
    return f"record-prefix_recal-close-{slug}"


def _ratio_close(observed: float, locked: float) -> bool:
    return abs(float(observed) - float(locked)) <= float(LOCKED_RATIO_ATOL)


def _realized_from_costs(
    actual_l: np.ndarray, actual_a: np.ndarray, actual_k: np.ndarray, models: Sequence[str]
) -> float:
    columns = np.asarray([MODEL_IDS.index(model_id) for model_id in models], dtype=np.int64)
    stacked = np.stack(
        [
            np.asarray(actual_l, dtype=np.float64),
            np.asarray(actual_a, dtype=np.float64),
            np.asarray(actual_k, dtype=np.float64),
        ],
        axis=1,
    )
    light_sum = float(actual_l.sum())
    if light_sum <= 0.0:
        return float("inf")
    spent = float(stacked[np.arange(int(actual_l.size)), columns].sum())
    return json_float(spent / light_sum)


def _mean_from_scores(
    score_l: np.ndarray, score_a: np.ndarray, score_k: np.ndarray, models: Sequence[str]
) -> float:
    columns = np.asarray([MODEL_IDS.index(model_id) for model_id in models], dtype=np.int64)
    stacked = np.stack(
        [
            np.asarray(score_l, dtype=np.float64),
            np.asarray(score_a, dtype=np.float64),
            np.asarray(score_k, dtype=np.float64),
        ],
        axis=1,
    )
    return json_float(float(stacked[np.arange(int(score_l.size)), columns].mean()))


def _argmax_cell(
    values: Sequence[float], names: Sequence[str], fractions: Sequence[float]
) -> Optional[dict[str, Any]]:
    if not values:
        return None
    best = max(range(len(values)), key=lambda index: (float(values[index]), names[index], fractions[index]))
    return {
        "f": json_float(fractions[best]),
        "kappa": json_float(values[best]),
        "view": names[best],
    }


def fit_and_evaluate(bundle: TrainBundle) -> dict[str, Any]:
    """Train-only the prefix recalibration layer fit. Q_A is never constructed. Dev is never opened."""

    folds = group_folds(bundle.episodes, folds=FOLDS, seed=FOLD_SEED_PREFIX_RECAL)
    fold_ids = np.asarray(list(folds), dtype=np.int64)
    families = bundle.families
    fam_arr = np.asarray(list(families))
    n_train = int(bundle.costs.shape[0])
    texts = bundle.texts
    digests = content_digests(texts)
    costs = np.asarray(bundle.costs, dtype=np.float64)
    scores = np.asarray(bundle.scores, dtype=np.float64)
    actual_l = costs[:, 0]
    actual_a = costs[:, 1]
    actual_k = costs[:, 2]
    score_l = scores[:, 0]
    score_a = scores[:, 1]
    score_k = scores[:, 2]
    actual_inc_a, actual_inc_k = actual_increments(costs)

    features = feature_matrix(texts, bins=LOCKED_BINS)
    qk_head = load_q_k_head(QUALITY_HEADS_PATH)
    if int(qk_head.bins) != LOCKED_BINS:
        features_qk = feature_matrix(texts, bins=int(qk_head.bins))
    else:
        features_qk = features

    oof_l, _oof_a, _oof_k, oof_inc_a_raw, oof_inc_k_raw, _clip = oof_incremental_costs(
        features, costs, folds, variant=LOCKED_VARIANT, alpha=LOCKED_ALPHA
    )
    rec_a_legacy = fit_recal_params(
        oof_inc_a_raw, actual_inc_a, n_bins=RECAL_BINS_LEGACY, clip=RECAL_CLIP_LEGACY
    )
    rec_k_legacy = fit_recal_params(
        oof_inc_k_raw, actual_inc_k, n_bins=RECAL_BINS_LEGACY, clip=RECAL_CLIP_LEGACY
    )
    oof_inc_a_legacy = apply_recal_params(rec_a_legacy, oof_inc_a_raw)
    oof_inc_k_legacy = apply_recal_params(rec_k_legacy, oof_inc_k_raw)
    ratio_a = float(actual_inc_a.sum()) / float(oof_inc_a_legacy.sum())
    ratio_k = float(actual_inc_k.sum()) / float(oof_inc_k_legacy.sum())
    if not _ratio_close(ratio_a, LOCKED_INC_A) or not _ratio_close(ratio_k, LOCKED_INC_K):
        return {
            "decision": decide(measurement_ok=False, reason="cost-head-ratio-drift"),
            "diagnostic": {
                "oof_recal_ratios_legacy": {
                    "inc_A": json_float(ratio_a),
                    "inc_K": json_float(ratio_k),
                }
            },
            "observed": {
                "adoption_deferred_to_operator": True,
                "dev_opened": False,
                "q_a_used": False,
            },
            "policies": {},
        }

    full_heads = fit_heads(features, costs, variant=LOCKED_VARIANT, alpha=LOCKED_ALPHA)
    _full_l, _fa, _fk, full_inc_a_raw, full_inc_k_raw = predict_heads(features, full_heads)

    qk_target = score_k - score_a
    oof_qk = oof_predict(features_qk, qk_target, folds, alpha=float(qk_head.alpha))
    lofo_raw = cache_lofo_raw(features, costs, families)
    lofo_qk = _lofo_qk(features_qk, qk_target, families, alpha=float(qk_head.alpha))

    views, catalogue = build_views(families, folds)
    layers = tuple(view_layer(view) for view in views)
    binding_idx = tuple(i for i, layer in enumerate(layers) if layer == "binding")
    redteam_idx = tuple(i for i, layer in enumerate(layers) if layer == "red-team")
    view_names = tuple(view.name for view in views)

    def _max_abs_on_binding(
        oof_ia: np.ndarray, lofo_ia: np.ndarray, oof_pl: np.ndarray, lofo_pl: np.ndarray
    ) -> float:
        worst = 0.0
        for index in binding_idx:
            view = views[index]
            src_ia = oof_ia if view.pred_source == "oof" else lofo_ia
            src_pl = oof_pl if view.pred_source == "oof" else lofo_pl
            idx = view.index
            digest = tuple(digests[int(item)] for item in idx)
            inc = _gather(src_ia, idx)
            order = sort_pred_inc(inc, digest)
            pred = predicted_ratio_curve(_gather(src_pl, idx), inc, order)
            actual = phi_view(_gather(actual_l, idx), _gather(actual_a, idx), order)
            worst = max(worst, float(np.max(np.abs(pred - actual))))
        return float(worst)

    recal_grid_rows: list[dict[str, Any]] = []
    selected_row: Optional[dict[str, Any]] = None
    selected_key: Optional[Tuple[float, int, float, str]] = None
    for n_bins in RECAL_GRID_N_BINS:
        for clip in RECAL_GRID_CLIPS:
            for fit_kind in RECAL_FIT_KINDS:
                if fit_kind == "prefix":
                    rec_a = fit_prefix_recal(
                        oof_inc_a_raw,
                        actual_inc_a,
                        oof_l,
                        actual_l,
                        actual_a,
                        fold_ids,
                        digests,
                        n_bins=n_bins,
                        clip=clip,
                    )
                else:
                    rec_a = fit_recal_params(
                        oof_inc_a_raw, actual_inc_a, n_bins=n_bins, clip=clip
                    )
                rec_k = fit_recal_params(
                    oof_inc_k_raw, actual_inc_k, n_bins=n_bins, clip=clip
                )
                oof_ia = apply_recal_params(rec_a, oof_inc_a_raw)
                oof_ik = apply_recal_params(rec_k, oof_inc_k_raw)
                lofo_pl, lofo_ia, lofo_ik = apply_lofo_recal(
                    lofo_raw,
                    costs,
                    families,
                    fold_ids,
                    digests,
                    n_bins=n_bins,
                    clip=clip,
                    fit_kind=fit_kind,
                )
                max_abs = _max_abs_on_binding(oof_ia, lofo_ia, oof_l, lofo_pl)
                clip_info = clip_binds(rec_a, clip)
                row = {
                    "clip": [json_float(clip[0]), json_float(clip[1])],
                    "clip_binds": clip_info,
                    "factors": json_floats(rec_a.clipped_factors),
                    "fit": fit_kind,
                    "max_abs_pred_minus_actual": json_float(max_abs),
                    "n_bins": int(n_bins),
                    "pav_factors": json_floats(rec_a.pav_factors),
                    "raw_factors": json_floats(rec_a.raw_factors),
                }
                recal_grid_rows.append(row)
                key = (
                    float(max_abs),
                    int(n_bins),
                    tighter_clip_key(clip),
                    fit_kind,
                )
                if selected_key is None or key < selected_key:
                    selected_key = key
                    selected_row = {
                        **row,
                        "_rec_a": rec_a,
                        "_rec_k": rec_k,
                        "_oof_ia": oof_ia,
                        "_oof_ik": oof_ik,
                        "_lofo_pl": lofo_pl,
                        "_lofo_ia": lofo_ia,
                        "_lofo_ik": lofo_ik,
                    }
    if selected_row is None:
        return {
            "decision": decide(measurement_ok=False, reason="recal-grid-empty"),
            "diagnostic": {},
            "observed": {
                "adoption_deferred_to_operator": True,
                "dev_opened": False,
                "q_a_used": False,
            },
            "policies": {},
        }

    selected_n_bins = int(selected_row["n_bins"])
    selected_clip = (float(selected_row["clip"][0]), float(selected_row["clip"][1]))
    selected_fit = str(selected_row["fit"])
    rec_a = selected_row["_rec_a"]
    rec_k = selected_row["_rec_k"]
    oof_inc_a = selected_row["_oof_ia"]
    oof_inc_k = selected_row["_oof_ik"]
    lofo_l = selected_row["_lofo_pl"]
    lofo_inc_a = selected_row["_lofo_ia"]
    lofo_inc_k = selected_row["_lofo_ik"]
    if selected_fit == "prefix":
        full_rec_a = fit_prefix_recal(
            full_inc_a_raw,
            actual_inc_a,
            _full_l,
            actual_l,
            actual_a,
            fold_ids,
            digests,
            n_bins=selected_n_bins,
            clip=selected_clip,
        )
    else:
        full_rec_a = fit_recal_params(
            full_inc_a_raw, actual_inc_a, n_bins=selected_n_bins, clip=selected_clip
        )
    full_rec_k = fit_recal_params(
        full_inc_k_raw, actual_inc_k, n_bins=selected_n_bins, clip=selected_clip
    )
    selected_recal_public = {
        "clip": [json_float(selected_clip[0]), json_float(selected_clip[1])],
        "clip_binds": selected_row["clip_binds"],
        "factors": selected_row["factors"],
        "fit": selected_fit,
        "max_abs_pred_minus_actual": selected_row["max_abs_pred_minus_actual"],
        "n_bins": selected_n_bins,
        "pav_factors": selected_row["pav_factors"],
        "raw_factors": selected_row["raw_factors"],
    }

    def _pred_pack(view: View) -> dict[str, np.ndarray]:
        src_l = oof_l if view.pred_source == "oof" else lofo_l
        src_ia = oof_inc_a if view.pred_source == "oof" else lofo_inc_a
        src_ik = oof_inc_k if view.pred_source == "oof" else lofo_inc_k
        src_q = oof_qk if view.pred_source == "oof" else lofo_qk
        idx = view.index
        return {
            "actual_a": _gather(actual_a, idx),
            "actual_k": _gather(actual_k, idx),
            "actual_l": _gather(actual_l, idx),
            "families": fam_arr[idx],
            "inc_a": _gather(src_ia, idx),
            "inc_k": _gather(src_ik, idx),
            "index": idx,
            "pred_l": _gather(src_l, idx),
            "q_k": _gather(src_q, idx),
            "score_a": _gather(score_a, idx),
            "score_k": _gather(score_k, idx),
            "score_l": _gather(score_l, idx),
        }

    view_packs = [_pred_pack(view) for view in views]
    view_digests = [tuple(digests[int(item)] for item in view.index) for view in views]
    orders = []
    pred_curves = []
    actual_curves = []
    monotone_flags = []
    for pack, digest in zip(view_packs, view_digests):
        order = sort_pred_inc(pack["inc_a"], digest)
        pred = predicted_ratio_curve(pack["pred_l"], pack["inc_a"], order)
        actual = phi_view(pack["actual_l"], pack["actual_a"], order)
        orders.append(order)
        pred_curves.append(pred)
        actual_curves.append(actual)
        monotone_flags.append(bool(np.all(np.diff(pred) >= -1e-15) and abs(float(pred[0]) - 1.0) <= 1e-15))

    binding_actual = [actual_curves[index] for index in binding_idx]
    phi_binding, phi_change = phi_from_actual_curves(binding_actual)
    phi_at_report = {
        f"{frac:.2f}": json_float(float(phi_binding[int(round(frac * 100.0))]))
        for frac in (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
    }

    n_kappa_included = 0
    n_kappa_excluded = 0
    binding_cells: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    redteam_cells: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    for index, (view, pred, actual, layer) in enumerate(
        zip(views, pred_curves, actual_curves, layers)
    ):
        kappa, rho, included = kappa_cells_from_curves(pred, actual)
        n_kappa_included += int(np.sum(included))
        n_kappa_excluded += int(np.sum(~included))
        for col, frac in enumerate(F_GRID):
            if not bool(included[col]):
                continue
            cell = {
                "actual_ratio": json_float(actual[col]),
                "f": json_float(frac),
                "index": int(index),
                "kappa": json_float(kappa[col]),
                "pred_ratio": json_float(pred[col]),
                "rho": json_float(rho[col]),
                "view": view.name,
            }
            for tier in TIERS:
                if in_tier_band(float(pred[col]), tier):
                    if layer == "binding":
                        binding_cells[tier].append(cell)
                    else:
                        redteam_cells[tier].append(cell)

    kappa_block: dict[str, Any] = {}
    kappa_used = {
        "candidate_kappa_max": {tier: 1.0 for tier in TIERS},
        "candidate_kappa_q9975": {tier: 1.0 for tier in TIERS},
    }
    for tier in TIERS:
        bind_k = np.asarray([cell["kappa"] for cell in binding_cells[tier]], dtype=np.float64)
        bind_rho = np.asarray([cell["rho"] for cell in binding_cells[tier]], dtype=np.float64)
        red_k = np.asarray([cell["kappa"] for cell in redteam_cells[tier]], dtype=np.float64)
        red_rho = np.asarray([cell["rho"] for cell in redteam_cells[tier]], dtype=np.float64)
        bind_summary = summarize_values(bind_k)
        rho_summary = summarize_values(bind_rho)
        red_summary = summarize_values(red_k)
        argmax = _argmax_cell(
            [cell["kappa"] for cell in binding_cells[tier]],
            [cell["view"] for cell in binding_cells[tier]],
            [cell["f"] for cell in binding_cells[tier]],
        )
        if bind_summary is None:
            return {
                "decision": decide(measurement_ok=False, reason=f"empty-kappa-{tier}"),
                "diagnostic": {"tier": tier},
                "observed": {
                    "adoption_deferred_to_operator": True,
                    "dev_opened": False,
                    "q_a_used": False,
                },
                "policies": {},
            }
        kappa_used["candidate_kappa_max"][tier] = float(bind_summary["max"])
        kappa_used["candidate_kappa_q9975"][tier] = float(bind_summary["q9975"])
        kappa_block[tier] = {
            "argmax_view": argmax,
            "binding": bind_summary,
            "binding_n_cells": int(bind_k.size),
            "ratio_of_ratios_binding": rho_summary,
            "ratio_of_ratios_redteam_max": (
                json_float(float(red_rho.max())) if red_rho.size else None
            ),
            "redteam": red_summary,
            "redteam_max": json_float(float(red_k.max())) if red_k.size else None,
        }

    # kappa_K on the K1 increment, binding layer only, pred increment >= 0.01.
    k_cells_max: list[float] = []
    k_cells_rho: list[float] = []
    k_cell_meta: list[dict[str, Any]] = []
    psi_max_m = max(1, int(math.floor(K1_COUNT_CAP_FRAC * float(n_train))))
    psi_columns: list[list[float]] = [[] for _ in range(psi_max_m)]
    for index in binding_idx:
        pack = view_packs[index]
        digest = view_digests[index]
        cons = conservative_inc_k(pack["inc_k"], pack["families"].tolist())
        eligible, _binds = k1_mask(
            pack["families"].tolist(), pack["q_k"], cons, pack["pred_l"]
        )
        ranked = order_k1(eligible, pack["q_k"], pack["inc_k"], digest, rule=K_ORDER)
        n_view = int(pack["actual_l"].size)
        count_cap = int(math.floor(K1_COUNT_CAP_FRAC * float(n_view)))
        ranked = ranked[:count_cap]
        if ranked.size == 0:
            continue
        dummy_prefix = np.ones(n_view, dtype=bool)
        pred_inc = predicted_k_increment(
            cons, pack["inc_a"], dummy_prefix, ranked, float(pack["pred_l"].sum())
        )
        act_inc = actual_k_increment(
            pack["actual_l"], pack["actual_a"], pack["actual_k"], dummy_prefix, ranked
        )
        for m_i in range(int(ranked.size)):
            if m_i < psi_max_m:
                psi_columns[m_i].append(float(act_inc[m_i]))
            pred_val = float(pred_inc[m_i])
            if pred_val < KAPPA_MIN_INCREMENT:
                continue
            kappa_k_val = float(act_inc[m_i]) / pred_val
            k_cells_max.append(kappa_k_val)
            k_cells_rho.append(float(act_inc[m_i]) / pred_val)
            k_cell_meta.append({"m": int(m_i + 1), "view": views[index].name})
    k_array = np.asarray(k_cells_max, dtype=np.float64)
    kappa_k_summary = summarize_values(k_array)
    kappa_k_available = kappa_k_summary is not None
    kappa_k_by_candidate = {
        "candidate_kappa_max": (
            float(kappa_k_summary["max"]) if kappa_k_available else float("nan")
        ),
        "candidate_kappa_q9975": (
            float(kappa_k_summary["q9975"]) if kappa_k_available else float("nan")
        ),
    }
    psi_raw = np.zeros(psi_max_m, dtype=np.float64)
    last_psi = 0.0
    for m_i, column in enumerate(psi_columns):
        if column:
            last_psi = float(quantile_higher(np.asarray(column, dtype=np.float64), KAPPA_Q))
        psi_raw[m_i] = last_psi
    psi_binding, psi_change = running_maximum(psi_raw)
    phi_tuple = tuple(json_floats(phi_binding))
    psi_tuple = tuple(json_floats(psi_binding))

    parent_ids = _parent_assignments(bundle)
    parent_weighted = float(PARENT_F_PINS["weighted"])
    fold_positions = [index for index, view in enumerate(views) if view.kind == "oof-fold"]
    family_names = tuple(sorted(dict.fromkeys(fam_arr.tolist())))

    def _make_policy(
        name: str,
        kappa_tier: Mapping[str, float],
        kappa_k: float,
        k1_enabled: Mapping[str, bool],
    ) -> SelectedPolicy:
        return SelectedPolicy(
            feature_version=FEATURE_VERSION,
            feature_signature=feature_signature(LOCKED_BINS),
            bins=LOCKED_BINS,
            alpha=LOCKED_ALPHA,
            variant=LOCKED_VARIANT,
            ridge_coefficients={
                key: tuple(json_floats(coef)) for key, coef in full_heads.coefs.items()
            },
            smearing_factors={
                key: json_float(value) for key, value in full_heads.smears.items()
            },
            recal_a_edges=tuple(json_floats(full_rec_a.edges)),
            recal_a_factors=tuple(json_floats(full_rec_a.clipped_factors)),
            recal_k_edges=tuple(json_floats(full_rec_k.edges)),
            recal_k_factors=tuple(json_floats(full_rec_k.clipped_factors)),
            qk_bins=int(qk_head.bins),
            qk_alpha=float(qk_head.alpha),
            qk_target_form=qk_head.target_form,
            qk_feature_signature=qk_head.feature_signature,
            qk_coefficients=tuple(json_floats(qk_head.coef)),
            kappa_tier={tier: json_float(kappa_tier[tier]) for tier in TIERS},
            kappa_k=json_float(kappa_k) if math.isfinite(float(kappa_k)) else 0.0,
            kappa_min_increment=KAPPA_MIN_INCREMENT,
            phi_binding_q9975=phi_tuple,
            psi_binding_q9975=psi_tuple,
            recal_n_bins=selected_n_bins,
            recal_clip=selected_clip,
            recal_fit=selected_fit,
            sort_rule=SORT_RULE,
            k_order=K_ORDER,
            q_elig=Q_ELIG,
            k1_denylist=K1_DENYLIST,
            k1_item_cap_frac=K1_ITEM_CAP_FRAC,
            k1_count_cap_frac=K1_COUNT_CAP_FRAC,
            k1_min_n=K1_MIN_N,
            k1_density_eps=K1_DENSITY_EPS,
            family_multipliers_inc_k={
                key: json_float(value) for key, value in LOCKED_FAMILY_MULT_INC_K.items()
            },
            m_unseen=K1_M_UNSEEN,
            family_mult_clip=K1_FAMILY_MULT_CLIP,
            operating_targets=dict(OPERATING_TARGETS),
            official_caps=dict(OFFICIAL_CAPS),
            k1_enabled={tier: bool(k1_enabled[tier]) for tier in TIERS},
            f_grid=tuple(json_floats(F_GRID)),
            intercept_policy=INTERCEPT_POLICY,
            candidate_name=name,
        )

    def _route_pack(
        pack: dict[str, np.ndarray],
        digest: Sequence[str],
        policy: SelectedPolicy,
        *,
        k1_on: bool,
        limits: Mapping[str, float],
    ) -> Tuple[dict[str, Tuple[str, ...]], dict[str, dict[str, Any]]]:
        assigned: dict[str, Tuple[str, ...]] = {}
        traces: dict[str, dict[str, Any]] = {}
        k1_map = {tier: (bool(k1_on) and bool(policy.k1_enabled.get(tier, False))) for tier in TIERS}
        local = SelectedPolicy(
            **{**policy.__dict__, "k1_enabled": k1_map}
        )
        for tier in TIERS:
            row: dict[str, Any] = {}
            assigned[tier] = allocate_from_arrays(
                tier,
                pack["inc_a"],
                pack["inc_k"],
                pack["pred_l"],
                pack["q_k"],
                pack["families"].tolist(),
                digest,
                local,
                limit_override=float(limits[tier]),
                trace=row,
            )
            traces[tier] = row
        return assigned, traces

    def _evaluate_one(name: str) -> dict[str, Any]:
        kappa_tier = kappa_used[name]
        kappa_k = kappa_k_by_candidate[name]
        k1_probe = {
            "fast": False,
            "balanced": bool(kappa_k_available),
            "premium": bool(kappa_k_available),
        }
        probe = _make_policy(name, kappa_tier, kappa_k, k1_probe)

        # Reference f*/k* on the full Train treated as one OOF batch.
        full_pack = {
            "actual_a": actual_a,
            "actual_k": actual_k,
            "actual_l": actual_l,
            "families": fam_arr,
            "inc_a": oof_inc_a,
            "inc_k": oof_inc_k,
            "pred_l": oof_l,
            "q_k": oof_qk,
            "score_a": score_a,
            "score_k": score_k,
            "score_l": score_l,
        }
        assigned_full_on, traces_full = _route_pack(
            full_pack, digests, probe, k1_on=True, limits=OPERATING_TARGETS
        )
        assigned_full_off, _tr_off = _route_pack(
            full_pack, digests, probe, k1_on=False, limits=OPERATING_TARGETS
        )
        assigned_full_official, traces_official = _route_pack(
            full_pack, digests, probe, k1_on=True, limits=OFFICIAL_CAPS
        )

        m_star_ref = {tier: int(traces_full[tier]["m_star"]) for tier in TIERS}
        premium_k1_full = int(sum(1 for mid in assigned_full_on["premium"] if mid == _K1))
        quality_from_k1 = {
            tier: json_float(
                _mean_from_scores(score_l, score_a, score_k, assigned_full_on[tier])
                - _mean_from_scores(score_l, score_a, score_k, assigned_full_off[tier])
            )
            for tier in TIERS
        }
        k1_quality_vacuous = bool(
            max(float(quality_from_k1["balanced"]), float(quality_from_k1["premium"]))
            <= float(K1_QUALITY_VACUOUS_MAX)
        )
        h2_10_probe = bool(premium_k1_full > 0)
        m_all_zero = all(int(m_star_ref[tier]) == 0 for tier in ("balanced", "premium"))
        k1_enabled = {
            "fast": False,
            "balanced": bool(
                kappa_k_available
                and int(m_star_ref["balanced"]) > 0
                and h2_10_probe
                and not k1_quality_vacuous
            ),
            "premium": bool(
                kappa_k_available
                and int(m_star_ref["premium"]) > 0
                and h2_10_probe
                and not k1_quality_vacuous
            ),
        }
        if m_all_zero or not h2_10_probe or k1_quality_vacuous:
            k1_enabled = {"fast": False, "balanced": False, "premium": False}
        adopted = _make_policy(name, kappa_tier, kappa_k, k1_enabled)
        use_k1 = any(k1_enabled.values())

        oof_models: dict[str, list[str]] = {tier: [""] * n_train for tier in TIERS}
        fold_traces: list[dict[str, Any]] = []
        for row in fold_positions:
            assigned, traces = _route_pack(
                view_packs[row],
                view_digests[row],
                adopted,
                k1_on=use_k1,
                limits=OPERATING_TARGETS,
            )
            fold_traces.append({"name": views[row].name, "traces": traces})
            for local, global_i in enumerate(views[row].index):
                for tier in TIERS:
                    oof_models[tier][int(global_i)] = assigned[tier][local]
        oof_models_t = {tier: tuple(oof_models[tier]) for tier in TIERS}
        if any(item == "" for tier in TIERS for item in oof_models[tier]):
            raise RuntimeError("OOF allocation left an unassigned episode")

        lofo_models: dict[str, list[str]] = {tier: [""] * n_train for tier in TIERS}
        lofo_family_gain: dict[str, dict[str, Any]] = {}
        for fam_name, held in family_folds(families):
            held_idx = np.asarray(held, dtype=np.int64)
            view_i = next(i for i, view in enumerate(views) if view.name == f"lofo-{fam_name}")
            assigned, _tr = _route_pack(
                view_packs[view_i],
                view_digests[view_i],
                adopted,
                k1_on=use_k1,
                limits=OPERATING_TARGETS,
            )
            for local, global_i in enumerate(held_idx):
                for tier in TIERS:
                    lofo_models[tier][int(global_i)] = assigned[tier][local]
            fam_gain = {}
            for tier in TIERS:
                ours = float(
                    _episode_scores(scores[held_idx], tuple(np.asarray(assigned[tier]))).mean()
                )
                parent = float(
                    _episode_scores(
                        scores[held_idx], tuple(np.asarray(parent_ids[tier])[held_idx])
                    ).mean()
                )
                fam_gain[tier] = {
                    "gain": json_float(ours - parent),
                    "n": int(held_idx.size),
                    "ours": json_float(ours),
                    "parent": json_float(parent),
                }
            weighted_ours = weighted_final(
                fam_gain["fast"]["ours"], fam_gain["balanced"]["ours"], fam_gain["premium"]["ours"]
            )
            weighted_parent = weighted_final(
                fam_gain["fast"]["parent"],
                fam_gain["balanced"]["parent"],
                fam_gain["premium"]["parent"],
            )
            lofo_family_gain[fam_name] = {
                "n": int(held_idx.size),
                "per_tier": fam_gain,
                "weighted_gain": json_float(weighted_ours - weighted_parent),
                "weighted_ours": json_float(weighted_ours),
                "weighted_parent": json_float(weighted_parent),
            }
        lofo_models_t = {tier: tuple(lofo_models[tier]) for tier in TIERS}
        if any(item == "" for tier in TIERS for item in lofo_models[tier]):
            raise RuntimeError("LOFO allocation left an unassigned episode")

        official_oof = official_score(bundle.inputs, bundle.outcomes, bundle.policy, oof_models_t)
        official_lofo = official_score(bundle.inputs, bundle.outcomes, bundle.policy, lofo_models_t)
        float_oof = {tier: _score_mean(scores, oof_models_t[tier]) for tier in TIERS}
        float_oof_weighted = json_float(
            weighted_final(float_oof["fast"], float_oof["balanced"], float_oof["premium"])
        )
        official_oof_q = {tier: float(official_oof["tiers"][tier]["quality_score"]) for tier in TIERS}
        official_agree = {
            tier: json_float(abs(float_oof[tier] - official_oof_q[tier])) for tier in TIERS
        }
        official_agree["max"] = json_float(max(official_agree[tier] for tier in TIERS))
        official_agree["within_1e_12"] = bool(official_agree["max"] <= 1e-12)

        oof_gain = {
            tier: json_float(official_oof_q[tier] - float(PARENT_F_PINS[tier]["quality"]))
            for tier in TIERS
        }
        oof_gain["weighted"] = json_float(float(official_oof["final_score"]) - parent_weighted)

        parent_fold_q = {tier: [] for tier in TIERS}
        ours_fold_q = {tier: [] for tier in TIERS}
        for fold in range(FOLDS):
            mask = fold_ids == fold
            for tier in TIERS:
                parent_fold_q[tier].append(
                    json_float(
                        float(
                            _episode_scores(
                                scores[mask], tuple(np.asarray(parent_ids[tier])[mask])
                            ).mean()
                        )
                    )
                )
                ours_fold_q[tier].append(
                    json_float(
                        float(
                            _episode_scores(
                                scores[mask], tuple(np.asarray(oof_models_t[tier])[mask])
                            ).mean()
                        )
                    )
                )
        fold_weighted_ours = [
            weighted_final(ours_fold_q["fast"][i], ours_fold_q["balanced"][i], ours_fold_q["premium"][i])
            for i in range(FOLDS)
        ]
        fold_weighted_parent = [
            weighted_final(
                parent_fold_q["fast"][i], parent_fold_q["balanced"][i], parent_fold_q["premium"][i]
            )
            for i in range(FOLDS)
        ]
        fold_gains = [
            json_float(fold_weighted_ours[i] - fold_weighted_parent[i]) for i in range(FOLDS)
        ]
        fold_wins = int(sum(1 for gain in fold_gains if gain > 0.0))

        ours_ep = (
            TIER_WEIGHTS["fast"] * _episode_scores(scores, oof_models_t["fast"])
            + TIER_WEIGHTS["balanced"] * _episode_scores(scores, oof_models_t["balanced"])
            + TIER_WEIGHTS["premium"] * _episode_scores(scores, oof_models_t["premium"])
        )
        parent_ep = (
            TIER_WEIGHTS["fast"] * _episode_scores(scores, parent_ids["fast"])
            + TIER_WEIGHTS["balanced"] * _episode_scores(scores, parent_ids["balanced"])
            + TIER_WEIGHTS["premium"] * _episode_scores(scores, parent_ids["premium"])
        )
        bootstrap = paired_group_bootstrap(
            ours_ep - parent_ep,
            bundle.group_keys,
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )

        lofo_official_q = {tier: float(official_lofo["tiers"][tier]["quality_score"]) for tier in TIERS}
        lofo_gain_zeroed = {
            tier: json_float(lofo_official_q[tier] - float(PARENT_F_PINS[tier]["quality"]))
            for tier in TIERS
        }
        lofo_gain_zeroed["weighted"] = json_float(
            float(official_lofo["final_score"]) - parent_weighted
        )
        lofo_quality = {tier: _score_mean(scores, lofo_models_t[tier]) for tier in TIERS}
        lofo_quality_weighted = json_float(
            weighted_final(lofo_quality["fast"], lofo_quality["balanced"], lofo_quality["premium"])
        )
        lofo_quality_gain = {
            tier: json_float(float(lofo_quality[tier]) - float(PARENT_F_PINS[tier]["quality"]))
            for tier in TIERS
        }
        lofo_quality_gain["weighted"] = json_float(float(lofo_quality_weighted) - parent_weighted)
        lofo_budget_exceeded: list[dict[str, Any]] = []
        lofo_fast_ratios: dict[str, float] = {}
        for fam_name, held in family_folds(families):
            held_idx = np.asarray(held, dtype=np.int64)
            for tier in TIERS:
                models = tuple(lofo_models_t[tier][int(item)] for item in held_idx)
                ratio = _realized_from_costs(
                    actual_l[held_idx], actual_a[held_idx], actual_k[held_idx], models
                )
                if tier == "fast":
                    lofo_fast_ratios[fam_name] = json_float(ratio)
                official = float(OFFICIAL_CAPS[tier])
                if float(ratio) > official + 1e-15:
                    lofo_budget_exceeded.append(
                        {
                            "family": fam_name,
                            "n": int(held_idx.size),
                            "official_limit": json_float(official),
                            "ratio": json_float(ratio),
                            "tier": tier,
                        }
                    )
        lofo_fast_inside_official = bool(
            all(float(value) <= float(OFFICIAL_CAPS["fast"]) + 1e-15 for value in lofo_fast_ratios.values())
        )
        lofo_fast_max = (
            json_float(max(lofo_fast_ratios.values())) if lofo_fast_ratios else None
        )
        _lofo_gain = lofo_quality_gain
        n50 = {
            fam: lofo_family_gain[fam]
            for fam in sorted(lofo_family_gain)
            if int(lofo_family_gain[fam]["n"]) < 50
        }
        n50_or_more = {
            fam: lofo_family_gain[fam]
            for fam in sorted(lofo_family_gain)
            if int(lofo_family_gain[fam]["n"]) >= 50
        }
        worst_family = (
            min(n50_or_more, key=lambda fam: n50_or_more[fam]["weighted_gain"]) if n50_or_more else None
        )
        worst_family_gain = (
            json_float(n50_or_more[worst_family]["weighted_gain"]) if worst_family else 0.0
        )

        guard_totals = {
            "all_light_fallback": 0,
            "count_cap_bind": 0,
            "denylist_bind": 0,
            "invariant_fail": 0,
            "item_cap_bind": 0,
            "kappa_k_cert_bind": 0,
            "n_lt_300_bind": 0,
            "qk_nonpositive_bind": 0,
            "self_cert_all_light": 0,
            "self_cert_shed": 0,
        }
        ruin_binding = {tier: {"n": 0, "n_ruin": 0} for tier in TIERS}
        ruin_redteam = {tier: {"n": 0, "n_ruin": 0} for tier in TIERS}
        h3_cost_fail = 0
        h2_9_fail = 0
        worst_psi = {tier: 0.0 for tier in TIERS}
        k1_inc_binding: dict[str, list[float]] = {tier: [] for tier in TIERS}
        premium_k1_views = 0
        n_bind_eval = 0
        realized_binding: list[dict[str, Any]] = []

        for index, view in enumerate(views):
            assigned, traces = _route_pack(
                view_packs[index],
                view_digests[index],
                adopted,
                k1_on=use_k1,
                limits=OPERATING_TARGETS,
            )
            layer = layers[index]
            pack = view_packs[index]
            row = {"kind": view.kind, "layer": layer, "n": int(view.index.size), "name": view.name}
            if int(sum(1 for mid in assigned["premium"] if mid == _K1)) > 0:
                premium_k1_views += 1
            for tier in TIERS:
                models = assigned[tier]
                tr = traces[tier]
                ratio = _realized_from_costs(
                    pack["actual_l"], pack["actual_a"], pack["actual_k"], models
                )
                pred_r = float(tr["pred_ratio"])
                cert = float(tr["certified_ax31"])
                official = float(OFFICIAL_CAPS[tier])
                k1_count = int(sum(1 for mid in models if mid == _K1))
                ax_count = int(sum(1 for mid in models if mid != _LIGHT))
                row[tier] = {
                    "certified_ax31": json_float(cert),
                    "f_star": json_float(tr["f_star"]),
                    "k1_count": k1_count,
                    "k_star": int(tr["k_star"]),
                    "m_star": int(tr["m_star"]),
                    "n_upgraded": ax_count,
                    "pred_k_increment": json_float(tr["pred_k_increment"]),
                    "pred_ratio": json_float(pred_r),
                    "ratio": json_float(ratio),
                }
                if bool(tr["all_light_fallback"]):
                    guard_totals["all_light_fallback"] += 1
                if bool(tr["invariant_fail"]):
                    guard_totals["invariant_fail"] += 1
                if bool(tr["self_cert_shed"]):
                    guard_totals["self_cert_shed"] += 1
                if bool(tr["self_cert_all_light"]):
                    guard_totals["self_cert_all_light"] += 1
                binds = tr.get("binds") or {}
                for key in (
                    "count_cap_bind",
                    "denylist_bind",
                    "item_cap_bind",
                    "kappa_k_cert_bind",
                    "n_lt_300_bind",
                    "qk_nonpositive_bind",
                ):
                    guard_totals[key] += int(binds.get(key, 0))
                ruin_bucket = ruin_binding if layer == "binding" else ruin_redteam
                ruin_bucket[tier]["n"] += 1
                if float(ratio) > official + 1e-15:
                    ruin_bucket[tier]["n_ruin"] += 1
                if layer == "binding":
                    n_bind_eval += 1
                    if float(ratio) > official + 1e-15 or cert > official + 1e-15:
                        h3_cost_fail += 1
                    full_cert = cert + float(adopted.kappa_k) * float(tr["pred_k_increment"])
                    if use_k1 and (
                        float(ratio) > official + 1e-15 or full_cert > official + 1e-15
                    ):
                        h2_9_fail += 1
                    k1_inc_binding[tier].append(float(tr["pred_k_increment"]))
                    if k1_count > 0:
                        psi = _realized_from_costs(
                            pack["actual_l"], pack["actual_a"], pack["actual_k"], models
                        ) - _realized_from_costs(
                            pack["actual_l"],
                            pack["actual_a"],
                            pack["actual_k"],
                            tuple(_AX31 if mid == _K1 else mid for mid in models),
                        )
                        worst_psi[tier] = max(worst_psi[tier], float(psi))
            realized_binding.append(row)

        # H2-6: held-out dual certificate from 4/5 folds and leave-one-family.
        def _kappa_from_indices(indices: Sequence[int], tier: str) -> Optional[float]:
            values = []
            for index in indices:
                pred = pred_curves[index]
                actual = actual_curves[index]
                kappa, _rho, included = kappa_cells_from_curves(pred, actual)
                for col in range(int(pred.size)):
                    if not bool(included[col]):
                        continue
                    if not in_tier_band(float(pred[col]), tier):
                        continue
                    values.append(float(kappa[col]))
            if not values:
                return None
            array = np.asarray(values, dtype=np.float64)
            if name == "candidate_kappa_max":
                return float(array.max())
            return float(quantile_higher(array, KAPPA_Q))

        def _phi_from_indices(indices: Sequence[int]) -> Optional[Tuple[float, ...]]:
            if not indices:
                return None
            curves = [actual_curves[index] for index in indices]
            phi_hat, _change = phi_from_actual_curves(curves)
            return tuple(json_floats(phi_hat))

        fold_checks: list[dict[str, Any]] = []
        family_checks: list[dict[str, Any]] = []
        for fold in range(FOLDS):
            held_name = f"oof-fold-{fold}"
            held_i = view_names.index(held_name)
            others = [
                i
                for i, view in enumerate(views)
                if view.kind == "oof-fold" and view.name != held_name
            ]
            phi_hat = _phi_from_indices(others)
            if phi_hat is None:
                continue
            for tier in TIERS:
                hat = _kappa_from_indices(others, tier)
                if hat is None:
                    continue
                local = SelectedPolicy(
                    **{
                        **adopted.__dict__,
                        "kappa_tier": {**dict(adopted.kappa_tier), tier: hat},
                        "phi_binding_q9975": phi_hat,
                    }
                )
                assigned, traces = _route_pack(
                    view_packs[held_i],
                    view_digests[held_i],
                    local,
                    k1_on=False,
                    limits=OPERATING_TARGETS,
                )
                ratio = _realized_from_costs(
                    view_packs[held_i]["actual_l"],
                    view_packs[held_i]["actual_a"],
                    view_packs[held_i]["actual_k"],
                    assigned[tier],
                )
                official = float(OFFICIAL_CAPS[tier])
                fold_checks.append(
                    {
                        "certified_ratio": json_float(traces[tier]["certified_ratio"]),
                        "exceeded": bool(float(ratio) > official + 1e-15),
                        "f_star": json_float(traces[tier]["f_star"]),
                        "held_pred_ratio": json_float(traces[tier]["pred_ratio"]),
                        "held_realized": json_float(ratio),
                        "heldout": held_name,
                        "kappa_hat": json_float(hat),
                        "tier": tier,
                    }
                )
        for fam_name in family_names:
            held_i = next(
                (i for i, view in enumerate(views) if view.name == f"lofo-{fam_name}"), None
            )
            if held_i is None:
                continue
            others = [
                i
                for i, view in enumerate(views)
                if layers[i] == "binding"
                and view.name != f"lofo-{fam_name}"
                and not view.name.startswith(f"famdom-{fam_name}-")
            ]
            phi_hat = _phi_from_indices(others)
            if phi_hat is None:
                continue
            for tier in TIERS:
                hat = _kappa_from_indices(others, tier)
                if hat is None:
                    continue
                local = SelectedPolicy(
                    **{
                        **adopted.__dict__,
                        "kappa_tier": {**dict(adopted.kappa_tier), tier: hat},
                        "phi_binding_q9975": phi_hat,
                    }
                )
                assigned, traces = _route_pack(
                    view_packs[held_i],
                    view_digests[held_i],
                    local,
                    k1_on=False,
                    limits=OPERATING_TARGETS,
                )
                ratio = _realized_from_costs(
                    view_packs[held_i]["actual_l"],
                    view_packs[held_i]["actual_a"],
                    view_packs[held_i]["actual_k"],
                    assigned[tier],
                )
                official = float(OFFICIAL_CAPS[tier])
                family_checks.append(
                    {
                        "certified_ratio": json_float(traces[tier]["certified_ratio"]),
                        "exceeded": bool(float(ratio) > official + 1e-15),
                        "f_star": json_float(traces[tier]["f_star"]),
                        "family": fam_name,
                        "held_pred_ratio": json_float(traces[tier]["pred_ratio"]),
                        "held_realized": json_float(ratio),
                        "heldout": f"lofo-{fam_name}",
                        "kappa_hat": json_float(hat),
                        "tier": tier,
                    }
                )
        all_h26 = fold_checks + family_checks
        n_h26 = int(len(all_h26))
        n_exceed = int(sum(1 for row in all_h26 if row["exceeded"]))
        exceed_rate = json_float(float(n_exceed) / float(n_h26) if n_h26 else 0.0)
        worst_held = (
            max(all_h26, key=lambda row: float(row["held_realized"])) if all_h26 else None
        )

        oof_counts = {tier: _count_models(oof_models_t[tier]) for tier in TIERS}
        oof_ratios = {tier: _realized_ratio(costs, oof_models_t[tier]) for tier in TIERS}
        per_tier = {}
        for tier in TIERS:
            tr = traces_full[tier]
            realized = oof_ratios[tier]
            pred_r = float(tr["pred_ratio"])
            per_tier[tier] = {
                "binds_term": tr.get("binds_term"),
                "certified_ax31": json_float(tr["certified_ax31"]),
                "certified_ratio": json_float(tr.get("certified_ratio", tr["certified_ax31"])),
                "empirical_term": tr.get("empirical_term"),
                "f_star": json_float(tr["f_star"]),
                "k_star": int(sum(1 for mid in oof_models_t[tier] if mid != _LIGHT)),
                "k_star_full_batch": int(tr["k_star"]),
                "kappa_term": tr.get("kappa_term"),
                "kappa_term_applied": bool(tr.get("kappa_term_applied", False)),
                "m_star": int(tr["m_star"]),
                "m_star_oof": int(sum(1 for mid in oof_models_t[tier] if mid == _K1)),
                "margin_internal": json_float(float(OPERATING_TARGETS[tier]) - float(realized)),
                "margin_official": json_float(float(OFFICIAL_CAPS[tier]) - float(realized)),
                "parent_k": int(PARENT_F_PINS[tier]["ax31_count"]),
                "pred_ratio": json_float(pred_r),
                "q_oof": json_float(official_oof_q[tier]),
                "realized_oof_ratio": json_float(realized),
                "ruin_binding": {
                    "frequency": json_float(
                        float(ruin_binding[tier]["n_ruin"]) / float(ruin_binding[tier]["n"])
                        if ruin_binding[tier]["n"]
                        else 0.0
                    ),
                    "n": int(ruin_binding[tier]["n"]),
                    "n_ruin": int(ruin_binding[tier]["n_ruin"]),
                },
                "ruin_redteam": {
                    "frequency": json_float(
                        float(ruin_redteam[tier]["n_ruin"]) / float(ruin_redteam[tier]["n"])
                        if ruin_redteam[tier]["n"]
                        else 0.0
                    ),
                    "n": int(ruin_redteam[tier]["n"]),
                    "n_ruin": int(ruin_redteam[tier]["n_ruin"]),
                },
            }
        official_variant = {}
        for tier in TIERS:
            tr = traces_official[tier]
            official_variant[tier] = {
                "binds_term": tr.get("binds_term"),
                "certified_ratio": json_float(tr.get("certified_ratio", tr["certified_ax31"])),
                "empirical_term": tr.get("empirical_term"),
                "f_star": json_float(tr["f_star"]),
                "k_star": int(tr["k_star"]),
                "kappa_term": tr.get("kappa_term"),
                "m_star": int(tr["m_star"]),
                "pred_ratio": json_float(tr["pred_ratio"]),
                "realized_full_batch": _realized_from_costs(
                    actual_l, actual_a, actual_k, assigned_full_official[tier]
                ),
            }

        gates = {
            "h2_1_oof_gain": {
                "pass": bool(oof_gain["weighted"] > 0.0),
                "per_tier": oof_gain,
                "threshold": 0.0,
                "value": oof_gain["weighted"],
            },
            "h2_2_fold_wins": {
                "fold_gains": fold_gains,
                "pass": bool(fold_wins >= 4),
                "threshold": ">= 4/5",
                "value": f"{fold_wins}/5",
            },
            "h2_3_bootstrap": {
                "pass": bool(float(bootstrap["q2_5"]) > 0.0),
                "threshold": 0.0,
                "value": bootstrap,
            },
            "h2_4_lofo": {
                "budget_component": {
                    "n_exceeded": int(len(lofo_budget_exceeded)),
                    "exceeded": lofo_budget_exceeded,
                    "fast_inside_official_1_25": bool(lofo_fast_inside_official),
                    "fast_max_ratio": lofo_fast_max,
                    "fast_per_family": lofo_fast_ratios,
                    "note": (
                        "Zeroing is counted under the cost gates, never as "
                        "quality regression. Charter §12.3 item 6."
                    ),
                    "official_zeroed_gain": lofo_gain_zeroed,
                },
                "pass": bool(lofo_quality_gain["weighted"] >= 0.0),
                "quality_component": {
                    "per_tier": lofo_quality_gain,
                    "weighted": lofo_quality_gain["weighted"],
                },
                "threshold": 0.0,
                "value": lofo_quality_gain["weighted"],
            },
            "h2_5_lofo_worst_family": {
                "n_lt_50": n50,
                "pass": bool(worst_family_gain >= H2_5_THRESHOLD),
                "threshold": json_float(H2_5_THRESHOLD),
                "value": worst_family_gain,
                "worst_family": worst_family,
            },
            "h2_6_heldout": {
                "exceedance_rate": exceed_rate,
                "n_checks": n_h26,
                "n_exceeded": n_exceed,
                "pass": bool(exceed_rate <= H2_6_EXCEEDANCE_MAX + 1e-15),
                "threshold": json_float(H2_6_EXCEEDANCE_MAX),
                "worst_held_out": worst_held,
            },
            "h2_9_k1_certificate": {
                "n_fail": int(h2_9_fail),
                "pass": bool(h2_9_fail == 0),
                "threshold": "K1 certificate holds on every binding view",
            },
            "h2_10_premium_k1": {
                "n_positive_views": int(premium_k1_views),
                "n_views": int(len(views)),
                "oof_premium_k1": int(sum(1 for mid in oof_models_t["premium"] if mid == _K1)),
                "pass": bool(sum(1 for mid in oof_models_t["premium"] if mid == _K1) > 0),
                "threshold": "Premium K1 count > 0",
            },
            "h3_cost": {
                "n_fail": int(h3_cost_fail),
                "pass": bool(h3_cost_fail == 0),
                "threshold": (
                    "certified_ratio(f*) <= official limit and every "
                    "binding view realized ratio <= official limit"
                ),
            },
        }

        return {
            "float64_vs_official": official_agree,
            "fold_checks_h2_6": fold_checks,
            "family_checks_h2_6": family_checks,
            "gates": gates,
            "guard_binds": guard_totals,
            "k1": {
                "h2_10_holds": bool(gates["h2_10_premium_k1"]["pass"]),
                "k1_enabled": {tier: bool(k1_enabled[tier]) for tier in TIERS},
                "k1_quality_vacuous": bool(k1_quality_vacuous),
                "kappa_k": json_float(kappa_k) if math.isfinite(float(kappa_k)) else None,
                "m_star": {tier: int(m_star_ref[tier]) for tier in TIERS},
                "quality_from_k1": quality_from_k1,
                "worst_realized_psi": {tier: json_float(worst_psi[tier]) for tier in TIERS},
            },
            "kappa_tier": {tier: json_float(kappa_tier[tier]) for tier in TIERS},
            "lofo_family_gain": lofo_family_gain,
            "oof_operating": {
                "model_counts": oof_counts,
                "official_per_tier_quality": official_oof_q,
                "official_weighted": json_float(float(official_oof["final_score"])),
                "parent_weighted": parent_weighted,
                "realized_ratios": oof_ratios,
                "weighted_float64": float_oof_weighted,
            },
            "official_limits_variant": official_variant,
            "per_tier": per_tier,
            "policy": adopted,
            "view_realized_head": realized_binding[:25],
        }

    candidates: dict[str, Any] = {}
    policies: dict[str, SelectedPolicy] = {}
    for cand_name in CANDIDATE_NAMES:
        evaluated = _evaluate_one(cand_name)
        policies[cand_name] = evaluated.pop("policy")
        candidates[cand_name] = evaluated

    observed = {
        "adoption_deferred_to_operator": True,
        "candidates": candidates,
        "dev_opened": False,
        "kappa": kappa_block,
        "kappa_cells": {
            "excluded": int(n_kappa_excluded),
            "included": int(n_kappa_included),
            "min_increment": json_float(KAPPA_MIN_INCREMENT),
        },
        "kappa_k": {
            "available": bool(kappa_k_available),
            "binding_n_cells": int(k_array.size),
            "summary": kappa_k_summary,
        },
        "phi_binding_q9975": {
            "at_f": phi_at_report,
            "running_maximum": phi_change,
            "values": json_floats(phi_binding),
        },
        "predicted_ratio_monotone_all_views": bool(all(monotone_flags)),
        "psi_binding_q9975": {
            "n": int(psi_binding.size),
            "running_maximum": psi_change,
            "values": json_floats(psi_binding),
        },
        "q_a_used": False,
        "quality_entered_calibration_selection": False,
        "quality_entered_cost_selection": False,
        "recalibration": {
            "grid": recal_grid_rows,
            "selected": selected_recal_public,
        },
        "view_layers": {
            "binding": int(len(binding_idx)),
            "red_team": int(len(redteam_idx)),
            "binding_kinds": {
                kind: int(sum(1 for i in binding_idx if views[i].kind == kind))
                for kind in ("oof-fold", "lofo", "famdom", "dirichlet", "half", "small")
            },
            "redteam_kinds": {
                kind: int(sum(1 for i in redteam_idx if views[i].kind == kind))
                for kind in ("lofo-combined", "small")
            },
        },
    }
    diagnostic = {
        "famdom_fallback_by_family": catalogue["famdom_fallback_by_family"],
        "float64_table_note": FLOAT64_NOTE,
        "imported_modeling_symbols": [
            "family_folds",
            "feature_matrix",
            "group_folds",
            "load_train",
            "official_score",
            "oof_predict",
            "paired_group_bootstrap",
            "quantile_higher",
            "ridge_predict",
        ],
        "imported_cost_cert_symbols": [
            "FittedHeads",
            "actual_increments",
            "fit_heads",
            "floor_inc",
            "oof_incremental_costs",
            "predict_heads",
        ],
        "legacy_oof_recal_ratios": {
            "inc_A": json_float(ratio_a),
            "inc_K": json_float(ratio_k),
        },
        "imported_prefix_cert_symbols": [
            "build_views",
            "phi_view",
            "prefix_k",
            "sort_pred_inc",
            "order_k1",
        ],
        "n_views": int(len(views)),
        "oof_recal_ratios_legacy": {
            "inc_A": json_float(ratio_a),
            "inc_K": json_float(ratio_k),
        },
        "predicted_ratio_monotone_violations": int(sum(1 for flag in monotone_flags if not flag)),
        "qk_head": {
            "alpha": json_float(qk_head.alpha),
            "bins": int(qk_head.bins),
            "feature_signature": qk_head.feature_signature,
            "n_positive_oof": int((oof_qk > 0.0).sum()),
            "target_form": qk_head.target_form,
        },
        "skipped_views": catalogue["skipped"],
        "view_kind_counts": catalogue["view_kind_counts"],
    }
    return {
        "decision": decide(measurement_ok=True),
        "diagnostic": diagnostic,
        "observed": observed,
        "policies": policies,
    }


__all__ = (
    "BOOTSTRAP_SEED",
    "CANDIDATE_NAMES",
    "SelectedPolicy",
    "DECISION_TWO",
    "EXPERIMENT",
    "F_GRID_ARRAY",
    "K1_COUNT_CAP_FRAC",
    "K1_DENYLIST",
    "K1_ITEM_CAP_FRAC",
    "K1_MIN_N",
    "KAPPA_MIN_INCREMENT",
    "LOCKED_ALPHA",
    "LOCKED_BINS",
    "LOCKED_VARIANT",
    "OPERATING_TARGETS",
    "View",
    "allocate",
    "allocate_from_arrays",
    "assemble_report",
    "conservative_inc_k",
    "decide",
    "family_of_text",
    "fit_and_evaluate",
    "k1_mask",
    "kappa_cells_from_curves",
    "load_train",
    "locked_record",
    "predicted_ratio_curve",
    "prefix_k",
    "reject_dev_reference",
    "certified_ratio_curve",
    "fit_prefix_recal",
    "select_f_star_certified",
    "select_f_star_kappa",
    "sort_pred_inc",
    "summarize_values",
    "view_layer",
)
