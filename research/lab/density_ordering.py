# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the density ordering layer H5 selected candidate: density ordering + dual certificate.

Reuses the prefix recalibration layer view construction, prefix sweeps, kappa estimation, official
scorer path, and recalibration fitters. Changes only what charter §14.4
specifies. K1 is off in every tier. Dev is never opened.

Required upstream changes (NOT applied; local shims only):
1. ``research.lab.prompt_features.ALLOWED_HASH_BINS`` / ``_require_bins`` must
   allow ``bins=64`` so the hash-64 quality block can share the module.
2. ``research.lab.modeling.feature_matrix`` / ``research.lab.prefix_certificates.design_matrix_g_features``
   must accept a structural-only (zero hash columns) layout.
3. ``research.lab.prefix_recalibration.fit_prefix_recal`` must accept a caller-supplied order
   so prefix-curve recal can follow density rather than cheapest-cost.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.lab.prompt_features import (
    FEATURE_VERSION,
    STRUCTURAL_FEATURE_NAMES,
    _SIGN_BIT,
    _fnv1a32,
    _hash_terms,
    _tokenize,
    feature_row,
    feature_signature,
    structural_features,
)
from ossp_router.protocol import MODEL_IDS, TIERS, Episode
from research.lab.modeling import (
    FOLD_SEED,
    FOLDS,
    HASH_BINS,
    INTERCEPT_POLICY,
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
    reject_dev_reference,
    ridge_fit,
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
    PARENT_F_EXACT,
    PARENT_F_PINS,
    View,
    _count_models,
    _episode_scores,
    _gather,
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
    models_from_masks,
    phi_view,
    prefix_k,
    prefix_mask,
)
from research.lab.prefix_recalibration import (
    KAPPA_MIN_INCREMENT,
    KAPPA_Q,
    H2_5_THRESHOLD,
    H2_6_EXCEEDANCE_MAX,
    LOCKED_ALPHA,
    LOCKED_BINS,
    LOCKED_INC_A,
    LOCKED_INC_K,
    LOCKED_RATIO_ATOL,
    LOCKED_VARIANT,
    RECAL_BINS_LEGACY,
    RECAL_CLIP_LEGACY,
    TIER_PRED_BANDS,
    apply_lofo_recal,
    apply_recal_params,
    binding_term_at,
    cache_lofo_raw,
    certified_ratio_curve,
    clip_binds,
    fit_recal_params,
    in_tier_band,
    kappa_cells_from_curves,
    phi_from_actual_curves,
    predicted_ratio_curve,
    summarize_values,
    tighter_clip_key,
)
from research.lab.quality_heads import (
    DecimalQuality,
    allocate_two_action,
    models_two_action,
    pearson,
    spearman,
)


EXPERIMENT = "the density ordering layer"
REPORT_TYPE = "scrooge-density-h5-selected-candidate-v1"
SCHEMA_VERSION = 1
DECISION_ARM_A = "record-density-arm-a-promote"
DECISION_ARM_B = "record-density-arm-b-promote"
DECISION_ROLLBACK = "record-density-close-rollback-ladder"
FOLD_SEED_DENSITY = FOLD_SEED  # 2026082202
BOOTSTRAP_SEED = 2026082203
BOOTSTRAP_DRAWS = 1000
QA_REFERENCE_CAPS: Tuple[str, ...] = ("1.05", "1.15")
FEATURE_BLOCKS: Tuple[str, ...] = (
    "structural_only",
    "structural+hash64",
    "structural+hash256",
)
QUALITY_ALPHAS: Tuple[float, ...] = (10.0, 30.0, 100.0, 300.0)
TARGET_FORM = "direct_signed"
RECAL_GRID_N_BINS: Tuple[int, ...] = (10, 20, 40)
RECAL_GRID_CLIPS: Tuple[Tuple[float, float], ...] = ((0.5, 6.0), (0.25, 200.0))
RECAL_FIT_KINDS: Tuple[str, ...] = ("aggregate", "prefix")
RECAL_LIGHT_FRAC = 0.10
# Absolute floor on the density denominator (recalibrated predicted inc_A).
# A floor, not a ratio-of-sums: each episode is guarded independently so a
# near-zero increment cannot produce a 1e15-scale density (the cost certificate layer trap).
DENSITY_INC_FLOOR = 1e-6
SORT_RULE = "sortA_density_uplift_over_inc"
ARM_NAMES: Tuple[str, ...] = ("arm_a", "arm_b")
_LIGHT = "ax31-light"
_AX31 = "ax31"
_K1 = "axk1-think"
QUALITY_SPEARMAN = 0.0014
QUALITY_SIGN_ACC = 0.6538
PREFIX_RECAL_WEIGHTED = 0.639460

# Local shim: research.lab.prompt_features.has no bins=64. Required change
# (NOT applied): add 64 to ALLOWED_HASH_BINS. Workaround: identical
# signed-FNV tokenization folded into 64 buckets below.

COST_HEAD_LOCK = (
    "Reuse the cost certificate layer's mechanically selected cost head: variant "
    "direct_log1p_inc, alpha=300.0, bins=512. Folds seed 2026082202. "
    "Assert the head reproduces OOF post-recalibration aggregate ratios "
    "inc_A = 1.029434 / inc_K = 1.002749 under the legacy "
    "(n_bins=10, clip=(0.5, 6.0)) aggregate-fit so the pipeline is "
    "unchanged. Recalibration is then re-selected on the H5 grid. "
    "Charter §10.4 / §14.4."
)
RECAL_SELECTION_CRITERION = (
    "Grid, pre-registered, exactly six cells: n_bins in (10, 20, 40) "
    "x clip in ((0.5, 6.0), (0.25, 200.0)). For each cell report both "
    "the standard aggregate-ratio fit and the prefix-curve fit. "
    "Selection criterion, pre-registered and mechanical: restrict to "
    "views whose light_total is at least 10% of the global median "
    "light_total, then minimize q99 across those views of "
    "max_f |predicted_ratio(f) - actual_ratio(f)|. Tie-break: smaller "
    "n_bins, then the tighter clip, then aggregate before prefix. "
    "quality_entered_calibration_selection: false. Charter §14.4."
)
PREFIX_FIT_SPEC = (
    "Choose the monotone non-increasing (PAV) per-bin factors that "
    "minimize the squared error of the CUMULATIVE prefix ratio curve "
    "predicted_ratio(f) against actual_ratio(f) over the pinned "
    "101-point f grid, aggregated over the OOF folds, with f indexing "
    "a density-ordered prefix. Keep PAV non-increasing and keep the clip."
)
H5_HYPOTHESIS = (
    "Working ordering (uplift/cost density) plus the verified dual "
    "certificate beats parent. certified_ratio(f) = max( "
    "1 + kappa_tier * (predicted_ratio(f) - 1), Phi_binding_q9975(f) ). "
    "f*_tier = argmax_f Q_oof(f) subject to certified_ratio(f) <= "
    "limit_tier; ties resolved toward the smaller f. "
    "Phi_binding_q9975(f) is the q99.75 of the ACTUAL realized ratio at "
    "prefix fraction f across the binding view layer. Measure kappa "
    "only where predicted_ratio(f) - 1 >= 0.01; at runtime, where "
    "predicted_ratio(f) - 1 < 0.01, drop the kappa term and use the "
    "empirical term alone. Risk standard kappa_q9975. Charter §13.1, "
    "§13.3, §14.4."
)
REOPENING_RATIONALE = (
    "Section 10.5 is reopened narrowly because the quality head study's grid always "
    "stacked 256 or 512 hash features on 53 structural ones and never "
    "tested structural-only, while the frozen the sparse uplift study head that works "
    "(Train OOF +0.02348, fold wins 5/5) is structural-only with "
    "alpha=100. The reopening is strictly toward FEWER hash features. "
    "Charter §14.3."
)
BINDING_LAYER_REASON = (
    "Single-family 100% batches (lofo-{family}, lofo-combined) and "
    "n=100 are red-team because the official evaluation builds batches "
    "with the same harness as Train/Dev; a batch that is 100% one "
    "family after that family was left out of training is a "
    "composition the harness cannot produce. This split is fixed by "
    "design, not by result. Charter §14.4."
)
BINDING_LAYER_SPEC = (
    "binding: oof-fold 5, famdom-{family} 75% 2000 draws "
    "(seed 2026082204), dirichlet 1000 (seed 2026082205), half 20 "
    "(seed 2026082206), small-{300,880} 200 each (seed 2026082207)."
)
REDTEAM_LAYER_SPEC = (
    "red-team: lofo-{family} single-family batches 10, "
    "lofo-combined 10, small-100 200. Measured and reported in full, "
    "never used to set kappa, Phi_binding_q9975, or any cost gate."
)
DENSITY_ORDER_SPEC = (
    "Descending predicted_uplift_A / max(recalibrated_predicted_inc_A, "
    "DENSITY_INC_FLOOR=1e-6). Episodes with predicted uplift <= 0 are "
    "never upgraded (sorted to the tail with density -inf; prefix k is "
    "clamped to the positive-uplift count). Final tie-break is the "
    "SHA-256 content digest. Episode id, split, and input order are "
    "never keys. Charter §14.4."
)
F_STAR_RULE = (
    "f*_tier = argmax_f Q_oof(f) subject to certified_ratio(f) <= "
    "limit_tier; ties resolved toward the smaller f. Primary limits "
    "are the internal operating targets Fast 1.15 / Balanced 1.80 / "
    "Premium 3.40; also report the official-limit variant "
    "1.25 / 2.0 / 4.0. Charter §13.3 / §14.4."
)
H5_GATE_SPEC = (
    "H2-1 weighted Train OOF gain > 0; H2-2 fold wins >= 4/5; "
    "H2-3 prompt-group paired bootstrap q2.5 > 0 (draws 1000, seed "
    "2026082203); H2-4 split reporting: quality component no-regression, "
    "budget component reported separately (zeroing counted under cost "
    "gates, never as quality regression); H2-5 LOFO worst-family "
    "quality gain >= -0.010 (n<50 families reported separately); "
    "H2-6 held-out exceedance <= 0.01 on the binding layer with the "
    "worst held-out realized ratio; H3-cost every binding view within "
    "the official limit, with per-tier ruin count and frequency on "
    "binding and red-team. Charter §14.4."
)
QUALITY_LOFO_SPEC = (
    "Hold family f out of the quality-head fit, route f's episodes as "
    "part of a normal MIXED batch, and measure quality only. Budget "
    "zeroing is counted under the cost gates, never as quality "
    "regression. Arm B cannot LOFO-refit the frozen the sparse uplift study head; its "
    "quality LOFO uses the frozen full-fit predictions and is so "
    "labelled. Charter §14.4."
)
ARM_A_SPEC = (
    "Pre-registered grid, exactly 12 configs, no extensions: "
    "feature_block in (structural_only, structural+hash64, "
    "structural+hash256) x alpha in (10.0, 30.0, 100.0, 300.0). "
    "Target form direct_signed only. Prompt-group 5-fold OOF, fold "
    "seed 2026082202, plus one full-fit. Selection criterion, identical "
    "to the quality head study: exact-cost greedy allocation quality, ranking by "
    "predicted uplift divided by exact incremental cost, upgrading "
    "while the exact realized ratio stays within the reference cap; "
    "config score = mean of the Train OOF qualities at reference caps "
    "1.05 and 1.15, scored with the OFFICIAL Decimal scorer. Episodes "
    "with predicted uplift <= 0 are never upgraded. Tie-break: fewer "
    "features, then lower alpha. Charter §14.4."
)
ARM_B_SPEC = (
    "Reuse the frozen the sparse uplift study/the calibrated base quality head's ORDERING exactly as the "
    "frozen routers compute it. Import ossp_router.feasibility_ladder / "
    "final_router READ-ONLY and extract per-episode predicted AX31 "
    "uplift via final_router._quality_prediction. Do not reimplement "
    "or refit. Arm B then applies the SAME dual certificate and the "
    "SAME f* rule as Arm A. Charter §14.4."
)
FLOAT64_NOTE = (
    FLOAT64_TABLE_NOTE
    + " the density ordering layer predicted_ratio(f) is the same cumsum construction on "
    "recalibrated predicted increments along a density-ordered prefix. "
    "Official Decimal is used only at selected OOF/LOFO operating "
    "points and H2-1..H2-5 gate numbers."
)

_FROZEN_ARTIFACT: Any = None


def view_layer(view: View) -> str:
    """Pre-registered binding / red-team split. Design property, not a result."""

    if view.kind in ("lofo", "lofo-combined"):
        return "red-team"
    if view.kind == "small" and str(view.name).startswith("small-100-"):
        return "red-team"
    return "binding"


def hash_features_64(text: str) -> dict[int, float]:
    """Local shim: signed-FNV unigram/bigram hashes folded into 64 buckets.

    Arithmetic is identical to ``g_features._hash_from`` except the
    width is 64 and ``_require_bins`` is not called. Required upstream
    change (NOT applied): allow ``bins=64`` in
    ``g_features.ALLOWED_HASH_BINS``.
    """

    tokens = _tokenize(text)
    width = 64
    mask = width - 1
    counts = [0] * width
    for term in _hash_terms(tokens):
        digest = _fnv1a32(term)
        bucket = digest & mask
        counts[bucket] += -1 if digest & _SIGN_BIT else 1
    hashed: dict[int, float] = {}
    for bucket, count in enumerate(counts):
        if count == 0:
            continue
        hashed[bucket] = math.copysign(math.log1p(abs(count)), count)
    return hashed


def quality_feature_signature(block: str) -> str:
    """SHA-256 pin. Local shim for blocks that g_features.feature_signature rejects."""

    if block == "structural+hash256":
        return feature_signature(256)
    if block == "structural_only":
        width_token = "0"
    elif block == "structural+hash64":
        width_token = "64"
    else:
        raise ValueError(f"unknown quality feature block {block!r}")
    payload = "\n".join((FEATURE_VERSION, width_token, *STRUCTURAL_FEATURE_NAMES, ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quality_n_features(block: str) -> int:
    n_struct = len(STRUCTURAL_FEATURE_NAMES)
    if block == "structural_only":
        return n_struct
    if block == "structural+hash64":
        return n_struct + 64
    if block == "structural+hash256":
        return n_struct + 256
    raise ValueError(f"unknown quality feature block {block!r}")


def quality_design_matrix(texts: Sequence[str], block: str) -> np.ndarray:
    """``[intercept | structural | optional hash]``. No call to feature_matrix for 0/64."""

    n_struct = len(STRUCTURAL_FEATURE_NAMES)
    extra = {"structural_only": 0, "structural+hash64": 64, "structural+hash256": 256}[block]
    matrix = np.zeros((len(texts), 1 + n_struct + extra), dtype=np.float64)
    matrix[:, 0] = 1.0
    hash_start = 1 + n_struct
    for row, text in enumerate(texts):
        if block == "structural+hash256":
            structural, hashed = feature_row(text, bins=256)
        else:
            structural = structural_features(text)
            hashed = hash_features_64(text) if extra == 64 else {}
        matrix[row, 1:hash_start] = structural
        for bucket in sorted(hashed):
            matrix[row, hash_start + int(bucket)] = float(hashed[bucket])
    return matrix


def frozen_artifact() -> Any:
    global _FROZEN_ARTIFACT
    if _FROZEN_ARTIFACT is None:
        from ossp_router.feasibility_ladder import load_bundled_artifact

        _FROZEN_ARTIFACT = load_bundled_artifact()
    return _FROZEN_ARTIFACT


def extract_frozen_uplift(episodes: Sequence[Episode]) -> Tuple[np.ndarray, dict[str, Any]]:
    """Read-only extraction of the frozen the sparse uplift study/the feasibility ladder AX31 uplift head.

    Drives ``final_router._quality_prediction`` on the bundled the feasibility ladder
    artifact. Does not reimplement or refit the head.
    """

    from ossp_router.cost_calibrated_router import _quality_prediction

    artifact = frozen_artifact()
    value = artifact.value
    quality = value.get("quality") if isinstance(value, Mapping) else None
    if not isinstance(quality, Mapping) or "coefficients" not in quality:
        raise RuntimeError(
            "frozen the sparse uplift study/the feasibility ladder quality head blocker: bundled artifact "
            "has no quality.coefficients; cannot extract without "
            "touching frozen code"
        )
    predicted = np.empty(len(episodes), dtype=np.float64)
    for index, episode in enumerate(episodes):
        predicted[index] = float(_quality_prediction(episode, value))
    meta = {
        "alpha": quality.get("alpha"),
        "extracted": True,
        "n_coefficients": int(len(quality["coefficients"])),
        "n_episodes": int(predicted.size),
        "source": "ossp_router.cost_calibrated_router._quality_prediction",
        "via": "ossp_router.feasibility_ladder.load_bundled_artifact",
    }
    return predicted, meta


def texts_to_episodes(texts: Sequence[str]) -> Tuple[Episode, ...]:
    return tuple(Episode(episode_id=f"density-route-{index:04d}", prompt=text) for index, text in enumerate(texts))


def sort_density(
    uplift: np.ndarray,
    inc_a: np.ndarray,
    digests: Sequence[str],
    *,
    floor: float = DENSITY_INC_FLOOR,
) -> np.ndarray:
    """Descending predicted_uplift / max(inc, floor); digest final tie-break.

    Non-positive predicted uplift is assigned density -inf so those
    episodes sort to the tail and are never taken by a prefix clamped
    to the positive-uplift count.
    """

    pred = np.asarray(uplift, dtype=np.float64).reshape(-1)
    increment = np.asarray(inc_a, dtype=np.float64).reshape(-1)
    keys = np.asarray(list(digests))
    if not (pred.size == increment.size == keys.size):
        raise ValueError("sort_density requires aligned uplift, inc_A, and digests")
    if pred.size == 0:
        return np.zeros(0, dtype=np.int64)
    denom = np.maximum(increment, float(floor))
    density = np.where(pred > 0.0, pred / denom, -np.inf)
    return np.lexsort((keys, -density))


def n_positive_uplift(uplift: np.ndarray) -> int:
    return int(np.sum(np.asarray(uplift, dtype=np.float64).reshape(-1) > 0.0))


def prefix_k_eligible(fraction: float, n: int, n_eligible: int) -> int:
    return int(min(prefix_k(fraction, n), int(n_eligible), int(n)))


def quality_prefix_curve(
    score_l: np.ndarray,
    score_a: np.ndarray,
    order: np.ndarray,
    n_eligible: int,
    grid: np.ndarray = F_GRID_ARRAY,
) -> np.ndarray:
    """Mean score of the density-ordered eligible prefix. Vectorized cumsum."""

    light = np.asarray(score_l, dtype=np.float64).reshape(-1)
    ax31 = np.asarray(score_a, dtype=np.float64).reshape(-1)
    ranked = np.asarray(order, dtype=np.int64).reshape(-1)
    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    n_rows = int(light.size)
    if n_rows == 0:
        return np.zeros(knots.size, dtype=np.float64)
    gain = (ax31 - light)[ranked]
    cap = min(int(n_eligible), n_rows)
    if cap < n_rows:
        gain[cap:] = 0.0
    cumulative = np.cumsum(gain)
    base = float(light.sum())
    ks = np.minimum(
        np.clip(np.floor(knots * float(n_rows) + 1e-15).astype(np.int64), 0, n_rows),
        cap,
    )
    quality = np.full(knots.size, base / float(n_rows), dtype=np.float64)
    nonempty = ks > 0
    quality[nonempty] = (base + cumulative[ks[nonempty] - 1]) / float(n_rows)
    return quality


def select_f_star_quality(
    q_oof: np.ndarray,
    certified: np.ndarray,
    limit: float,
    grid: np.ndarray = F_GRID_ARRAY,
) -> float:
    """f* = argmax_f Q_oof(f) s.t. certified_ratio(f) <= limit; ties → smaller f."""

    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    quality = np.asarray(q_oof, dtype=np.float64).reshape(-1)
    cert = np.asarray(certified, dtype=np.float64).reshape(-1)
    if not (quality.size == cert.size == knots.size):
        raise ValueError("select_f_star_quality requires aligned curves")
    feasible = (cert <= float(limit) + 1e-15) & (knots <= 1.0 + 1e-15)
    if not np.any(feasible):
        return 0.0
    masked = np.where(feasible, quality, -np.inf)
    best = float(np.max(masked))
    winners = np.flatnonzero(np.abs(masked - best) <= 1e-15)
    return json_float(knots[int(winners[0])])


def recal_view_eligible(
    light_totals: np.ndarray,
    layers: Sequence[str],
    *,
    min_frac: float = RECAL_LIGHT_FRAC,
) -> Tuple[np.ndarray, float, float]:
    """Binding views whose light_total is at least min_frac of the global median."""

    totals = np.asarray(light_totals, dtype=np.float64).reshape(-1)
    if totals.size == 0:
        return np.zeros(0, dtype=bool), 0.0, 0.0
    median = float(np.median(totals))
    threshold = float(min_frac) * median
    binding = np.array([layer == "binding" for layer in layers], dtype=bool)
    eligible = binding & (totals >= threshold - 1e-15)
    return eligible, json_float(median), json_float(threshold)


def fit_prefix_recal_density(
    pred_inc: np.ndarray,
    actual_inc: np.ndarray,
    pred_light: np.ndarray,
    actual_light: np.ndarray,
    actual_ax31: np.ndarray,
    uplift: np.ndarray,
    fold_ids: np.ndarray,
    digests: Sequence[str],
    *,
    n_bins: int,
    clip: Tuple[float, float],
    floor: float = DENSITY_INC_FLOOR,
) -> RankRecal:
    """Prefix-curve PAV recalibration on a density-ordered prefix.

    Local shim over research.lab.prefix_recalibration.fit_prefix_recal (hardcoded cheapest
    order). Required change (NOT applied): parameterize the order.
    """

    predicted = floor_inc(pred_inc)
    actual = np.asarray(actual_inc, dtype=np.float64).reshape(-1)
    pred_u = np.asarray(uplift, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape or predicted.shape != pred_u.shape:
        raise ValueError("fit_prefix_recal_density requires aligned inputs")
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
        local_order = sort_density(pred_u[idx], local_pred, local_digests, floor=floor)
        n_local = int(idx.size)
        n_elig = n_positive_uplift(pred_u[idx])
        ordered_bins = bin_index[idx][local_order]
        ordered_pred = local_pred[local_order]
        if n_elig < n_local:
            ordered_pred = ordered_pred.copy()
            ordered_pred[n_elig:] = 0.0
        onehot = np.zeros((n_local, bins), dtype=np.float64)
        onehot[np.arange(n_local), ordered_bins] = ordered_pred
        cum_mass = np.cumsum(onehot, axis=0)
        actual_curve = phi_view(a_light[idx], a_ax31[idx], local_order)
        ks = np.minimum(
            np.clip(np.floor(F_GRID_ARRAY * float(n_local) + 1e-15).astype(np.int64), 0, n_local),
            n_elig,
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


def correlation_block(pred: np.ndarray, target: np.ndarray, *, unequal_mask: np.ndarray) -> dict[str, Any]:
    pred_pos = np.asarray(pred, dtype=np.float64) > 0.0
    actual_pos = np.asarray(target, dtype=np.float64) > 0.0
    unequal_n = int(np.count_nonzero(unequal_mask))
    if unequal_n:
        sign_accuracy = json_float(float(np.mean(pred_pos[unequal_mask] == actual_pos[unequal_mask])))
    else:
        sign_accuracy = None
    return {
        "n_unequal": unequal_n,
        "pearson_target": None if pearson(pred, target) is None else json_float(pearson(pred, target)),
        "sign_accuracy_unequal": sign_accuracy,
        "spearman_target": None if spearman(pred, target) is None else json_float(spearman(pred, target)),
    }


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


def _curve_gaps(pred: np.ndarray, actual: np.ndarray, fractions: Sequence[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for frac in fractions:
        col = int(round(float(frac) * 100.0))
        out[f"{frac:.2f}"] = json_float(float(pred[col] - actual[col]))
    return out


@dataclass(frozen=True)
class SelectedPolicy:
    """Deployable H5 allocator. route() uses g_features plus documented shims."""

    feature_version: str
    feature_signature: str
    bins: int
    alpha: float
    variant: str
    ridge_coefficients: Mapping[str, Tuple[float, ...]]
    smearing_factors: Mapping[str, float]
    recal_a_edges: Tuple[float, ...]
    recal_a_factors: Tuple[float, ...]
    quality_source: str
    quality_block: str
    quality_alpha: float
    quality_target_form: str
    quality_feature_signature: str
    quality_coefficients: Tuple[float, ...]
    kappa_tier: Mapping[str, float]
    kappa_min_increment: float
    phi_binding_q9975: Tuple[float, ...]
    recal_n_bins: int
    recal_clip: Tuple[float, float]
    recal_fit: str
    sort_rule: str
    density_inc_floor: float
    f_star: Mapping[str, float]
    operating_targets: Mapping[str, float]
    official_caps: Mapping[str, float]
    k1_enabled: Mapping[str, bool]
    f_grid: Tuple[float, ...]
    intercept_policy: str
    arm_name: str

    def to_dict(self) -> dict[str, Any]:
        return sort_mapping(
            {
                "alpha": json_float(self.alpha),
                "arm_name": self.arm_name,
                "bins": int(self.bins),
                "density_inc_floor": json_float(self.density_inc_floor),
                "density_ordering_rule": DENSITY_ORDER_SPEC,
                "f_grid": json_floats(self.f_grid),
                "f_star": {tier: json_float(self.f_star[tier]) for tier in TIERS},
                "feature_signature": self.feature_signature,
                "feature_version": self.feature_version,
                "intercept_policy": self.intercept_policy,
                "k1_enabled": {tier: bool(self.k1_enabled[tier]) for tier in TIERS},
                "kappa_min_increment": json_float(self.kappa_min_increment),
                "kappa_tier": {tier: json_float(self.kappa_tier[tier]) for tier in TIERS},
                "official_caps": {tier: json_float(self.official_caps[tier]) for tier in TIERS},
                "operating_targets": {
                    tier: json_float(self.operating_targets[tier]) for tier in TIERS
                },
                "phi_binding_q9975": json_floats(self.phi_binding_q9975),
                "quality_alpha": json_float(self.quality_alpha),
                "quality_block": self.quality_block,
                "quality_coefficients": json_floats(self.quality_coefficients),
                "quality_feature_signature": self.quality_feature_signature,
                "quality_source": self.quality_source,
                "quality_target_form": self.quality_target_form,
                "recal_a_edges": json_floats(self.recal_a_edges),
                "recal_a_factors": json_floats(self.recal_a_factors),
                "recal_clip": [
                    json_float(self.recal_clip[0]),
                    json_float(self.recal_clip[1]),
                ],
                "recal_fit": self.recal_fit,
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
            raise ValueError("serialized cost bins are outside the the modeling foundation closed list")
        if payload["feature_signature"] != feature_signature(bins):
            raise ValueError("feature signature mismatch")
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError("feature_version mismatch")
        block = str(payload["quality_block"])
        if block != "frozen_sparse_uplift" and payload["quality_feature_signature"] != quality_feature_signature(
            block
        ):
            raise ValueError("quality feature signature mismatch")
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
            quality_source=str(payload["quality_source"]),
            quality_block=block,
            quality_alpha=float(payload["quality_alpha"]),
            quality_target_form=str(payload["quality_target_form"]),
            quality_feature_signature=str(payload["quality_feature_signature"]),
            quality_coefficients=tuple(float(item) for item in payload["quality_coefficients"]),
            kappa_tier={tier: float(payload["kappa_tier"][tier]) for tier in TIERS},
            kappa_min_increment=float(payload["kappa_min_increment"]),
            phi_binding_q9975=tuple(float(item) for item in payload["phi_binding_q9975"]),
            recal_n_bins=int(payload["recal_n_bins"]),
            recal_clip=(
                float(payload["recal_clip"][0]),
                float(payload["recal_clip"][1]),
            ),
            recal_fit=str(payload["recal_fit"]),
            sort_rule=str(payload["sort_rule"]),
            density_inc_floor=float(payload["density_inc_floor"]),
            f_star={tier: float(payload["f_star"][tier]) for tier in TIERS},
            operating_targets={tier: float(payload["operating_targets"][tier]) for tier in TIERS},
            official_caps={tier: float(payload["official_caps"][tier]) for tier in TIERS},
            k1_enabled={tier: bool(payload["k1_enabled"][tier]) for tier in TIERS},
            f_grid=tuple(float(item) for item in payload["f_grid"]),
            intercept_policy=str(payload["intercept_policy"]),
            arm_name=str(payload["arm_name"]),
        )

    def predict_uplift(self, texts: Sequence[str]) -> np.ndarray:
        if self.quality_source == "frozen_sparse_uplift_ladder":
            episodes = texts_to_episodes(texts)
            predicted, _meta = extract_frozen_uplift(episodes)
            return predicted
        features = quality_design_matrix(texts, self.quality_block)
        coef = np.asarray(self.quality_coefficients, dtype=np.float64)
        if coef.size == 0:
            return np.zeros(len(texts), dtype=np.float64)
        return ridge_predict(coef, features)

    def predict_arrays(self, texts: Sequence[str]) -> dict[str, Any]:
        features = design_matrix_g_features(texts, bins=int(self.bins))
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
        pred_l, _pred_a, _pred_k, inc_a, _inc_k = predict_heads(features, heads)
        recal_a = apply_recal_baked(
            inc_a,
            np.asarray(self.recal_a_edges, dtype=np.float64),
            np.asarray(self.recal_a_factors, dtype=np.float64),
        )
        return {
            "digests": content_digests(texts),
            "families": tuple(family_of_text(text) for text in texts),
            "inc_a": recal_a,
            "pred_l": pred_l,
            "uplift": self.predict_uplift(texts),
        }

    def allocate(self, tier: str, texts: Sequence[str]) -> Tuple[str, ...]:
        predicted = self.predict_arrays(texts)
        return allocate_from_arrays(
            tier,
            predicted["uplift"],
            predicted["inc_a"],
            predicted["pred_l"],
            predicted["digests"],
            self,
        )

    def route(self, tier: str, texts: Sequence[str]) -> Tuple[str, ...]:
        """Runtime path: cost features through g_features; quality via block or frozen shim."""

        return self.allocate(tier, texts)


def allocate_from_arrays(
    tier: str,
    uplift: np.ndarray,
    inc_a: np.ndarray,
    pred_l: np.ndarray,
    digests: Sequence[str],
    policy: SelectedPolicy,
    *,
    force_invariant_fail: bool = False,
    limit_override: Optional[float] = None,
    f_star_override: Optional[float] = None,
    trace: Optional[dict[str, Any]] = None,
) -> Tuple[str, ...]:
    """Baked f* on a density-ordered prefix, dual certificate, self-cert shed."""

    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    pred_u = np.asarray(uplift, dtype=np.float64).reshape(-1)
    pred_inc_a = np.asarray(inc_a, dtype=np.float64).reshape(-1)
    light = np.asarray(pred_l, dtype=np.float64).reshape(-1)
    n_rows = int(pred_inc_a.size)
    empty_trace = {
        "all_light_fallback": False,
        "binds_term": "empirical",
        "certified_ratio": 1.0,
        "empirical_term": 1.0,
        "f_star": 0.0,
        "invariant_fail": False,
        "k_star": 0,
        "kappa_term": None,
        "kappa_term_applied": False,
        "pred_ratio": 1.0,
        "self_cert_all_light": False,
        "self_cert_shed": False,
    }
    if n_rows == 0:
        if trace is not None:
            trace.update(empty_trace)
        return tuple()
    if not (pred_u.size == light.size == len(digests) == n_rows):
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

    finite = np.all(np.isfinite(pred_inc_a)) and np.all(np.isfinite(light)) and np.all(np.isfinite(pred_u))
    light_sum = float(light.sum())
    kappa = float(policy.kappa_tier[tier])
    if force_invariant_fail or not finite or light_sum <= 0.0 or not math.isfinite(kappa):
        return _all_light(invariant=True, shed=False)
    if any(bool(policy.k1_enabled.get(name, False)) for name in TIERS):
        return _all_light(invariant=True, shed=False)

    order = sort_density(
        pred_u, pred_inc_a, digests, floor=float(policy.density_inc_floor)
    )
    n_elig = n_positive_uplift(pred_u)
    grid = np.asarray(policy.f_grid, dtype=np.float64)
    pred_ratio = predicted_ratio_curve(light, pred_inc_a, order, grid)
    if n_elig < n_rows:
        # Increments of non-positive-uplift tail must not enter the curve.
        pred_ratio = predicted_ratio_curve(
            light,
            np.where(pred_u > 0.0, pred_inc_a, 0.0),
            order,
            grid,
        )
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
    certified = certified_ratio_curve(pred_ratio, kappa, phi, min_increment=min_inc)
    baked = float(policy.f_star[tier]) if f_star_override is None else float(f_star_override)
    if baked < 0.0:
        baked = 0.0
    if baked > 1.0:
        baked = 1.0
    col = int(round(baked * 100.0))
    col = min(max(col, 0), int(grid.size) - 1)
    f_star = json_float(grid[col])
    if float(certified[col]) > limit + 1e-15:
        ok = (certified <= limit + 1e-15) & (grid <= f_star + 1e-15)
        if not np.any(ok):
            return _all_light(invariant=False, shed=True)
        f_star = json_float(grid[int(np.flatnonzero(ok)[-1])])
        col = int(round(float(f_star) * 100.0))
        shed = True
    else:
        shed = False
    k_star = prefix_k_eligible(f_star, n_rows, n_elig)
    upgrade_a = prefix_mask(order, k_star, n_rows)
    upgrade_a &= pred_u > 0.0
    upgrade_k = np.zeros(n_rows, dtype=bool)
    pred_at = json_float(pred_ratio[col])
    term = binding_term_at(pred_at, kappa, float(phi[col]), min_increment=min_inc)
    certified_at = float(term["certified_ratio"])
    if certified_at > limit + 1e-15:
        return _all_light(invariant=True, shed=False)

    accounted = certified_at
    k_used = int(np.sum(upgrade_a))
    if accounted > limit + 1e-15 or k_used > k_star:
        shed = True
        for index in order[::-1]:
            if not upgrade_a[int(index)]:
                continue
            upgrade_a[int(index)] = False
            k_used = int(np.sum(upgrade_a))
            frac = float(k_used) / float(n_rows) if n_rows else 0.0
            pred_used = predicted_ratio_curve(
                light,
                np.where(pred_u > 0.0, pred_inc_a, 0.0),
                order,
                np.asarray([frac], dtype=np.float64),
            )
            phi_at = float(np.interp(frac, grid, phi))
            term_a = binding_term_at(float(pred_used[0]), kappa, phi_at, min_increment=min_inc)
            accounted = float(term_a["certified_ratio"])
            if accounted <= limit + 1e-15:
                break
        if accounted > limit + 1e-15:
            return _all_light(invariant=False, shed=True)

    if np.any(upgrade_k) or any(mid == _K1 for mid in ()):
        return _all_light(invariant=True, shed=False)
    models = models_from_masks(upgrade_a, upgrade_k)
    if _K1 in models:
        return _all_light(invariant=True, shed=False)
    if trace is not None:
        trace.update(
            {
                "all_light_fallback": False,
                "binds_term": term["binds"],
                "certified_ratio": json_float(certified_at),
                "empirical_term": term["empirical_term"],
                "f_star": json_float(f_star),
                "invariant_fail": False,
                "k_star": int(np.sum(upgrade_a)),
                "kappa_term": term["kappa_term"],
                "kappa_term_applied": bool(term["kappa_term_applied"]),
                "pred_ratio": pred_at,
                "self_cert_all_light": False,
                "self_cert_shed": bool(shed),
            }
        )
    return models


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "arm_a_spec": ARM_A_SPEC,
            "arm_b_spec": ARM_B_SPEC,
            "binding_layer_reason": BINDING_LAYER_REASON,
            "binding_layer_spec": BINDING_LAYER_SPEC,
            "bootstrap_draws": int(BOOTSTRAP_DRAWS),
            "bootstrap_seed": int(BOOTSTRAP_SEED),
            "cost_head_lock": COST_HEAD_LOCK,
            "density_inc_floor": json_float(DENSITY_INC_FLOOR),
            "density_ordering_rule": DENSITY_ORDER_SPEC,
            "f_grid": json_floats(F_GRID),
            "f_star_rule": F_STAR_RULE,
            "feature_blocks": list(FEATURE_BLOCKS),
            "feature_version": FEATURE_VERSION,
            "float64_table_note": FLOAT64_NOTE,
            "fold_seed": int(FOLD_SEED_DENSITY),
            "folds": int(FOLDS),
            "h2_5_threshold": json_float(H2_5_THRESHOLD),
            "h2_6_exceedance_max": json_float(H2_6_EXCEEDANCE_MAX),
            "h5_gate_spec": H5_GATE_SPEC,
            "h5_hypothesis": H5_HYPOTHESIS,
            "hash_bins_allowed": list(HASH_BINS),
            "intercept_policy": INTERCEPT_POLICY,
            "k1_enabled": {tier: False for tier in TIERS},
            "kappa_min_increment": json_float(KAPPA_MIN_INCREMENT),
            "kappa_q": json_float(KAPPA_Q),
            "locked_alpha": json_float(LOCKED_ALPHA),
            "locked_bins": int(LOCKED_BINS),
            "locked_inc_A": json_float(LOCKED_INC_A),
            "locked_inc_K": json_float(LOCKED_INC_K),
            "locked_variant": LOCKED_VARIANT,
            "official_caps": {tier: json_float(OFFICIAL_CAPS[tier]) for tier in TIERS},
            "operating_targets": {tier: json_float(OPERATING_TARGETS[tier]) for tier in TIERS},
            "parent_f_exact": PARENT_F_EXACT,
            "parent_f_pins": PARENT_F_PINS,
            "prefix_fit_spec": PREFIX_FIT_SPEC,
            "qa_reference_caps": list(QA_REFERENCE_CAPS),
            "quality_alphas": [json_float(alpha) for alpha in QUALITY_ALPHAS],
            "quality_lofo_spec": QUALITY_LOFO_SPEC,
            "quality_target_form": TARGET_FORM,
            "recal_clip_legacy": [
                json_float(RECAL_CLIP_LEGACY[0]),
                json_float(RECAL_CLIP_LEGACY[1]),
            ],
            "recal_grid_clips": [
                [json_float(clip[0]), json_float(clip[1])] for clip in RECAL_GRID_CLIPS
            ],
            "recal_grid_n_bins": list(RECAL_GRID_N_BINS),
            "recal_light_frac": json_float(RECAL_LIGHT_FRAC),
            "recal_n_bins_legacy": int(RECAL_BINS_LEGACY),
            "recal_selection_criterion": RECAL_SELECTION_CRITERION,
            "redteam_layer_spec": REDTEAM_LAYER_SPEC,
            "reopening_rationale": REOPENING_RATIONALE,
            "sort_rule": SORT_RULE,
            "target_form": TARGET_FORM,
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
        "k1_enabled": False,
        "locked": locked,
        "observed": observed,
        "quality_entered_calibration_selection": False,
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
    }
    if report["dev_opened"] is not False:
        raise RuntimeError("the density ordering layer report must assert dev_opened is false")
    if report["k1_enabled"] is not False:
        raise RuntimeError("the density ordering layer report must assert k1_enabled is false")
    if report["adoption_deferred_to_operator"] is not True:
        raise RuntimeError("the density ordering layer must defer adoption to the operator")
    return sort_mapping(report)


def decide(*, arm_a_pass: bool, arm_b_pass: bool, arm_a_weighted: float, arm_b_weighted: float) -> str:
    if arm_a_pass and arm_b_pass:
        if float(arm_b_weighted) > float(arm_a_weighted) + 1e-15:
            return DECISION_ARM_B
        return DECISION_ARM_A
    if arm_a_pass:
        return DECISION_ARM_A
    if arm_b_pass:
        return DECISION_ARM_B
    return DECISION_ROLLBACK


def _select_arm_a_winner(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            -float(row["selection_score"]),
            int(row["n_features"]),
            float(row["alpha"]),
            str(row["feature_block"]),
        )

    return min(rows, key=key)


def _score_arm_a_grid(
    bundle: TrainBundle,
    fold_ids: np.ndarray,
    texts: Sequence[str],
    score_l: np.ndarray,
    score_a: np.ndarray,
) -> Tuple[list[dict[str, Any]], np.ndarray, dict[str, Any], np.ndarray]:
    decimal_quality = DecimalQuality(bundle)
    tie_keys = content_digests(texts)
    uplift_target = score_a - score_l
    unequal = score_a != score_l
    costs = np.asarray(bundle.costs, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    oof_store: dict[Tuple[str, float], np.ndarray] = {}
    full_store: dict[Tuple[str, float], np.ndarray] = {}
    for block in FEATURE_BLOCKS:
        features = quality_design_matrix(texts, block)
        for alpha in QUALITY_ALPHAS:
            oof = oof_predict(features, uplift_target, fold_ids, alpha=float(alpha))
            full_coef = ridge_fit(features, uplift_target, alpha=float(alpha))
            oof_store[(block, float(alpha))] = oof
            full_store[(block, float(alpha))] = full_coef
            per_cap = []
            qualities = []
            for cap_text in QA_REFERENCE_CAPS:
                upgrade = allocate_two_action(
                    oof, costs, bundle.light_total, float(cap_text), tie_keys
                )
                model_ids = models_two_action(upgrade)
                quality = decimal_quality.quality_float(model_ids)
                qualities.append(quality)
                realized = float(
                    np.where(upgrade, costs[:, 1], costs[:, 0]).sum() / float(bundle.light_total)
                )
                per_cap.append(
                    {
                        "cap": cap_text,
                        "n_upgraded": int(np.count_nonzero(upgrade)),
                        "quality": json_float(quality),
                        "quality_official": decimal_quality.quality_text(model_ids),
                        "realized_cost_ratio": json_float(realized),
                    }
                )
            corr = correlation_block(oof, uplift_target, unequal_mask=unequal)
            rows.append(
                {
                    "alpha": json_float(alpha),
                    "feature_block": block,
                    "n_features": int(quality_n_features(block)),
                    "per_cap": per_cap,
                    "selected": False,
                    "selection_score": json_float(float(sum(qualities) / len(qualities))),
                    "sign_accuracy_A_ne_L": corr["sign_accuracy_unequal"],
                    "spearman": corr["spearman_target"],
                    "pearson": corr["pearson_target"],
                    "n_A_ne_L": corr["n_unequal"],
                    "target_form": TARGET_FORM,
                }
            )
    winner = dict(_select_arm_a_winner(rows))
    for row in rows:
        if (
            row["feature_block"] == winner["feature_block"]
            and abs(float(row["alpha"]) - float(winner["alpha"])) <= 1e-15
        ):
            row["selected"] = True
            winner = dict(row)
    oof_selected = oof_store[(str(winner["feature_block"]), float(winner["alpha"]))]
    full_coef = full_store[(str(winner["feature_block"]), float(winner["alpha"]))]
    return rows, oof_selected, winner, full_coef


def _evaluate_arm(
    *,
    name: str,
    bundle: TrainBundle,
    oof_uplift: np.ndarray,
    full_uplift: np.ndarray,
    quality_source: str,
    quality_block: str,
    quality_alpha: float,
    quality_signature: str,
    quality_coef: np.ndarray,
    oof_l: np.ndarray,
    oof_inc_a_raw: np.ndarray,
    oof_inc_k_raw: np.ndarray,
    full_l: np.ndarray,
    full_inc_a_raw: np.ndarray,
    full_heads: FittedHeads,
    actual_l: np.ndarray,
    actual_a: np.ndarray,
    actual_k: np.ndarray,
    actual_inc_a: np.ndarray,
    actual_inc_k: np.ndarray,
    score_l: np.ndarray,
    score_a: np.ndarray,
    score_k: np.ndarray,
    fold_ids: np.ndarray,
    families: Sequence[str],
    fam_arr: np.ndarray,
    digests: Sequence[str],
    texts: Sequence[str],
    views: Sequence[View],
    layers: Sequence[str],
    view_names: Sequence[str],
    lofo_raw: Mapping[str, Any],
    parent_ids: Mapping[str, Sequence[str]],
    lofo_uplift: Optional[np.ndarray],
    lofo_note: str,
) -> dict[str, Any]:
    n_train = int(actual_l.size)
    costs = np.asarray(bundle.costs, dtype=np.float64)
    scores = np.asarray(bundle.scores, dtype=np.float64)
    parent_weighted = float(PARENT_F_PINS["weighted"])

    light_totals = np.asarray(
        [float(actual_l[view.index].sum()) for view in views], dtype=np.float64
    )
    eligible_mask, median_light, light_threshold = recal_view_eligible(light_totals, layers)
    eligible_idx = tuple(int(index) for index in np.flatnonzero(eligible_mask))

    def _max_abs_curve(oof_ia: np.ndarray, oof_pl: np.ndarray, view_i: int) -> float:
        view = views[view_i]
        idx = view.index
        digest = tuple(digests[int(item)] for item in idx)
        inc = _gather(oof_ia, idx)
        upl = _gather(oof_uplift, idx)
        order = sort_density(upl, inc, digest)
        n_elig = n_positive_uplift(upl)
        inc_eff = np.where(upl > 0.0, inc, 0.0)
        pred = predicted_ratio_curve(_gather(oof_pl, idx), inc_eff, order)
        actual = phi_view(_gather(actual_l, idx), _gather(actual_a, idx), order)
        if n_elig < int(idx.size):
            # actual upgrades also stop at the eligible prefix
            actual = phi_view(
                _gather(actual_l, idx),
                np.where(upl > 0.0, _gather(actual_a, idx), _gather(actual_l, idx)),
                order,
            )
        return float(np.max(np.abs(pred - actual)))

    recal_grid_rows: list[dict[str, Any]] = []
    selected_row: Optional[dict[str, Any]] = None
    selected_key: Optional[Tuple[float, int, float, str]] = None
    for n_bins in RECAL_GRID_N_BINS:
        for clip in RECAL_GRID_CLIPS:
            cell_fits: dict[str, dict[str, Any]] = {}
            for fit_kind in RECAL_FIT_KINDS:
                if fit_kind == "prefix":
                    rec_a = fit_prefix_recal_density(
                        oof_inc_a_raw,
                        actual_inc_a,
                        oof_l,
                        actual_l,
                        actual_a,
                        oof_uplift,
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
                    fit_kind="aggregate" if fit_kind == "aggregate" else "prefix",
                )
                # apply_lofo_recal prefix path uses cheapest order (the prefix recalibration layer).
                # Binding views are all pred_source=oof, so selection uses OOF.
                diffs = [_max_abs_curve(oof_ia, oof_l, index) for index in eligible_idx]
                arr = np.asarray(diffs, dtype=np.float64) if diffs else np.zeros(0)
                q99 = float(quantile_higher(arr, 0.99)) if arr.size else float("inf")
                clip_info = clip_binds(rec_a, clip)
                fit_row = {
                    "clip": [json_float(clip[0]), json_float(clip[1])],
                    "clip_binds": clip_info,
                    "factors": json_floats(rec_a.clipped_factors),
                    "fit": fit_kind,
                    "n_bins": int(n_bins),
                    "n_eligible_views": int(arr.size),
                    "pav_factors": json_floats(rec_a.pav_factors),
                    "q99_max_abs_pred_minus_actual": json_float(q99),
                    "raw_factors": json_floats(rec_a.raw_factors),
                    "raw_max": clip_info.get("raw_max"),
                    "clip_still_binds": bool(
                        clip_info.get("clip_high_binds") or clip_info.get("clip_low_binds")
                    ),
                }
                cell_fits[fit_kind] = {
                    **fit_row,
                    "_rec_a": rec_a,
                    "_rec_k": rec_k,
                    "_oof_ia": oof_ia,
                    "_oof_ik": oof_ik,
                    "_lofo_pl": lofo_pl,
                    "_lofo_ia": lofo_ia,
                    "_lofo_ik": lofo_ik,
                }
                key = (
                    float(q99),
                    int(n_bins),
                    tighter_clip_key(clip),
                    0 if fit_kind == "aggregate" else 1,
                )
                if selected_key is None or key < selected_key:
                    selected_key = key
                    selected_row = cell_fits[fit_kind]
            recal_grid_rows.append(
                {
                    "aggregate": {
                        k: v
                        for k, v in cell_fits["aggregate"].items()
                        if not str(k).startswith("_")
                    },
                    "clip": [json_float(clip[0]), json_float(clip[1])],
                    "n_bins": int(n_bins),
                    "prefix": {
                        k: v
                        for k, v in cell_fits["prefix"].items()
                        if not str(k).startswith("_")
                    },
                }
            )
    if selected_row is None:
        raise RuntimeError(f"{name} recalibration grid was empty")

    selected_n_bins = int(selected_row["n_bins"])
    selected_clip = (float(selected_row["clip"][0]), float(selected_row["clip"][1]))
    selected_fit = str(selected_row["fit"])
    rec_a = selected_row["_rec_a"]
    rec_k = selected_row["_rec_k"]
    oof_inc_a = selected_row["_oof_ia"]
    _oof_inc_k = selected_row["_oof_ik"]
    lofo_l = selected_row["_lofo_pl"]
    lofo_inc_a = selected_row["_lofo_ia"]
    _lofo_inc_k = selected_row["_lofo_ik"]
    if selected_fit == "prefix":
        full_rec_a = fit_prefix_recal_density(
            full_inc_a_raw,
            actual_inc_a,
            full_l,
            actual_l,
            actual_a,
            full_uplift,
            fold_ids,
            digests,
            n_bins=selected_n_bins,
            clip=selected_clip,
        )
    else:
        full_rec_a = fit_recal_params(
            full_inc_a_raw, actual_inc_a, n_bins=selected_n_bins, clip=selected_clip
        )
    selected_recal_public = {
        k: v for k, v in selected_row.items() if not str(k).startswith("_")
    }

    # Before/after gap on the full Train OOF density order.
    raw_order = sort_density(oof_uplift, oof_inc_a_raw, digests)
    raw_inc_eff = np.where(oof_uplift > 0.0, oof_inc_a_raw, 0.0)
    raw_pred = predicted_ratio_curve(oof_l, raw_inc_eff, raw_order)
    raw_actual = phi_view(
        actual_l,
        np.where(oof_uplift > 0.0, actual_a, actual_l),
        raw_order,
    )
    post_order = sort_density(oof_uplift, oof_inc_a, digests)
    post_inc_eff = np.where(oof_uplift > 0.0, oof_inc_a, 0.0)
    post_pred = predicted_ratio_curve(oof_l, post_inc_eff, post_order)
    post_actual = phi_view(
        actual_l,
        np.where(oof_uplift > 0.0, actual_a, actual_l),
        post_order,
    )
    gap_fractions = (0.05, 0.10, 0.25)
    gap_block = {
        "after": {
            "actual": _curve_gaps(post_actual, np.zeros_like(post_actual), gap_fractions),
            "pred_minus_actual": _curve_gaps(post_pred, post_actual, gap_fractions),
            "predicted": _curve_gaps(post_pred, np.zeros_like(post_pred), gap_fractions),
        },
        "before": {
            "actual": _curve_gaps(raw_actual, np.zeros_like(raw_actual), gap_fractions),
            "pred_minus_actual": _curve_gaps(raw_pred, raw_actual, gap_fractions),
            "predicted": _curve_gaps(raw_pred, np.zeros_like(raw_pred), gap_fractions),
        },
    }
    # Store actual ratio values (not gaps-from-zero) under predicted/actual.
    for tag, pred_c, act_c in (
        ("before", raw_pred, raw_actual),
        ("after", post_pred, post_actual),
    ):
        gap_block[tag]["predicted"] = {
            f"{frac:.2f}": json_float(float(pred_c[int(round(frac * 100.0))]))
            for frac in gap_fractions
        }
        gap_block[tag]["actual"] = {
            f"{frac:.2f}": json_float(float(act_c[int(round(frac * 100.0))]))
            for frac in gap_fractions
        }

    def _pred_pack(view: View) -> dict[str, np.ndarray]:
        src_l = oof_l if view.pred_source == "oof" else lofo_l
        src_ia = oof_inc_a if view.pred_source == "oof" else lofo_inc_a
        src_u = oof_uplift if view.pred_source == "oof" else (
            lofo_uplift if lofo_uplift is not None else oof_uplift
        )
        idx = view.index
        return {
            "actual_a": _gather(actual_a, idx),
            "actual_k": _gather(actual_k, idx),
            "actual_l": _gather(actual_l, idx),
            "families": fam_arr[idx],
            "inc_a": _gather(src_ia, idx),
            "index": idx,
            "pred_l": _gather(src_l, idx),
            "score_a": _gather(score_a, idx),
            "score_k": _gather(score_k, idx),
            "score_l": _gather(score_l, idx),
            "uplift": _gather(src_u, idx),
        }

    view_packs = [_pred_pack(view) for view in views]
    view_digests = [tuple(digests[int(item)] for item in view.index) for view in views]
    orders = []
    pred_curves = []
    actual_curves = []
    monotone_flags = []
    for pack, digest in zip(view_packs, view_digests):
        order = sort_density(pack["uplift"], pack["inc_a"], digest)
        inc_eff = np.where(pack["uplift"] > 0.0, pack["inc_a"], 0.0)
        pred = predicted_ratio_curve(pack["pred_l"], inc_eff, order)
        actual = phi_view(
            pack["actual_l"],
            np.where(pack["uplift"] > 0.0, pack["actual_a"], pack["actual_l"]),
            order,
        )
        orders.append(order)
        pred_curves.append(pred)
        actual_curves.append(actual)
        monotone_flags.append(
            bool(np.all(np.diff(pred) >= -1e-15) and abs(float(pred[0]) - 1.0) <= 1e-15)
        )

    binding_idx = tuple(i for i, layer in enumerate(layers) if layer == "binding")
    redteam_idx = tuple(i for i, layer in enumerate(layers) if layer == "red-team")
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
            bucket = binding_cells if layer == "binding" else redteam_cells
            for tier in TIERS:
                if in_tier_band(float(pred[col]), tier):
                    bucket[tier].append(cell)

    kappa_tier: dict[str, float] = {}
    kappa_block: dict[str, Any] = {}
    for tier in TIERS:
        bind_k = np.asarray([cell["kappa"] for cell in binding_cells[tier]], dtype=np.float64)
        bind_summary = summarize_values(bind_k)
        if bind_summary is None:
            raise RuntimeError(f"{name} empty kappa cells for {tier}")
        kappa_tier[tier] = float(bind_summary["q9975"])
        kappa_block[tier] = {
            "binding": bind_summary,
            "binding_n_cells": int(bind_k.size),
            "redteam": summarize_values(
                np.asarray([cell["kappa"] for cell in redteam_cells[tier]], dtype=np.float64)
            ),
        }

    # Train OOF Q curve + f* under the dual certificate.
    full_order = sort_density(oof_uplift, oof_inc_a, digests)
    n_elig = n_positive_uplift(oof_uplift)
    q_oof_curve = quality_prefix_curve(score_l, score_a, full_order, n_elig)
    full_pred_ratio = predicted_ratio_curve(
        oof_l, np.where(oof_uplift > 0.0, oof_inc_a, 0.0), full_order
    )
    f_star: dict[str, float] = {}
    f_star_official: dict[str, float] = {}
    certified_full: dict[str, np.ndarray] = {}
    for tier in TIERS:
        cert = certified_ratio_curve(
            full_pred_ratio, kappa_tier[tier], phi_binding, min_increment=KAPPA_MIN_INCREMENT
        )
        certified_full[tier] = cert
        f_star[tier] = select_f_star_quality(q_oof_curve, cert, float(OPERATING_TARGETS[tier]))
        f_star_official[tier] = select_f_star_quality(
            q_oof_curve, cert, float(OFFICIAL_CAPS[tier])
        )

    k1_off = {tier: False for tier in TIERS}
    policy = SelectedPolicy(
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
        quality_source=quality_source,
        quality_block=quality_block,
        quality_alpha=float(quality_alpha),
        quality_target_form=TARGET_FORM,
        quality_feature_signature=quality_signature,
        quality_coefficients=tuple(json_floats(quality_coef)) if quality_coef.size else tuple(),
        kappa_tier={tier: json_float(kappa_tier[tier]) for tier in TIERS},
        kappa_min_increment=KAPPA_MIN_INCREMENT,
        phi_binding_q9975=tuple(json_floats(phi_binding)),
        recal_n_bins=selected_n_bins,
        recal_clip=selected_clip,
        recal_fit=selected_fit,
        sort_rule=SORT_RULE,
        density_inc_floor=DENSITY_INC_FLOOR,
        f_star={tier: json_float(f_star[tier]) for tier in TIERS},
        operating_targets=dict(OPERATING_TARGETS),
        official_caps=dict(OFFICIAL_CAPS),
        k1_enabled=k1_off,
        f_grid=tuple(json_floats(F_GRID)),
        intercept_policy=INTERCEPT_POLICY,
        arm_name=name,
    )

    def _route_pack(
        pack: dict[str, np.ndarray],
        digest: Sequence[str],
        local: SelectedPolicy,
        *,
        limits: Mapping[str, float],
        f_star_map: Optional[Mapping[str, float]] = None,
    ) -> Tuple[dict[str, Tuple[str, ...]], dict[str, dict[str, Any]]]:
        assigned: dict[str, Tuple[str, ...]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for tier in TIERS:
            row: dict[str, Any] = {}
            assigned[tier] = allocate_from_arrays(
                tier,
                pack["uplift"],
                pack["inc_a"],
                pack["pred_l"],
                digest,
                local,
                limit_override=float(limits[tier]),
                f_star_override=None if f_star_map is None else float(f_star_map[tier]),
                trace=row,
            )
            traces[tier] = row
        return assigned, traces

    full_pack = {
        "actual_a": actual_a,
        "actual_k": actual_k,
        "actual_l": actual_l,
        "inc_a": oof_inc_a,
        "pred_l": oof_l,
        "score_a": score_a,
        "score_k": score_k,
        "score_l": score_l,
        "uplift": oof_uplift,
    }
    assigned_full, traces_full = _route_pack(
        full_pack, digests, policy, limits=OPERATING_TARGETS
    )
    assigned_official, traces_official = _route_pack(
        full_pack, digests, policy, limits=OFFICIAL_CAPS, f_star_map=f_star_official
    )

    fold_positions = [index for index, view in enumerate(views) if view.kind == "oof-fold"]
    oof_models: dict[str, list[str]] = {tier: [""] * n_train for tier in TIERS}
    for row in fold_positions:
        assigned, _tr = _route_pack(
            view_packs[row], view_digests[row], policy, limits=OPERATING_TARGETS
        )
        for local_i, global_i in enumerate(views[row].index):
            for tier in TIERS:
                oof_models[tier][int(global_i)] = assigned[tier][local_i]
    oof_models_t = {tier: tuple(oof_models[tier]) for tier in TIERS}
    if any(item == "" for tier in TIERS for item in oof_models[tier]):
        raise RuntimeError(f"{name} OOF allocation left an unassigned episode")
    if any(mid == _K1 for tier in TIERS for mid in oof_models_t[tier]):
        raise RuntimeError(f"{name} selected axk1-think")

    # Quality LOFO: hold family out of the quality fit, route as a MIXED batch.
    lofo_models: dict[str, list[str]] = {tier: [""] * n_train for tier in TIERS}
    lofo_family_gain: dict[str, dict[str, Any]] = {}
    lofo_budget_exceeded: list[dict[str, Any]] = []
    lofo_fast_ratios: dict[str, float] = {}
    if lofo_uplift is None:
        mixed_uplift = oof_uplift
    else:
        mixed_uplift = lofo_uplift
    for fam_name, held in family_folds(families):
        held_idx = np.asarray(held, dtype=np.int64)
        mixed_pack = dict(full_pack)
        mixed_pack["uplift"] = mixed_uplift
        assigned, _tr = _route_pack(mixed_pack, digests, policy, limits=OPERATING_TARGETS)
        for global_i in held_idx:
            for tier in TIERS:
                lofo_models[tier][int(global_i)] = assigned[tier][int(global_i)]
        fam_gain = {}
        for tier in TIERS:
            fam_models = tuple(assigned[tier][int(item)] for item in held_idx)
            ours = float(_episode_scores(scores[held_idx], fam_models).mean())
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
            ratio = _realized_from_costs(
                actual_l[held_idx], actual_a[held_idx], actual_k[held_idx], fam_models
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
        raise RuntimeError(f"{name} LOFO allocation left an unassigned episode")

    official_oof = official_score(bundle.inputs, bundle.outcomes, bundle.policy, oof_models_t)
    official_lofo = official_score(bundle.inputs, bundle.outcomes, bundle.policy, lofo_models_t)
    float_oof = {tier: _score_mean(scores, oof_models_t[tier]) for tier in TIERS}
    float_oof_weighted = json_float(
        weighted_final(float_oof["fast"], float_oof["balanced"], float_oof["premium"])
    )
    official_oof_q = {tier: float(official_oof["tiers"][tier]["quality_score"]) for tier in TIERS}
    official_agree = {tier: json_float(abs(float_oof[tier] - official_oof_q[tier])) for tier in TIERS}
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
    fold_gains = [json_float(fold_weighted_ours[i] - fold_weighted_parent[i]) for i in range(FOLDS)]
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
        ours_ep - parent_ep, bundle.group_keys, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
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
    lofo_official_q = {tier: float(official_lofo["tiers"][tier]["quality_score"]) for tier in TIERS}
    lofo_gain_zeroed = {
        tier: json_float(lofo_official_q[tier] - float(PARENT_F_PINS[tier]["quality"]))
        for tier in TIERS
    }
    lofo_gain_zeroed["weighted"] = json_float(float(official_lofo["final_score"]) - parent_weighted)
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

    ruin_binding = {tier: {"n": 0, "n_ruin": 0} for tier in TIERS}
    ruin_redteam = {tier: {"n": 0, "n_ruin": 0} for tier in TIERS}
    h3_cost_fail = 0
    redteam_rows: list[dict[str, Any]] = []
    for index, view in enumerate(views):
        assigned, traces = _route_pack(
            view_packs[index], view_digests[index], policy, limits=OPERATING_TARGETS
        )
        layer = layers[index]
        pack = view_packs[index]
        row: dict[str, Any] = {
            "kind": view.kind,
            "layer": layer,
            "n": int(view.index.size),
            "name": view.name,
        }
        for tier in TIERS:
            models = assigned[tier]
            tr = traces[tier]
            ratio = _realized_from_costs(
                pack["actual_l"], pack["actual_a"], pack["actual_k"], models
            )
            official = float(OFFICIAL_CAPS[tier])
            cert = float(tr["certified_ratio"])
            row[tier] = {
                "certified_ratio": json_float(cert),
                "f_star": json_float(tr["f_star"]),
                "k_star": int(tr["k_star"]),
                "n_upgraded": int(sum(1 for mid in models if mid != _LIGHT)),
                "pred_ratio": json_float(tr["pred_ratio"]),
                "ratio": json_float(ratio),
            }
            bucket = ruin_binding if layer == "binding" else ruin_redteam
            bucket[tier]["n"] += 1
            if float(ratio) > official + 1e-15:
                bucket[tier]["n_ruin"] += 1
            if layer == "binding" and (
                float(ratio) > official + 1e-15 or cert > official + 1e-15
            ):
                h3_cost_fail += 1
        if layer == "red-team":
            redteam_rows.append(row)

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
        return float(quantile_higher(np.asarray(values, dtype=np.float64), KAPPA_Q))

    def _phi_from_indices(indices: Sequence[int]) -> Optional[Tuple[float, ...]]:
        if not indices:
            return None
        phi_hat, _change = phi_from_actual_curves([actual_curves[index] for index in indices])
        return tuple(json_floats(phi_hat))

    fold_checks: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        held_name = f"oof-fold-{fold}"
        if held_name not in view_names:
            continue
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
                    **policy.__dict__,
                    "kappa_tier": {**dict(policy.kappa_tier), tier: hat},
                    "phi_binding_q9975": phi_hat,
                }
            )
            assigned, traces = _route_pack(
                view_packs[held_i],
                view_digests[held_i],
                local,
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
    n_h26 = int(len(fold_checks))
    n_exceed = int(sum(1 for row in fold_checks if row["exceeded"]))
    exceed_rate = json_float(float(n_exceed) / float(n_h26) if n_h26 else 0.0)
    worst_held = (
        max(fold_checks, key=lambda row: float(row["held_realized"])) if fold_checks else None
    )

    oof_counts = {tier: _count_models(oof_models_t[tier]) for tier in TIERS}
    oof_ratios = {tier: _realized_ratio(costs, oof_models_t[tier]) for tier in TIERS}
    per_tier = {}
    for tier in TIERS:
        tr = traces_full[tier]
        realized = oof_ratios[tier]
        col = int(round(float(f_star[tier]) * 100.0))
        col = min(max(col, 0), 100)
        pred_r = json_float(full_pred_ratio[col])
        term = binding_term_at(
            pred_r, kappa_tier[tier], float(phi_binding[col]), min_increment=KAPPA_MIN_INCREMENT
        )
        per_tier[tier] = {
            "binds_term": term["binds"],
            "certified_ratio": term["certified_ratio"],
            "empirical_term": term["empirical_term"],
            "f_star": json_float(f_star[tier]),
            "k_star": int(sum(1 for mid in oof_models_t[tier] if mid != _LIGHT)),
            "k_star_full_batch": int(tr["k_star"]),
            "kappa_term": term["kappa_term"],
            "kappa_term_applied": bool(term["kappa_term_applied"]),
            "margin_internal": json_float(float(OPERATING_TARGETS[tier]) - float(realized)),
            "margin_official": json_float(float(OFFICIAL_CAPS[tier]) - float(realized)),
            "parent_k": int(PARENT_F_PINS[tier]["ax31_count"]),
            "pred_ratio": pred_r,
            "q_oof": json_float(official_oof_q[tier]),
            "q_oof_curve": json_float(q_oof_curve[col]),
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
            "certified_ratio": json_float(tr.get("certified_ratio", 1.0)),
            "f_star": json_float(f_star_official[tier]),
            "k_star": int(tr["k_star"]),
            "pred_ratio": json_float(tr["pred_ratio"]),
            "realized_full_batch": _realized_from_costs(
                actual_l, actual_a, actual_k, assigned_official[tier]
            ),
        }

    redteam_fast_overruns = {
        row["name"]: json_float(row["fast"]["ratio"])
        for row in redteam_rows
        if row["kind"] == "lofo" and float(row["fast"]["ratio"]) > float(OFFICIAL_CAPS["fast"]) + 1e-15
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
                "exceeded": lofo_budget_exceeded,
                "fast_inside_official_1_25": bool(
                    all(
                        float(value) <= float(OFFICIAL_CAPS["fast"]) + 1e-15
                        for value in lofo_fast_ratios.values()
                    )
                ),
                "fast_max_ratio": (
                    json_float(max(lofo_fast_ratios.values())) if lofo_fast_ratios else None
                ),
                "fast_per_family": lofo_fast_ratios,
                "n_exceeded": int(len(lofo_budget_exceeded)),
                "note": (
                    "Zeroing is counted under the cost gates, never as "
                    "quality regression. Charter §12.3 item 6 / §14.4. "
                    + lofo_note
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
        "h3_cost": {
            "n_fail": int(h3_cost_fail),
            "pass": bool(h3_cost_fail == 0),
            "threshold": (
                "certified_ratio(f*) <= official limit and every "
                "binding view realized ratio <= official limit"
            ),
        },
    }
    gates_pass = bool(
        gates["h2_1_oof_gain"]["pass"]
        and gates["h2_2_fold_wins"]["pass"]
        and gates["h2_3_bootstrap"]["pass"]
        and gates["h2_4_lofo"]["pass"]
        and gates["h2_5_lofo_worst_family"]["pass"]
        and gates["h2_6_heldout"]["pass"]
        and gates["h3_cost"]["pass"]
    )

    return {
        "float64_vs_official": official_agree,
        "fold_checks_h2_6": fold_checks,
        "gates": gates,
        "gates_pass": gates_pass,
        "kappa": kappa_block,
        "kappa_cells": {
            "excluded": int(n_kappa_excluded),
            "included": int(n_kappa_included),
            "min_increment": json_float(KAPPA_MIN_INCREMENT),
        },
        "kappa_tier": {tier: json_float(kappa_tier[tier]) for tier in TIERS},
        "lofo_family_gain": lofo_family_gain,
        "lofo_note": lofo_note,
        "official_limits_variant": official_variant,
        "oof_operating": {
            "model_counts": oof_counts,
            "official_per_tier_quality": official_oof_q,
            "official_weighted": json_float(float(official_oof["final_score"])),
            "parent_weighted": parent_weighted,
            "realized_ratios": oof_ratios,
            "vs_prefix_recal_weighted": json_float(float(official_oof["final_score"]) - PREFIX_RECAL_WEIGHTED),
            "weighted_float64": float_oof_weighted,
        },
        "per_tier": per_tier,
        "phi_binding_q9975": {
            "at_f": phi_at_report,
            "running_maximum": phi_change,
            "values": json_floats(phi_binding),
        },
        "policy": policy,
        "predicted_ratio_gaps": gap_block,
        "predicted_ratio_monotone_all_views": bool(all(monotone_flags)),
        "predicted_ratio_monotone_violations": int(
            sum(1 for flag in monotone_flags if not flag)
        ),
        "recalibration": {
            "eligible_n": int(eligible_mask.sum()),
            "grid": recal_grid_rows,
            "light_total_median": median_light,
            "light_total_threshold": light_threshold,
            "selected": selected_recal_public,
        },
        "redteam": {
            "fast_single_family_overruns": redteam_fast_overruns,
            "n": int(len(redteam_idx)),
            "rows": redteam_rows,
        },
        "view_layers": {
            "binding": int(len(binding_idx)),
            "binding_kinds": {
                kind: int(sum(1 for i in binding_idx if views[i].kind == kind))
                for kind in ("oof-fold", "famdom", "dirichlet", "half", "small")
            },
            "red_team": int(len(redteam_idx)),
            "redteam_kinds": {
                kind: int(sum(1 for i in redteam_idx if views[i].kind == kind))
                for kind in ("lofo", "lofo-combined", "small")
            },
        },
    }


def fit_and_evaluate(bundle: TrainBundle) -> dict[str, Any]:
    """Train-only the density ordering layer fit. K1 is never constructed. Dev is never opened."""

    folds = group_folds(bundle.episodes, folds=FOLDS, seed=FOLD_SEED_DENSITY)
    fold_ids = np.asarray(list(folds), dtype=np.int64)
    families = bundle.families
    fam_arr = np.asarray(list(families))
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
            "decision": DECISION_ROLLBACK,
            "diagnostic": {
                "oof_recal_ratios_legacy": {
                    "inc_A": json_float(ratio_a),
                    "inc_K": json_float(ratio_k),
                }
            },
            "observed": {
                "adoption_deferred_to_operator": True,
                "dev_opened": False,
                "k1_enabled": False,
            },
            "policies": {},
        }

    full_heads = fit_heads(features, costs, variant=LOCKED_VARIANT, alpha=LOCKED_ALPHA)
    full_l, _fa, _fk, full_inc_a_raw, _full_inc_k_raw = predict_heads(features, full_heads)
    lofo_raw = cache_lofo_raw(features, costs, families)
    views, catalogue = build_views(families, folds)
    layers = tuple(view_layer(view) for view in views)
    view_names = tuple(view.name for view in views)
    parent_ids = _parent_assignments(bundle)

    arm_a_rows, arm_a_oof, arm_a_winner, arm_a_coef = _score_arm_a_grid(
        bundle, fold_ids, texts, score_l, score_a
    )
    hash_dropped = str(arm_a_winner["feature_block"]) == "structural_only"
    recovered = bool(
        arm_a_winner["spearman"] is not None
        and (
            float(arm_a_winner["spearman"]) > QUALITY_SPEARMAN + 1e-6
            or (
                arm_a_winner["sign_accuracy_A_ne_L"] is not None
                and float(arm_a_winner["sign_accuracy_A_ne_L"]) > QUALITY_SIGN_ACC + 1e-6
            )
        )
    )

    # Arm A quality LOFO predictions for the selected config.
    arm_a_lofo = np.empty(int(score_l.size), dtype=np.float64)
    qa_features = quality_design_matrix(texts, str(arm_a_winner["feature_block"]))
    uplift_target = score_a - score_l
    for _fam, held in family_folds(families):
        train = np.ones(qa_features.shape[0], dtype=bool)
        train[held] = False
        coef = ridge_fit(
            qa_features[train], uplift_target[train], alpha=float(arm_a_winner["alpha"])
        )
        arm_a_lofo[held] = ridge_predict(coef, qa_features[held])

    frozen_meta: dict[str, Any]
    try:
        arm_b_uplift, frozen_meta = extract_frozen_uplift(bundle.episodes)
        arm_b_ok = True
        arm_b_blocker = None
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        arm_b_uplift = np.zeros(int(score_l.size), dtype=np.float64)
        frozen_meta = {"extracted": False}
        arm_b_ok = False
        arm_b_blocker = f"{type(exc).__name__}: {exc}"

    shared = dict(
        bundle=bundle,
        oof_l=oof_l,
        oof_inc_a_raw=oof_inc_a_raw,
        oof_inc_k_raw=oof_inc_k_raw,
        full_l=full_l,
        full_inc_a_raw=full_inc_a_raw,
        full_heads=full_heads,
        actual_l=actual_l,
        actual_a=actual_a,
        actual_k=actual_k,
        actual_inc_a=actual_inc_a,
        actual_inc_k=actual_inc_k,
        score_l=score_l,
        score_a=score_a,
        score_k=score_k,
        fold_ids=fold_ids,
        families=families,
        fam_arr=fam_arr,
        digests=digests,
        texts=texts,
        views=views,
        layers=layers,
        view_names=view_names,
        lofo_raw=lofo_raw,
        parent_ids=parent_ids,
    )
    evaluated: dict[str, Any] = {}
    policies: dict[str, SelectedPolicy] = {}

    arm_a = _evaluate_arm(
        name="arm_a",
        oof_uplift=arm_a_oof,
        full_uplift=ridge_predict(
            arm_a_coef, quality_design_matrix(texts, str(arm_a_winner["feature_block"]))
        ),
        quality_source="ridge_direct_signed",
        quality_block=str(arm_a_winner["feature_block"]),
        quality_alpha=float(arm_a_winner["alpha"]),
        quality_signature=quality_feature_signature(str(arm_a_winner["feature_block"])),
        quality_coef=arm_a_coef,
        lofo_uplift=arm_a_lofo,
        lofo_note="Arm A quality LOFO holds family f out of the ridge fit.",
        **shared,
    )
    policies["arm_a"] = arm_a.pop("policy")
    evaluated["arm_a"] = arm_a

    if arm_b_ok:
        arm_b = _evaluate_arm(
            name="arm_b",
            oof_uplift=arm_b_uplift,
            full_uplift=arm_b_uplift,
            quality_source="frozen_sparse_uplift_ladder",
            quality_block="frozen_sparse_uplift",
            quality_alpha=100.0,
            quality_signature="frozen-sparse_uplift-ladder-structural-direct",
            quality_coef=np.zeros(0, dtype=np.float64),
            lofo_uplift=None,
            lofo_note=(
                "Arm B cannot LOFO-refit the frozen the sparse uplift study head; quality "
                "LOFO uses the frozen full-fit predictions."
            ),
            **shared,
        )
        policies["arm_b"] = arm_b.pop("policy")
        evaluated["arm_b"] = arm_b
    else:
        evaluated["arm_b"] = {
            "blocker": arm_b_blocker,
            "extracted": False,
            "gates_pass": False,
            "oof_operating": {"official_weighted": 0.0},
        }

    arm_a_pass = bool(evaluated["arm_a"]["gates_pass"])
    arm_b_pass = bool(evaluated["arm_b"].get("gates_pass"))
    decision = decide(
        arm_a_pass=arm_a_pass,
        arm_b_pass=arm_b_pass,
        arm_a_weighted=float(evaluated["arm_a"]["oof_operating"]["official_weighted"]),
        arm_b_weighted=float(evaluated["arm_b"].get("oof_operating", {}).get("official_weighted") or 0.0),
    )
    observed = {
        "adoption_deferred_to_operator": True,
        "arm_a": {
            "configs": arm_a_rows,
            "hash_block_dropped": hash_dropped,
            "selected": arm_a_winner,
            "signal_vs_quality_head": {
                "quality_head_sign_accuracy": json_float(QUALITY_SIGN_ACC),
                "quality_head_spearman": json_float(QUALITY_SPEARMAN),
                "recovered_uplift_signal": recovered,
                "selected_sign_accuracy": arm_a_winner["sign_accuracy_A_ne_L"],
                "selected_spearman": arm_a_winner["spearman"],
            },
            **evaluated["arm_a"],
        },
        "arm_b": {
            "extracted": bool(arm_b_ok),
            "frozen_head": frozen_meta,
            "blocker": arm_b_blocker,
            **{k: v for k, v in evaluated["arm_b"].items() if k != "blocker"},
        },
        "dev_opened": False,
        "k1_enabled": False,
        "quality_entered_calibration_selection": False,
    }
    diagnostic = {
        "famdom_fallback_by_family": catalogue["famdom_fallback_by_family"],
        "float64_table_note": FLOAT64_NOTE,
        "legacy_oof_recal_ratios": {
            "inc_A": json_float(ratio_a),
            "inc_K": json_float(ratio_k),
        },
        "n_views": int(len(views)),
        "required_upstream_changes": [
            "g_features.ALLOWED_HASH_BINS must include 64",
            "modeling.feature_matrix / prefix_cert.design_matrix_g_features must accept structural-only",
            "prefix_recal.fit_prefix_recal must accept a caller-supplied order",
        ],
        "skipped_views": catalogue["skipped"],
        "view_kind_counts": catalogue["view_kind_counts"],
        "view_layer_counts": {
            "binding": int(sum(1 for layer in layers if layer == "binding")),
            "red-team": int(sum(1 for layer in layers if layer == "red-team")),
        },
    }
    return {
        "decision": decision,
        "diagnostic": diagnostic,
        "observed": observed,
        "policies": policies,
    }


__all__ = (
    "BOOTSTRAP_SEED",
    "SelectedPolicy",
    "DECISION_ARM_A",
    "DECISION_ARM_B",
    "DECISION_ROLLBACK",
    "DENSITY_INC_FLOOR",
    "EXPERIMENT",
    "F_GRID_ARRAY",
    "KAPPA_MIN_INCREMENT",
    "RECAL_LIGHT_FRAC",
    "allocate_from_arrays",
    "assemble_report",
    "certified_ratio_curve",
    "decide",
    "extract_frozen_uplift",
    "fit_and_evaluate",
    "fit_prefix_recal_density",
    "hash_features_64",
    "load_train",
    "locked_record",
    "predicted_ratio_curve",
    "quality_design_matrix",
    "quality_prefix_curve",
    "recal_view_eligible",
    "reject_dev_reference",
    "select_f_star_quality",
    "sort_density",
    "view_layer",
)
