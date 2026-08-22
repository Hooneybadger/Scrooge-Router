# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E2/E3 — OOF point-cost, item sigma, and two-price settlement.

Quality stays locked at the E1 ``baseline_continuous_uplift`` OOF signal.
Cost heads predict per-model input/output tokens on the runtime feature
layout (intercept + 14-d structural + 256 hash), assemble policy rates,
and apply raise-only monotonicity. Spread/tail scalars come from inner
grouped OOF residuals inside each outer-train complement — never from
in-sample residuals or outer held-out labels.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.cost_calibrated_router import hashed_features, structural_features
from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.cost_certificates import clamp_price_ladder, duan_smearing_factor
from research.lab.e1_objectives import (
    BASELINE_NAME,
    FEATURE_DIM,
    canonical_json_text,
    current_quality_matrix,
    oof_candidate_predictions,
    score_decisions,
    sha256_text,
    stress_views,
    write_json_atomic,
)
from research.lab.e3_two_price import (
    allocate_all_tiers_single_price,
    allocate_all_tiers_two_price,
    selection_spend,
)
from research.lab.modeling import (
    OFFICIAL_CAPS,
    ridge_fit,
    ridge_predict,
    sort_mapping,
)
from research.lab.public_pool import ROOT, PublicPool
from research.lab.quality_heads import content_tie_keys
from research.lab.validation import public_arrays


EXPERIMENT = "e2-cost-uncertainty"
REPORT_TYPE = "scrooge-e2-cost-uncertainty-v1"
SCHEMA_VERSION = 1
QUALITY_SIGNAL = BASELINE_NAME
HASH_BINS = 256
COST_ALPHA = 1.0
SIGMA_ALPHA = 100.0
STEP_RATIO = 1.05
TOKEN_EXPM1_CLIP = math.log1p(10_000_000.0)
LOG_EPS = 1e-12
SIGMA_FLOOR = 1e-6
FAMILY_SMEAR_MIN_N = 20
FAMILY_SIGMA_MIN_N = 10
Z90 = 1.2815515655446004
Z99 = 2.3263478740408408
K_SINGLE_PRICE = Z90
GLOBAL_RATIO_Q = 0.75
DENOM_RATIO_Q = 0.50
DENOM_SCALE_CLIP = (0.5, 2.0)
KAPPA_CLIP = (1.0, 3.0)
BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SEED = 20260821
MIXTURE_DRAWS = 40
MIXTURE_SEED = 20260821
MIXTURE_N = 880
VIEW_MIN_N = 20
GATE_VIEW_KINDS: Tuple[str, ...] = ("family", "fold", "language", "length", "split")
GATE_VIEW_DROP = 0.003
GATE_QUALITY_SLACK = 0.001
GATE_QUALITY_GAIN = 0.002
QUALITY_REFERENCE = "point_cost_baseline"
QUALITY_REFERENCE_POLICY = (
    "Deltas are historical versus point_cost_baseline quality_weighted. "
    "That reference may itself violate official hard caps "
    "(reference_budget_valid). quality_ok / quality slack is necessary "
    "but never sufficient for promotion. A candidate still fails on "
    "pooled/fold official caps, grouped-bootstrap q99.9, or gated "
    "actual-ratio views. The next experiment needs a budget-valid "
    "quality baseline."
)
RATIO_VIEW_DENOMINATOR = (
    "Each gated ratio view divides selected actual cost on the slice by "
    "that slice's actual light total. This is scoring/stress evidence. "
    "The allocator never receives an actual light bill."
)
COVERAGE_SLACK = 0.02
STRESS_95_CAPS = {"fast": 1.1875, "balanced": 1.90, "premium": 3.80}
AUDIT_RELATIVE_PATH = "build/compare-e2-cost-uncertainty/episode-audit.json"
FAMILY_GUARD_PATH = (
    ROOT / "src" / "ossp_router" / "resources" / "family-guard-router.v1.json"
)

CANDIDATE_ORDER: Tuple[str, ...] = (
    "point_cost_baseline",
    "item_sigma_single_price",
    "two_price_q50_q90",
    "two_price_tier_tail",
)

CANDIDATE_DEFINITIONS: Mapping[str, Mapping[str, str]] = {
    "point_cost_baseline": {
        "allocation": "single_price",
        "buy": "smeared mean * inner-OOF q75 actual/pred kappa",
        "settle": "same as buy",
        "summary": "OOF token mean + Duan smear + global conservative kappa.",
    },
    "item_sigma_single_price": {
        "allocation": "single_price",
        "buy": "smeared mean * exp(z90 * sigma_i)",
        "settle": "same as buy",
        "summary": "Item-wise log-cost tail at a pre-registered z90.",
    },
    "two_price_q50_q90": {
        "allocation": "two_price",
        "buy": "smeared mean (q50)",
        "settle": "q90 for every tier",
        "summary": "Buy q50, settle q90, exact-SHA group rollback.",
    },
    "two_price_tier_tail": {
        "allocation": "two_price",
        "buy": "smeared mean (q50)",
        "settle": "q90 Fast/Balanced, q99 Premium",
        "summary": "Same buy surface; Premium settles at pre-registered z99.",
    },
}

RUNTIME_DIFFS = (
    "Runtime cost uses the frozen Train artifact (ridge_alpha=1, hash256, "
    "log1p tokens, family Duan smear, 1.05 step). This experiment refits "
    "the same layout on each outer-train complement with closed-form ridge "
    "and an unpenalized intercept column. Feature mean/scale, smear, kappa, "
    "sigma, token clamps, and the light-denominator scale are computed on "
    "outer-train only. Frozen artifact coefficients are never read for "
    "allocation. Premium overlay residual_upper / kappa_q999 and family-guard "
    "multipliers are not applied — they are diagnostics only."
)


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def _json_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(np.float64(value))
    if not np.isfinite(number):
        return None
    return number


def policy_rates(policy: Any) -> np.ndarray:
    """Rows: (fixed, input_rate/unit, output_rate/unit) for Light, AX31, K1."""

    unit = float(policy.token_unit)
    rows = []
    for model_id in MODEL_IDS:
        rates = policy.models[model_id]
        rows.append(
            (
                float(rates.fixed_cost),
                float(rates.input_token_rate) / unit,
                float(rates.output_token_rate) / unit,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def cost_feature_matrix(episodes: Sequence[Any]) -> np.ndarray:
    rows = len(episodes)
    matrix = np.ones((rows, 1 + FEATURE_DIM + HASH_BINS), dtype=np.float64)
    for row, episode in enumerate(episodes):
        structural = structural_features(episode)
        if len(structural) != FEATURE_DIM:
            raise RuntimeError("structural_features width drifted")
        hashed = hashed_features(episode, HASH_BINS, "fnv1a64")
        matrix[row, 1 : 1 + FEATURE_DIM] = structural
        matrix[row, 1 + FEATURE_DIM :] = hashed
    return matrix


def standardize_apply(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train[:, 1:].mean(axis=0)
    scale = train[:, 1:].std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    out_train = train.copy()
    out_test = test.copy()
    out_train[:, 1:] = (train[:, 1:] - mean) / scale
    out_test[:, 1:] = (test[:, 1:] - mean) / scale
    return out_train, out_test, mean, scale


def token_log1p_targets(input_tokens: np.ndarray, output_tokens: np.ndarray) -> np.ndarray:
    stacked = np.column_stack(
        [
            input_tokens[:, 0],
            output_tokens[:, 0],
            input_tokens[:, 1],
            output_tokens[:, 1],
            input_tokens[:, 2],
            output_tokens[:, 2],
        ]
    )
    return np.log1p(np.maximum(np.asarray(stacked, dtype=np.float64), 0.0))


def expm1_tokens(log_pred: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(log_pred, dtype=np.float64), 0.0, TOKEN_EXPM1_CLIP)
    return np.expm1(clipped)


def assemble_costs(tokens: np.ndarray, rates: np.ndarray) -> np.ndarray:
    matrix = np.asarray(tokens, dtype=np.float64)
    costs = np.empty((matrix.shape[0], 3), dtype=np.float64)
    for model in range(3):
        costs[:, model] = (
            rates[model, 0]
            + matrix[:, 2 * model] * rates[model, 1]
            + matrix[:, 2 * model + 1] * rates[model, 2]
        )
    return costs


def clamp_predicted_costs(costs: np.ndarray) -> np.ndarray:
    light, ax31, k1 = clamp_price_ladder(costs[:, 0], costs[:, 1], costs[:, 2])
    ax31 = np.maximum(ax31, light * STEP_RATIO)
    k1 = np.maximum(k1, ax31 * STEP_RATIO)
    return np.column_stack([light, ax31, k1])


def _fit_heads(features: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
    return [
        ridge_fit(features, targets[:, column], alpha=COST_ALPHA)
        for column in range(targets.shape[1])
    ]


def _predict_heads(coefs: Sequence[np.ndarray], features: np.ndarray) -> np.ndarray:
    return np.column_stack([ridge_predict(coef, features) for coef in coefs])


def _smear_tokens(
    tokens: np.ndarray,
    families: Sequence[str],
    global_smear: np.ndarray,
    family_smear: Mapping[str, np.ndarray],
) -> np.ndarray:
    factors = np.vstack(
        [family_smear.get(family, global_smear) for family in families]
    )
    return np.maximum(0.0, (tokens + 1.0) * factors - 1.0)


def _quantile(values: np.ndarray, prob: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return _json_float(np.quantile(finite, float(prob)))


def token_matrices(pool: PublicPool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = public_arrays(pool.inputs, pool.outcomes, pool.policy)
    return (
        np.asarray(arrays.input_tokens, dtype=np.float64),
        np.asarray(arrays.output_tokens, dtype=np.float64),
        np.asarray(arrays.costs, dtype=np.float64),
    )


def oof_quality_baseline(pool: PublicPool) -> Tuple[np.ndarray, np.ndarray]:
    features = current_quality_matrix(pool.episodes)
    predicted = oof_candidate_predictions(features, pool.scores, pool.folds)
    return predicted[QUALITY_SIGNAL]


def oof_cost_surfaces(pool: PublicPool) -> dict[str, Any]:
    """Outer grouped OOF cost surfaces. Held-out tokens/costs never enter fits."""

    input_tokens, output_tokens, actual_costs = token_matrices(pool)
    raw_features = cost_feature_matrix(pool.episodes)
    targets = token_log1p_targets(input_tokens, output_tokens)
    rates = policy_rates(pool.policy)
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    families = tuple(pool.families)
    n_rows = raw_features.shape[0]
    point = np.empty((n_rows, 3), dtype=np.float64)
    conservative = np.empty((n_rows, 3), dtype=np.float64)
    q90 = np.empty((n_rows, 3), dtype=np.float64)
    q99 = np.empty((n_rows, 3), dtype=np.float64)
    sigma = np.empty((n_rows, 3), dtype=np.float64)
    pred_tokens = np.empty((n_rows, 6), dtype=np.float64)
    denom_scale = np.empty(n_rows, dtype=np.float64)
    kappa_rows = np.empty((n_rows, 3), dtype=np.float64)
    inner_resid_abs = np.empty((n_rows, 3), dtype=np.float64)
    inner_resid_abs[:] = np.nan
    fold_calibration = []
    inner_train: list[dict[str, Any]] = []
    n_inner_fits = 0

    for fold in range(int(fold_ids.max()) + 1):
        train = fold_ids != fold
        test = fold_ids == fold
        raw_train = raw_features[train]
        raw_test = raw_features[test]
        x_train, x_test, _mean, _scale = standardize_apply(raw_train, raw_test)
        y_train = targets[train]
        actual_train = actual_costs[train]
        tokens_train_actual = np.expm1(y_train)
        token_lo = tokens_train_actual.min(axis=0)
        token_hi = tokens_train_actual.max(axis=0)

        inner_pred_log = np.empty_like(y_train)
        inner_ids = fold_ids[train]
        for inner in sorted(np.unique(inner_ids)):
            inner_fit = inner_ids != inner
            inner_val = inner_ids == inner
            x_in, x_val, _, _ = standardize_apply(
                raw_train[inner_fit], raw_train[inner_val]
            )
            coefs = _fit_heads(x_in, y_train[inner_fit])
            inner_pred_log[inner_val] = _predict_heads(coefs, x_val)
            n_inner_fits += 6

        inner_resid = y_train - inner_pred_log
        global_smear = np.asarray(
            [duan_smearing_factor(inner_resid[:, column]) for column in range(6)],
            dtype=np.float64,
        )
        family_smear: dict[str, np.ndarray] = {}
        train_families = tuple(families[index] for index in np.flatnonzero(train))
        for name in sorted(set(train_families)):
            mask = np.asarray([family == name for family in train_families])
            if int(mask.sum()) < FAMILY_SMEAR_MIN_N:
                continue
            family_smear[name] = np.asarray(
                [
                    duan_smearing_factor(inner_resid[mask, column])
                    for column in range(6)
                ],
                dtype=np.float64,
            )
        inner_tokens = _smear_tokens(
            expm1_tokens(inner_pred_log), train_families, global_smear, family_smear
        )
        inner_point = assemble_costs(inner_tokens, rates)
        log_resid = np.log(np.maximum(actual_train, LOG_EPS)) - np.log(
            np.maximum(inner_point, LOG_EPS)
        )
        inner_resid_abs[train] = np.abs(log_resid)
        sigma_targets = np.log(np.abs(log_resid) + SIGMA_FLOOR)
        sigma_coefs = [
            ridge_fit(x_train, sigma_targets[:, model], alpha=SIGMA_ALPHA)
            for model in range(3)
        ]
        family_sigma_floor: dict[str, np.ndarray] = {}
        global_sigma_floor = np.median(np.abs(log_resid), axis=0)
        for name in sorted(set(train_families)):
            mask = np.asarray([family == name for family in train_families])
            if int(mask.sum()) < FAMILY_SIGMA_MIN_N:
                continue
            family_sigma_floor[name] = np.median(np.abs(log_resid[mask]), axis=0)

        inner_mono = clamp_predicted_costs(inner_point)
        inner_train.append(
            {
                "denom_scale": None,
                "fold": int(fold),
                "point": inner_mono.copy(),
                "train_index": np.flatnonzero(train).astype(np.int64),
            }
        )
        kappa = np.asarray(
            [
                float(
                    np.clip(
                        _quantile(actual_train[:, model] / inner_mono[:, model], GLOBAL_RATIO_Q),
                        KAPPA_CLIP[0],
                        KAPPA_CLIP[1],
                    )
                )
                for model in range(3)
            ],
            dtype=np.float64,
        )
        light_ratios = actual_train[:, 0] / np.maximum(inner_mono[:, 0], LOG_EPS)
        scale = float(
            np.clip(_quantile(light_ratios, DENOM_RATIO_Q), *DENOM_SCALE_CLIP)
        )

        mean_coefs = _fit_heads(x_train, y_train)
        held_log = _predict_heads(mean_coefs, x_test)
        held_tokens = np.clip(
            _smear_tokens(
                expm1_tokens(held_log),
                tuple(families[index] for index in np.flatnonzero(test)),
                global_smear,
                family_smear,
            ),
            token_lo,
            token_hi,
        )
        held_point = clamp_predicted_costs(assemble_costs(held_tokens, rates))
        held_sigma = np.column_stack(
            [np.exp(ridge_predict(coef, x_test)) for coef in sigma_coefs]
        )
        test_families = tuple(families[index] for index in np.flatnonzero(test))
        floors = np.vstack(
            [family_sigma_floor.get(family, global_sigma_floor) for family in test_families]
        )
        held_sigma = np.maximum(held_sigma, floors)
        held_sigma = np.maximum(held_sigma, SIGMA_FLOOR)
        point[test] = held_point
        conservative[test] = clamp_predicted_costs(held_point * kappa)
        q90[test] = clamp_predicted_costs(held_point * np.exp(Z90 * held_sigma))
        q99[test] = clamp_predicted_costs(held_point * np.exp(Z99 * held_sigma))
        sigma[test] = held_sigma
        pred_tokens[test] = held_tokens
        denom_scale[test] = scale
        kappa_rows[test] = kappa
        inner_train[-1]["denom_scale"] = _json_float(scale)
        fold_calibration.append(
            {
                "denom_scale": _json_float(scale),
                "fold": int(fold),
                "global_smear": [_json_float(value) for value in global_smear],
                "kappa": [_json_float(value) for value in kappa],
                "n_inner_rows": int(train.sum()),
                "n_test": int(test.sum()),
            }
        )

    if not np.all(np.isfinite(point)):
        raise RuntimeError("OOF cost surfaces contain non-finite values")
    return {
        "actual_costs": actual_costs,
        "conservative": conservative,
        "denom_scale": denom_scale,
        "fold_calibration": fold_calibration,
        "inner_abs_log_resid": inner_resid_abs,
        "inner_train": inner_train,
        "input_tokens": input_tokens,
        "kappa": kappa_rows,
        "n_inner_ridge_fits": n_inner_fits,
        "n_outer_mean_fits": 6 * (int(fold_ids.max()) + 1),
        "output_tokens": output_tokens,
        "point": point,
        "pred_tokens": pred_tokens,
        "q90": q90,
        "q99": q99,
        "sigma": sigma,
    }


def predicted_light_total(point_light: np.ndarray, denom_scale: np.ndarray) -> float:
    """OOF predicted light bill. Never pass an actual light total here."""

    light = np.asarray(point_light, dtype=np.float64).reshape(-1)
    scale = np.asarray(denom_scale, dtype=np.float64).reshape(-1)
    if light.shape != scale.shape:
        raise ValueError("predicted light and denom_scale must align")
    return float((light * scale).sum())


def attach_predicted_light(costs: np.ndarray, alloc_light: np.ndarray) -> np.ndarray:
    """Keep the OOF light bill in column 0 so denom and current costs match."""

    aligned = np.asarray(costs, dtype=np.float64).copy()
    aligned[:, 0] = np.asarray(alloc_light, dtype=np.float64)
    return clamp_predicted_costs(aligned)


def candidate_surfaces(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    alloc_light = bundle["point"][:, 0] * bundle["denom_scale"]
    point = attach_predicted_light(bundle["point"], alloc_light)
    conservative = attach_predicted_light(bundle["conservative"], alloc_light)
    q90 = attach_predicted_light(bundle["q90"], alloc_light)
    q99 = attach_predicted_light(bundle["q99"], alloc_light)
    return {
        "point_cost_baseline": {
            "buy": conservative,
            "kind": "single_price",
            "light": alloc_light,
            "settle": {
                "balanced": conservative,
                "fast": conservative,
                "premium": conservative,
            },
        },
        "item_sigma_single_price": {
            "buy": q90,
            "kind": "single_price",
            "light": alloc_light,
            "settle": {"balanced": q90, "fast": q90, "premium": q90},
        },
        "two_price_q50_q90": {
            "buy": point,
            "kind": "two_price",
            "light": alloc_light,
            "settle": {"balanced": q90, "fast": q90, "premium": q90},
        },
        "two_price_tier_tail": {
            "buy": point,
            "kind": "two_price",
            "light": alloc_light,
            "settle": {"balanced": q90, "fast": q90, "premium": q99},
        },
    }


def allocate_candidate(
    name: str,
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    surface: Mapping[str, Any],
    predicted_light_total: float,
    tie_keys: Sequence[str],
    group_keys: Sequence[str],
) -> dict[str, Tuple[str, ...]]:
    if float(predicted_light_total) <= 0.0:
        raise ValueError("allocator refused an actual or non-positive light total")
    if surface["kind"] == "single_price":
        return allocate_all_tiers_single_price(
            pred_qa, pred_qk, surface["buy"], predicted_light_total, OFFICIAL_CAPS, tie_keys
        )
    return allocate_all_tiers_two_price(
        pred_qa,
        pred_qk,
        surface["buy"],
        surface["settle"],
        predicted_light_total,
        OFFICIAL_CAPS,
        tie_keys,
        group_keys,
    )


def _error_block(pred: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    residual = pred - actual
    ratio = actual / np.maximum(pred, LOG_EPS)
    return {
        "mae": _json_float(np.mean(np.abs(residual))),
        "mean_actual_over_pred": _json_float(np.mean(ratio)),
        "rmse": _json_float(np.sqrt(np.mean(residual * residual))),
        "rmsle": _json_float(
            np.sqrt(
                np.mean(
                    (
                        np.log(np.maximum(pred, LOG_EPS))
                        - np.log(np.maximum(actual, LOG_EPS))
                    )
                    ** 2
                )
            )
        ),
    }


def coverage_block(pred: np.ndarray, actual: np.ndarray, nominal: float) -> dict[str, Any]:
    covered = actual <= pred
    rate = _json_float(np.mean(covered))
    return {
        "empirical": rate,
        "nominal": _json_float(nominal),
        "shortfall": _json_float(max(0.0, nominal - rate)),
        "slack_ok": bool(rate + 1e-15 >= nominal - COVERAGE_SLACK),
    }


def cost_accuracy(bundle: Mapping[str, Any]) -> dict[str, Any]:
    actual = bundle["actual_costs"]
    models = {}
    for index, model_id in enumerate(MODEL_IDS):
        models[model_id] = {
            "point": _error_block(bundle["point"][:, index], actual[:, index]),
            "q90_coverage": coverage_block(bundle["q90"][:, index], actual[:, index], 0.90),
            "q99_coverage": coverage_block(bundle["q99"][:, index], actual[:, index], 0.99),
        }
    light_pred = bundle["point"][:, 0] * bundle["denom_scale"]
    return {
        "light_denominator": {
            "actual_total": _json_float(actual[:, 0].sum()),
            "predicted_total": _json_float(light_pred.sum()),
            "scale_used": "per-fold inner-OOF q50(actual_light/pred_light) on point costs",
            "used_actual_light_in_allocator": False,
            "bias": _json_float(light_pred.sum() / max(float(actual[:, 0].sum()), LOG_EPS) - 1.0),
        },
        "models": models,
    }


def _slice_quality(
    scores: np.ndarray, models_by_tier: Mapping[str, Sequence[str]], mask: np.ndarray
) -> float | None:
    from research.lab.modeling import weighted_final

    if not np.any(mask):
        return None
    qualities = []
    for tier in TIERS:
        columns = np.asarray(
            [MODEL_IDS.index(model_id) for model_id in models_by_tier[tier]],
            dtype=np.int64,
        )
        selected = scores[np.arange(scores.shape[0]), columns]
        qualities.append(float(selected[mask].mean()))
    return weighted_final(qualities[0], qualities[1], qualities[2])


def actual_ratio_of(
    models: Sequence[str], actual_costs: np.ndarray, actual_light_total: float
) -> float:
    return selection_spend(models, actual_costs) / float(actual_light_total)


def grouped_ratio_bootstrap(
    models: Sequence[str],
    actual_costs: np.ndarray,
    actual_light: np.ndarray,
    group_keys: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    columns = np.asarray(
        [{"ax31-light": 0, "ax31": 1, "axk1-think": 2}[model] for model in models],
        dtype=np.int64,
    )
    spend = actual_costs[np.arange(actual_costs.shape[0]), columns]
    members: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        members[key].append(index)
    keys = tuple(sorted(members))
    group_spend = np.asarray(
        [float(spend[members[key]].sum()) for key in keys], dtype=np.float64
    )
    group_light = np.asarray(
        [float(actual_light[members[key]].sum()) for key in keys], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    n_groups = len(keys)
    for draw in range(int(draws)):
        chosen = rng.integers(0, n_groups, size=n_groups)
        samples[draw] = float(group_spend[chosen].sum() / max(group_light[chosen].sum(), LOG_EPS))
    return {
        "draws": int(draws),
        "max": _json_float(samples.max()),
        "mean": _json_float(samples.mean()),
        "q99_9": _json_float(np.quantile(samples, 0.999)),
        "samples": samples,
        "seed": int(seed),
    }


def family_mixture_ratios(
    models: Sequence[str],
    actual_costs: np.ndarray,
    actual_light: np.ndarray,
    families: Sequence[str],
    *,
    draws: int,
    seed: int,
    batch: int,
) -> dict[str, Any]:
    names = tuple(sorted(set(families)))
    pools = {
        name: np.flatnonzero(np.asarray(list(families)) == name) for name in names
    }
    columns = np.asarray(
        [{"ax31-light": 0, "ax31": 1, "axk1-think": 2}[model] for model in models],
        dtype=np.int64,
    )
    spend = actual_costs[np.arange(actual_costs.shape[0]), columns]
    rng = np.random.default_rng(int(seed))
    samples = []
    take = min(int(batch), spend.shape[0])
    for _draw in range(int(draws)):
        weights = rng.dirichlet(np.ones(len(names)))
        counts = rng.multinomial(take, weights)
        chosen = []
        for name, count in zip(names, counts):
            pool = pools[name]
            if pool.size == 0 or count == 0:
                continue
            chosen.append(rng.choice(pool, size=int(count), replace=True))
        if not chosen:
            continue
        index = np.concatenate(chosen)
        samples.append(float(spend[index].sum() / max(float(actual_light[index].sum()), LOG_EPS)))
    array = np.asarray(samples, dtype=np.float64)
    return {
        "draws": int(array.size),
        "max": _json_float(array.max()) if array.size else None,
        "mean": _json_float(array.mean()) if array.size else None,
        "q99_9": _json_float(np.quantile(array, 0.999)) if array.size else None,
        "seed": int(seed),
    }


def family_guard_overlap(pool: PublicPool, sigma: np.ndarray) -> dict[str, Any]:
    payload = json.loads(FAMILY_GUARD_PATH.read_text(encoding="utf-8"))
    guard = payload["family_guard"]
    ratios = {
        name: float(value) for name, value in guard["train_family_ratios"].items()
    }
    multipliers = {name: float(value) for name, value in guard["multipliers"].items()}
    rows = []
    mean_sigma = []
    ratio_values = []
    for name in sorted(set(pool.families)):
        mask = np.asarray([family == name for family in pool.families])
        sigma_mean = float(sigma[mask].mean())
        rows.append(
            {
                "family": name,
                "guard_multiplier": multipliers.get(name),
                "mean_sigma": _json_float(sigma_mean),
                "n": int(mask.sum()),
                "train_family_ratio": ratios.get(name),
            }
        )
        if name in ratios:
            mean_sigma.append(sigma_mean)
            ratio_values.append(ratios[name])
    corr = None
    if len(mean_sigma) >= 3:
        corr = _json_optional_float(np.corrcoef(mean_sigma, ratio_values)[0, 1])
    return {
        "correlation_sigma_vs_train_family_ratio": corr,
        "families": rows,
        "note": (
            "Item sigma and family-guard multipliers are reported only. "
            "They are not multiplied together in this phase."
        ),
        "runtime_multipliers": multipliers,
    }


def evaluate_allocation(
    pool: PublicPool,
    models_by_tier: Mapping[str, Sequence[str]],
    *,
    indexes: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    return score_decisions(pool, models_by_tier, indexes=indexes)


def _quality_views(
    pool: PublicPool,
    baseline_models: Mapping[str, Sequence[str]],
    candidate_models: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    return stress_views(pool, baseline_models, candidate_models)


def _hard_caps_ok(scored: Mapping[str, Any]) -> bool:
    return all(scored["tiers"][tier]["within_hard_cap"] for tier in TIERS)


def _slice_actual_ratio(
    models: Sequence[str], actual_costs: np.ndarray, mask: np.ndarray
) -> float | None:
    if not np.any(mask):
        return None
    subset_models = [models[index] for index, flag in enumerate(mask) if flag]
    subset_costs = actual_costs[mask]
    light = float(subset_costs[:, 0].sum())
    if light <= 0.0:
        return None
    return selection_spend(subset_models, subset_costs) / light


def deterministic_ratio_views(
    pool: PublicPool, models_by_tier: Mapping[str, Sequence[str]]
) -> list[dict[str, Any]]:
    actual = pool.costs
    views: list[tuple[str, str, np.ndarray]] = []
    for name in sorted(set(pool.families)):
        views.append(("family", name, np.asarray([family == name for family in pool.families])))
    for name in sorted(set(pool.languages)):
        views.append(
            ("language", name, np.asarray([label == name for label in pool.languages]))
        )
    for name in sorted(set(pool.length_views)):
        views.append(
            ("length", name, np.asarray([label == name for label in pool.length_views]))
        )
    for name in sorted(set(pool.split_labels)):
        views.append(("split", name, np.asarray([label == name for label in pool.split_labels])))
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    for fold in range(int(fold_ids.max()) + 1):
        views.append(("fold", str(fold), fold_ids == fold))
    rows = []
    for kind, name, mask in views:
        n = int(np.count_nonzero(mask))
        gated = bool(kind in GATE_VIEW_KINDS and n >= VIEW_MIN_N)
        slice_light = float(actual[mask, 0].sum()) if n else 0.0
        tier_rows = {}
        overrun = False
        for tier in TIERS:
            ratio = _slice_actual_ratio(models_by_tier[tier], actual, mask)
            hard = OFFICIAL_CAPS[tier]
            stress = STRESS_95_CAPS[tier]
            over_hard = bool(ratio is not None and ratio > hard)
            over_95 = bool(ratio is not None and ratio >= stress)
            overrun = overrun or (gated and over_hard)
            tier_rows[tier] = {
                "actual_ratio": _json_optional_float(ratio),
                "hard_cap_over": over_hard,
                "stress_95_over": over_95,
            }
        rows.append(
            {
                "denominator": "slice_actual_light_total",
                "denominator_value": _json_optional_float(slice_light if n else None),
                "gated": gated,
                "hard_cap_overrun": overrun,
                "kind": kind,
                "n": n,
                "name": name,
                "tiers": tier_rows,
            }
        )
    rows.sort(key=lambda row: (row["kind"], row["name"]))
    return rows


def promotion_gate(
    results: Mapping[str, Mapping[str, Any]],
    views_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    ratio_views: Mapping[str, Sequence[Mapping[str, Any]]],
    stress: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    reference = results[QUALITY_REFERENCE]["pooled"]
    baseline_q = float(reference["quality_weighted"])
    reference_budget_valid = _hard_caps_ok(reference)
    rows = []
    for name in CANDIDATE_ORDER:
        pooled = results[name]["pooled"]
        quality = float(pooled["quality_weighted"])
        delta = quality - baseline_q
        view_fail = [
            f"{row['kind']}:{row['name']}"
            for row in views_by_candidate[name]
            if row["kind"] in GATE_VIEW_KINDS
            and row["gated"]
            and row["delta"] is not None
            and row["delta"] < -GATE_VIEW_DROP
        ]
        ratio_fail = [
            f"{row['kind']}:{row['name']}"
            for row in ratio_views[name]
            if row["kind"] in GATE_VIEW_KINDS and row["hard_cap_overrun"]
        ]
        fold_caps = all(_hard_caps_ok(row) for row in results[name]["per_fold"])
        pooled_caps = _hard_caps_ok(pooled)
        stress_ok = True
        stress_fail = []
        for tier in TIERS:
            block = stress[name]["bootstrap"][tier]
            if float(block["q99_9"]) >= STRESS_95_CAPS[tier]:
                stress_ok = False
                stress_fail.append(f"bootstrap_q99_9:{tier}")
        coverage_ok = all(
            coverage["models"][model_id]["q90_coverage"]["slack_ok"]
            and coverage["models"][model_id]["q99_coverage"]["slack_ok"]
            for model_id in MODEL_IDS
        )
        quality_ok = delta >= -GATE_QUALITY_SLACK
        preferred = delta >= GATE_QUALITY_GAIN
        independent_safety_ok = bool(
            pooled_caps and fold_caps and not ratio_fail and stress_ok
        )
        passed = bool(
            independent_safety_ok
            and not view_fail
            and coverage_ok
            and quality_ok
        )
        rows.append(
            {
                "candidate": name,
                "coverage_ok": coverage_ok,
                "delta_vs_point_baseline": _json_float(delta),
                "fold_caps_ok": fold_caps,
                "independent_safety_ok": independent_safety_ok,
                "pass": passed,
                "pooled_caps_ok": pooled_caps,
                "preferred_quality_gain": preferred,
                "quality_ok": quality_ok,
                "quality_weighted": _json_float(quality),
                "ratio_view_failures": ratio_fail,
                "stress_failures": stress_fail,
                "stress_ok": stress_ok,
                "view_failures": view_fail,
                "views_ok": not view_fail,
            }
        )
    winners = [row for row in rows if row["pass"]]
    recommended = None
    if winners:
        recommended = max(
            winners,
            key=lambda row: (
                row["preferred_quality_gain"],
                row["quality_weighted"],
                row["candidate"],
            ),
        )["candidate"]
    independent_failures = [
        row["candidate"]
        for row in rows
        if not row["independent_safety_ok"]
    ]
    return {
        "baseline": QUALITY_REFERENCE,
        "baseline_quality_weighted": _json_float(baseline_q),
        "candidates": rows,
        "next_experiment_needs_safe_quality_baseline": True,
        "passed": bool(winners),
        "quality_ok_is_not_sufficient": True,
        "quality_reference_policy": QUALITY_REFERENCE_POLICY,
        "recommended": recommended,
        "reference_budget_valid": bool(reference_budget_valid),
        "safety_independent_failures": independent_failures,
        "thresholds": {
            "coverage_slack": COVERAGE_SLACK,
            "gated_quality_view_kinds": list(GATE_VIEW_KINDS),
            "gated_ratio_view_kinds": list(GATE_VIEW_KINDS),
            "quality_gain_preferred": GATE_QUALITY_GAIN,
            "quality_slack": GATE_QUALITY_SLACK,
            "ratio_denominator": RATIO_VIEW_DENOMINATOR,
            "stress_95_gate_statistic": "grouped_bootstrap_q99_9",
            "stress_95_max_reported": True,
            "stress_hard_cap_views": (
                "pooled official + fold-local official + deterministic "
                "family/fold/language/length/split actual-ratio slices "
                "(denominator = slice actual light total)"
            ),
            "view_drop": GATE_VIEW_DROP,
        },
    }


def episode_audit_document(
    pool: PublicPool,
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    bundle: Mapping[str, Any],
    models_by_name: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    rows = []
    for index, episode in enumerate(pool.episodes):
        selected = {
            name: {tier: str(models_by_name[name][tier][index]) for tier in TIERS}
            for name in CANDIDATE_ORDER
        }
        rows.append(
            {
                "episode_id": episode.episode_id,
                "family": pool.families[index],
                "fold": int(pool.folds[index]),
                "group_key": pool.group_keys[index],
                "language": pool.languages[index],
                "length_view": pool.length_views[index],
                "pred_qa": _json_float(pred_qa[index]),
                "pred_qk": _json_float(pred_qk[index]),
                "pred_costs": {
                    "conservative": [_json_float(value) for value in bundle["conservative"][index]],
                    "point": [_json_float(value) for value in bundle["point"][index]],
                    "q90": [_json_float(value) for value in bundle["q90"][index]],
                    "q99": [_json_float(value) for value in bundle["q99"][index]],
                },
                "selected": selected,
                "sigma": [_json_float(value) for value in bundle["sigma"][index]],
                "split": pool.split_labels[index],
            }
        )
    return {
        "experiment": EXPERIMENT,
        "n_rows": len(rows),
        "prompt_text_included": False,
        "quality_signal": QUALITY_SIGNAL,
        "rows": rows,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "audit": report["audit"],
            "candidates": report["candidates"],
            "cost_accuracy": report["cost_accuracy"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "family_guard_overlap": report["family_guard_overlap"],
            "feature": report["feature"],
            "fold_calibration": report["fold_calibration"],
            "identity": report["identity"],
            "inner_protocol": report["inner_protocol"],
            "limitations": report["limitations"],
            "promotion_gate": report["promotion_gate"],
            "quality_signal": report["quality_signal"],
            "report_type": report["report_type"],
            "results": report["results"],
            "schema_version": report["schema_version"],
            "stress": report["stress"],
            "stress_views": report["stress_views"],
        }
    )


def assemble(pool: PublicPool) -> Tuple[dict[str, Any], dict[str, Any]]:
    pred_qa, pred_qk = oof_quality_baseline(pool)
    bundle = oof_cost_surfaces(pool)
    surfaces = candidate_surfaces(bundle)
    tie_keys = content_tie_keys(pool.texts)
    content_groups = pool.exact_keys
    shared_denom = predicted_light_total(bundle["point"][:, 0], bundle["denom_scale"])
    results = {}
    models_by_name = {}
    for name in CANDIDATE_ORDER:
        surface = surfaces[name]
        denom = shared_denom
        models = allocate_candidate(
            name, pred_qa, pred_qk, surface, denom, tie_keys, content_groups
        )
        models_by_name[name] = models
        pooled = evaluate_allocation(pool, models)
        per_fold = []
        fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
        for fold in range(int(fold_ids.max()) + 1):
            indexes = [index for index, value in enumerate(pool.folds) if value == fold]
            mask = fold_ids == fold
            local_denom = predicted_light_total(
                bundle["point"][mask, 0], bundle["denom_scale"][mask]
            )
            local_surface = {
                "buy": surface["buy"][mask],
                "kind": surface["kind"],
                "settle": {
                    tier: matrix[mask] for tier, matrix in surface["settle"].items()
                },
            }
            local_models = allocate_candidate(
                name,
                pred_qa[mask],
                pred_qk[mask],
                local_surface,
                local_denom,
                tuple(tie_keys[index] for index in indexes),
                tuple(content_groups[index] for index in indexes),
            )
            local = evaluate_allocation(pool, local_models, indexes=indexes)
            per_fold.append(
                {
                    "fold": fold,
                    "n": int(mask.sum()),
                    "official_final_score": local["official_final_score"],
                    "predicted_light_total": _json_float(local_denom),
                    "quality_weighted": local["quality_weighted"],
                    "tiers": local["tiers"],
                }
            )
        results[name] = {
            "definition": dict(CANDIDATE_DEFINITIONS[name]),
            "name": name,
            "per_fold": per_fold,
            "per_fold_note": (
                "Fold-local reallocation uses that fold's predicted light "
                "denominator. Observational alongside pooled allocation."
            ),
            "pooled": pooled,
            "predicted_light_total": _json_float(denom),
        }

    views = {
        name: _quality_views(pool, models_by_name["point_cost_baseline"], models_by_name[name])
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
                bundle["actual_costs"],
                bundle["actual_costs"][:, 0],
                pool.group_keys,
                draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
            bootstrap[tier] = _bootstrap_with_caps(
                block, OFFICIAL_CAPS[tier], STRESS_95_CAPS[tier]
            )
            mix = family_mixture_ratios(
                models_by_name[name][tier],
                bundle["actual_costs"],
                bundle["actual_costs"][:, 0],
                pool.families,
                draws=MIXTURE_DRAWS,
                seed=MIXTURE_SEED,
                batch=MIXTURE_N,
            )
            mixture[tier] = _mixture_with_caps(mix, OFFICIAL_CAPS[tier], STRESS_95_CAPS[tier])
        stress[name] = {"bootstrap": bootstrap, "family_mixture": mixture}

    coverage = cost_accuracy(bundle)
    gate = promotion_gate(results, views, ratio_views, stress, coverage)
    decision = (
        f"record-e2-promote-{gate['recommended']}"
        if gate["recommended"]
        else "record-e2-no-promote"
    )
    decision_reason = (
        "no-promote holds from independent hard-cap, bootstrap q99.9, "
        "and gated actual-ratio failures "
        f"({', '.join(gate['safety_independent_failures']) or 'none'}). "
        "Quality slack versus point_cost_baseline is not a promotion "
        "license because that reference itself is not budget-valid. "
        "The next experiment needs a budget-valid quality baseline."
        if decision == "record-e2-no-promote"
        else (
            f"promote {gate['recommended']} after independent safety and "
            "quality checks; runtime export still requires a later phase."
        )
    )
    audit_document = episode_audit_document(
        pool, pred_qa, pred_qk, bundle, models_by_name
    )
    audit_sha = sha256_text(canonical_json_text(audit_document))
    limitations = [
        RUNTIME_DIFFS,
        "Quality is frozen at E1 baseline_continuous_uplift OOF. Failed E1 "
        "objectives are not mixed into this comparison.",
        "Spread/tail uses inner grouped OOF residuals on each outer-train "
        "complement (the other four Phase-1 fold labels). In-sample residuals "
        "are not used. k/z/quantiles are pre-registered or fit on that inner "
        "evidence only.",
        "Allocator denominators are OOF predicted light totals. Actual light "
        "bills enter official scoring and stress only.",
        "Exact public costs remain non-monotone on 24 AX31<Light and 1 "
        "K1<AX31 rows. Predicted costs are raise-only monotone. Actual "
        "scores still use the raw public cost matrix.",
        "Family-guard multipliers are not applied. Item sigma vs family "
        "ratio correlation is diagnostic only.",
        "Grouped-bootstrap q99.9 is the pre-registered 95%-cap gate "
        "statistic; bootstrap max is reported and is not the gate. Hard-cap "
        "overrun must be zero on pooled official scores, fold-local official "
        "scores, and deterministic family/fold/language/length/split "
        "actual-ratio slices (slice actual light total).",
        QUALITY_REFERENCE_POLICY,
        "A passing candidate is not exported into src/ in this phase.",
    ]
    report = {
        "audit": {
            "n_rows": int(audit_document["n_rows"]),
            "relative_path": AUDIT_RELATIVE_PATH,
            "sha256": audit_sha,
        },
        "candidates": {
            name: dict(CANDIDATE_DEFINITIONS[name]) for name in CANDIDATE_ORDER
        },
        "cost_accuracy": coverage,
        "decision": decision,
        "decision_reason": decision_reason,
        "experiment": EXPERIMENT,
        "family_guard_overlap": family_guard_overlap(pool, bundle["sigma"]),
        "feature": {
            "alpha_mean": COST_ALPHA,
            "alpha_sigma": SIGMA_ALPHA,
            "dimension": 1 + FEATURE_DIM + HASH_BINS,
            "hash_bins": HASH_BINS,
            "name": "runtime structural_features/14 + hashed_features/256",
            "runtime_artifact_changed": False,
            "standardize": "outer-train columns 1.. excluding intercept",
            "target": "log1p input/output tokens per model, assembled with policy rates",
        },
        "fold_calibration": bundle["fold_calibration"],
        "identity": dict(pool.identity),
        "inner_protocol": {
            "inner_folds": "the other Phase-1 group-fold labels inside each outer-train complement",
            "n_inner_ridge_fits": bundle["n_inner_ridge_fits"],
            "n_outer_mean_fits": bundle["n_outer_mean_fits"],
            "residual_source": "inner OOF token/cost residuals, never in-sample, never outer held-out",
        },
        "limitations": limitations,
        "promotion_gate": gate,
        "quality_signal": QUALITY_SIGNAL,
        "report_type": REPORT_TYPE,
        "results": results,
        "runtime": {
            "complexity": {
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "mixture_draws": MIXTURE_DRAWS,
                "n_candidates": len(CANDIDATE_ORDER),
            },
            "excluded_from_core": ["elapsed_s"],
        },
        "schema_version": SCHEMA_VERSION,
        "stress": stress,
        "stress_views": {
            "quality": views,
            "ratio": ratio_views,
            "ratio_denominator": RATIO_VIEW_DENOMINATOR,
        },
    }
    report["decision_core_sha256"] = decision_core_sha256_report(report)
    return sort_mapping(report), audit_document


def decision_core_sha256_report(report: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        decision_core_payload(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(encoded)


def _bootstrap_with_caps(
    block: Mapping[str, Any], hard: float, stress: float
) -> dict[str, Any]:
    samples = np.asarray(block["samples"], dtype=np.float64)
    return {
        "draws": block["draws"],
        "hard_cap": _json_float(hard),
        "hard_cap_overrun": int(np.count_nonzero(samples > hard)),
        "max": block["max"],
        "max_over_hard": bool(float(block["max"]) > hard),
        "mean": block["mean"],
        "q99_9": block["q99_9"],
        "q99_9_over_stress_95": bool(float(block["q99_9"]) >= stress),
        "seed": block["seed"],
        "stress_95_cap": _json_float(stress),
        "stress_95_overrun": int(np.count_nonzero(samples >= stress)),
    }


def _mixture_with_caps(
    block: Mapping[str, Any], hard: float, stress: float
) -> dict[str, Any]:
    row = dict(block)
    row["hard_cap"] = _json_float(hard)
    row["stress_95_cap"] = _json_float(stress)
    if block["max"] is None:
        row["max_over_hard"] = None
        row["q99_9_over_stress_95"] = None
        return row
    row["max_over_hard"] = bool(float(block["max"]) > hard)
    row["q99_9_over_stress_95"] = bool(float(block["q99_9"]) >= stress)
    return row


def measure(pool: PublicPool) -> dict[str, Any]:
    report, _audit = assemble(pool)
    return report


__all__ = (
    "AUDIT_RELATIVE_PATH",
    "CANDIDATE_ORDER",
    "GATE_VIEW_KINDS",
    "QUALITY_REFERENCE",
    "QUALITY_SIGNAL",
    "allocate_candidate",
    "assemble",
    "clamp_predicted_costs",
    "cost_feature_matrix",
    "measure",
    "oof_cost_surfaces",
    "oof_quality_baseline",
    "predicted_light_total",
    "write_json_atomic",
)
