# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1B — hashed / logistic / residual quality heads on exact public costs.

Quality reference is the E1 ``baseline_continuous_uplift`` exact-cost policy.
New heads may use hashed structural features or a shallow residual tree, but
every candidate is scored with the same exact-public-cost greedy allocator.
Phase-2 predicted-cost / two-price surfaces are never mixed in.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ossp_router.cost_calibrated_router import hashed_features, structural_features
from ossp_router.protocol import TIERS
from research.lab.e1_objectives import (
    ALLOCATOR,
    BASELINE_NAME as E1_BASELINE,
    FEATURE_DIM,
    GATE_VIEW_DROP,
    GATE_VIEW_KINDS,
    GATE_WEIGHTED_GAIN,
    RIDGE_ALPHA,
    VIEW_MIN_N,
    allocate_all_tiers,
    canonical_json_text,
    current_quality_matrix,
    exact_cost_diagnostic,
    oof_candidate_predictions,
    score_decisions,
    sha256_text,
    stress_views,
    write_json_atomic,
)
from research.lab.modeling import ridge_fit, ridge_predict, sort_mapping
from research.lab.public_pool import PublicPool
from research.lab.quality_heads import content_tie_keys, pearson, spearman


EXPERIMENT = "e1b-quality-models"
REPORT_TYPE = "scrooge-e1b-quality-models-v1"
SCHEMA_VERSION = 1
BASELINE_NAME = "structural_baseline"
HASH_BINS = 256
HASH_ALGORITHM = "fnv1a64"
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 500
TREE_N_ESTIMATORS = 32
TREE_MAX_DEPTH = 3
TREE_MIN_SAMPLES_LEAF = 8
TREE_MAX_FEATURES = 1.0
TREE_RANDOM_STATE = 20260821
SEED = 20260821
CHAMPION_ABS = 0.690
COMPLEXITY_MIN_GAIN = 0.001
EXPECTED_E1_BASELINE_QUALITY = 0.6877178030302
AUDIT_RELATIVE_PATH = "build/compare-e1b-quality-models/episode-audit.json"
_LIGHT = 0
_AX31 = 1
_K1 = 2

CANDIDATE_ORDER: Tuple[str, ...] = (
    "structural_baseline",
    "hashed_adjacent_ridge",
    "hashed_logistic_hybrid",
    "robust_hashed_hybrid",
    "shallow_structural_residual",
)

COMPLEXITY_RANK: Mapping[str, int] = {
    "structural_baseline": 0,
    "hashed_adjacent_ridge": 1,
    "hashed_logistic_hybrid": 2,
    "robust_hashed_hybrid": 3,
    "shallow_structural_residual": 4,
}

CANDIDATE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "structural_baseline": {
        "complexity_rank": 0,
        "features": "intercept + 14-d runtime structural_features",
        "pred_qa": "Ridge(score(ax31)-score(light)), alpha=100",
        "pred_qk": "Ridge(score(k1)-score(light)), alpha=100",
        "summary": "Byte-equivalent E1 baseline_continuous_uplift exact-cost policy.",
    },
    "hashed_adjacent_ridge": {
        "complexity_rank": 1,
        "features": "intercept + 14-d structural + 256 signed FNV unigram/bigram",
        "pred_qa": "Ridge(Δ31), alpha=100, outer-train standardized",
        "pred_qk": "Ridge(Δk1 adjacent), alpha=100, outer-train standardized",
        "summary": "Adjacent-step hashed Ridge. Alpha is pre-fixed, not nested-tuned.",
    },
    "hashed_logistic_hybrid": {
        "complexity_rank": 2,
        "features": "same hashed matrix as hashed_adjacent_ridge",
        "pred_qa": "Ridge(Δ31) * LogisticP(Δ31>0)",
        "pred_qk": "Ridge(Δk1) * LogisticP(Δk1>0)",
        "summary": (
            "Proper logistic P(Δ>0) times adjacent magnitude Ridge. "
            "Not the E1 Ridge-on-indicator hybrid."
        ),
    },
    "robust_hashed_hybrid": {
        "complexity_rank": 3,
        "features": "same hashed matrix; inverse family/language/length frequency weights",
        "pred_qa": "weighted Ridge(Δ31) * weighted LogisticP(Δ31>0)",
        "pred_qk": "weighted Ridge(Δk1) * weighted LogisticP(Δk1>0)",
        "summary": (
            "Same hybrid with outer-train frequency weights decided before "
            "targets are read. Not a post-hoc Korean/MC patch of E1 hybrid."
        ),
    },
    "shallow_structural_residual": {
        "complexity_rank": 4,
        "features": "14-d structural only (no hash-256 in the tree)",
        "pred_qa": "structural Ridge(Δ31-from-light) + ExtraTrees(inner-OOF residual)",
        "pred_qk": "structural Ridge(Δk1-from-light) + ExtraTrees(inner-OOF residual)",
        "summary": (
            "Shallow residual on the E1 baseline targets. Residuals are "
            "inner grouped OOF inside each outer-train complement."
        ),
    },
}


@dataclass(frozen=True)
class HeadPred:
    pred_qa: np.ndarray
    pred_qk: np.ndarray
    mag_qa: Optional[np.ndarray] = None
    mag_qk: Optional[np.ndarray] = None
    prob_qa: Optional[np.ndarray] = None
    prob_qk: Optional[np.ndarray] = None
    residual_qa: Optional[np.ndarray] = None
    residual_qk: Optional[np.ndarray] = None


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def _json_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(np.float64(value))
    if not np.isfinite(number):
        return None
    return number


def hashed_quality_matrix(episodes: Sequence[Any]) -> np.ndarray:
    rows = len(episodes)
    matrix = np.ones((rows, 1 + FEATURE_DIM + HASH_BINS), dtype=np.float64)
    for row, episode in enumerate(episodes):
        structural = structural_features(episode)
        if len(structural) != FEATURE_DIM:
            raise RuntimeError("structural_features width drifted")
        hashed = hashed_features(episode, HASH_BINS, HASH_ALGORITHM)
        matrix[row, 1 : 1 + FEATURE_DIM] = structural
        matrix[row, 1 + FEATURE_DIM :] = hashed
    return matrix


def standardize_apply(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train[:, 1:].mean(axis=0)
    scale = train[:, 1:].std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    out_train = train.copy()
    out_test = test.copy()
    out_train[:, 1:] = (train[:, 1:] - mean) / scale
    out_test[:, 1:] = (test[:, 1:] - mean) / scale
    return out_train, out_test


def ridge_fit_weighted(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weights is None:
        return ridge_fit(features, target, alpha=float(alpha))
    scale = np.sqrt(np.asarray(weights, dtype=np.float64).reshape(-1))
    return ridge_fit(features * scale[:, None], target * scale, alpha=float(alpha))


def frequency_weights(
    families: Sequence[str],
    languages: Sequence[str],
    length_views: Sequence[str],
) -> np.ndarray:
    """Inverse family/language/length frequency. Targets are not consulted."""

    def _inv(values: Sequence[str]) -> np.ndarray:
        counts = Counter(values)
        return np.asarray([1.0 / float(counts[value]) for value in values], dtype=np.float64)

    weights = _inv(families) * _inv(languages) * _inv(length_views)
    mean = float(weights.mean())
    if mean <= 0.0:
        return np.ones(len(families), dtype=np.float64)
    return weights / mean


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> LogisticRegression | float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.min() == y.max():
        return float(y[0])
    model = LogisticRegression(
        C=LOGISTIC_C,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=SEED,
        tol=1e-6,
    )
    model.fit(features, y, sample_weight=weights)
    return model


def _predict_logistic(model: LogisticRegression | float, features: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(features.shape[0], model, dtype=np.float64)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def _oof_hashed_ridge(
    features: np.ndarray,
    target: np.ndarray,
    folds: Sequence[int],
    *,
    weight_rows: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    fold_ids = np.asarray(list(folds), dtype=np.int64)
    predicted = np.empty(y.shape[0], dtype=np.float64)
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        test = fold_ids == fold
        x_train, x_test = standardize_apply(matrix[train], matrix[test])
        w_train = None
        if weight_rows is not None:
            rows = [weight_rows[index] for index in np.flatnonzero(train)]
            w_train = frequency_weights(
                [row[0] for row in rows],
                [row[1] for row in rows],
                [row[2] for row in rows],
            )
        coef = ridge_fit_weighted(x_train, y[train], alpha=RIDGE_ALPHA, weights=w_train)
        predicted[test] = ridge_predict(coef, x_test)
    return predicted


def _oof_hashed_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[int],
    *,
    weight_rows: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    fold_ids = np.asarray(list(folds), dtype=np.int64)
    predicted = np.empty(y.shape[0], dtype=np.float64)
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        test = fold_ids == fold
        x_train, x_test = standardize_apply(matrix[train], matrix[test])
        w_train = None
        if weight_rows is not None:
            rows = [weight_rows[index] for index in np.flatnonzero(train)]
            w_train = frequency_weights(
                [row[0] for row in rows],
                [row[1] for row in rows],
                [row[2] for row in rows],
            )
        model = _fit_logistic(x_train, y[train], weights=w_train)
        predicted[test] = _predict_logistic(model, x_test)
    return np.clip(predicted, 0.0, 1.0)


def _inner_oof_ridge(
    features: np.ndarray, target: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    predicted = np.empty(target.shape[0], dtype=np.float64)
    for fold in np.unique(folds):
        train = folds != fold
        test = folds == fold
        coef = ridge_fit(features[train], target[train], alpha=RIDGE_ALPHA)
        predicted[test] = ridge_predict(coef, features[test])
    return predicted


def _fit_residual_tree(features: np.ndarray, residual: np.ndarray) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=TREE_N_ESTIMATORS,
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        max_features=TREE_MAX_FEATURES,
        criterion="squared_error",
        bootstrap=False,
        n_jobs=1,
        random_state=TREE_RANDOM_STATE,
    )
    model.fit(features, residual)
    return model


def oof_structural_residual(
    structural: np.ndarray,
    target: np.ndarray,
    folds: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (base + residual, residual) with inner-OOF residual labels."""

    features = np.asarray(structural, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    fold_ids = np.asarray(list(folds), dtype=np.int64)
    combined = np.empty(y.shape[0], dtype=np.float64)
    residual_hat = np.empty(y.shape[0], dtype=np.float64)
    tree_x = features[:, 1:]
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        test = fold_ids == fold
        inner = _inner_oof_ridge(features[train], y[train], fold_ids[train])
        resid = y[train] - inner
        tree = _fit_residual_tree(tree_x[train], resid)
        base = ridge_fit(features[train], y[train], alpha=RIDGE_ALPHA)
        residual_hat[test] = tree.predict(tree_x[test])
        combined[test] = ridge_predict(base, features[test]) + residual_hat[test]
    return combined, residual_hat


def oof_all_heads(
    pool: PublicPool,
    *,
    scores: Optional[np.ndarray] = None,
) -> dict[str, HeadPred]:
    """Grouped OOF heads. ``scores`` defaults to the pool; tests may swap labels."""

    y = pool.scores if scores is None else np.asarray(scores, dtype=np.float64)
    structural = current_quality_matrix(pool.episodes)
    hashed = hashed_quality_matrix(pool.episodes)
    delta_al = y[:, _AX31] - y[:, _LIGHT]
    delta_kl = y[:, _K1] - y[:, _LIGHT]
    delta_ka = y[:, _K1] - y[:, _AX31]
    sign_al = (delta_al > 0.0).astype(np.int64)
    sign_ka = (delta_ka > 0.0).astype(np.int64)
    weight_rows = tuple(zip(pool.families, pool.languages, pool.length_views))

    e1 = oof_candidate_predictions(structural, y, pool.folds)
    base_qa, base_qk = e1[E1_BASELINE]
    hashed_qa = _oof_hashed_ridge(hashed, delta_al, pool.folds)
    hashed_qk = _oof_hashed_ridge(hashed, delta_ka, pool.folds)
    prob_qa = _oof_hashed_logistic(hashed, sign_al, pool.folds)
    prob_qk = _oof_hashed_logistic(hashed, sign_ka, pool.folds)
    robust_qa = _oof_hashed_ridge(hashed, delta_al, pool.folds, weight_rows=weight_rows)
    robust_qk = _oof_hashed_ridge(hashed, delta_ka, pool.folds, weight_rows=weight_rows)
    robust_p_qa = _oof_hashed_logistic(hashed, sign_al, pool.folds, weight_rows=weight_rows)
    robust_p_qk = _oof_hashed_logistic(hashed, sign_ka, pool.folds, weight_rows=weight_rows)
    resid_qa, resid_hat_qa = oof_structural_residual(structural, delta_al, pool.folds)
    resid_qk, resid_hat_qk = oof_structural_residual(structural, delta_kl, pool.folds)
    return {
        "structural_baseline": HeadPred(base_qa, base_qk, mag_qa=base_qa, mag_qk=base_qk),
        "hashed_adjacent_ridge": HeadPred(
            hashed_qa, hashed_qk, mag_qa=hashed_qa, mag_qk=hashed_qk
        ),
        "hashed_logistic_hybrid": HeadPred(
            hashed_qa * prob_qa,
            hashed_qk * prob_qk,
            mag_qa=hashed_qa,
            mag_qk=hashed_qk,
            prob_qa=prob_qa,
            prob_qk=prob_qk,
        ),
        "robust_hashed_hybrid": HeadPred(
            robust_qa * robust_p_qa,
            robust_qk * robust_p_qk,
            mag_qa=robust_qa,
            mag_qk=robust_qk,
            prob_qa=robust_p_qa,
            prob_qk=robust_p_qk,
        ),
        "shallow_structural_residual": HeadPred(
            resid_qa,
            resid_qk,
            mag_qa=base_qa,
            mag_qk=base_qk,
            residual_qa=resid_hat_qa,
            residual_qk=resid_hat_qk,
        ),
    }


def _ranking_block(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    unequal = target != 0.0
    pred_pos = pred > 0.0
    actual_pos = target > 0.0
    unequal_n = int(np.count_nonzero(unequal))
    residual = pred - target
    return {
        "mae": _json_float(np.mean(np.abs(residual))),
        "mse": _json_float(np.mean(residual * residual)),
        "n_unequal": unequal_n,
        "pearson": _json_optional_float(pearson(pred, target)),
        "sign_accuracy_unequal": (
            _json_float(np.mean(pred_pos[unequal] == actual_pos[unequal]))
            if unequal_n
            else None
        ),
        "spearman": _json_optional_float(spearman(pred, target)),
    }


def _classifier_block(prob: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    labels = (np.asarray(target, dtype=np.float64) > 0.0).astype(np.int64)
    values = np.clip(np.asarray(prob, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    finite = bool(np.all(np.isfinite(prob)) and np.all((prob >= 0.0) & (prob <= 1.0)))
    auc = None
    if labels.min() != labels.max():
        auc = _json_optional_float(roc_auc_score(labels, values))
    return {
        "brier": _json_float(brier_score_loss(labels, values)),
        "finite_unit_interval": finite,
        "log_loss": _json_float(log_loss(labels, np.column_stack([1.0 - values, values]))),
        "roc_auc": auc,
        "sign_accuracy": _json_float(np.mean((values >= 0.5) == labels.astype(bool))),
    }


def _hard_caps_ok(scored: Mapping[str, Any]) -> bool:
    return all(scored["tiers"][tier]["within_hard_cap"] for tier in TIERS)


def evaluate_candidate(
    pool: PublicPool,
    name: str,
    head: HeadPred,
    tie_keys: Sequence[str],
) -> dict[str, Any]:
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    pooled_models = allocate_all_tiers(
        head.pred_qa, head.pred_qk, pool.costs, pool.light_total, tie_keys
    )
    pooled = score_decisions(pool, pooled_models)
    per_fold = []
    for fold in range(int(max(pool.folds)) + 1):
        indexes = [index for index, value in enumerate(pool.folds) if value == fold]
        mask = fold_ids == fold
        local_models = allocate_all_tiers(
            head.pred_qa[mask],
            head.pred_qk[mask],
            pool.costs[mask],
            float(pool.costs[mask, _LIGHT].sum()),
            tuple(tie_keys[index] for index in indexes),
        )
        local = score_decisions(pool, local_models, indexes=indexes)
        per_fold.append(
            {
                "fold": fold,
                "n": int(mask.sum()),
                "official_final_score": local["official_final_score"],
                "quality_weighted": local["quality_weighted"],
                "tiers": local["tiers"],
            }
        )
    delta_al = pool.scores[:, _AX31] - pool.scores[:, _LIGHT]
    delta_ka = pool.scores[:, _K1] - pool.scores[:, _AX31]
    ranking = {
        "qa_vs_delta_ax31_light": _ranking_block(head.pred_qa, delta_al),
        "qk_vs_delta_k1_ax31": _ranking_block(head.pred_qk, delta_ka),
    }
    classifier = None
    if head.prob_qa is not None and head.prob_qk is not None:
        classifier = {
            "qa_p_delta_ax31_light": _classifier_block(head.prob_qa, delta_al),
            "qk_p_delta_k1_ax31": _classifier_block(head.prob_qk, delta_ka),
        }
    return {
        "classifier": classifier,
        "definition": dict(CANDIDATE_DEFINITIONS[name]),
        "fold_caps_ok": all(_hard_caps_ok(row) for row in per_fold),
        "name": name,
        "per_fold": per_fold,
        "per_fold_note": (
            "Fold-local reallocation is observational for quality views but "
            "is gated for official hard caps."
        ),
        "pooled": pooled,
        "ranking": ranking,
    }


def _split_deltas(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {
        row["name"]: row
        for row in views
        if row["kind"] == "split"
    }
    return {
        name: {
            "delta": row["delta"],
            "n": row["n"],
            "quality": row["candidate_quality_weighted"],
        }
        for name, row in sorted(rows.items())
    }


def promotion_gate(
    results: Mapping[str, Mapping[str, Any]],
    views_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    baseline = results[BASELINE_NAME]["pooled"]
    baseline_quality = float(baseline["quality_weighted"])
    rows = []
    for name in CANDIDATE_ORDER:
        pooled = results[name]["pooled"]
        quality = float(pooled["quality_weighted"])
        delta = quality - baseline_quality
        views = list(views_by_candidate[name])
        view_fail = [
            row
            for row in views
            if row["kind"] in GATE_VIEW_KINDS and row["worse_than_gate"]
        ]
        pooled_caps = _hard_caps_ok(pooled)
        fold_caps = bool(results[name]["fold_caps_ok"])
        gain_ok = delta >= GATE_WEIGHTED_GAIN
        abs_ok = quality >= CHAMPION_ABS
        views_ok = not view_fail
        passed = bool(
            name != BASELINE_NAME
            and gain_ok
            and abs_ok
            and views_ok
            and pooled_caps
            and fold_caps
        )
        rows.append(
            {
                "absolute_ok": abs_ok,
                "candidate": name,
                "complexity_rank": COMPLEXITY_RANK[name],
                "delta_vs_baseline": _json_float(delta),
                "fold_caps_ok": fold_caps,
                "gain_ok": gain_ok,
                "pass": passed,
                "pooled_caps_ok": pooled_caps,
                "quality_weighted": _json_float(quality),
                "view_failures": [f"{row['kind']}:{row['name']}" for row in view_fail],
                "views_ok": views_ok,
            }
        )
    winners = [row for row in rows if row["pass"]]
    if winners:
        simplest = min(row["complexity_rank"] for row in winners)
        kept = []
        for row in winners:
            if row["complexity_rank"] == simplest:
                kept.append(row)
                continue
            simpler_best = max(
                item["quality_weighted"]
                for item in winners
                if item["complexity_rank"] < row["complexity_rank"]
            )
            if row["quality_weighted"] - simpler_best >= COMPLEXITY_MIN_GAIN:
                kept.append(row)
        winners = kept
    recommended = None
    if winners:
        recommended = min(
            winners,
            key=lambda row: (row["complexity_rank"], -row["quality_weighted"], row["candidate"]),
        )["candidate"]
    return {
        "baseline": BASELINE_NAME,
        "baseline_e1_name": E1_BASELINE,
        "baseline_quality_weighted": _json_float(baseline_quality),
        "candidates": rows,
        "champion_absolute": CHAMPION_ABS,
        "complexity_min_gain": COMPLEXITY_MIN_GAIN,
        "passed": bool(winners),
        "recommended": recommended,
        "thresholds": {
            "both_gain_and_absolute_required": True,
            "fold_caps": "fold-local official hard caps",
            "gated_view_kinds": list(GATE_VIEW_KINDS),
            "stress_95_not_gated": True,
            "view_drop": GATE_VIEW_DROP,
            "view_min_n": VIEW_MIN_N,
            "weighted_gain": GATE_WEIGHTED_GAIN,
        },
    }


def episode_audit_document(
    pool: PublicPool, heads: Mapping[str, HeadPred], models_by_name: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    for index, episode in enumerate(pool.episodes):
        predictions = {}
        selected = {}
        for name in CANDIDATE_ORDER:
            head = heads[name]
            payload = {
                "pred_qa": _json_float(head.pred_qa[index]),
                "pred_qk": _json_float(head.pred_qk[index]),
            }
            if head.prob_qa is not None and head.prob_qk is not None:
                payload["prob_qa"] = _json_float(head.prob_qa[index])
                payload["prob_qk"] = _json_float(head.prob_qk[index])
            if head.residual_qa is not None and head.residual_qk is not None:
                payload["residual_qa"] = _json_float(head.residual_qa[index])
                payload["residual_qk"] = _json_float(head.residual_qk[index])
            predictions[name] = payload
            selected[name] = {
                tier: str(models_by_name[name][tier][index]) for tier in TIERS
            }
        rows.append(
            {
                "episode_id": episode.episode_id,
                "family": pool.families[index],
                "fold": int(pool.folds[index]),
                "group_key": pool.group_keys[index],
                "language": pool.languages[index],
                "length_view": pool.length_views[index],
                "predictions": predictions,
                "selected": selected,
                "split": pool.split_labels[index],
            }
        )
    return {
        "allocation": "pooled_exact_cost_greedy",
        "experiment": EXPERIMENT,
        "n_rows": len(rows),
        "prompt_text_included": False,
        "quality_reference": BASELINE_NAME,
        "rows": rows,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "allocator": report["allocator"],
            "audit": report["audit"],
            "candidates": report["candidates"],
            "cost_diagnostic": report["cost_diagnostic"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "feature": report["feature"],
            "fold_table": report["fold_table"],
            "grouping": report["grouping"],
            "identity": report["identity"],
            "limitations": report["limitations"],
            "promotion_gate": report["promotion_gate"],
            "report_type": report["report_type"],
            "results": report["results"],
            "schema_version": report["schema_version"],
            "split_deltas": report["split_deltas"],
            "stress_view_kinds": report["stress_view_kinds"],
            "stress_views": report["stress_views"],
        }
    )


def decision_core_sha256(report: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        decision_core_payload(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(encoded)


def _complexity_block() -> dict[str, Any]:
    hashed_dim = 1 + FEATURE_DIM + HASH_BINS
    return {
        "hashed_adjacent_ridge": {
            "feature_columns": hashed_dim,
            "heads": "2 ridge",
            "param_floats": hashed_dim * 2,
        },
        "hashed_logistic_hybrid": {
            "feature_columns": hashed_dim,
            "heads": "2 ridge + 2 logistic",
            "param_floats": hashed_dim * 4,
        },
        "robust_hashed_hybrid": {
            "feature_columns": hashed_dim,
            "heads": "2 weighted ridge + 2 weighted logistic",
            "param_floats": hashed_dim * 4,
        },
        "shallow_structural_residual": {
            "feature_columns": 1 + FEATURE_DIM,
            "heads": "2 ridge + 2 ExtraTrees(32x depth 3)",
            "param_floats": (1 + FEATURE_DIM) * 2,
            "tree": {
                "max_depth": TREE_MAX_DEPTH,
                "min_samples_leaf": TREE_MIN_SAMPLES_LEAF,
                "n_estimators": TREE_N_ESTIMATORS,
            },
        },
        "structural_baseline": {
            "feature_columns": 1 + FEATURE_DIM,
            "heads": "2 ridge",
            "param_floats": (1 + FEATURE_DIM) * 2,
        },
    }


def assemble(pool: PublicPool) -> Tuple[dict[str, Any], dict[str, Any]]:
    heads = oof_all_heads(pool)
    tie_keys = content_tie_keys(pool.texts)
    results = {}
    models_by_name = {}
    for name in CANDIDATE_ORDER:
        head = heads[name]
        results[name] = evaluate_candidate(pool, name, head, tie_keys)
        models_by_name[name] = allocate_all_tiers(
            head.pred_qa, head.pred_qk, pool.costs, pool.light_total, tie_keys
        )
    views = {
        name: stress_views(pool, models_by_name[BASELINE_NAME], models_by_name[name])
        for name in CANDIDATE_ORDER
    }
    split_deltas = {name: _split_deltas(views[name]) for name in CANDIDATE_ORDER}
    gate = promotion_gate(results, views)
    baseline_q = float(results[BASELINE_NAME]["pooled"]["quality_weighted"])
    baseline_match = abs(baseline_q - EXPECTED_E1_BASELINE_QUALITY) < 1e-12
    decision = (
        f"record-e1b-promote-{gate['recommended']}"
        if gate["recommended"]
        else "record-e1b-no-promote"
    )
    if gate["recommended"]:
        decision_reason = (
            f"promote {gate['recommended']}: gain and 0.690 absolute, "
            "gated views, and exact-cost hard caps all passed. "
            "Runtime export still requires a later phase."
        )
    else:
        decision_reason = (
            "no-promote: no candidate cleared +0.002 gain AND 0.690 "
            "absolute AND gated views AND exact-cost pooled/fold hard caps "
            "after complexity filtering. 95% stress caps are not a gate."
        )
    audit_document = episode_audit_document(pool, heads, models_by_name)
    audit_sha = sha256_text(canonical_json_text(audit_document))
    present_kinds = sorted({row["kind"] for rows in views.values() for row in rows})
    cost_diagnostic = exact_cost_diagnostic(pool.costs)
    cost_diagnostic = dict(cost_diagnostic)
    cost_diagnostic["note"] = (
        "Exact public costs are not monotone on every row "
        f"(AX31<Light {cost_diagnostic['ax31_lt_light_rows']}, "
        f"K1<AX31 {cost_diagnostic['k1_lt_ax31_rows']}). "
        "Every E1B candidate uses this same cost matrix. "
        "Phase-2 predicted costs are not used."
    )
    limitations = [
        "Quality reference is E1 baseline_continuous_uplift exact-cost OOF, "
        f"pinned at {EXPECTED_E1_BASELINE_QUALITY}. reproduced={baseline_match}.",
        "Alpha=100, logistic C=1.0, hash bins=256, ExtraTrees 32x depth 3 "
        "are pre-registered. No outer held-out labels enter scaling, "
        "weights, logistic fit, or residual labels.",
        "Residual trees train on inner grouped OOF residuals inside each "
        "outer-train complement, not in-sample residuals.",
        "robust_hashed_hybrid weights are inverse family/language/length "
        "frequency computed on each outer-train complement only. Counts do "
        "not use scores. This is not a post-hoc Korean/MC correction of "
        "the E1 hybrid.",
        "E1 hybrid_magnitude_sign is not a candidate and is not renamed.",
        "95% stress ratio caps are observational and are not a promotion gate.",
        "Hashed unigrams/bigrams can memorize public wording; train/dev and "
        "grouped-OOF view gaps are the leakage check.",
        "A passing candidate is not exported into src/ in this phase.",
    ]
    report = {
        "allocator": dict(ALLOCATOR),
        "audit": {
            "n_rows": int(audit_document["n_rows"]),
            "relative_path": AUDIT_RELATIVE_PATH,
            "sha256": audit_sha,
        },
        "baseline_reproduction": {
            "e1_name": E1_BASELINE,
            "expected_quality_weighted": EXPECTED_E1_BASELINE_QUALITY,
            "matched": baseline_match,
            "observed_quality_weighted": _json_float(baseline_q),
        },
        "candidates": {
            name: dict(CANDIDATE_DEFINITIONS[name]) for name in CANDIDATE_ORDER
        },
        "cost_diagnostic": cost_diagnostic,
        "decision": decision,
        "decision_reason": decision_reason,
        "experiment": EXPERIMENT,
        "feature": {
            "hash_algorithm": HASH_ALGORITHM,
            "hash_bins": HASH_BINS,
            "logistic_c": LOGISTIC_C,
            "ridge_alpha": RIDGE_ALPHA,
            "runtime_artifact_changed": False,
            "standardize_hashed": "outer-train columns 1.. per fold",
            "structural_dim": FEATURE_DIM,
            "tree": {
                "max_depth": TREE_MAX_DEPTH,
                "min_samples_leaf": TREE_MIN_SAMPLES_LEAF,
                "n_estimators": TREE_N_ESTIMATORS,
                "random_state": TREE_RANDOM_STATE,
            },
        },
        "fold_table": list(pool.fold_table),
        "grouping": dict(pool.grouping),
        "identity": dict(pool.identity),
        "limitations": limitations,
        "promotion_gate": gate,
        "report_type": REPORT_TYPE,
        "results": results,
        "runtime": {
            "complexity": _complexity_block(),
            "excluded_from_core": ["elapsed_s"],
        },
        "schema_version": SCHEMA_VERSION,
        "split_deltas": split_deltas,
        "stress_view_kinds": present_kinds,
        "stress_views": views,
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    return sort_mapping(report), audit_document


def measure(pool: PublicPool) -> dict[str, Any]:
    report, _audit = assemble(pool)
    return report


__all__ = (
    "AUDIT_RELATIVE_PATH",
    "BASELINE_NAME",
    "CANDIDATE_ORDER",
    "HeadPred",
    "assemble",
    "frequency_weights",
    "hashed_quality_matrix",
    "measure",
    "oof_all_heads",
    "oof_structural_residual",
    "promotion_gate",
    "write_json_atomic",
)
