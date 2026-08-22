# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1D — frozen bhmg-v1 binomial hierarchical multi-task GLM.

One pre-registered quality candidate. Coefficients are fit by numpy
IRLS/Newton on outer-train binomial counts only. Predicted-cost Phase 2
and runtime export are out of scope for this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.cost_calibrated_router import structural_features
from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.e1_objectives import (
    ALLOCATOR,
    BASELINE_NAME as E1_BASELINE,
    FEATURE_DIM,
    FEATURE_NAME,
    GATE_VIEW_DROP,
    GATE_VIEW_KINDS,
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
    _ranking_block,
)
from research.lab.e1b_quality_models import CHAMPION_ABS
from research.lab.e1c_regime_residual import relabel_folds
from research.lab.modeling import sort_mapping
from research.lab.public_pool import PublicPool, load_public_pool
from research.lab.quality_heads import content_tie_keys


EXPERIMENT = "e1d-binomial-hierarchical-multitask"
REPORT_TYPE = "scrooge-e1d-bhmg-v1"
SCHEMA_VERSION = 1
CANDIDATE_NAME = "bhmg-v1"
BASELINE_NAME = "baseline_continuous_uplift"
FOLD_SEEDS: Tuple[int, ...] = (20260821, 20260822, 20260823, 20260824, 20260825)
LAMBDA_BETA = 10.0
LAMBDA_GAMMA = 100.0
TAU = 0.25
MAX_ITERS = 200
GTOL = 1e-8
N_MODELS = 3
N_FREE = 45
SCALE_ZERO_STD = 1e-12
ALLOWED_GENERATIONS = (2, 4)
GATE_MEAN_DELTA = 0.002
GATE_WORST_DELTA = 0.001
EXPECTED_BASELINE_20260821 = 0.6877178030302
AUDIT_RELATIVE_PATH = "build/compare-e1d-bhmg/episode-audit.json"
EXPORT_PREVIEW_KEYS: Tuple[str, ...] = (
    "alpha",
    "beta",
    "gamma",
    "tau",
    "feature_name",
)
_LIGHT = 0
_AX31 = 1
_K1 = 2
_DIAG_H1 = (
    ("family", "english_multiple_choice"),
    ("family", "korean_multiple_choice"),
    ("language", "korean"),
    ("family", "long_context"),
    ("length", "len_ge_8000"),
)
SEQUENTIAL_TESTING = (
    "This phase is a single sequential follow-up after E1/E2/E1B/E1C/E4. "
    "Type-I error is not family-wise controlled. A Phase-1 pass is not a "
    "runtime export and does not authorize Phase 2 in this invocation."
)
SOLVER_NOTE = (
    "numpy IRLS/Newton on the 45 free parameters. Line search halves the "
    "Newton step when the penalized NLL would increase. That damping is "
    "numerical stability only; lambda_beta, lambda_gamma, and tau are "
    "frozen. A singular Hessian is fail-closed (no pseudoinverse)."
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


def structural_feature_matrix(episodes: Sequence[Any]) -> np.ndarray:
    """Raw 14-d ``structural_features``. No intercept, hash, or family flags."""

    matrix = np.empty((len(episodes), FEATURE_DIM), dtype=np.float64)
    for row, episode in enumerate(episodes):
        features = structural_features(episode)
        if len(features) != FEATURE_DIM:
            raise RuntimeError(
                f"structural_features width drifted: {len(features)} != {FEATURE_DIM}"
            )
        matrix[row] = features
    return matrix


def column_scales(train_x: np.ndarray) -> np.ndarray:
    """Outer-train std (ddof=0). Numerically zero std keeps scale 1.

    ``SCALE_ZERO_STD`` is the float stand-in for the frozen ``std==0 → 1``
    rule. The public ``log1p(message_count)`` column is constant
    ``log1p(1)`` but ``numpy.std`` can return ~1e-14; dividing by that
    is not a scale, it is overflow. Held-out rows still do not set scale.
    """

    matrix = np.asarray(train_x, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (n, {FEATURE_DIM}) features, got {matrix.shape}")
    scales = np.std(matrix, axis=0, ddof=0)
    scales = np.where(scales <= SCALE_ZERO_STD, 1.0, scales)
    return np.asarray(scales, dtype=np.float64)


def scale_features(features: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) / np.asarray(scales, dtype=np.float64)


def binomial_counts(
    pool: PublicPool,
) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``n=num_generations``, ``k=score*n`` from outcome rows only."""

    n_rows = len(pool.episodes)
    n = np.zeros((n_rows, N_MODELS), dtype=np.int64)
    k = np.zeros((n_rows, N_MODELS), dtype=np.int64)
    index_of = {episode.episode_id: row for row, episode in enumerate(pool.episodes)}
    model_of = {model_id: column for column, model_id in enumerate(MODEL_IDS)}
    seen = np.zeros((n_rows, N_MODELS), dtype=np.bool_)
    non_integer = 0
    illegal_n = 0
    for outcome in pool.outcomes.outcomes:
        row = index_of.get(outcome.episode_id)
        column = model_of.get(outcome.model_id)
        if row is None or column is None:
            continue
        generations = int(outcome.num_generations)
        if generations not in ALLOWED_GENERATIONS:
            illegal_n += 1
        product = Decimal(outcome.score) * generations
        if product != product.to_integral_value():
            non_integer += 1
        n[row, column] = generations
        k[row, column] = int(product)
        seen[row, column] = True
    if not bool(np.all(seen)):
        raise RuntimeError("binomial counts missing an episode/model outcome row")
    if illegal_n:
        raise RuntimeError(f"num_generations outside {{2,4}}: {illegal_n}")
    if non_integer:
        raise RuntimeError(f"score*num_generations is not integer on {non_integer} rows")
    n_mismatch = int(np.count_nonzero(n.max(axis=1) != n.min(axis=1)))
    if n_mismatch:
        raise RuntimeError(f"per-model num_generations mismatch: {n_mismatch}")
    diagnostic = {
        "illegal_n": illegal_n,
        "k_non_integer": non_integer,
        "n_mismatch": n_mismatch,
        "n_outcome_rows": int(n.size),
        "n_values": {str(value): int(np.count_nonzero(n == value)) for value in ALLOWED_GENERATIONS},
    }
    return n, k, diagnostic


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def log_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = -np.log1p(np.exp(-values[positive]))
    out[~positive] = values[~positive] - np.log1p(np.exp(values[~positive]))
    return out


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def unpack_theta(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vector = np.asarray(theta, dtype=np.float64).reshape(N_FREE)
    alpha = vector[0:3].copy()
    beta = vector[3:17].copy()
    gamma_l = vector[17:31].copy()
    gamma_a = vector[31:45].copy()
    return alpha, beta, gamma_l, gamma_a


def pack_theta(
    alpha: np.ndarray, beta: np.ndarray, gamma_l: np.ndarray, gamma_a: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(alpha, dtype=np.float64).reshape(3),
            np.asarray(beta, dtype=np.float64).reshape(FEATURE_DIM),
            np.asarray(gamma_l, dtype=np.float64).reshape(FEATURE_DIM),
            np.asarray(gamma_a, dtype=np.float64).reshape(FEATURE_DIM),
        ]
    )


def _etas(x_scaled: np.ndarray, theta: np.ndarray) -> np.ndarray:
    alpha, beta, gamma_l, gamma_a = unpack_theta(theta)
    shared = x_scaled @ beta
    eta = np.empty((x_scaled.shape[0], N_MODELS), dtype=np.float64)
    eta[:, _LIGHT] = alpha[_LIGHT] + shared + x_scaled @ gamma_l
    eta[:, _AX31] = alpha[_AX31] + shared + x_scaled @ gamma_a
    eta[:, _K1] = alpha[_K1] + shared - x_scaled @ (gamma_l + gamma_a)
    return eta


def _penalized_nll(x_scaled: np.ndarray, n: np.ndarray, k: np.ndarray, theta: np.ndarray) -> float:
    eta = _etas(x_scaled, theta)
    n_float = np.asarray(n, dtype=np.float64)
    k_float = np.asarray(k, dtype=np.float64)
    nll = float(
        np.sum(
            -k_float * log_sigmoid(eta) - (n_float - k_float) * log_sigmoid(-eta)
        )
    )
    _alpha, beta, gamma_l, gamma_a = unpack_theta(theta)
    gamma_k = -(gamma_l + gamma_a)
    penalty = 0.5 * LAMBDA_BETA * float(np.dot(beta, beta)) + 0.5 * LAMBDA_GAMMA * (
        float(np.dot(gamma_l, gamma_l))
        + float(np.dot(gamma_a, gamma_a))
        + float(np.dot(gamma_k, gamma_k))
    )
    return nll + penalty


def _gradient_hessian(
    x_scaled: np.ndarray, n: np.ndarray, k: np.ndarray, theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    eta = _etas(x_scaled, theta)
    probability = sigmoid(eta)
    n_float = np.asarray(n, dtype=np.float64)
    k_float = np.asarray(k, dtype=np.float64)
    residual = n_float * probability - k_float
    weight = n_float * probability * (1.0 - probability)
    _alpha, beta, gamma_l, gamma_a = unpack_theta(theta)

    gradient = np.zeros(N_FREE, dtype=np.float64)
    gradient[0:3] = residual.sum(axis=0)
    gradient[3:17] = x_scaled.T @ residual.sum(axis=1) + LAMBDA_BETA * beta
    gradient[17:31] = x_scaled.T @ (
        residual[:, _LIGHT] - residual[:, _K1]
    ) + LAMBDA_GAMMA * (2.0 * gamma_l + gamma_a)
    gradient[31:45] = x_scaled.T @ (
        residual[:, _AX31] - residual[:, _K1]
    ) + LAMBDA_GAMMA * (2.0 * gamma_a + gamma_l)

    hessian = np.zeros((N_FREE, N_FREE), dtype=np.float64)
    w_l = weight[:, _LIGHT]
    w_a = weight[:, _AX31]
    w_k = weight[:, _K1]
    hessian[0, 0] = float(w_l.sum())
    hessian[1, 1] = float(w_a.sum())
    hessian[2, 2] = float(w_k.sum())
    xw_l = x_scaled.T @ w_l
    xw_a = x_scaled.T @ w_a
    xw_k = x_scaled.T @ w_k
    hessian[0, 3:17] = xw_l
    hessian[3:17, 0] = xw_l
    hessian[1, 3:17] = xw_a
    hessian[3:17, 1] = xw_a
    hessian[2, 3:17] = xw_k
    hessian[3:17, 2] = xw_k
    hessian[0, 17:31] = xw_l
    hessian[17:31, 0] = xw_l
    hessian[1, 31:45] = xw_a
    hessian[31:45, 1] = xw_a
    hessian[2, 17:31] = -xw_k
    hessian[17:31, 2] = -xw_k
    hessian[2, 31:45] = -xw_k
    hessian[31:45, 2] = -xw_k

    xx_l = (x_scaled.T * w_l) @ x_scaled
    xx_a = (x_scaled.T * w_a) @ x_scaled
    xx_k = (x_scaled.T * w_k) @ x_scaled
    identity = np.eye(FEATURE_DIM, dtype=np.float64)
    hessian[3:17, 3:17] = xx_l + xx_a + xx_k + LAMBDA_BETA * identity
    hessian[3:17, 17:31] = xx_l - xx_k
    hessian[17:31, 3:17] = hessian[3:17, 17:31]
    hessian[3:17, 31:45] = xx_a - xx_k
    hessian[31:45, 3:17] = hessian[3:17, 31:45]
    hessian[17:31, 17:31] = xx_l + xx_k + 2.0 * LAMBDA_GAMMA * identity
    hessian[31:45, 31:45] = xx_a + xx_k + 2.0 * LAMBDA_GAMMA * identity
    hessian[17:31, 31:45] = xx_k + LAMBDA_GAMMA * identity
    hessian[31:45, 17:31] = hessian[17:31, 31:45]
    return gradient, hessian


@dataclass(frozen=True)
class BhmgFit:
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    scale: np.ndarray
    iters: int
    converged: bool
    singular: bool
    n_backtracks: int
    step_inf: float
    nll: float


def _gamma_matrix(gamma_l: np.ndarray, gamma_a: np.ndarray) -> np.ndarray:
    gamma = np.empty((N_MODELS, FEATURE_DIM), dtype=np.float64)
    gamma[_LIGHT] = gamma_l
    gamma[_AX31] = gamma_a
    gamma[_K1] = -(gamma_l + gamma_a)
    return gamma


def fit_bhmg(x_scaled: np.ndarray, n: np.ndarray, k: np.ndarray) -> BhmgFit:
    """IRLS/Newton on scaled 14-d features. Returns coefficients in scaled space."""

    features = np.asarray(x_scaled, dtype=np.float64)
    trials = np.asarray(n, dtype=np.int64)
    successes = np.asarray(k, dtype=np.int64)
    if features.shape[1] != FEATURE_DIM:
        raise ValueError("BHMG features must be 14-d")
    totals = trials.sum(axis=0).astype(np.float64)
    totals = np.where(totals <= 0.0, 1.0, totals)
    alpha0 = _logit(successes.sum(axis=0).astype(np.float64) / totals)
    theta = pack_theta(
        alpha0,
        np.zeros(FEATURE_DIM, dtype=np.float64),
        np.zeros(FEATURE_DIM, dtype=np.float64),
        np.zeros(FEATURE_DIM, dtype=np.float64),
    )
    n_backtracks = 0
    step_inf = float("inf")
    converged = False
    singular = False
    iters = 0
    for iters in range(1, MAX_ITERS + 1):
        gradient, hessian = _gradient_hessian(features, trials, successes, theta)
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            singular = True
            break
        if not np.all(np.isfinite(direction)):
            singular = True
            break
        step_inf = float(np.max(np.abs(direction)))
        if step_inf < GTOL:
            converged = True
            break
        loss0 = _penalized_nll(features, trials, successes, theta)
        step = 1.0
        accepted = False
        while step >= 2.0 ** -20:
            trial = theta - step * direction
            loss1 = _penalized_nll(features, trials, successes, trial)
            if np.isfinite(loss1) and loss1 < loss0:
                theta = trial
                accepted = True
                break
            step *= 0.5
            n_backtracks += 1
        if not accepted:
            break
    alpha, beta, gamma_l, gamma_a = unpack_theta(theta)
    return BhmgFit(
        alpha=alpha,
        beta=beta,
        gamma=_gamma_matrix(gamma_l, gamma_a),
        scale=np.ones(FEATURE_DIM, dtype=np.float64),
        iters=int(iters),
        converged=bool(converged),
        singular=bool(singular),
        n_backtracks=int(n_backtracks),
        step_inf=_json_float(step_inf) if np.isfinite(step_inf) else float("inf"),
        nll=_json_float(_penalized_nll(features, trials, successes, theta)),
    )


def unscale_fit(fit: BhmgFit, scales: np.ndarray) -> BhmgFit:
    scale = np.asarray(scales, dtype=np.float64)
    return BhmgFit(
        alpha=np.asarray(fit.alpha, dtype=np.float64),
        beta=np.asarray(fit.beta, dtype=np.float64) / scale,
        gamma=np.asarray(fit.gamma, dtype=np.float64) / scale,
        scale=scale,
        iters=fit.iters,
        converged=fit.converged,
        singular=fit.singular,
        n_backtracks=fit.n_backtracks,
        step_inf=fit.step_inf,
        nll=fit.nll,
    )


def model_logits(features: np.ndarray, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    raw = np.asarray(features, dtype=np.float64)
    eta = np.empty((raw.shape[0], N_MODELS), dtype=np.float64)
    shared = raw @ np.asarray(beta, dtype=np.float64)
    for model in range(N_MODELS):
        eta[:, model] = alpha[model] + shared + raw @ gamma[model]
    return eta


def model_probabilities(
    features: np.ndarray, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    return sigmoid(model_logits(features, alpha, beta, gamma))


def upgrade_from_probabilities(
    probabilities: np.ndarray, *, tau: float = TAU
) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Runtime contract: mu if mu >= tau * s else 0. n/k are not used."""

    p_l = probabilities[:, _LIGHT]
    p_a = probabilities[:, _AX31]
    p_k = probabilities[:, _K1]
    mu31 = p_a - p_l
    muk1 = p_k - p_a
    s31 = np.sqrt(p_a * (1.0 - p_a) + p_l * (1.0 - p_l))
    sk1 = np.sqrt(p_k * (1.0 - p_k) + p_a * (1.0 - p_a))
    pred_qa = np.where(mu31 >= tau * s31, mu31, 0.0)
    pred_qk = np.where(muk1 >= tau * sk1, muk1, 0.0)
    extras = {
        "mu31": mu31,
        "muk1": muk1,
        "pA": p_a,
        "pK": p_k,
        "pL": p_l,
        "s31": s31,
        "sk1": sk1,
    }
    return pred_qa, pred_qk, extras


def export_preview_coefficients(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, *, tau: float = TAU
) -> dict[str, Any]:
    preview = {
        "alpha": [_json_float(value) for value in np.asarray(alpha, dtype=np.float64)],
        "beta": [_json_float(value) for value in np.asarray(beta, dtype=np.float64)],
        "feature_name": FEATURE_NAME,
        "gamma": [
            [_json_float(value) for value in row]
            for row in np.asarray(gamma, dtype=np.float64)
        ],
        "tau": _json_float(tau),
    }
    if tuple(sorted(preview)) != tuple(sorted(EXPORT_PREVIEW_KEYS)):
        raise RuntimeError("export preview keys drifted")
    return preview


@dataclass(frozen=True)
class HeadPred:
    pred_qa: np.ndarray
    pred_qk: np.ndarray
    extras: Mapping[str, np.ndarray]


def oof_bhmg_heads(
    pool: PublicPool,
    *,
    scores: Optional[np.ndarray] = None,
    n: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
) -> Tuple[HeadPred, HeadPred, list[dict[str, Any]]]:
    y = pool.scores if scores is None else np.asarray(scores, dtype=np.float64)
    if n is None or k is None:
        trials, successes, _diagnostic = binomial_counts(pool)
    else:
        trials = np.asarray(n, dtype=np.int64)
        successes = np.asarray(k, dtype=np.int64)
    raw = structural_feature_matrix(pool.episodes)
    structural = current_quality_matrix(pool.episodes)
    baseline_qa, baseline_qk = oof_candidate_predictions(structural, y, pool.folds)[
        E1_BASELINE
    ]
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    pred_qa = np.zeros(y.shape[0], dtype=np.float64)
    pred_qk = np.zeros(y.shape[0], dtype=np.float64)
    extras = {
        name: np.zeros(y.shape[0], dtype=np.float64)
        for name in ("pL", "pA", "pK", "mu31", "s31", "muk1", "sk1")
    }
    fold_rows = []
    for fold in range(int(fold_ids.max()) + 1):
        train = fold_ids != int(fold)
        test = fold_ids == int(fold)
        scales = column_scales(raw[train])
        fit = fit_bhmg(scale_features(raw[train], scales), trials[train], successes[train])
        fitted = unscale_fit(fit, scales)
        if fitted.singular:
            fold_rows.append(_fold_record(fold, train, test, fitted))
            continue
        probabilities = model_probabilities(
            raw[test], fitted.alpha, fitted.beta, fitted.gamma
        )
        qa, qk, detail = upgrade_from_probabilities(probabilities, tau=TAU)
        pred_qa[test] = qa
        pred_qk[test] = qk
        extras["pL"][test] = detail["pL"]
        extras["pA"][test] = detail["pA"]
        extras["pK"][test] = detail["pK"]
        extras["mu31"][test] = detail["mu31"]
        extras["s31"][test] = detail["s31"]
        extras["muk1"][test] = detail["muk1"]
        extras["sk1"][test] = detail["sk1"]
        fold_rows.append(_fold_record(fold, train, test, fitted))
    return (
        HeadPred(baseline_qa, baseline_qk, extras={}),
        HeadPred(pred_qa, pred_qk, extras=extras),
        fold_rows,
    )


def _fold_record(
    fold: int, train: np.ndarray, test: np.ndarray, fitted: BhmgFit
) -> dict[str, Any]:
    return {
        "alpha": [_json_float(value) for value in fitted.alpha],
        "beta": [_json_float(value) for value in fitted.beta],
        "converged": bool(fitted.converged),
        "export_preview": {
            "coefficients": export_preview_coefficients(
                fitted.alpha, fitted.beta, fitted.gamma, tau=TAU
            ),
            "selection_use": False,
        },
        "fold": int(fold),
        "gamma": [[_json_float(value) for value in row] for row in fitted.gamma],
        "gamma_sum_maxabs": _json_float(np.max(np.abs(fitted.gamma.sum(axis=0)))),
        "iters": int(fitted.iters),
        "n_backtracks": int(fitted.n_backtracks),
        "n_test": int(test.sum()),
        "n_train": int(train.sum()),
        "nll": fitted.nll if np.isfinite(fitted.nll) else None,
        "scale": [_json_float(value) for value in fitted.scale],
        "singular": bool(fitted.singular),
        "step_inf": None if not np.isfinite(fitted.step_inf) else _json_float(fitted.step_inf),
    }


def _caps_ok(scored: Mapping[str, Any]) -> bool:
    return all(bool(scored["tiers"][tier]["within_hard_cap"]) for tier in TIERS)


def _k1_fast_balanced(scored: Mapping[str, Any]) -> Tuple[int, int]:
    fast = int(scored["tiers"]["fast"]["model_counts"]["axk1-think"])
    balanced = int(scored["tiers"]["balanced"]["model_counts"]["axk1-think"])
    return fast, balanced


def _evaluate_head(pool: PublicPool, name: str, head: HeadPred, tie_keys: Sequence[str]) -> dict[str, Any]:
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
    fast_k1, balanced_k1 = _k1_fast_balanced(pooled)
    return {
        "fold_caps_ok": all(_caps_ok(row) for row in per_fold),
        "k1_balanced": balanced_k1,
        "k1_fast": fast_k1,
        "k1_fast_balanced_zero": bool(fast_k1 == 0 and balanced_k1 == 0),
        "name": name,
        "per_fold": per_fold,
        "pooled": pooled,
    }


def _worst_view(views: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    gated = [
        row
        for row in views
        if row["kind"] in GATE_VIEW_KINDS and row["gated"] and row["delta"] is not None
    ]
    if not gated:
        return None
    row = min(gated, key=lambda item: (item["delta"], item["kind"], item["name"]))
    return {
        "delta": row["delta"],
        "kind": row["kind"],
        "n": row["n"],
        "name": row["name"],
        "worse_than_gate": row["worse_than_gate"],
    }


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0 or not np.all(np.isfinite(finite)):
        return {"max": None, "mean": None, "median": None, "min": None, "std": None}
    return {
        "max": _json_float(np.max(finite)),
        "mean": _json_float(np.mean(finite)),
        "median": _json_float(np.median(finite)),
        "min": _json_float(np.min(finite)),
        "std": _json_float(np.std(finite, ddof=0)),
    }


def _diagnostics(
    pool: PublicPool, candidate: HeadPred, views: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    extras = candidate.extras
    qa_zero = float(np.mean(candidate.pred_qa == 0.0))
    qk_zero = float(np.mean(candidate.pred_qk == 0.0))
    mu31 = extras.get("mu31", np.zeros(0))
    muk1 = extras.get("muk1", np.zeros(0))
    view_focus = []
    for kind, name in _DIAG_H1:
        match = next(
            (row for row in views if row["kind"] == kind and row["name"] == name),
            None,
        )
        if match is None:
            continue
        view_focus.append(
            {
                "delta": match["delta"],
                "gated": match["gated"],
                "kind": kind,
                "n": match["n"],
                "name": name,
                "worse_than_gate": match["worse_than_gate"],
            }
        )
    ranking = {
        "mu31_vs_delta_ax31_light": _ranking_block(
            extras.get("mu31", candidate.pred_qa),
            pool.scores[:, _AX31] - pool.scores[:, _LIGHT],
        ),
        "muk1_vs_delta_k1_ax31": _ranking_block(
            extras.get("muk1", candidate.pred_qk),
            pool.scores[:, _K1] - pool.scores[:, _AX31],
        ),
    }
    return {
        "abstention": {
            "pred_qa_zero_fraction": _json_float(qa_zero),
            "pred_qk_zero_fraction": _json_float(qk_zero),
            "qa_positive_mu_zeroed": _json_optional_float(
                float(np.mean((mu31 > 0.0) & (candidate.pred_qa == 0.0)))
                if mu31.size
                else None
            ),
            "qk_positive_mu_zeroed": _json_optional_float(
                float(np.mean((muk1 > 0.0) & (candidate.pred_qk == 0.0)))
                if muk1.size
                else None
            ),
        },
        "h1_h2_views": view_focus,
        "ranking": ranking,
        "summaries": {name: _array_summary(extras[name]) for name in extras},
    }


def promotion_gate(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta"]) for row in seed_reports]
    qualities = [
        float(row["candidate"]["pooled"]["quality_weighted"]) for row in seed_reports
    ]
    baseline_qualities = [
        float(row["baseline"]["pooled"]["quality_weighted"]) for row in seed_reports
    ]
    view_fail = []
    cap_fail = []
    k1_fail = []
    singular_fail = []
    matched = None
    for row in seed_reports:
        seed = row["fold_seed"]
        if seed == 20260821:
            matched = bool(row.get("matched_e1_baseline"))
        pooled_ok = _caps_ok(row["candidate"]["pooled"]) and _caps_ok(
            row["baseline"]["pooled"]
        )
        fold_ok = bool(row["candidate"]["fold_caps_ok"] and row["baseline"]["fold_caps_ok"])
        if not (pooled_ok and fold_ok):
            cap_fail.append(seed)
        if not (
            row["candidate"].get("k1_fast_balanced_zero", False)
            and row["baseline"].get("k1_fast_balanced_zero", False)
        ):
            k1_fail.append(seed)
        if row.get("singular"):
            singular_fail.append(seed)
        fails = [
            f"{item['kind']}:{item['name']}"
            for item in row["views"]
            if item["kind"] in GATE_VIEW_KINDS and item["worse_than_gate"]
        ]
        if fails:
            view_fail.append({"failures": fails, "seed": seed})
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    worst_delta = float(np.min(deltas)) if deltas else float("nan")
    mean_quality = float(np.mean(qualities)) if qualities else float("nan")
    worst_quality = float(np.min(qualities)) if qualities else float("nan")
    baseline_matched = True if matched is None else bool(matched)
    experiment_valid = bool(baseline_matched and not singular_fail)
    phase1 = bool(
        experiment_valid
        and not cap_fail
        and not view_fail
        and not k1_fail
        and mean_delta >= GATE_MEAN_DELTA
        and worst_delta >= GATE_WORST_DELTA
        and mean_quality >= CHAMPION_ABS
    )
    return {
        "baseline_mean_quality": _json_float(float(np.mean(baseline_qualities)))
        if baseline_qualities
        else None,
        "cap_failures": cap_fail,
        "experiment_valid": experiment_valid,
        "k1_failures": k1_fail,
        "matched_e1_baseline_20260821": matched,
        "mean_absolute": _json_float(mean_quality),
        "mean_delta": _json_float(mean_delta),
        "passed": phase1,
        "phase1_passed": phase1,
        "phase2_executed": False,
        "singular_failures": singular_fail,
        "thresholds": {
            "mean_absolute": CHAMPION_ABS,
            "mean_delta": GATE_MEAN_DELTA,
            "stress_95_not_gated": True,
            "view_drop": GATE_VIEW_DROP,
            "view_min_n": VIEW_MIN_N,
            "worst_delta": GATE_WORST_DELTA,
        },
        "view_failures": view_fail,
        "worst_absolute": _json_float(worst_quality),
        "worst_delta": _json_float(worst_delta),
    }


def episode_audit_document(
    seed_pools: Mapping[int, PublicPool],
    seed_reports: Mapping[int, Mapping[str, Any]],
    heads: Mapping[int, Tuple[HeadPred, HeadPred]],
) -> dict[str, Any]:
    seed_blocks = {}
    for seed, pool in seed_pools.items():
        base_head, cand_head = heads[seed]
        ties = content_tie_keys(pool.texts)
        base_models = allocate_all_tiers(
            base_head.pred_qa, base_head.pred_qk, pool.costs, pool.light_total, ties
        )
        cand_models = allocate_all_tiers(
            cand_head.pred_qa, cand_head.pred_qk, pool.costs, pool.light_total, ties
        )
        extras = cand_head.extras
        rows = []
        for index, episode in enumerate(pool.episodes):
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "fold": int(pool.folds[index]),
                    "group_key": pool.group_keys[index],
                    "mu31": _json_float(extras["mu31"][index]),
                    "muk1": _json_float(extras["muk1"][index]),
                    "pA": _json_float(extras["pA"][index]),
                    "pK": _json_float(extras["pK"][index]),
                    "pL": _json_float(extras["pL"][index]),
                    "pred_qa": _json_float(cand_head.pred_qa[index]),
                    "pred_qk": _json_float(cand_head.pred_qk[index]),
                    "s31": _json_float(extras["s31"][index]),
                    "seed": int(seed),
                    "selected": {
                        BASELINE_NAME: {
                            tier: str(base_models[tier][index]) for tier in TIERS
                        },
                        CANDIDATE_NAME: {
                            tier: str(cand_models[tier][index]) for tier in TIERS
                        },
                    },
                    "sk1": _json_float(extras["sk1"][index]),
                    "split": pool.split_labels[index],
                }
            )
        seed_blocks[str(seed)] = {"n_rows": len(rows), "rows": rows}
    return {
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "seeds": seed_blocks,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "allocator": report["allocator"],
            "audit": report["audit"],
            "candidate": report["candidate"],
            "constants": report["constants"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "export_preview": report["export_preview"],
            "feature": report["feature"],
            "fold_seeds": report["fold_seeds"],
            "identity": report["identity"],
            "label_checks": report["label_checks"],
            "limitations": report["limitations"],
            "promotion_gate": report["promotion_gate"],
            "report_type": report["report_type"],
            "schema_version": report["schema_version"],
            "seed_results": report["seed_results"],
            "sequential_testing": report["sequential_testing"],
            "solver": report["solver"],
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


def _example_preview(seed_reports: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    for seed in FOLD_SEEDS:
        report = seed_reports.get(int(seed))
        if report is None:
            continue
        for row in report["fold_fits"]:
            if row.get("singular"):
                continue
            coefficients = dict(row["export_preview"]["coefficients"])
            return {
                "coefficients": coefficients,
                "example_fold": row["fold"],
                "example_seed": int(seed),
                "note": (
                    "example coefficients from one outer-train fold; not a "
                    "full-public refit and not for selection"
                ),
                "selection_use": False,
            }
    return {
        "coefficients": export_preview_coefficients(
            np.zeros(3), np.zeros(FEATURE_DIM), np.zeros((N_MODELS, FEATURE_DIM))
        ),
        "example_fold": None,
        "example_seed": None,
        "note": "no non-singular fold fit; preview is zeros and not for selection",
        "selection_use": False,
    }


def assemble(
    pool: PublicPool | None = None,
    *,
    seeds: Sequence[int] = FOLD_SEEDS,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    base_pool = pool or load_public_pool()
    trials, successes, label_checks = binomial_counts(base_pool)
    seed_pools: dict[int, PublicPool] = {}
    seed_reports: dict[int, dict[str, Any]] = {}
    heads: dict[int, Tuple[HeadPred, HeadPred]] = {}
    public_n = int(base_pool.identity.get("n_episodes", 0))
    for seed in seeds:
        current = relabel_folds(base_pool, int(seed))
        seed_pools[int(seed)] = current
        baseline_head, candidate_head, fold_rows = oof_bhmg_heads(
            current, n=trials, k=successes
        )
        heads[int(seed)] = (baseline_head, candidate_head)
        tie_keys = content_tie_keys(current.texts)
        baseline = _evaluate_head(current, BASELINE_NAME, baseline_head, tie_keys)
        candidate = _evaluate_head(current, CANDIDATE_NAME, candidate_head, tie_keys)
        models_base = allocate_all_tiers(
            baseline_head.pred_qa,
            baseline_head.pred_qk,
            current.costs,
            current.light_total,
            tie_keys,
        )
        models_cand = allocate_all_tiers(
            candidate_head.pred_qa,
            candidate_head.pred_qk,
            current.costs,
            current.light_total,
            tie_keys,
        )
        views = stress_views(current, models_base, models_cand)
        baseline_quality = float(baseline["pooled"]["quality_weighted"])
        matched = None
        if int(seed) == 20260821 and public_n == 2640:
            matched = bool(
                _json_float(baseline_quality) == EXPECTED_BASELINE_20260821
            )
        seed_reports[int(seed)] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": _json_float(
                float(candidate["pooled"]["quality_weighted"]) - baseline_quality
            ),
            "diagnostics": _diagnostics(current, candidate_head, views),
            "fold_fits": fold_rows,
            "fold_seed": int(seed),
            "matched_e1_baseline": matched,
            "singular": any(row["singular"] for row in fold_rows),
            "views": views,
            "worst_view": _worst_view(views),
        }
    ordered = [seed_reports[int(seed)] for seed in seeds]
    gate = promotion_gate(ordered)
    if gate["phase1_passed"]:
        decision = "record-e1d-quality-pass-await-phase2"
        decision_reason = (
            "Phase-1 exact-cost gates passed. This invocation does not run "
            "predicted-cost Phase 2 and does not export runtime artifacts. "
            "Hand off to independent review only."
        )
    else:
        decision = "record-e1d-no-promote"
        decision_reason = (
            "Phase-1 exact-cost gates failed. STOP the quality-architecture "
            "line. Do not open Phase 2, do not retune lambda/tau, and do not "
            "add a second candidate. Keep the current runtime."
        )
    audit_document = episode_audit_document(seed_pools, seed_reports, heads)
    audit_sha = sha256_text(canonical_json_text(audit_document))
    export_block = _example_preview(seed_reports)
    seed_payload = {}
    for seed, report in seed_reports.items():
        seed_payload[str(seed)] = {
            "baseline_quality": report["baseline"]["pooled"]["quality_weighted"],
            "candidate_quality": report["candidate"]["pooled"]["quality_weighted"],
            "delta": report["delta"],
            "diagnostics": report["diagnostics"],
            "fold_fits": report["fold_fits"],
            "fold_table": list(seed_pools[seed].fold_table),
            "matched_e1_baseline": report["matched_e1_baseline"],
            "results": {
                BASELINE_NAME: report["baseline"],
                CANDIDATE_NAME: report["candidate"],
            },
            "singular": report["singular"],
            "views": report["views"],
            "worst_view": report["worst_view"],
        }
    report = {
        "allocator": dict(ALLOCATOR),
        "audit": {
            "n_rows": sum(block["n_rows"] for block in audit_document["seeds"].values()),
            "relative_path": AUDIT_RELATIVE_PATH,
            "sha256": audit_sha,
        },
        "candidate": CANDIDATE_NAME,
        "constants": {
            "gtol": GTOL,
            "lambda_beta": LAMBDA_BETA,
            "lambda_gamma": LAMBDA_GAMMA,
            "max_iters": MAX_ITERS,
            "n_free": N_FREE,
            "tau": TAU,
        },
        "cost_diagnostic": exact_cost_diagnostic(base_pool.costs),
        "decision": decision,
        "decision_reason": decision_reason,
        "experiment": EXPERIMENT,
        "export_preview": export_block,
        "feature": {
            "dim": FEATURE_DIM,
            "hash_bins": 0,
            "name": FEATURE_NAME,
            "runtime_artifact_changed": False,
            "scale": (
                "outer-train std ddof=0 on columns 1..14; std<=1e-12 -> 1 "
                "(float realization of zero-std); intercept/alpha not "
                "scaled; held-out rows do not set scale"
            ),
            "scale_zero_std": SCALE_ZERO_STD,
        },
        "fold_seeds": [int(seed) for seed in seeds],
        "identity": dict(base_pool.identity),
        "label_checks": label_checks,
        "limitations": [
            SEQUENTIAL_TESTING,
            "Outer held-out score / n / k never enter scale or IRLS.",
            "95% stress caps are observational.",
            "Phase 2 predicted-cost evaluation is not executed here.",
            "A pass is not a runtime export.",
            "exact-cost OOF and the current public Dev runtime replay "
            "are different protocols and must not be subtracted.",
        ],
        "phase2": {
            "executed": False,
            "reason": "this invocation never opens predicted-cost Phase 2",
        },
        "promotion_gate": gate,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_payload,
        "sequential_testing": SEQUENTIAL_TESTING,
        "solver": {
            "damping": "backtracking_halving_on_nll_increase",
            "gtol": GTOL,
            "max_iters": MAX_ITERS,
            "name": "numpy_irls_newton",
            "note": SOLVER_NOTE,
            "external_optimizer": False,
            "singular": "fail_closed",
        },
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    return sort_mapping(report), audit_document


__all__ = (
    "AUDIT_RELATIVE_PATH",
    "BASELINE_NAME",
    "CANDIDATE_NAME",
    "EXPECTED_BASELINE_20260821",
    "EXPORT_PREVIEW_KEYS",
    "FOLD_SEEDS",
    "GTOL",
    "LAMBDA_BETA",
    "LAMBDA_GAMMA",
    "MAX_ITERS",
    "TAU",
    "assemble",
    "binomial_counts",
    "column_scales",
    "export_preview_coefficients",
    "fit_bhmg",
    "model_probabilities",
    "oof_bhmg_heads",
    "promotion_gate",
    "structural_feature_matrix",
    "upgrade_from_probabilities",
    "write_json_atomic",
)
