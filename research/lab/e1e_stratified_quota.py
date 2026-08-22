# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1E — frozen ebsq-v1 empirical-Bayes stratified quota router.

AX31 uses the outer-train global Beta-Binomial population posterior only.
K1 uses content-only family empirical-Bayes shrinkage. There is no item
score model, no feature matrix, and no hyperparameter search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.e1_objectives import (
    ALLOCATOR,
    BASELINE_NAME as E1_BASELINE,
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
)
from research.lab.e1b_quality_models import CHAMPION_ABS
from research.lab.e1c_regime_residual import relabel_folds
from research.lab.modeling import sort_mapping
from research.lab.public_pool import PublicPool, load_public_pool
from research.lab.quality_heads import content_tie_keys


EXPERIMENT = "e1e-empirical-bayes-stratified-quota"
REPORT_TYPE = "scrooge-e1e-ebsq-v1"
SCHEMA_VERSION = 1
CANDIDATE_NAME = "ebsq-v1"
BASELINE_NAME = "baseline_continuous_uplift"
FOLD_SEEDS: Tuple[int, ...] = (20260821, 20260822, 20260823, 20260824, 20260825)
BETA_PRIOR_A = 0.5
BETA_PRIOR_B = 0.5
MIN_FAMILY_GROUPS = 20
N_MODELS = 3
ALLOWED_GENERATIONS = (2, 4)
GATE_MEAN_DELTA = 0.002
GATE_WORST_DELTA = 0.001
EXPECTED_BASELINE_20260821 = 0.6877178030302
AUDIT_RELATIVE_PATH = "build/compare-e1e-stratified-quota/episode-audit.json"
FAMILY_DEFINITION = "ossp_router.cost_calibrated_router.prompt_family"
AX31_POLICY = "global_posterior_cost_quota"
K1_POLICY = "family_empirical_bayes_posterior"
EXPORT_PREVIEW_KEYS: Tuple[str, ...] = (
    "global_model_success_posterior",
    "global_adjacent_uplift",
    "family_k1_uplift",
    "family_weights",
    "family_lambda",
    "family_group_counts",
    "min_family_groups",
    "family_definition",
)
_LIGHT = 0
_AX31 = 1
_K1 = 2
SEQUENTIAL_TESTING = (
    "This phase is a single sequential follow-up after E1/E2/E1B/E1C/E4/E1D. "
    "Type-I error is not family-wise controlled. A Phase-1 pass is not a "
    "runtime export and does not authorize Phase 2 in this invocation."
)
POSTERIOR_FORMULA = {
    "ax31_selection": (
        "pred_qa = max(u31_global, 0) is identical on every held-out row "
        "of a fold. No prompt feature enters AX31 quality."
    ),
    "between": "max(0, between_obs - within)",
    "between_obs": "sum_f G_f (raw_u_k,f - u_bar)^2 / sum_f G_f",
    "clip": "[-1, 1] on shrunk probability uplifts only",
    "family_raw": (
        "p_m,f = (sum_{i in f, train} k_{i,m}) / (sum_{i in f, train} "
        "n_{i,m}); raw_u_k,f = p_K,f - p_A,f"
    ),
    "global": (
        "p_m = (sum_train k_m + 0.5) / (sum_train n_m + 1) with "
        "Beta(0.5, 0.5)"
    ),
    "guards": [
        "empty train totals keep the Beta prior only: p_m = 0.5",
        "family with N_A=0 or N_K=0 is unseen for K1",
        "n_families used in MoM < 2 => lambda_k = +inf",
        "between == 0 => lambda_k = +inf and every w_f = 0",
        "G_f < MIN_FAMILY_GROUPS => w_f = 0",
        "unseen held-out family uses uk_global",
        "AX31 family MoM is diagnostic and never used for pred_qa",
    ],
    "lambda_k": "within / between if between > 0 else +inf",
    "mom_population": (
        "all outer-train families with N_A>0 and N_K>0, including "
        "G_f < MIN_FAMILY_GROUPS; those families still get w_f = 0"
    ),
    "mom_weights": "G_f = n unique group_keys in the family on outer-train",
    "pred_qk": "max(u_k,family, 0)",
    "sampling_variance": (
        "v_f = p_A,f(1-p_A,f)/N_A,f + p_K,f(1-p_K,f)/N_K,f using raw p=k/n"
    ),
    "u31_global": "p_A - p_L",
    "u_k,f": "clip(uk_global + w_f * (raw_u_k,f - uk_global), -1, 1)",
    "uk_global": "p_K - p_A",
    "unseen": "uk_global",
    "w_f": "0 if G_f < 20 or lambda_k=+inf else G_f / (G_f + lambda_k)",
    "weighted_mean": "u_bar = sum_f G_f raw_u_k,f / sum_f G_f",
    "within": "sum_f G_f v_f / sum_f G_f",
}


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def _json_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(np.float64(value))
    if not np.isfinite(number):
        return None
    return number


def _clip_uplift(value: float) -> float:
    return float(np.clip(float(value), -1.0, 1.0))


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
        "n_values": {
            str(value): int(np.count_nonzero(n == value)) for value in ALLOWED_GENERATIONS
        },
    }
    return n, k, diagnostic


def global_success_posterior(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Beta(0.5, 0.5) posterior mean per model. Empty totals stay at 0.5."""

    trials = np.asarray(n, dtype=np.float64).reshape(-1, N_MODELS)
    successes = np.asarray(k, dtype=np.float64).reshape(-1, N_MODELS)
    numer = successes.sum(axis=0) + BETA_PRIOR_A
    denom = trials.sum(axis=0) + BETA_PRIOR_A + BETA_PRIOR_B
    return numer / denom


def adjacent_uplift(probability: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(probability, dtype=np.float64).reshape(N_MODELS)
    u31 = _clip_uplift(float(values[_AX31] - values[_LIGHT]))
    uk = _clip_uplift(float(values[_K1] - values[_AX31]))
    return u31, uk


def method_of_moments_lambda(
    uplifts: np.ndarray,
    sampling_variances: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, dict[str, Any]]:
    """Weighted between variance minus binomial sampling variance.

    ``lambda = within / between`` with ``between = max(0, between_obs -
    within)``. ``between == 0`` or fewer than two families yields +inf.
    """

    u = np.asarray(uplifts, dtype=np.float64).reshape(-1)
    var = np.asarray(sampling_variances, dtype=np.float64).reshape(-1)
    g = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(u) & np.isfinite(var) & np.isfinite(g) & (g > 0.0)
    n_fam = int(np.count_nonzero(finite))
    record = {
        "between": None,
        "between_obs": None,
        "lambda_infinite": True,
        "n_families": n_fam,
        "u_bar": None,
        "weight_sum": None,
        "within": None,
    }
    if n_fam < 2:
        return float("inf"), record
    u = u[finite]
    var = var[finite]
    g = g[finite]
    weight_sum = float(g.sum())
    u_bar = float(np.dot(g, u) / weight_sum)
    between_obs = float(np.dot(g, (u - u_bar) ** 2) / weight_sum)
    within = float(np.dot(g, var) / weight_sum)
    between = max(0.0, between_obs - within)
    record.update(
        {
            "between": _json_float(between),
            "between_obs": _json_float(between_obs),
            "u_bar": _json_float(u_bar),
            "weight_sum": _json_float(weight_sum),
            "within": _json_float(within),
        }
    )
    if between <= 0.0:
        return float("inf"), record
    record["lambda_infinite"] = False
    return float(within / between), record


def _family_raw_table(
    families: Sequence[str],
    group_keys: Sequence[str],
    n: np.ndarray,
    k: np.ndarray,
) -> dict[str, dict[str, Any]]:
    trials = np.asarray(n, dtype=np.int64)
    successes = np.asarray(k, dtype=np.int64)
    buckets: dict[str, dict[str, Any]] = {}
    for family, group, row_n, row_k in zip(families, group_keys, trials, successes):
        bucket = buckets.setdefault(
            str(family),
            {
                "groups": set(),
                "k": np.zeros(N_MODELS, dtype=np.int64),
                "n": np.zeros(N_MODELS, dtype=np.int64),
                "n_episodes": 0,
            },
        )
        bucket["groups"].add(str(group))
        bucket["n"] += row_n
        bucket["k"] += row_k
        bucket["n_episodes"] += 1
    table = {}
    for family, bucket in buckets.items():
        totals_n = np.asarray(bucket["n"], dtype=np.int64)
        totals_k = np.asarray(bucket["k"], dtype=np.int64)
        raw_p = np.full(N_MODELS, np.nan, dtype=np.float64)
        defined = totals_n > 0
        raw_p[defined] = totals_k[defined].astype(np.float64) / totals_n[defined]
        raw_u31 = (
            float(raw_p[_AX31] - raw_p[_LIGHT])
            if defined[_AX31] and defined[_LIGHT]
            else None
        )
        raw_uk = (
            float(raw_p[_K1] - raw_p[_AX31])
            if defined[_K1] and defined[_AX31]
            else None
        )
        var31 = None
        var_k = None
        if raw_u31 is not None:
            p_l = float(raw_p[_LIGHT])
            p_a = float(raw_p[_AX31])
            var31 = p_a * (1.0 - p_a) / float(totals_n[_AX31]) + p_l * (
                1.0 - p_l
            ) / float(totals_n[_LIGHT])
        if raw_uk is not None:
            p_a = float(raw_p[_AX31])
            p_k = float(raw_p[_K1])
            var_k = p_a * (1.0 - p_a) / float(totals_n[_AX31]) + p_k * (
                1.0 - p_k
            ) / float(totals_n[_K1])
        table[family] = {
            "k": totals_k,
            "n": totals_n,
            "n_episodes": int(bucket["n_episodes"]),
            "n_unique_groups": int(len(bucket["groups"])),
            "p_raw": raw_p,
            "raw_u31": raw_u31,
            "raw_uk": raw_uk,
            "sampling_variance_u31": var31,
            "sampling_variance_uk": var_k,
        }
    return table


def _shrink_map(
    table: Mapping[str, Mapping[str, Any]],
    *,
    global_uplift: float,
    raw_key: str,
    variance_key: str,
) -> Tuple[float, dict[str, Any], dict[str, float], dict[str, float]]:
    names = []
    uplifts = []
    variances = []
    groups = []
    for family, row in table.items():
        if row[raw_key] is None or row[variance_key] is None:
            continue
        names.append(family)
        uplifts.append(float(row[raw_key]))
        variances.append(float(row[variance_key]))
        groups.append(float(row["n_unique_groups"]))
    lam, mom = method_of_moments_lambda(
        np.asarray(uplifts, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(groups, dtype=np.float64),
    )
    weights: dict[str, float] = {}
    shrunk: dict[str, float] = {}
    for family, row in table.items():
        count = int(row["n_unique_groups"])
        raw = row[raw_key]
        if raw is None or count < MIN_FAMILY_GROUPS or not np.isfinite(lam):
            weight = 0.0
        else:
            weight = float(count) / (float(count) + float(lam))
        weights[family] = weight
        if raw is None:
            value = float(global_uplift)
        else:
            value = float(global_uplift) + weight * (float(raw) - float(global_uplift))
        shrunk[family] = _clip_uplift(value)
    return lam, mom, weights, shrunk


@dataclass(frozen=True)
class FoldPosterior:
    p_global: np.ndarray
    u31_global: float
    uk_global: float
    lambda_k: float
    lambda_31: float
    family_uk: dict[str, float]
    family_weights: dict[str, float]
    family_group_counts: dict[str, int]
    family_raw_uk: dict[str, float]
    family_u31: dict[str, float]
    family_weights_31: dict[str, float]
    mom_k: dict[str, Any]
    mom_31: dict[str, Any]
    family_table: dict[str, dict[str, Any]]
    split_shift: dict[str, Any]

    def u_k_for(self, family: str) -> float:
        return float(self.family_uk.get(str(family), self.uk_global))


def _split_shift(
    n: np.ndarray,
    k: np.ndarray,
    split_labels: Sequence[str] | None,
) -> dict[str, Any]:
    if split_labels is None:
        return {"dev": None, "train": None}
    labels = np.asarray(list(split_labels))
    out = {}
    for name in ("train", "dev"):
        mask = labels == name
        if not np.any(mask):
            out[name] = None
            continue
        probability = global_success_posterior(n[mask], k[mask])
        u31, uk = adjacent_uplift(probability)
        out[name] = {
            "p_global": {
                model_id: _json_float(probability[index])
                for index, model_id in enumerate(MODEL_IDS)
            },
            "u31_global": _json_float(u31),
            "uk_global": _json_float(uk),
        }
    return out


def fit_fold_posterior(
    families: Sequence[str],
    group_keys: Sequence[str],
    n: np.ndarray,
    k: np.ndarray,
    *,
    split_labels: Sequence[str] | None = None,
) -> FoldPosterior:
    """Outer-train posteriors. Held-out rows must not be passed in."""

    probability = global_success_posterior(n, k)
    u31_global, uk_global = adjacent_uplift(probability)
    table = _family_raw_table(families, group_keys, n, k)
    lambda_k, mom_k, weights_k, shrunk_k = _shrink_map(
        table, global_uplift=uk_global, raw_key="raw_uk", variance_key="sampling_variance_uk"
    )
    lambda_31, mom_31, weights_31, shrunk_31 = _shrink_map(
        table,
        global_uplift=u31_global,
        raw_key="raw_u31",
        variance_key="sampling_variance_u31",
    )
    group_counts = {
        family: int(row["n_unique_groups"]) for family, row in table.items()
    }
    raw_uk = {
        family: float(row["raw_uk"])
        for family, row in table.items()
        if row["raw_uk"] is not None
    }
    return FoldPosterior(
        p_global=probability,
        u31_global=u31_global,
        uk_global=uk_global,
        lambda_k=float(lambda_k),
        lambda_31=float(lambda_31),
        family_uk=shrunk_k,
        family_weights=weights_k,
        family_group_counts=group_counts,
        family_raw_uk=raw_uk,
        family_u31=shrunk_31,
        family_weights_31=weights_31,
        mom_k=mom_k,
        mom_31=mom_31,
        family_table=table,
        split_shift=_split_shift(n, k, split_labels),
    )


def predict_from_posterior(
    families: Sequence[str], posterior: FoldPosterior
) -> Tuple[np.ndarray, np.ndarray]:
    pred_qa = np.full(len(families), max(float(posterior.u31_global), 0.0), dtype=np.float64)
    pred_qk = np.asarray(
        [max(posterior.u_k_for(family), 0.0) for family in families],
        dtype=np.float64,
    )
    return pred_qa, pred_qk


def export_preview_coefficients(posterior: FoldPosterior) -> dict[str, Any]:
    return {
        "family_definition": FAMILY_DEFINITION,
        "family_group_counts": {
            name: int(posterior.family_group_counts[name])
            for name in sorted(posterior.family_group_counts)
        },
        "family_k1_uplift": {
            name: _json_float(posterior.family_uk[name])
            for name in sorted(posterior.family_uk)
        },
        "family_lambda": _json_optional_float(posterior.lambda_k),
        "family_weights": {
            name: _json_float(posterior.family_weights[name])
            for name in sorted(posterior.family_weights)
        },
        "global_adjacent_uplift": {
            "u31": _json_float(posterior.u31_global),
            "uk": _json_float(posterior.uk_global),
        },
        "global_model_success_posterior": {
            model_id: _json_float(posterior.p_global[index])
            for index, model_id in enumerate(MODEL_IDS)
        },
        "min_family_groups": MIN_FAMILY_GROUPS,
    }


def _family_records(posterior: FoldPosterior) -> list[dict[str, Any]]:
    rows = []
    for family in sorted(posterior.family_table):
        row = posterior.family_table[family]
        raw_p = row["p_raw"]
        rows.append(
            {
                "below_min_groups": bool(row["n_unique_groups"] < MIN_FAMILY_GROUPS),
                "family": family,
                "n_episodes_train": int(row["n_episodes"]),
                "n_unique_groups": int(row["n_unique_groups"]),
                "p_raw": {
                    model_id: _json_optional_float(raw_p[index])
                    for index, model_id in enumerate(MODEL_IDS)
                },
                "raw_u31": _json_optional_float(row["raw_u31"]),
                "raw_uk": _json_optional_float(row["raw_uk"]),
                "sampling_variance_u31": _json_optional_float(row["sampling_variance_u31"]),
                "sampling_variance_uk": _json_optional_float(row["sampling_variance_uk"]),
                "u31_diagnostic": _json_optional_float(posterior.family_u31.get(family)),
                "u_k": _json_optional_float(posterior.family_uk.get(family)),
                "weight_31_diagnostic": _json_optional_float(
                    posterior.family_weights_31.get(family)
                ),
                "weight_k": _json_optional_float(posterior.family_weights.get(family)),
            }
        )
    return rows


def _posterior_record(fold: int, train: np.ndarray, test: np.ndarray, posterior: FoldPosterior) -> dict[str, Any]:
    return {
        "ax31_family_shrinkage_used_in_selection": False,
        "export_preview": {
            "coefficients": export_preview_coefficients(posterior),
            "selection_use": False,
        },
        "families": _family_records(posterior),
        "fold": int(fold),
        "lambda_31_diagnostic": _json_optional_float(posterior.lambda_31),
        "lambda_31_infinite": not np.isfinite(posterior.lambda_31),
        "lambda_k": _json_optional_float(posterior.lambda_k),
        "lambda_k_infinite": not np.isfinite(posterior.lambda_k),
        "mom_31_diagnostic": dict(posterior.mom_31),
        "mom_k": dict(posterior.mom_k),
        "n_test": int(test.sum()),
        "n_train": int(train.sum()),
        "p_global": {
            model_id: _json_float(posterior.p_global[index])
            for index, model_id in enumerate(MODEL_IDS)
        },
        "split_shift_outer_train": posterior.split_shift,
        "u31_global": _json_float(posterior.u31_global),
        "uk_global": _json_float(posterior.uk_global),
    }


@dataclass(frozen=True)
class HeadPred:
    pred_qa: np.ndarray
    pred_qk: np.ndarray
    extras: dict[str, np.ndarray]


def oof_ebsq_heads(
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
    structural = current_quality_matrix(pool.episodes)
    baseline_qa, baseline_qk = oof_candidate_predictions(structural, y, pool.folds)[
        E1_BASELINE
    ]
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    pred_qa = np.zeros(y.shape[0], dtype=np.float64)
    pred_qk = np.zeros(y.shape[0], dtype=np.float64)
    extras = {
        "family_uk": np.zeros(y.shape[0], dtype=np.float64),
        "u31_global": np.zeros(y.shape[0], dtype=np.float64),
        "uk_global": np.zeros(y.shape[0], dtype=np.float64),
    }
    fold_rows = []
    families = tuple(pool.families)
    groups = tuple(pool.group_keys)
    splits = tuple(pool.split_labels)
    for fold in range(int(fold_ids.max()) + 1):
        train = fold_ids != int(fold)
        test = fold_ids == int(fold)
        train_idx = np.flatnonzero(train)
        posterior = fit_fold_posterior(
            tuple(families[index] for index in train_idx),
            tuple(groups[index] for index in train_idx),
            trials[train],
            successes[train],
            split_labels=tuple(splits[index] for index in train_idx),
        )
        qa, qk = predict_from_posterior(
            tuple(families[index] for index in np.flatnonzero(test)),
            posterior,
        )
        pred_qa[test] = qa
        pred_qk[test] = qk
        extras["u31_global"][test] = posterior.u31_global
        extras["uk_global"][test] = posterior.uk_global
        extras["family_uk"][test] = np.asarray(
            [posterior.u_k_for(families[index]) for index in np.flatnonzero(test)],
            dtype=np.float64,
        )
        fold_rows.append(_posterior_record(fold, train, test, posterior))
    return (
        HeadPred(baseline_qa, baseline_qk, extras={}),
        HeadPred(pred_qa, pred_qk, extras=extras),
        fold_rows,
    )


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


def _regret_block(
    pool: PublicPool, models: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    actual_qa = pool.scores[:, _AX31] - pool.scores[:, _LIGHT]
    actual_qk = pool.scores[:, _K1] - pool.scores[:, _AX31]
    fast = np.asarray(list(models["fast"]))
    premium = np.asarray(list(models["premium"]))
    selected = fast == "ax31"
    positive = actual_qa > 0.0
    tie = actual_qa == 0.0
    negative = actual_qa < 0.0
    missed = (~selected) & positive
    harm = selected & negative
    family_k1 = []
    for family in sorted(set(pool.families)):
        mask = np.asarray([item == family for item in pool.families])
        k1 = (premium == "axk1-think") & mask
        ax31 = (premium == "ax31") & mask
        family_k1.append(
            {
                "actual_quality_k1": _json_optional_float(
                    float(pool.scores[k1, _K1].mean()) if np.any(k1) else None
                ),
                "actual_uk_k1_mean": _json_optional_float(
                    float(actual_qk[k1].mean()) if np.any(k1) else None
                ),
                "family": family,
                "n": int(mask.sum()),
                "n_ax31": int(np.count_nonzero(ax31)),
                "n_k1": int(np.count_nonzero(k1)),
                "n_light": int(np.count_nonzero((premium == "ax31-light") & mask)),
            }
        )
    return {
        "fast": {
            "n_missed_positive": int(np.count_nonzero(missed)),
            "n_selected_negative": int(np.count_nonzero(harm)),
            "n_selected_positive": int(np.count_nonzero(selected & positive)),
            "n_selected_tie": int(np.count_nonzero(selected & tie)),
            "sum_missed_gain": _json_float(float(actual_qa[missed].sum())),
            "sum_selected_harm": _json_float(float(actual_qa[harm].sum())),
        },
        "premium_k1_by_family": family_k1,
    }


def _family_quota(
    pool: PublicPool, models: Mapping[str, Sequence[str]]
) -> list[dict[str, Any]]:
    rows = []
    for family in sorted(set(pool.families)):
        mask = np.asarray([item == family for item in pool.families])
        block = {"family": family, "n": int(mask.sum()), "tiers": {}}
        for tier in TIERS:
            chosen = np.asarray(list(models[tier]))
            counts = {model_id: 0 for model_id in MODEL_IDS}
            qualities = {model_id: None for model_id in MODEL_IDS}
            for model_id in MODEL_IDS:
                picked = mask & (chosen == model_id)
                counts[model_id] = int(np.count_nonzero(picked))
                if np.any(picked):
                    column = MODEL_IDS.index(model_id)
                    qualities[model_id] = _json_float(float(pool.scores[picked, column].mean()))
            block["tiers"][tier] = {
                "actual_quality": qualities,
                "model_counts": counts,
            }
        rows.append(block)
    return rows


def _diagnostics(
    pool: PublicPool,
    candidate: HeadPred,
    views: Sequence[Mapping[str, Any]],
    models: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "ax31_policy": AX31_POLICY,
        "family_quota": _family_quota(pool, models),
        "k1_policy": K1_POLICY,
        "pred_qa_unique_per_fold": {
            str(fold): int(np.unique(candidate.pred_qa[np.asarray(pool.folds) == fold]).size)
            for fold in sorted(set(pool.folds))
        },
        "quality_feature_dimension": 0,
        "regret": _regret_block(pool, models),
        "uplift_bounds": {
            "pred_qa_max": _json_float(float(np.max(candidate.pred_qa))),
            "pred_qa_min": _json_float(float(np.min(candidate.pred_qa))),
            "pred_qk_max": _json_float(float(np.max(candidate.pred_qk))),
            "pred_qk_min": _json_float(float(np.min(candidate.pred_qk))),
        },
        "views": [
            {
                "delta": row["delta"],
                "gated": row["gated"],
                "kind": row["kind"],
                "n": row["n"],
                "name": row["name"],
                "worse_than_gate": row["worse_than_gate"],
            }
            for row in views
            if row["kind"] in GATE_VIEW_KINDS
        ],
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
    experiment_valid = bool(baseline_matched)
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
        rows = []
        for index, episode in enumerate(pool.episodes):
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "family": pool.families[index],
                    "fold": int(pool.folds[index]),
                    "group_key": pool.group_keys[index],
                    "pred_qa": _json_float(cand_head.pred_qa[index]),
                    "pred_qk": _json_float(cand_head.pred_qk[index]),
                    "seed": int(seed),
                    "selected": {
                        BASELINE_NAME: {
                            tier: str(base_models[tier][index]) for tier in TIERS
                        },
                        CANDIDATE_NAME: {
                            tier: str(cand_models[tier][index]) for tier in TIERS
                        },
                    },
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
            "posterior_formula": report["posterior_formula"],
            "promotion_gate": report["promotion_gate"],
            "report_type": report["report_type"],
            "schema_version": report["schema_version"],
            "seed_results": report["seed_results"],
            "sequential_testing": report["sequential_testing"],
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
        for row in report["fold_posteriors"]:
            coefficients = dict(row["export_preview"]["coefficients"])
            return {
                "coefficients": coefficients,
                "example_fold": row["fold"],
                "example_seed": int(seed),
                "note": (
                    "example outer-train posteriors from one fold; not a "
                    "full-public refit and not for selection"
                ),
                "selection_use": False,
            }
    empty = FoldPosterior(
        p_global=np.full(N_MODELS, 0.5),
        u31_global=0.0,
        uk_global=0.0,
        lambda_k=float("inf"),
        lambda_31=float("inf"),
        family_uk={},
        family_weights={},
        family_group_counts={},
        family_raw_uk={},
        family_u31={},
        family_weights_31={},
        mom_k={},
        mom_31={},
        family_table={},
        split_shift={},
    )
    return {
        "coefficients": export_preview_coefficients(empty),
        "example_fold": None,
        "example_seed": None,
        "note": "no fold posterior; preview is empty and not for selection",
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
        baseline_head, candidate_head, fold_rows = oof_ebsq_heads(
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
            matched = bool(_json_float(baseline_quality) == EXPECTED_BASELINE_20260821)
        seed_reports[int(seed)] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": _json_float(
                float(candidate["pooled"]["quality_weighted"]) - baseline_quality
            ),
            "diagnostics": _diagnostics(current, candidate_head, views, models_cand),
            "fold_posteriors": fold_rows,
            "fold_seed": int(seed),
            "matched_e1_baseline": matched,
            "views": views,
            "worst_view": _worst_view(views),
        }
    ordered = [seed_reports[int(seed)] for seed in seeds]
    gate = promotion_gate(ordered)
    if gate["phase1_passed"]:
        decision = "record-e1e-quality-pass-await-phase2"
        decision_reason = (
            "Phase-1 exact-cost gates passed. This invocation does not run "
            "predicted-cost Phase 2 and does not export runtime artifacts. "
            "Hand off to independent review only."
        )
    else:
        decision = "record-e1e-no-promote"
        decision_reason = (
            "Phase-1 exact-cost gates failed. STOP this quota-architecture "
            "line. Do not open Phase 2, do not retune priors or "
            "MIN_FAMILY_GROUPS, and do not add a second candidate. Keep "
            "the current runtime."
        )
    audit_document = episode_audit_document(seed_pools, heads)
    audit_sha = sha256_text(canonical_json_text(audit_document))
    export_block = _example_preview(seed_reports)
    seed_payload = {}
    for seed, report in seed_reports.items():
        seed_payload[str(seed)] = {
            "baseline_quality": report["baseline"]["pooled"]["quality_weighted"],
            "candidate_quality": report["candidate"]["pooled"]["quality_weighted"],
            "delta": report["delta"],
            "diagnostics": report["diagnostics"],
            "fold_posteriors": report["fold_posteriors"],
            "fold_table": list(seed_pools[seed].fold_table),
            "matched_e1_baseline": report["matched_e1_baseline"],
            "results": {
                BASELINE_NAME: report["baseline"],
                CANDIDATE_NAME: report["candidate"],
            },
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
            "beta_prior_a": BETA_PRIOR_A,
            "beta_prior_b": BETA_PRIOR_B,
            "min_family_groups": MIN_FAMILY_GROUPS,
        },
        "cost_diagnostic": exact_cost_diagnostic(base_pool.costs),
        "decision": decision,
        "decision_reason": decision_reason,
        "experiment": EXPERIMENT,
        "export_preview": export_block,
        "feature": {
            "ax31_policy": AX31_POLICY,
            "family_definition": FAMILY_DEFINITION,
            "item_score_model": False,
            "k1_policy": K1_POLICY,
            "quality_feature_dimension": 0,
            "runtime_artifact_changed": False,
        },
        "fold_seeds": [int(seed) for seed in seeds],
        "identity": dict(base_pool.identity),
        "label_checks": label_checks,
        "limitations": [
            SEQUENTIAL_TESTING,
            "Outer held-out score / n / k never enter the fold posterior.",
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
        "posterior_formula": dict(POSTERIOR_FORMULA),
        "promotion_gate": gate,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_payload,
        "sequential_testing": SEQUENTIAL_TESTING,
        "solver": {"name": None, "note": "closed-form posterior; no iterative solver"},
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    return sort_mapping(report), audit_document


__all__ = (
    "AUDIT_RELATIVE_PATH",
    "BASELINE_NAME",
    "BETA_PRIOR_A",
    "BETA_PRIOR_B",
    "CANDIDATE_NAME",
    "EXPECTED_BASELINE_20260821",
    "EXPORT_PREVIEW_KEYS",
    "FAMILY_DEFINITION",
    "FOLD_SEEDS",
    "MIN_FAMILY_GROUPS",
    "adjacent_uplift",
    "assemble",
    "binomial_counts",
    "export_preview_coefficients",
    "fit_fold_posterior",
    "global_success_posterior",
    "method_of_moments_lambda",
    "oof_ebsq_heads",
    "predict_from_posterior",
    "promotion_gate",
    "write_json_atomic",
)
