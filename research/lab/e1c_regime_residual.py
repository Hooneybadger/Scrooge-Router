# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1C — nested-regime structural residual on exact public costs.

One pre-registered candidate. Residual blend λ is chosen per outer fold
from outer-train inner grouped OOF only. Hash heads are not used.
This is a sequential follow-up to E1B and is documented as such.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.heuristic import extract_features
from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.e1_objectives import (
    ALLOCATOR,
    BASELINE_NAME as E1_BASELINE,
    GATE_VIEW_DROP,
    GATE_VIEW_KINDS,
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
    _slice_quality,
)
from research.lab.e1b_quality_models import (
    CHAMPION_ABS,
    TREE_MAX_DEPTH,
    TREE_MIN_SAMPLES_LEAF,
    TREE_N_ESTIMATORS,
    TREE_RANDOM_STATE,
    HeadPred,
    _fit_residual_tree,
    _hard_caps_ok,
    _inner_oof_ridge,
)
from research.lab.grouped_crossfit import (
    FOLDS,
    assign_balanced_group_folds,
    fold_balance,
    fold_leakage_count,
)
from research.lab.modeling import ridge_fit, ridge_predict, sort_mapping, weighted_final
from research.lab.public_pool import PublicPool, load_public_pool
from research.lab.quality_heads import content_tie_keys


EXPERIMENT = "e1c-regime-residual"
REPORT_TYPE = "scrooge-e1c-regime-residual-v1"
SCHEMA_VERSION = 1
BASELINE_NAME = "structural_baseline"
CANDIDATE_NAME = "nested_regime_structural_residual"
FOLD_SEEDS: Tuple[int, ...] = (20260821, 20260822, 20260823)
LAMBDA_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
CLIP_QUANTILE = 0.95
REGIME_LONG = "long"
REGIME_SHORT = "short"
GATE_MEAN_DELTA = 0.002
GATE_WORST_DELTA = 0.001
COMPLEXITY_NOTE_GAIN = 0.001
AUDIT_RELATIVE_PATH = "build/compare-e1c-regime-residual/episode-audit.json"
_LIGHT = 0
_AX31 = 1
_K1 = 2

SEQUENTIAL_TESTING = (
    "This phase is a single sequential follow-up to the E1B residual "
    "result (long_context = len_ge_8000 drop). λ is not a hard-coded "
    "long-off switch; it is chosen from a pre-registered grid on "
    "outer-train inner OOF. The Type-I error is not family-wise "
    "controlled across E1/E1B/E1C. A pass still needs an independent "
    "review before any runtime integration."
)


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def regime_label(episode: Any) -> str:
    """Runtime content-only long_context: character_count >= 8000."""

    return REGIME_LONG if extract_features(episode).long_context else REGIME_SHORT


def regimes_of(episodes: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(regime_label(episode) for episode in episodes)


def relabel_folds(pool: PublicPool, seed: int, *, folds: int | None = None) -> PublicPool:
    n_folds = int(folds if folds is not None else pool.identity.get("folds", FOLDS))
    folds = assign_balanced_group_folds(
        pool.group_keys, pool.families, folds=n_folds, seed=int(seed)
    )
    leaked = fold_leakage_count(pool.group_keys, folds)
    if leaked:
        raise RuntimeError(f"grouped fold leakage at seed {seed}: {leaked}")
    identity = dict(pool.identity)
    identity["fold_seed"] = int(seed)
    identity["folds"] = n_folds
    return replace(
        pool,
        folds=folds,
        identity=identity,
        fold_table=fold_balance(pool.group_keys, folds, pool.families),
    )


def _clip_scale(values: np.ndarray) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if absolute.size == 0:
        return 1.0
    return float(max(np.quantile(absolute, CLIP_QUANTILE), 1e-12))


def _blend(
    base: np.ndarray,
    residual: np.ndarray,
    regimes: Sequence[str],
    lam_short: float,
    lam_long: float,
    clip: float,
) -> np.ndarray:
    clipped = np.clip(np.asarray(residual, dtype=np.float64), -float(clip), float(clip))
    scale = np.asarray(
        [lam_long if regime == REGIME_LONG else lam_short for regime in regimes],
        dtype=np.float64,
    )
    return np.asarray(base, dtype=np.float64) + scale * clipped


def _oof_resid_hat(
    features: np.ndarray, resid_labels: np.ndarray, inner_folds: np.ndarray
) -> np.ndarray:
    hat = np.empty(resid_labels.shape[0], dtype=np.float64)
    tree_x = features[:, 1:]
    for fold in np.unique(inner_folds):
        train = inner_folds != fold
        test = inner_folds == fold
        tree = _fit_residual_tree(tree_x[train], resid_labels[train])
        hat[test] = tree.predict(tree_x[test])
    return hat


def _inner_views(
    scores: np.ndarray,
    families: Sequence[str],
    languages: Sequence[str],
    length_views: Sequence[str],
    baseline_models: Mapping[str, Sequence[str]],
    candidate_models: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    views: list[tuple[str, str, np.ndarray]] = []
    for name in sorted(set(families)):
        views.append(("family", name, np.asarray([item == name for item in families])))
    for name in sorted(set(languages)):
        views.append(("language", name, np.asarray([item == name for item in languages])))
    for name in sorted(set(length_views)):
        views.append(("length", name, np.asarray([item == name for item in length_views])))
    rows = []
    for kind, name, mask in views:
        n = int(np.count_nonzero(mask))
        base = _slice_quality(scores, baseline_models, mask)
        cand = _slice_quality(scores, candidate_models, mask)
        delta = None if base is None or cand is None else cand - base
        gated = n >= VIEW_MIN_N
        rows.append(
            {
                "delta": None if delta is None else float(delta),
                "gated": gated,
                "kind": kind,
                "n": n,
                "name": name,
                "worse": bool(gated and delta is not None and delta < -GATE_VIEW_DROP),
            }
        )
    return rows


def _block_quality(scores: np.ndarray, models: Mapping[str, Sequence[str]]) -> float:
    qualities = []
    for tier in TIERS:
        columns = np.asarray(
            [MODEL_IDS.index(model_id) for model_id in models[tier]], dtype=np.int64
        )
        selected = scores[np.arange(scores.shape[0]), columns]
        qualities.append(float(selected.mean()))
    return weighted_final(qualities[0], qualities[1], qualities[2])


def select_lambdas(
    scores: np.ndarray,
    costs: np.ndarray,
    families: Sequence[str],
    languages: Sequence[str],
    length_views: Sequence[str],
    regimes: Sequence[str],
    tie_keys: Sequence[str],
    pred_qa_base: np.ndarray,
    pred_qk_base: np.ndarray,
    resid_qa: np.ndarray,
    resid_qk: np.ndarray,
    clip_qa: float,
    clip_qk: float,
    quality_fn: Any = None,
) -> dict[str, Any]:
    """Pick (λ_short, λ_long) on this outer-train block only."""

    light = float(costs[:, _LIGHT].sum())
    baseline_models = allocate_all_tiers(
        pred_qa_base, pred_qk_base, costs, light, tie_keys
    )
    score_fn = quality_fn or (lambda models: _block_quality(scores, models))
    feasible = []
    rejected = 0
    for lam_short, lam_long in itertools.product(LAMBDA_GRID, LAMBDA_GRID):
        pred_qa = _blend(pred_qa_base, resid_qa, regimes, lam_short, lam_long, clip_qa)
        pred_qk = _blend(pred_qk_base, resid_qk, regimes, lam_short, lam_long, clip_qk)
        models = allocate_all_tiers(pred_qa, pred_qk, costs, light, tie_keys)
        views = _inner_views(
            scores, families, languages, length_views, baseline_models, models
        )
        worse = [f"{row['kind']}:{row['name']}" for row in views if row["worse"]]
        record = {
            "lambda_long": float(lam_long),
            "lambda_short": float(lam_short),
            "quality": _json_float(score_fn(models)),
            "view_failures": worse,
        }
        if worse:
            rejected += 1
            continue
        feasible.append(record)
    if not feasible:
        chosen = {
            "fallback": True,
            "lambda_long": 0.0,
            "lambda_short": 0.0,
            "quality": _json_float(score_fn(baseline_models)),
            "view_failures": [],
        }
    else:
        chosen = min(
            feasible,
            key=lambda row: (
                -row["quality"],
                row["lambda_short"] + row["lambda_long"],
                row["lambda_short"],
                row["lambda_long"],
            ),
        )
        chosen = dict(chosen)
        chosen["fallback"] = False
    return {
        "chosen": chosen,
        "clip_qa": _json_float(clip_qa),
        "clip_qk": _json_float(clip_qk),
        "n_feasible": len(feasible),
        "n_rejected": rejected,
    }


@dataclass(frozen=True)
class FoldFit:
    clip_qa: float
    clip_qk: float
    lambda_long: float
    lambda_short: float
    pred_qa: np.ndarray
    pred_qk: np.ndarray
    residual_qa: np.ndarray
    residual_qk: np.ndarray
    selection: Mapping[str, Any]


def fit_outer_fold(
    structural: np.ndarray,
    scores: np.ndarray,
    costs: np.ndarray,
    folds: Sequence[int],
    fold: int,
    regimes: Sequence[str],
    families: Sequence[str],
    languages: Sequence[str],
    length_views: Sequence[str],
    tie_keys: Sequence[str],
    pool: PublicPool | None = None,
) -> FoldFit:
    fold_ids = np.asarray(list(folds), dtype=np.int64)
    train = fold_ids != int(fold)
    test = fold_ids == int(fold)
    x_train = structural[train]
    y_train = scores[train]
    inner_folds = fold_ids[train]
    delta_al = y_train[:, _AX31] - y_train[:, _LIGHT]
    delta_kl = y_train[:, _K1] - y_train[:, _LIGHT]
    base_qa = _inner_oof_ridge(x_train, delta_al, inner_folds)
    base_qk = _inner_oof_ridge(x_train, delta_kl, inner_folds)
    resid_al = delta_al - base_qa
    resid_kl = delta_kl - base_qk
    resid_hat_qa = _oof_resid_hat(x_train, resid_al, inner_folds)
    resid_hat_qk = _oof_resid_hat(x_train, resid_kl, inner_folds)
    clip_qa = _clip_scale(resid_al)
    clip_qk = _clip_scale(resid_kl)
    train_idx = [index for index, flag in enumerate(train) if flag]
    quality_fn = None
    if pool is not None:
        train_indexes = list(train_idx)

        def quality_fn(models, _idx=train_indexes):
            return float(score_decisions(pool, models, indexes=_idx)["quality_weighted"])
    selection = select_lambdas(
        y_train,
        costs[train],
        tuple(families[index] for index in train_idx),
        tuple(languages[index] for index in train_idx),
        tuple(length_views[index] for index in train_idx),
        tuple(regimes[index] for index in train_idx),
        tuple(tie_keys[index] for index in train_idx),
        base_qa,
        base_qk,
        resid_hat_qa,
        resid_hat_qk,
        clip_qa,
        clip_qk,
        quality_fn=quality_fn,
    )
    lam_short = float(selection["chosen"]["lambda_short"])
    lam_long = float(selection["chosen"]["lambda_long"])
    tree_x_train = x_train[:, 1:]
    tree_qa = _fit_residual_tree(tree_x_train, resid_al)
    tree_qk = _fit_residual_tree(tree_x_train, resid_kl)
    coef_qa = ridge_fit(x_train, delta_al, alpha=RIDGE_ALPHA)
    coef_qk = ridge_fit(x_train, delta_kl, alpha=RIDGE_ALPHA)
    x_test = structural[test]
    base_qa_h = ridge_predict(coef_qa, x_test)
    base_qk_h = ridge_predict(coef_qk, x_test)
    resid_qa_h = tree_qa.predict(x_test[:, 1:])
    resid_qk_h = tree_qk.predict(x_test[:, 1:])
    regimes_h = tuple(regimes[index] for index, flag in enumerate(test) if flag)
    pred_qa = _blend(base_qa_h, resid_qa_h, regimes_h, lam_short, lam_long, clip_qa)
    pred_qk = _blend(base_qk_h, resid_qk_h, regimes_h, lam_short, lam_long, clip_qk)
    return FoldFit(
        clip_qa=clip_qa,
        clip_qk=clip_qk,
        lambda_long=lam_long,
        lambda_short=lam_short,
        pred_qa=np.asarray(pred_qa, dtype=np.float64),
        pred_qk=np.asarray(pred_qk, dtype=np.float64),
        residual_qa=np.asarray(resid_qa_h, dtype=np.float64),
        residual_qk=np.asarray(resid_qk_h, dtype=np.float64),
        selection=selection,
    )


def oof_regime_heads(
    pool: PublicPool,
    *,
    scores: Optional[np.ndarray] = None,
) -> Tuple[HeadPred, HeadPred, list[dict[str, Any]]]:
    y = pool.scores if scores is None else np.asarray(scores, dtype=np.float64)
    structural = current_quality_matrix(pool.episodes)
    regimes = regimes_of(pool.episodes)
    tie_keys = content_tie_keys(pool.texts)
    e1 = oof_candidate_predictions(structural, y, pool.folds)
    base_qa, base_qk = e1[E1_BASELINE]
    pred_qa = np.empty(y.shape[0], dtype=np.float64)
    pred_qk = np.empty(y.shape[0], dtype=np.float64)
    resid_qa = np.empty(y.shape[0], dtype=np.float64)
    resid_qk = np.empty(y.shape[0], dtype=np.float64)
    fold_rows = []
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    for fold in range(int(fold_ids.max()) + 1):
        fitted = fit_outer_fold(
            structural,
            y,
            pool.costs,
            pool.folds,
            fold,
            regimes,
            pool.families,
            pool.languages,
            pool.length_views,
            tie_keys,
            pool=pool,
        )
        test = fold_ids == fold
        pred_qa[test] = fitted.pred_qa
        pred_qk[test] = fitted.pred_qk
        resid_qa[test] = fitted.residual_qa
        resid_qk[test] = fitted.residual_qk
        fold_rows.append(
            {
                "clip_qa": _json_float(fitted.clip_qa),
                "clip_qk": _json_float(fitted.clip_qk),
                "fold": int(fold),
                "lambda_long": _json_float(fitted.lambda_long),
                "lambda_short": _json_float(fitted.lambda_short),
                "n_test": int(test.sum()),
                "n_train": int((~test).sum()),
                "selection": fitted.selection,
            }
        )
    baseline = HeadPred(base_qa, base_qk, mag_qa=base_qa, mag_qk=base_qk)
    candidate = HeadPred(
        pred_qa,
        pred_qk,
        mag_qa=base_qa,
        mag_qk=base_qk,
        residual_qa=resid_qa,
        residual_qk=resid_qk,
    )
    return baseline, candidate, fold_rows


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
    return {
        "fold_caps_ok": all(_hard_caps_ok(row) for row in per_fold),
        "name": name,
        "per_fold": per_fold,
        "pooled": pooled,
    }


def _worst_view(views: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    gated = [row for row in views if row["gated"] and row["delta"] is not None]
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
    complexity_notes = []
    for row in seed_reports:
        seed = row["fold_seed"]
        pooled_ok = _hard_caps_ok(row["candidate"]["pooled"])
        fold_ok = bool(row["candidate"]["fold_caps_ok"])
        base_pooled_ok = _hard_caps_ok(row["baseline"]["pooled"])
        base_fold_ok = bool(row["baseline"]["fold_caps_ok"])
        if not (pooled_ok and fold_ok and base_pooled_ok and base_fold_ok):
            cap_fail.append(seed)
        fails = [
            f"{item['kind']}:{item['name']}"
            for item in row["views"]
            if item["kind"] in GATE_VIEW_KINDS and item["worse_than_gate"]
        ]
        if fails:
            view_fail.append({"failures": fails, "seed": seed})
        if float(row["delta"]) < COMPLEXITY_NOTE_GAIN:
            complexity_notes.append(
                {
                    "delta": row["delta"],
                    "reason": (
                        "complexity increased over structural baseline but "
                        f"seed {seed} gain {row['delta']} < {COMPLEXITY_NOTE_GAIN}"
                    ),
                    "seed": seed,
                }
            )
    mean_delta = float(np.mean(deltas))
    worst_delta = float(np.min(deltas))
    mean_quality = float(np.mean(qualities))
    worst_quality = float(np.min(qualities))
    passed = bool(
        not cap_fail
        and not view_fail
        and mean_delta >= GATE_MEAN_DELTA
        and worst_delta >= GATE_WORST_DELTA
        and mean_quality >= CHAMPION_ABS
    )
    return {
        "baseline_mean_quality": _json_float(float(np.mean(baseline_qualities))),
        "cap_failures": cap_fail,
        "complexity_notes": complexity_notes,
        "mean_absolute": _json_float(mean_quality),
        "mean_delta": _json_float(mean_delta),
        "passed": passed,
        "thresholds": {
            "mean_absolute": CHAMPION_ABS,
            "mean_delta": GATE_MEAN_DELTA,
            "stress_95_not_gated": True,
            "view_drop": GATE_VIEW_DROP,
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
        report = seed_reports[seed]
        base_head, cand_head = heads[seed]
        ties = content_tie_keys(pool.texts)
        base_models = allocate_all_tiers(
            base_head.pred_qa, base_head.pred_qk, pool.costs, pool.light_total, ties
        )
        cand_models = allocate_all_tiers(
            cand_head.pred_qa, cand_head.pred_qk, pool.costs, pool.light_total, ties
        )
        lam = {row["fold"]: row for row in report["lambda_by_fold"]}
        rows = []
        regimes = regimes_of(pool.episodes)
        for index, episode in enumerate(pool.episodes):
            fold = int(pool.folds[index])
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "family": pool.families[index],
                    "fold": fold,
                    "group_key": pool.group_keys[index],
                    "lambda_long": lam[fold]["lambda_long"],
                    "lambda_short": lam[fold]["lambda_short"],
                    "language": pool.languages[index],
                    "length_view": pool.length_views[index],
                    "pred_qa": _json_float(cand_head.pred_qa[index]),
                    "pred_qk": _json_float(cand_head.pred_qk[index]),
                    "regime": regimes[index],
                    "residual_qa": _json_float(cand_head.residual_qa[index]),
                    "residual_qk": _json_float(cand_head.residual_qk[index]),
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
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "feature": report["feature"],
            "fold_seeds": report["fold_seeds"],
            "identity": report["identity"],
            "lambda_distribution": report["lambda_distribution"],
            "limitations": report["limitations"],
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


def assemble(
    pool: PublicPool | None = None,
    *,
    seeds: Sequence[int] = FOLD_SEEDS,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    base_pool = pool or load_public_pool()
    seed_pools: dict[int, PublicPool] = {}
    seed_reports: dict[int, dict[str, Any]] = {}
    heads: dict[int, Tuple[HeadPred, HeadPred]] = {}
    for seed in seeds:
        current = relabel_folds(base_pool, int(seed))
        seed_pools[int(seed)] = current
        baseline_head, candidate_head, fold_rows = oof_regime_heads(current)
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
        seed_reports[int(seed)] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": _json_float(
                float(candidate["pooled"]["quality_weighted"])
                - float(baseline["pooled"]["quality_weighted"])
            ),
            "fold_fits": fold_rows,
            "fold_seed": int(seed),
            "lambda_by_fold": [
                {
                    "fold": row["fold"],
                    "lambda_long": row["lambda_long"],
                    "lambda_short": row["lambda_short"],
                }
                for row in fold_rows
            ],
            "views": views,
            "worst_view": _worst_view(views),
        }
    ordered = [seed_reports[seed] for seed in sorted(seed_reports)]
    gate = promotion_gate(ordered)
    decision = (
        f"record-e1c-promote-{CANDIDATE_NAME}"
        if gate["passed"]
        else "record-e1c-no-promote"
    )
    if gate["passed"]:
        decision_reason = (
            "promote nested_regime_structural_residual on repeated-seed "
            "exact-cost gates. Runtime export still requires independent review."
        )
    else:
        decision_reason = (
            "no-promote: the nested-regime residual did not clear the "
            "repeated-seed exact-cost gates. Quality-head search stops here. "
            "Keep the current runtime. Do not add another view-specific patch."
        )
    audit_document = episode_audit_document(seed_pools, seed_reports, heads)
    audit_sha = sha256_text(canonical_json_text(audit_document))
    lambda_rows = []
    for seed, report in seed_reports.items():
        for row in report["lambda_by_fold"]:
            lambda_rows.append(
                {
                    "fold": row["fold"],
                    "lambda_long": row["lambda_long"],
                    "lambda_short": row["lambda_short"],
                    "seed": int(seed),
                }
            )
    seed_payload = {}
    for seed, report in seed_reports.items():
        seed_payload[str(seed)] = {
            "baseline_quality": report["baseline"]["pooled"]["quality_weighted"],
            "candidate_quality": report["candidate"]["pooled"]["quality_weighted"],
            "delta": report["delta"],
            "fold_fits": report["fold_fits"],
            "fold_table": list(seed_pools[seed].fold_table),
            "lambda_by_fold": report["lambda_by_fold"],
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
        "cost_diagnostic": exact_cost_diagnostic(base_pool.costs),
        "decision": decision,
        "decision_reason": decision_reason,
        "experiment": EXPERIMENT,
        "feature": {
            "clip_quantile": CLIP_QUANTILE,
            "inner_objective": (
                "exact-cost weighted quality on outer-train; exclude "
                "family/language/length view drop < -0.003 versus λ=(0,0)"
            ),
            "lambda_grid": [float(value) for value in LAMBDA_GRID],
            "regime": "extract_features.long_context (character_count >= 8000)",
            "ridge_alpha": RIDGE_ALPHA,
            "runtime_artifact_changed": False,
            "tie_break": "higher quality, then smaller λ_short+λ_long, then λ_short, then λ_long",
            "tree": {
                "max_depth": TREE_MAX_DEPTH,
                "min_samples_leaf": TREE_MIN_SAMPLES_LEAF,
                "n_estimators": TREE_N_ESTIMATORS,
                "random_state": TREE_RANDOM_STATE,
            },
        },
        "fold_seeds": [int(seed) for seed in sorted(seed_reports)],
        "identity": dict(base_pool.identity),
        "lambda_distribution": lambda_rows,
        "limitations": [
            SEQUENTIAL_TESTING,
            "Outer held-out score labels never enter λ, clip, or residual fits.",
            "Hash candidates from E1B are discarded and are not re-run.",
            "95% stress caps are observational.",
            "A pass is not a runtime export.",
        ],
        "promotion_gate": gate,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_payload,
        "sequential_testing": SEQUENTIAL_TESTING,
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    return sort_mapping(report), audit_document


__all__ = (
    "AUDIT_RELATIVE_PATH",
    "BASELINE_NAME",
    "CANDIDATE_NAME",
    "FOLD_SEEDS",
    "LAMBDA_GRID",
    "assemble",
    "fit_outer_fold",
    "oof_regime_heads",
    "promotion_gate",
    "regime_label",
    "regimes_of",
    "relabel_folds",
    "select_lambdas",
    "write_json_atomic",
)
