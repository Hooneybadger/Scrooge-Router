# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1 quality-objective comparison on a locked feature space.

The current Scrooge quality head is already a continuous uplift Ridge
(Light→AX31). This module compares that single-reference objective with
adjacent-step delta regression, a sign/ranking surrogate, and a hybrid of
magnitude and positive probability. Features stay at the 14-d structural
vector used by the frozen runtime quality head. Allocation uses the existing
research exact-public-cost greedy density allocator so family-guard /
feasibility / budget-brake runtime paths are never called on retrain data.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ossp_router.cost_calibrated_router import structural_features
from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.modeling import (
    OFFICIAL_CAPS,
    STRESS_BACKSTOP,
    official_score,
    oof_predict,
    ridge_fit,
    ridge_predict,
    sort_mapping,
    weighted_final,
)
from research.lab.public_pool import (
    PublicPool,
    subset_inputs,
    subset_outcomes,
)
from research.lab.quality_heads import (
    allocate_k1_on_top,
    allocate_two_action,
    content_tie_keys,
    models_three_action,
    models_two_action,
    pearson,
    spearman,
)


EXPERIMENT = "e1-quality-objectives"
REPORT_TYPE = "scrooge-e1-quality-objectives-v2"
SCHEMA_VERSION = 2
FEATURE_NAME = "ossp_router.cost_calibrated_router.structural_features/14"
FEATURE_DIM = 14
RIDGE_ALPHA = 100.0
BASELINE_NAME = "baseline_continuous_uplift"
GATE_WEIGHTED_GAIN = 0.002
GATE_VIEW_DROP = 0.003
VIEW_MIN_N = 20
GATE_VIEW_KINDS: Tuple[str, ...] = ("family", "fold", "language", "length", "split")
AUDIT_RELATIVE_PATH = "build/compare-e1-quality-objectives/episode-audit.json"
STRESS_RATIO_CAPS = {"fast": 1.1875, "balanced": 1.90, "premium": 3.80}
_LIGHT = 0
_AX31 = 1
_K1 = 2

CANDIDATE_ORDER: Tuple[str, ...] = (
    "baseline_continuous_uplift",
    "direct_adjacent_delta",
    "delta_sign_ridge",
    "hybrid_magnitude_sign",
)

CANDIDATE_DEFINITIONS: Mapping[str, Mapping[str, str]] = {
    "baseline_continuous_uplift": {
        "pred_qa": "Ridge(score(ax31) - score(light))",
        "pred_qk": "Ridge(score(k1) - score(light))",
        "summary": (
            "Current continuous uplift family: both upgrades are scored as "
            "uplift from the cheapest model."
        ),
    },
    "direct_adjacent_delta": {
        "pred_qa": "Ridge(score(ax31) - score(light))",
        "pred_qk": "Ridge(score(k1) - score(ax31))",
        "summary": (
            "Adjacent-step direct delta: Light→AX31 and AX31→K1 are "
            "separate regression targets."
        ),
    },
    "delta_sign_ridge": {
        "pred_qa": "Ridge(I[score(ax31) > score(light)])",
        "pred_qk": "Ridge(I[score(k1) > score(ax31)])",
        "summary": (
            "Linear ranking/sign surrogate: closed-form ridge on the "
            "positive-delta indicator. Not a pairwise RankNet."
        ),
    },
    "hybrid_magnitude_sign": {
        "pred_qa": "Ridge(Δ31) * clip(Ridge(I[Δ31>0]), 0, 1)",
        "pred_qk": "Ridge(Δk1) * clip(Ridge(I[Δk1>0]), 0, 1)",
        "summary": (
            "Minimum hybrid: adjacent-step magnitude times clipped "
            "positive probability."
        ),
    },
}

ALLOCATOR = {
    "caps": dict(OFFICIAL_CAPS),
    "cost_evidence": "exact public outcome costs",
    "fast_balanced": "two-action Light/AX31 only (K1 hard-off)",
    "name": "exact_public_cost_greedy_density",
    "premium": "two-action then K1-on-top of AX31",
    "reason": (
        "research.lab.quality_heads greedy_upgrade_mask / allocate_two_action / "
        "allocate_k1_on_top. This is the existing quality-head candidate "
        "evaluation convention. Every candidate sees the same exact public "
        "costs and the same official hard caps. Predicted-cost family-guard, "
        "feasibility-ladder, and budget-brake runtime paths are not called, "
        "so those layers cannot leak fold labels or turn budget-bust=0 into "
        "a quality training loss. Ties break on content SHA-256."
    ),
    "tie_break": "content SHA-256",
    "upgrade_rule": "predicted score > 0 and density prefix stays within cap",
}


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def canonical_json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(canonical_json_text(value), encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(np.float64(value))
    if not np.isfinite(number):
        return None
    return number


def current_quality_matrix(episodes: Sequence[Any]) -> np.ndarray:
    """Intercept plus the 14-d runtime structural quality features."""

    rows = len(episodes)
    matrix = np.ones((rows, 1 + FEATURE_DIM), dtype=np.float64)
    for row, episode in enumerate(episodes):
        features = structural_features(episode)
        if len(features) != FEATURE_DIM:
            raise RuntimeError(
                f"structural_features width drifted: {len(features)} != {FEATURE_DIM}"
            )
        matrix[row, 1:] = features
    return matrix


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def oof_candidate_predictions(
    features: np.ndarray,
    scores: np.ndarray,
    folds: Sequence[int],
    *,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """OOF (pred_qa, pred_qk) for every locked candidate."""

    score_l = scores[:, _LIGHT]
    score_a = scores[:, _AX31]
    score_k = scores[:, _K1]
    delta_al = score_a - score_l
    delta_kl = score_k - score_l
    delta_ka = score_k - score_a
    sign_al = (delta_al > 0.0).astype(np.float64)
    sign_ka = (delta_ka > 0.0).astype(np.float64)

    pred_al = oof_predict(features, delta_al, folds, alpha=alpha)
    pred_kl = oof_predict(features, delta_kl, folds, alpha=alpha)
    pred_ka = oof_predict(features, delta_ka, folds, alpha=alpha)
    pred_sign_al = oof_predict(features, sign_al, folds, alpha=alpha)
    pred_sign_ka = oof_predict(features, sign_ka, folds, alpha=alpha)

    return {
        "baseline_continuous_uplift": (pred_al, pred_kl),
        "direct_adjacent_delta": (pred_al, pred_ka),
        "delta_sign_ridge": (pred_sign_al, pred_sign_ka),
        "hybrid_magnitude_sign": (
            pred_al * _clip01(pred_sign_al),
            pred_ka * _clip01(pred_sign_ka),
        ),
    }


def allocate_tier(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    tier: str,
    tie_keys: Sequence[str],
) -> Tuple[str, ...]:
    cap = float(OFFICIAL_CAPS[tier])
    upgrade_a = allocate_two_action(pred_qa, costs, light_total, cap, tie_keys)
    if tier == "premium":
        upgrade_k = allocate_k1_on_top(
            pred_qk, upgrade_a, costs, light_total, cap, tie_keys
        )
        return models_three_action(upgrade_a, upgrade_k)
    return models_two_action(upgrade_a)


def allocate_all_tiers(
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    tie_keys: Sequence[str],
) -> dict[str, Tuple[str, ...]]:
    return {
        tier: allocate_tier(pred_qa, pred_qk, costs, light_total, tier, tie_keys)
        for tier in TIERS
    }


def _count_models(model_ids: Sequence[str]) -> dict[str, int]:
    counts = {model_id: 0 for model_id in MODEL_IDS}
    for model_id in model_ids:
        counts[model_id] += 1
    return counts


def _tier_block(official: Mapping[str, Any], tier: str) -> dict[str, Any]:
    row = official["tiers"][tier]
    realized = float(row["budget_ratio"])
    return {
        "budget_passed": bool(row["budget_passed"]),
        "budget_ratio": row["budget_ratio"],
        "model_counts": {
            model_id: int(row["model_counts"].get(model_id, 0)) for model_id in MODEL_IDS
        },
        "near_budget": bool(row["near_budget"]),
        "quality_score": row["quality_score"],
        "realized_times_1054": _json_float(realized * float(STRESS_BACKSTOP)),
        "stress_95_cap": STRESS_RATIO_CAPS[tier],
        "stress_95_observed": bool(realized >= STRESS_RATIO_CAPS[tier]),
        "tier_score": row["tier_score"],
        "within_hard_cap": bool(row["budget_passed"]),
    }


def score_decisions(
    pool: PublicPool,
    models_by_tier: Mapping[str, Sequence[str]],
    *,
    indexes: Sequence[int] | None = None,
) -> dict[str, Any]:
    if indexes is None:
        inputs = pool.inputs
        outcomes = pool.outcomes
        chosen = {tier: tuple(models_by_tier[tier]) for tier in TIERS}
    else:
        inputs = subset_inputs(pool.inputs, indexes)
        outcomes = subset_outcomes(pool.inputs, pool.outcomes, indexes)
        n_subset = len(indexes)
        n_full = len(pool.inputs.episodes)
        chosen = {}
        for tier in TIERS:
            models = tuple(models_by_tier[tier])
            if len(models) == n_subset:
                chosen[tier] = models
            elif len(models) == n_full:
                chosen[tier] = tuple(models[index] for index in indexes)
            else:
                raise ValueError(
                    f"{tier} model list length {len(models)} matches neither "
                    f"subset {n_subset} nor full {n_full}"
                )
    official = official_score(inputs, outcomes, pool.policy, chosen)
    qualities = {tier: float(official["tiers"][tier]["quality_score"]) for tier in TIERS}
    return {
        "model_counts": {
            tier: _count_models(chosen[tier]) for tier in TIERS
        },
        "official_final_score": official["final_score"],
        "quality_weighted": _json_float(
            weighted_final(qualities["fast"], qualities["balanced"], qualities["premium"])
        ),
        "tiers": {tier: _tier_block(official, tier) for tier in TIERS},
    }


def _ranking_block(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    unequal = target != 0.0
    pred_pos = pred > 0.0
    actual_pos = target > 0.0
    unequal_n = int(np.count_nonzero(unequal))
    if unequal_n:
        sign_accuracy = _json_float(np.mean(pred_pos[unequal] == actual_pos[unequal]))
    else:
        sign_accuracy = None
    residual = pred - target
    return {
        "mae": _json_float(np.mean(np.abs(residual))),
        "mse": _json_float(np.mean(residual * residual)),
        "n_unequal": unequal_n,
        "pearson": _json_optional_float(pearson(pred, target)),
        "sign_accuracy_unequal": sign_accuracy,
        "spearman": _json_optional_float(spearman(pred, target)),
    }


def _slice_quality(
    scores: np.ndarray, models_by_tier: Mapping[str, Sequence[str]], mask: np.ndarray
) -> float | None:
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


def stress_views(
    pool: PublicPool,
    baseline_models: Mapping[str, Sequence[str]],
    candidate_models: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Pooled-allocation slices. Per-fold reallocations are not gate views."""

    views: list[tuple[str, str, np.ndarray]] = []
    for name in sorted(set(pool.families)):
        views.append(
            ("family", name, np.asarray([family == name for family in pool.families]))
        )
    for name in sorted(set(pool.length_views)):
        views.append(
            (
                "length",
                name,
                np.asarray([label == name for label in pool.length_views]),
            )
        )
    for name in sorted(set(pool.languages)):
        views.append(
            (
                "language",
                name,
                np.asarray([label == name for label in pool.languages]),
            )
        )
    for name in sorted(set(pool.split_labels)):
        views.append(
            (
                "split",
                name,
                np.asarray([label == name for label in pool.split_labels]),
            )
        )
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    if fold_ids.size:
        for fold in range(int(fold_ids.max()) + 1):
            views.append(("fold", str(fold), fold_ids == fold))
    rows = []
    for kind, name, mask in views:
        n = int(np.count_nonzero(mask))
        baseline = _slice_quality(pool.scores, baseline_models, mask)
        candidate = _slice_quality(pool.scores, candidate_models, mask)
        if baseline is None or candidate is None:
            delta = None
        else:
            delta = candidate - baseline
        gated = n >= VIEW_MIN_N
        drop_fail = bool(
            gated and delta is not None and delta < -GATE_VIEW_DROP
        )
        rows.append(
            {
                "allocation": "pooled",
                "baseline_quality_weighted": _json_optional_float(baseline),
                "candidate_quality_weighted": _json_optional_float(candidate),
                "delta": _json_optional_float(delta),
                "gated": gated,
                "kind": kind,
                "n": n,
                "name": name,
                "worse_than_gate": drop_fail,
            }
        )
    rows.sort(key=lambda row: (row["kind"], row["name"]))
    return rows


def exact_cost_diagnostic(costs: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(costs, dtype=np.float64)
    ax31_lt_light = int(np.count_nonzero(matrix[:, _AX31] < matrix[:, _LIGHT]))
    k1_lt_ax31 = int(np.count_nonzero(matrix[:, _K1] < matrix[:, _AX31]))
    return {
        "ax31_lt_light_rows": ax31_lt_light,
        "clamped": False,
        "k1_lt_ax31_rows": k1_lt_ax31,
        "n_rows": int(matrix.shape[0]),
        "note": (
            "Exact public costs are not monotone on every row "
            f"(AX31<Light {ax31_lt_light}, K1<AX31 {k1_lt_ax31}). "
            "Every E1 candidate uses this same cost matrix, so relative "
            "comparisons stay valid. This phase does not clamp costs."
        ),
        "shared_across_candidates": True,
    }


def episode_audit_document(
    pool: PublicPool,
    predicted: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    models_by_name: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    """Per-episode audit rows. No prompt text and no generated time."""

    rows = []
    for index, episode in enumerate(pool.episodes):
        predictions = {}
        selected = {}
        for name in CANDIDATE_ORDER:
            pred_qa, pred_qk = predicted[name]
            predictions[name] = {
                "pred_qa": _json_float(pred_qa[index]),
                "pred_qk": _json_float(pred_qk[index]),
            }
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
        "rows": rows,
    }


def _hard_caps_ok(scored: Mapping[str, Any]) -> bool:
    return all(scored["tiers"][tier]["within_hard_cap"] for tier in TIERS)


def evaluate_candidate(
    pool: PublicPool,
    name: str,
    pred_qa: np.ndarray,
    pred_qk: np.ndarray,
    tie_keys: Sequence[str],
) -> dict[str, Any]:
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    pooled_models = allocate_all_tiers(
        pred_qa, pred_qk, pool.costs, pool.light_total, tie_keys
    )
    pooled = score_decisions(pool, pooled_models)
    per_fold = []
    for fold in range(int(max(pool.folds)) + 1):
        indexes = [index for index, value in enumerate(pool.folds) if value == fold]
        mask = fold_ids == fold
        local_models = allocate_all_tiers(
            pred_qa[mask],
            pred_qk[mask],
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
    return {
        "definition": dict(CANDIDATE_DEFINITIONS[name]),
        "name": name,
        "per_fold": per_fold,
        "per_fold_note": (
            "Each fold is re-allocated as its own batch. These rows are "
            "observational and are not promotion-gate views. Gate fold "
            "views slice the pooled allocation."
        ),
        "pooled": pooled,
        "ranking": {
            "qa_vs_delta_ax31_light": _ranking_block(pred_qa, delta_al),
            "qk_vs_delta_k1_ax31": _ranking_block(pred_qk, delta_ka),
        },
    }


def promotion_gate(
    results: Mapping[str, Mapping[str, Any]],
    views_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    baseline = results[BASELINE_NAME]["pooled"]
    baseline_quality = float(baseline["quality_weighted"])
    rows = []
    for name in CANDIDATE_ORDER:
        if name == BASELINE_NAME:
            continue
        pooled = results[name]["pooled"]
        quality = float(pooled["quality_weighted"])
        delta = quality - baseline_quality
        views = list(views_by_candidate[name])
        view_fail = [row for row in views if row["worse_than_gate"]]
        caps_ok = _hard_caps_ok(pooled)
        quality_ok = delta >= GATE_WEIGHTED_GAIN
        views_ok = not view_fail
        passed = bool(quality_ok and views_ok and caps_ok)
        rows.append(
            {
                "candidate": name,
                "caps_ok": caps_ok,
                "delta_vs_baseline": _json_float(delta),
                "pass": passed,
                "quality_ok": quality_ok,
                "quality_weighted": _json_float(quality),
                "view_failures": [
                    f"{row['kind']}:{row['name']}" for row in view_fail
                ],
                "views_ok": views_ok,
            }
        )
    winners = [row for row in rows if row["pass"]]
    recommended = None
    if winners:
        recommended = max(
            winners,
            key=lambda row: (row["quality_weighted"], row["candidate"]),
        )["candidate"]
    return {
        "baseline": BASELINE_NAME,
        "baseline_quality_weighted": _json_float(baseline_quality),
        "candidates": rows,
        "passed": bool(winners),
        "recommended": recommended,
        "thresholds": {
            "fold_view": "pooled_allocation_slice",
            "gated_view_kinds": list(GATE_VIEW_KINDS),
            "per_fold_reallocation": "observational_not_gated",
            "stress_95_caps_observational_only": dict(STRESS_RATIO_CAPS),
            "view_drop": GATE_VIEW_DROP,
            "view_min_n": VIEW_MIN_N,
            "weighted_gain": GATE_WEIGHTED_GAIN,
        },
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fields that must be byte-stable across reruns on the same inputs."""

    return sort_mapping(
        {
            "allocator": report["allocator"],
            "audit": report["audit"],
            "candidates": report["candidates"],
            "cost_diagnostic": report["cost_diagnostic"],
            "decision": report["decision"],
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
            "stress_view_kinds": report["stress_view_kinds"],
            "stress_views": report["stress_views"],
        }
    )


def decision_core_sha256(report: Mapping[str, Any]) -> str:
    payload = decision_core_payload(report)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assemble(pool: PublicPool) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Build the E1 report and the episode audit document. Does not write files."""

    features = current_quality_matrix(pool.episodes)
    if features.shape != (len(pool.episodes), 1 + FEATURE_DIM):
        raise RuntimeError("quality design matrix has the wrong shape")
    tie_keys = content_tie_keys(pool.texts)
    predicted = oof_candidate_predictions(features, pool.scores, pool.folds)
    results = {}
    models_by_name: dict[str, dict[str, Tuple[str, ...]]] = {}
    for name in CANDIDATE_ORDER:
        pred_qa, pred_qk = predicted[name]
        results[name] = evaluate_candidate(pool, name, pred_qa, pred_qk, tie_keys)
        models_by_name[name] = allocate_all_tiers(
            pred_qa, pred_qk, pool.costs, pool.light_total, tie_keys
        )

    views = {
        name: stress_views(pool, models_by_name[BASELINE_NAME], models_by_name[name])
        for name in CANDIDATE_ORDER
    }

    gate = promotion_gate(results, views)
    decision = (
        f"record-e1-promote-{gate['recommended']}"
        if gate["recommended"]
        else "record-e1-no-promote"
    )
    n_ridge_oof = 5 * int(max(pool.folds) + 1)
    cost_diagnostic = exact_cost_diagnostic(pool.costs)
    audit_document = episode_audit_document(pool, predicted, models_by_name)
    audit_text = canonical_json_text(audit_document)
    audit_sha = sha256_text(audit_text)
    present_kinds = sorted(
        {
            row["kind"]
            for rows in views.values()
            for row in rows
        }
    )
    limitations = [
        "Source IDs are absent from the public Episode schema, so gated "
        "stress views are family / length / language / split(train|dev) / "
        "fold(0..k pooled-allocation slices) rather than leave-one-source-out.",
        "Fold gate views slice the pooled 2,640 allocation. "
        "results[].per_fold re-allocates each fold as its own batch and is "
        "observational only — it is not a promotion-gate input.",
        "Pooled allocation treats the public episodes as one batch. "
        "That is not the hidden evaluation batch size.",
        "Exact public costs isolate the quality objective. They are more "
        "optimistic than runtime predicted-cost family-guard / brake paths, "
        "so these scores are not the public Dev runtime 0.669517 replay.",
        cost_diagnostic["note"],
        "The 14-d structural matrix matches the frozen quality-head features "
        "but is not standardized with the frozen artifact scale; all "
        "candidates share this locked design.",
        "delta_sign_ridge is a linear indicator surrogate, not a pairwise "
        "RankNet or sklearn logistic model.",
        "Stress 95% ratio caps are observational in this quality-objective "
        "phase and are not a promotion requirement.",
        "A passing candidate is not exported into src/ in this phase.",
    ]
    report = {
        "allocator": ALLOCATOR,
        "audit": {
            "n_rows": int(audit_document["n_rows"]),
            "relative_path": AUDIT_RELATIVE_PATH,
            "sha256": audit_sha,
        },
        "candidates": {
            name: dict(CANDIDATE_DEFINITIONS[name]) for name in CANDIDATE_ORDER
        },
        "cost_diagnostic": cost_diagnostic,
        "decision": decision,
        "experiment": EXPERIMENT,
        "feature": {
            "alpha": RIDGE_ALPHA,
            "dimension": FEATURE_DIM,
            "intercept": True,
            "name": FEATURE_NAME,
            "runtime_artifact_changed": False,
        },
        "fold_table": list(pool.fold_table),
        "grouping": dict(pool.grouping),
        "identity": dict(pool.identity),
        "limitations": limitations,
        "promotion_gate": gate,
        "report_type": REPORT_TYPE,
        "results": results,
        "schema_version": SCHEMA_VERSION,
        "stress_views": views,
        "stress_view_kinds": present_kinds,
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    report["runtime"] = {
        "complexity": {
            "feature_columns": 1 + FEATURE_DIM,
            "n_candidates": len(CANDIDATE_ORDER),
            "n_oof_ridge_fits": n_ridge_oof,
            "note": (
                "Five unique Ridge targets (Δ31, Δk1-from-light, Δk1-adjacent, "
                "I[Δ31>0], I[Δk1>0]) each fit once per fold via closed-form "
                "ridge. Allocator is exact-cost greedy, not the runtime stack."
            ),
        },
        "excluded_from_core": ["elapsed_s"],
    }
    return sort_mapping(report), audit_document


def measure(pool: PublicPool) -> dict[str, Any]:
    report, _audit = assemble(pool)
    return report


def fit_full_public(
    pool: PublicPool, candidate: str, *, alpha: float = RIDGE_ALPHA
) -> dict[str, Any]:
    """Frozen-setting full-public refit helper for a later phase. Not used to select."""

    if candidate not in CANDIDATE_DEFINITIONS:
        raise ValueError(f"unknown candidate: {candidate}")
    features = current_quality_matrix(pool.episodes)
    scores = pool.scores
    delta_al = scores[:, _AX31] - scores[:, _LIGHT]
    delta_kl = scores[:, _K1] - scores[:, _LIGHT]
    delta_ka = scores[:, _K1] - scores[:, _AX31]
    sign_al = (delta_al > 0.0).astype(np.float64)
    sign_ka = (delta_ka > 0.0).astype(np.float64)
    heads: dict[str, np.ndarray] = {}
    if candidate == "baseline_continuous_uplift":
        heads["qa"] = ridge_fit(features, delta_al, alpha=alpha)
        heads["qk"] = ridge_fit(features, delta_kl, alpha=alpha)
    elif candidate == "direct_adjacent_delta":
        heads["qa"] = ridge_fit(features, delta_al, alpha=alpha)
        heads["qk"] = ridge_fit(features, delta_ka, alpha=alpha)
    elif candidate == "delta_sign_ridge":
        heads["qa"] = ridge_fit(features, sign_al, alpha=alpha)
        heads["qk"] = ridge_fit(features, sign_ka, alpha=alpha)
    else:
        heads["qa_mag"] = ridge_fit(features, delta_al, alpha=alpha)
        heads["qk_mag"] = ridge_fit(features, delta_ka, alpha=alpha)
        heads["qa_sign"] = ridge_fit(features, sign_al, alpha=alpha)
        heads["qk_sign"] = ridge_fit(features, sign_ka, alpha=alpha)
    return {
        "alpha": float(alpha),
        "candidate": candidate,
        "feature": FEATURE_NAME,
        "heads": {name: [float(value) for value in vector] for name, vector in heads.items()},
        "n_episodes": int(features.shape[0]),
        "selection_use": False,
    }


def predict_full_public(
    pool: PublicPool, fitted: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    features = current_quality_matrix(pool.episodes)
    heads = {name: np.asarray(vector, dtype=np.float64) for name, vector in fitted["heads"].items()}
    candidate = str(fitted["candidate"])
    if candidate in {"baseline_continuous_uplift", "direct_adjacent_delta", "delta_sign_ridge"}:
        return ridge_predict(heads["qa"], features), ridge_predict(heads["qk"], features)
    pred_qa = ridge_predict(heads["qa_mag"], features) * _clip01(
        ridge_predict(heads["qa_sign"], features)
    )
    pred_qk = ridge_predict(heads["qk_mag"], features) * _clip01(
        ridge_predict(heads["qk_sign"], features)
    )
    return pred_qa, pred_qk


__all__ = (
    "ALLOCATOR",
    "AUDIT_RELATIVE_PATH",
    "BASELINE_NAME",
    "CANDIDATE_DEFINITIONS",
    "CANDIDATE_ORDER",
    "EXPERIMENT",
    "FEATURE_NAME",
    "GATE_VIEW_DROP",
    "GATE_VIEW_KINDS",
    "GATE_WEIGHTED_GAIN",
    "REPORT_TYPE",
    "RIDGE_ALPHA",
    "SCHEMA_VERSION",
    "allocate_all_tiers",
    "allocate_tier",
    "assemble",
    "canonical_json_text",
    "current_quality_matrix",
    "decision_core_payload",
    "decision_core_sha256",
    "episode_audit_document",
    "evaluate_candidate",
    "exact_cost_diagnostic",
    "fit_full_public",
    "measure",
    "oof_candidate_predictions",
    "predict_full_public",
    "promotion_gate",
    "score_decisions",
    "sha256_text",
    "stress_views",
    "write_json_atomic",
)
