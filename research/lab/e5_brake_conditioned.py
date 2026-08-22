# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E5 — brake-conditioned Premium ranking evaluated inside the serving allocator.

The frozen-runtime fidelity failure showed that heads selected against
exact-cost OOF allocators do not transfer to the frozen brake allocator.
E5 therefore varies only the promotion rank key and evaluates every arm
through the actual serving path: shared parent allocation from the shipped
artifact, then ``budget_brake_router.promote_premium_brake`` with an
injected quality vector whose order encodes the arm. Scoring uses public
outcomes only after decisions are frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from ossp_router import budget_brake_router, family_guard_router
from ossp_router.cost_calibrated_router import structural_features
from ossp_router.feasibility_ladder import _select_premium_configured
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    RoutingPolicy,
    load_bundled_policy,
)
from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.grouped_crossfit import assign_balanced_group_folds, fold_leakage_count
from research.lab.modeling import official_score, sort_mapping
from research.lab.public_pool import (
    EXPECTED_N_DEV,
    EXPECTED_N_TRAIN,
    PublicPool,
    load_public_pool,
    subset_inputs,
    subset_outcomes,
)
from research.lab.validation import public_arrays


EXPERIMENT = "e5-brake-conditioned-v1"
REPORT_TYPE = "scrooge-e5-brake-conditioned-v1"
SCHEMA_VERSION = 1
BASELINE_NAME = "uplift-xtrees-refit"
PRIMARY_NAME = "density-xtrees-refit"
SECONDARY_NAME = "density-ridge-standardized-wls"
EVAL_ARMS: Tuple[str, ...] = (BASELINE_NAME, PRIMARY_NAME, SECONDARY_NAME)
GATED_ARMS: Tuple[str, ...] = (BASELINE_NAME, PRIMARY_NAME)
MODEL_INDEX = {model: index for index, model in enumerate(MODEL_IDS)}
K1_MODEL = MODEL_IDS[2]
AUDIT_RELATIVE = "build/run-e5-brake-conditioned/episode-audit.json"
REPORT_RELATIVE = "build/run-e5-brake-conditioned/report.json"

FOLDS = 5
N_STRUCTURAL = 14
DENSITY_FLOOR = 1e-12
RIDGE_ALPHA = 100.0
PINNED_DEV_FINAL_SCORE = 0.669517045455
PIN_TOLERANCE = 5e-13
TVBALL_EPSILON = 0.014204545454545449
VIEW_MIN_N = 20
XTREES_PARAMS: Mapping[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "max_features": 1.0,
    "criterion": "squared_error",
    "bootstrap": False,
    "n_jobs": 1,
    "random_state": 20260816,
}


class ProtocolError(RuntimeError):
    """Sealed protocol or harness contract failure."""


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(dict(protocol)))


def derive_fresh_seeds(
    prefix: str, core_sha256: str, count: int, forbidden: Sequence[int]
) -> Tuple[int, ...]:
    """Fail-closed seed derivation; collisions must never be skipped."""

    forbidden_set = {int(value) for value in forbidden}
    seen: set[int] = set()
    seeds: list[int] = []
    for index in range(count):
        digest = hashlib.sha256(
            prefix.encode("utf-8")
            + b"\x00"
            + bytes.fromhex(core_sha256)
            + index.to_bytes(4, "big")
        ).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed in seen or seed in forbidden_set:
            raise ProtocolError(f"fresh seed collision at index {index}: {seed}")
        seen.add(seed)
        seeds.append(seed)
    return tuple(seeds)


def verify_protocol(
    protocol: Mapping[str, Any],
    expected_sha256: str,
    *,
    pool_identity: Optional[Mapping[str, Any]] = None,
) -> str:
    digest = protocol_sha256(protocol)
    if digest != expected_sha256:
        raise ProtocolError(
            f"protocol sha mismatch: got {digest}, expected {expected_sha256}"
        )
    if (
        protocol.get("experiment") != EXPERIMENT
        or protocol.get("protocol_id") != EXPERIMENT
    ):
        raise ProtocolError("protocol experiment id drifted")
    derivation = protocol["seed_derivation"]
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    expected_seeds = derive_fresh_seeds(
        str(derivation["prefix"]),
        str(derivation["core_sha256"]),
        int(derivation["n"]),
        [int(value) for value in derivation["forbidden_previous_seeds"]],
    )
    if fresh != expected_seeds or len(set(fresh)) != len(fresh):
        raise ProtocolError("sealed fresh seeds drifted from the derivation")
    if pool_identity is not None:
        pins = protocol["pins"]
        for key in (
            "train_inputs_sha256",
            "train_outcomes_sha256",
            "dev_inputs_sha256",
            "dev_outcomes_sha256",
            "policy_sha256",
        ):
            if str(pins[key]) != str(pool_identity[key]):
                raise ProtocolError(f"pinned {key} drifted")
    thresholds = protocol["thresholds"]
    if float(thresholds["mean_delta_min"]) <= 0.0:
        raise ProtocolError("mean_delta_min must be positive")
    if float(thresholds["stress_actual_ratio_max"]) <= 0.0:
        raise ProtocolError("stress_actual_ratio_max must be positive")
    return digest


@dataclass(frozen=True)
class ArmFit:
    """Fold-local heads fitted on outer-train labels only."""

    forest: Any
    ridge_coefficients: np.ndarray
    ridge_mean: np.ndarray
    ridge_scale: np.ndarray

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tree = self.forest.predict(features)
        design = _ridge_design(features, self.ridge_mean, self.ridge_scale)
        ridge = design @ self.ridge_coefficients
        return tree, ridge


def _ridge_design(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    standardized = (features - mean) / scale
    return np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), standardized],
        axis=1,
    )


def _weighted_ridge(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray, alpha: float
) -> np.ndarray:
    root = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted = design * root[:, None]
    gram = weighted.T @ weighted
    gram = 0.5 * (gram + gram.T)
    penalty = np.full(design.shape[1], float(alpha), dtype=np.float64)
    penalty[0] = 0.0
    gram = gram + np.diag(penalty)
    rhs = weighted.T @ (np.asarray(target, dtype=np.float64).reshape(-1) * root)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        gram = gram.copy()
        gram.flat[:: gram.shape[0] + 1] += 1e-12
        return np.linalg.solve(gram, rhs)


@dataclass(frozen=True)
class Harness:
    """Frozen-context serving replica shared by every arm."""

    pool: PublicPool
    policy: RoutingPolicy
    brake: budget_brake_router.BrakeArtifact
    features: np.ndarray
    families: Tuple[str, ...]
    digests: Tuple[str, ...]
    rows: Tuple[Tuple[float, Tuple[float, float, float]], ...]
    increments: np.ndarray
    predicted_cap_premium: float
    shipped_quality: np.ndarray
    generations: np.ndarray

    @classmethod
    def build(cls, pool: PublicPool) -> "Harness":
        policy = load_bundled_policy()
        brake = budget_brake_router.load_bundled_artifact()
        rows = tuple(
            budget_brake_router.premium_prediction_row(episode, policy, brake)
            for episode in pool.episodes
        )
        arrays = public_arrays(pool.inputs, pool.outcomes, policy)
        if not np.array_equal(arrays.costs, pool.costs):
            raise ProtocolError("exact cost matrices drifted from the pool")
        features = np.asarray(
            [structural_features(episode) for episode in pool.episodes],
            dtype=np.float64,
        )
        if features.ndim != 2 or features.shape[1] != N_STRUCTURAL:
            raise ProtocolError(f"structural_features must be {N_STRUCTURAL}-d")
        shipped = np.asarray(
            [
                budget_brake_router.predict_quality_features(row, brake)
                for row in features
            ],
            dtype=np.float64,
        )
        costs = np.asarray([row[1] for row in rows], dtype=np.float64)
        return cls(
            pool=pool,
            policy=policy,
            brake=brake,
            features=features,
            families=tuple(
                family_guard_router.prompt_family(episode) for episode in pool.episodes
            ),
            digests=tuple(
                budget_brake_router.content_digest(episode)
                for episode in pool.episodes
            ),
            rows=rows,
            increments=costs[:, 2] - costs[:, 1],
            predicted_cap_premium=float(brake.value["predicted_caps"]["premium"]),
            shipped_quality=shipped,
            generations=np.asarray(arrays.generations, dtype=np.int64),
        )

    def parent_allocation(self, indexes: Sequence[int]) -> Tuple[str, ...]:
        batch = subset_inputs(self.pool.inputs, indexes)
        rows = tuple(self.rows[index] for index in indexes)
        selected, _ratio = _select_premium_configured(
            batch, rows, self.predicted_cap_premium, self.brake.family_guard.base
        )
        return selected

    def promote(
        self,
        indexes: Sequence[int],
        parent: Sequence[str],
        quality_raw: np.ndarray,
        arm: str,
    ) -> Tuple[str, ...]:
        """Inject an order-encoding quality vector into the frozen promote loop."""

        if arm not in EVAL_ARMS:
            raise ProtocolError(f"unknown arm: {arm}")
        vector = np.asarray(quality_raw, dtype=np.float64)
        if arm in (PRIMARY_NAME, SECONDARY_NAME):
            vector = vector / np.maximum(
                self.increments[np.asarray(indexes, dtype=np.int64)], DENSITY_FLOOR
            )
        costs = [list(self.rows[index][1]) for index in indexes]
        digests = [self.digests[index] for index in indexes]
        families = [self.families[index] for index in indexes]
        return budget_brake_router.promote_premium_brake(
            list(parent),
            [float(value) for value in vector],
            families,
            costs,
            digests,
            self.brake.budget_brake,
        )


def fit_arms(harness: Harness, outer: Sequence[int]) -> ArmFit:
    outer_array = np.asarray(outer, dtype=np.int64)
    features = harness.features[outer_array]
    target = harness.pool.scores[outer_array, 2] - harness.pool.scores[outer_array, 1]
    weights = np.minimum(
        harness.generations[outer_array, 1].astype(np.float64),
        harness.generations[outer_array, 2].astype(np.float64),
    )
    forest = ExtraTreesRegressor(**XTREES_PARAMS)
    forest.fit(features, target, sample_weight=weights)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    design = _ridge_design(features, mean, scale)
    coefficients = _weighted_ridge(design, target, weights, RIDGE_ALPHA)
    return ArmFit(
        forest=forest,
        ridge_coefficients=coefficients,
        ridge_mean=mean,
        ridge_scale=scale,
    )


def equivalence_check(harness: Harness) -> Mapping[str, Any]:
    """Composed path must reproduce make_submission on the Train replay."""

    train_indexes = [
        index
        for index, label in enumerate(harness.pool.split_labels)
        if label == "train"
    ]
    if len(train_indexes) != EXPECTED_N_TRAIN:
        raise ProtocolError("train subset size drifted")
    batch = subset_inputs(harness.pool.inputs, train_indexes)
    runtime_plan = budget_brake_router.make_submission(
        batch, harness.policy, harness.brake, "premium"
    )
    runtime_models = tuple(
        decision.model_id for decision in runtime_plan.submission.decisions
    )
    parent = harness.parent_allocation(train_indexes)
    composed_models = harness.promote(
        train_indexes, parent, harness.shipped_quality[train_indexes], BASELINE_NAME
    )
    return {
        "matched": bool(runtime_models == tuple(composed_models)),
        "split": "train",
        "n_episodes": len(train_indexes),
        "runtime_n_k1": int(sum(model == K1_MODEL for model in runtime_models)),
        "composed_n_k1": int(sum(model == K1_MODEL for model in composed_models)),
    }


def dev_pin_replay(harness: Harness) -> Mapping[str, Any]:
    """Reproduce the pinned public Dev final score through the runtime path."""

    dev_indexes = [
        index
        for index, label in enumerate(harness.pool.split_labels)
        if label == "dev"
    ]
    if len(dev_indexes) != EXPECTED_N_DEV:
        raise ProtocolError("dev subset size drifted")
    batch = subset_inputs(harness.pool.inputs, dev_indexes)
    plans = {
        tier: budget_brake_router.make_submission(
            batch, harness.policy, harness.brake, tier
        ).submission
        for tier in TIERS
    }
    outcomes = subset_outcomes(harness.pool.inputs, harness.pool.outcomes, dev_indexes)
    official = official_score(batch, outcomes, harness.policy, plans)
    final = float(official["final_score"])
    pinned = float(PINNED_DEV_FINAL_SCORE)
    return {
        "final_score": final,
        "pinned_final_score": pinned,
        "matched": bool(abs(final - pinned) <= PIN_TOLERANCE),
        "premium_budget_ratio": float(official["tiers"]["premium"]["budget_ratio"]),
        "premium_n_k1": int(
            sum(
                decision.model_id == K1_MODEL
                for decision in plans["premium"].decisions
            )
        ),
    }


@dataclass(frozen=True)
class FoldOutcome:
    held: Tuple[int, ...]
    parent: Tuple[str, ...]
    selections: Mapping[str, Tuple[str, ...]]
    fold_ratios: Mapping[str, float]
    tree_quality: np.ndarray
    ridge_quality: np.ndarray


def evaluate_fold(
    harness: Harness, folds: Sequence[int], fold: int
) -> FoldOutcome:
    held = tuple(index for index, value in enumerate(folds) if value == fold)
    outer = [index for index, value in enumerate(folds) if value != fold]
    fit = fit_arms(harness, outer)
    tree_quality, ridge_quality = fit.predict(
        harness.features[np.asarray(held, dtype=np.int64)]
    )
    parent = harness.parent_allocation(held)
    selections = {
        BASELINE_NAME: harness.promote(held, parent, tree_quality, BASELINE_NAME),
        PRIMARY_NAME: harness.promote(held, parent, tree_quality, PRIMARY_NAME),
        SECONDARY_NAME: harness.promote(
            held, parent, ridge_quality, SECONDARY_NAME
        ),
    }
    ratios: dict[str, float] = {}
    for arm in EVAL_ARMS:
        models = selections[arm]
        numerator = math.fsum(
            float(harness.pool.costs[index, MODEL_INDEX[model]])
            for index, model in zip(held, models)
        )
        denominator = math.fsum(float(harness.pool.costs[index, 0]) for index in held)
        if denominator <= 0.0:
            raise ProtocolError("fold light denominator is not positive")
        ratios[arm] = numerator / denominator
    return FoldOutcome(
        held=held,
        parent=parent,
        selections=selections,
        fold_ratios=ratios,
        tree_quality=np.asarray(tree_quality, dtype=np.float64),
        ridge_quality=np.asarray(ridge_quality, dtype=np.float64),
    )


def _batch_ratio(
    harness: Harness,
    indexes: Sequence[int],
    models: Sequence[str],
) -> float:
    numerator = math.fsum(
        float(harness.pool.costs[index, MODEL_INDEX[model]])
        for index, model in zip(indexes, models)
    )
    denominator = math.fsum(float(harness.pool.costs[index, 0]) for index in indexes)
    return numerator / max(denominator, DENSITY_FLOOR)


def evaluate_seed(
    harness: Harness, seed: int, thresholds: Mapping[str, Any]
) -> Mapping[str, Any]:
    pool = harness.pool
    n_episodes = len(pool.episodes)
    folds = assign_balanced_group_folds(
        pool.group_keys, pool.families, folds=FOLDS, seed=seed
    )
    leaked = fold_leakage_count(pool.group_keys, folds)
    if leaked:
        raise ProtocolError(f"grouped fold leakage: {leaked}")

    tree_quality_all = np.full(n_episodes, np.nan, dtype=np.float64)
    ridge_quality_all = np.full(n_episodes, np.nan, dtype=np.float64)
    chosen_scores: dict[str, list[float]] = {arm: [] for arm in EVAL_ARMS}
    chosen_costs: dict[str, list[float]] = {arm: [] for arm in EVAL_ARMS}
    family_order: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    fold_ratio_rows: list[dict[str, Any]] = []
    k1_counts: dict[str, int] = {arm: 0 for arm in EVAL_ARMS}

    first_outcome: Optional[FoldOutcome] = None
    repeat_outcome: Optional[FoldOutcome] = None
    for fold in range(FOLDS):
        outcome = evaluate_fold(harness, folds, fold)
        if fold == 0:
            first_outcome = outcome
            repeat_outcome = evaluate_fold(harness, folds, fold)
        for position, index in enumerate(outcome.held):
            tree_quality_all[index] = float(outcome.tree_quality[position])
            ridge_quality_all[index] = float(outcome.ridge_quality[position])
        for arm in EVAL_ARMS:
            models = outcome.selections[arm]
            k1_counts[arm] += sum(model == K1_MODEL for model in models)
            for index, model in zip(outcome.held, models):
                column = MODEL_INDEX[model]
                chosen_scores[arm].append(float(pool.scores[index, column]))
                chosen_costs[arm].append(float(pool.costs[index, column]))
        for position, index in enumerate(outcome.held):
            family_order.append(harness.families[index])
            audit_rows.append(
                {
                    "episode_id": pool.episodes[index].episode_id,
                    "family": harness.families[index],
                    "fold": int(fold),
                    **{arm: outcome.selections[arm][position] for arm in EVAL_ARMS},
                }
            )
        for arm in GATED_ARMS:
            fold_ratio_rows.append(
                {
                    "arm": arm,
                    "fold": int(fold),
                    "ratio": float(outcome.fold_ratios[arm]),
                }
            )

    assert first_outcome is not None and repeat_outcome is not None
    deterministic = bool(
        list(repeat_outcome.parent) == list(first_outcome.parent)
        and all(
            list(repeat_outcome.selections[arm])
            == list(first_outcome.selections[arm])
            for arm in EVAL_ARMS
        )
    )
    del first_outcome, repeat_outcome

    if np.isnan(tree_quality_all).any() or np.isnan(ridge_quality_all).any():
        raise ProtocolError("fold predictions did not cover every episode")

    pooled = {
        arm: math.fsum(values) / float(n_episodes)
        for arm, values in chosen_scores.items()
    }
    delta = {
        arm: float(pooled[arm] - pooled[BASELINE_NAME])
        for arm in (PRIMARY_NAME, SECONDARY_NAME)
    }

    family_deltas: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for arm in (PRIMARY_NAME, SECONDARY_NAME):
        buckets: dict[str, list[float]] = {}
        for family_value, arm_score, base_score in zip(
            family_order, chosen_scores[arm], chosen_scores[BASELINE_NAME]
        ):
            buckets.setdefault(family_value, []).append(arm_score - base_score)
        family_deltas[arm] = {
            name: {
                "delta": float(math.fsum(values) / len(values)),
                "n": int(len(values)),
            }
            for name, values in sorted(buckets.items())
        }

    def tvball_worst(arm: str) -> float:
        eligible = [
            float(row["delta"])
            for row in family_deltas[arm].values()
            if int(row["n"]) >= VIEW_MIN_N
        ]
        if not eligible:
            raise ProtocolError("no eligible family view for the tv-ball")
        spread = min(eligible) - max(eligible)
        return float(delta[arm] + TVBALL_EPSILON * spread)

    stress_rows: list[dict[str, Any]] = []
    stress_max: dict[str, float] = {arm: 0.0 for arm in EVAL_ARMS}
    for family_name in sorted(set(harness.families)):
        indexes = [
            index
            for index, value in enumerate(harness.families)
            if value == family_name
        ]
        if len(indexes) < VIEW_MIN_N:
            continue
        parent = harness.parent_allocation(indexes)
        selections = {
            BASELINE_NAME: harness.promote(
                indexes, parent, tree_quality_all[indexes], BASELINE_NAME
            ),
            PRIMARY_NAME: harness.promote(
                indexes, parent, tree_quality_all[indexes], PRIMARY_NAME
            ),
            SECONDARY_NAME: harness.promote(
                indexes, parent, ridge_quality_all[indexes], SECONDARY_NAME
            ),
        }
        row: dict[str, Any] = {"view": f"family:{family_name}", "n": len(indexes)}
        for arm, models in selections.items():
            row[f"{arm}_ratio"] = float(_batch_ratio(harness, indexes, models))
            row[f"{arm}_n_k1"] = int(sum(model == K1_MODEL for model in models))
            stress_max[arm] = max(stress_max[arm], row[f"{arm}_ratio"])
        stress_rows.append(row)

    return {
        "seed": int(seed),
        "quality_premium": {arm: float(pooled[arm]) for arm in EVAL_ARMS},
        "delta_vs_baseline": delta,
        "tvball_worst": {
            PRIMARY_NAME: tvball_worst(PRIMARY_NAME),
            SECONDARY_NAME: tvball_worst(SECONDARY_NAME),
        },
        "family_deltas": family_deltas,
        "k1_counts": k1_counts,
        "fold_ratios": fold_ratio_rows,
        "stress_views": stress_rows,
        "stress_max_ratio": {arm: float(value) for arm, value in stress_max.items()},
        "determinism_passed": deterministic,
        "audit_rows": audit_rows,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "audit": report["audit"],
            "candidate_primary": report["candidate_primary"],
            "constants": report["constants"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "equivalence": report["equivalence"],
            "experiment": report["experiment"],
            "fold_seeds": report["fold_seeds"],
            "gate": report["gate"],
            "pin_dev_replay": report["pin_dev_replay"],
            "protocol_sha256": report["protocol_sha256"],
            "report_type": report["report_type"],
            "schema_version": report["schema_version"],
            "seed_results": report["seed_results"],
            "thresholds": report["thresholds"],
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
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("e5 output exists; refuse overwrite")

    pool = load_public_pool()
    harness = Harness.build(pool)
    thresholds = dict(protocol["thresholds"])
    seeds = [int(seed) for seed in protocol["fresh_seeds"]]

    equivalence = equivalence_check(harness)
    pin = dev_pin_replay(harness)

    seed_results: dict[str, Any] = {}
    audits: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        result = evaluate_seed(harness, seed, thresholds)
        audits[str(seed)] = result.pop("audit_rows")
        seed_results[str(seed)] = result

    primary_deltas = [
        float(seed_results[str(seed)]["delta_vs_baseline"][PRIMARY_NAME])
        for seed in seeds
    ]
    tvball_values = [
        float(seed_results[str(seed)]["tvball_worst"][PRIMARY_NAME])
        for seed in seeds
    ]
    secondary_deltas = [
        float(seed_results[str(seed)]["delta_vs_baseline"][SECONDARY_NAME])
        for seed in seeds
    ]
    mean_delta = math.fsum(primary_deltas) / len(primary_deltas)
    worst_delta = min(primary_deltas)
    ratio_limit = float(thresholds["stress_actual_ratio_max"])

    safety_failures: list[Mapping[str, Any]] = []
    for seed in seeds:
        block = seed_results[str(seed)]
        for row in block["fold_ratios"]:
            if row["arm"] in GATED_ARMS and float(row["ratio"]) >= ratio_limit:
                safety_failures.append(
                    {"kind": "fold", "name": f"fold:{row['fold']}", "seed": seed,
                     "arm": row["arm"], "ratio": float(row["ratio"])}
                )
        for view_row in block["stress_views"]:
            for arm in GATED_ARMS:
                if float(view_row[f"{arm}_ratio"]) >= ratio_limit:
                    safety_failures.append(
                        {"kind": "stress", "name": view_row["view"], "seed": seed,
                         "arm": arm, "ratio": float(view_row[f"{arm}_ratio"])}
                    )

    determinism_passed = all(
        bool(block["determinism_passed"]) for block in seed_results.values()
    )
    gates_ok = (
        mean_delta >= float(thresholds["mean_delta_min"])
        and worst_delta > float(thresholds["worst_seed_delta_min_exclusive"])
        and min(tvball_values) >= float(thresholds["tvball_worst_min"])
        and not safety_failures
        and determinism_passed
        and bool(equivalence["matched"])
        and bool(pin["matched"])
    )

    if gates_ok:
        decision = str(protocol["decisions"]["pass"])
        reason = str(protocol["decision_reasons"]["pass"])
    else:
        decision = str(protocol["decisions"]["fail"])
        reason = str(protocol["decision_reasons"]["fail"])

    audit_document = {
        "arms": list(EVAL_ARMS),
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "rows": {seed: audits[seed] for seed in sorted(audits)},
    }
    audit_payload = {
        "n_rows": sum(len(rows) for rows in audits.values()),
        "relative_path": AUDIT_RELATIVE,
        "sha256": sha256_text(canonical_json_text(audit_document)),
    }

    gate_block = {
        "mean_delta": float(mean_delta),
        "worst_delta": float(worst_delta),
        "tvball_worst_min": float(min(tvball_values)),
        "secondary_mean_delta_evidence_only": math.fsum(secondary_deltas)
        / len(secondary_deltas),
        "primary_deltas_by_seed": {
            str(seed): value for seed, value in zip(seeds, primary_deltas)
        },
        "safety_failures": [dict(item) for item in safety_failures],
        "determinism_passed": bool(determinism_passed),
        "equivalence_matched": bool(equivalence["matched"]),
        "pin_matched": bool(pin["matched"]),
        "passed": bool(gates_ok),
    }

    report = {
        "audit": audit_payload,
        "candidate_primary": PRIMARY_NAME,
        "constants": {
            "brake_ratio": float(harness.brake.budget_brake["brake_ratio"]),
            "count_cap": int(harness.brake.budget_brake["count_cap"]),
            "denylist_families": list(
                harness.brake.budget_brake["denylist_families"]
            ),
            "density_floor": DENSITY_FLOOR,
            "predicted_cap_premium": float(harness.predicted_cap_premium),
            "ridge_alpha": RIDGE_ALPHA,
            "runaway_absolute": float(
                harness.brake.budget_brake["runaway_absolute"]
            ),
            "xtrees_params": dict(XTREES_PARAMS),
        },
        "decision": decision,
        "decision_reason": reason,
        "equivalence": dict(equivalence),
        "experiment": EXPERIMENT,
        "fold_seeds": seeds,
        "gate": gate_block,
        "pin_dev_replay": dict(pin),
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": {
            seed: {
                key: value
                for key, value in seed_results[seed].items()
            }
            for seed in seed_results
        },
        "thresholds": thresholds,
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    write_json_atomic(audit_output, audit_document)
    write_json_atomic(output, report)
    return report, audit_document


def run_from_protocol(
    protocol_path: Path,
    expected_protocol_sha256: str,
    *,
    output: Path,
    audit_output: Path,
) -> Mapping[str, Any]:
    import json as _json

    payload = _json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("protocol is not a JSON object")
    digest = verify_protocol(payload, expected_protocol_sha256)
    report, _audit = assemble(
        payload,
        digest,
        output=output,
        audit_output=audit_output,
    )
    return report
