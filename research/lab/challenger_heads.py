# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Challenger-heads — competitor head classes inside the serving allocator.

The three competitor quality-head classes never tested in this repo
(RouterX high-dim TF-IDF, Shin shared-ability, GBM boosted trees) are
transplanted as uplift vectors feeding the frozen fast/balanced ladder.
Everything else stays shipped: costs, caps, guard multipliers, allocator.
Paired deltas against the incumbent recipe refit per fold isolate the one
question the unified position scoreboard needs: does any competitor head
class generalize better through our serving path?
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router import family_guard_router
from ossp_router.cost_calibrated_router import hashed_features, structural_features
from ossp_router.feasibility_ladder import select_fast_balanced
from ossp_router.protocol import load_bundled_policy
from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.e5_brake_conditioned import ProtocolError, _weighted_ridge
from research.lab.grouped_crossfit import assign_balanced_group_folds, fold_leakage_count
from research.lab.modeling import sort_mapping
from research.lab.public_pool import load_public_pool


EXPERIMENT = "challenger-heads-v1"
REPORT_TYPE = "scrooge-challenger-heads-v1"
SCHEMA_VERSION = 1
BASELINE_ARM = "incumbent-ridge-refit"
ARMS: Tuple[str, ...] = (
    BASELINE_ARM,
    "routerx-tfidf-uplift",
    "shin-irt-ability",
    "gbm-histgb-uplift",
)
CHALLENGER_ARMS: Tuple[str, ...] = ARMS[1:]
AUDIT_RELATIVE = "build/run-challenger-heads/episode-audit.json"
REPORT_RELATIVE = "build/run-challenger-heads/report.json"

FOLDS = 5
RIDGE_ALPHA = 100.0
TVBALL_EPSILON = 0.014204545454545449
VIEW_MIN_N = 20
TIER_WEIGHTS = {"fast": 0.4, "balanced": 0.3}
TIERS_EVAL: Tuple[str, ...] = ("fast", "balanced")


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(dict(protocol)))


def derive_fresh_seeds(
    prefix: str, core_sha256: str, count: int, forbidden: Sequence[int]
) -> Tuple[int, ...]:
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
    return digest


@dataclass(frozen=True)
class Context:
    """Frozen serving context shared by every arm."""

    pool: Any
    policy: Any
    guard: family_guard_router.GuardedArtifact
    texts: Tuple[str, ...]
    features: np.ndarray
    hashed512: np.ndarray
    families: Tuple[str, ...]
    guarded_costs: Tuple[Tuple[float, float], ...]
    tier_caps: Mapping[str, float]

    @classmethod
    def build(cls) -> "Context":
        pool = load_public_pool()
        policy = load_bundled_policy()
        guard = family_guard_router.load_bundled_artifact()
        features = np.asarray(
            [structural_features(episode) for episode in pool.episodes],
            dtype=np.float64,
        )
        hashed = np.asarray(
            [hashed_features(episode, 512) for episode in pool.episodes],
            dtype=np.float64,
        )
        guarded_costs = []
        for episode in pool.episodes:
            _uplift, costs = family_guard_router.guarded_prediction(
                episode, policy, guard
            )
            guarded_costs.append(costs)
        return cls(
            pool=pool,
            policy=policy,
            guard=guard,
            texts=tuple(pool.texts),
            features=features,
            hashed512=hashed,
            families=tuple(
                family_guard_router.prompt_family(episode) for episode in pool.episodes
            ),
            guarded_costs=tuple(guarded_costs),
            tier_caps={
                tier: float(guard.value["predicted_caps"][tier])
                for tier in TIERS_EVAL
            },
        )


def _standardize(train: np.ndarray, apply_to: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train - mean) / scale, (apply_to - mean) / scale


def uplift_incumbent(ctx: Context, outer: Sequence[int], held: Sequence[int]) -> np.ndarray:
    outer_array = np.asarray(outer, dtype=np.int64)
    z_outer, z_held = _standardize(
        ctx.features[outer_array], ctx.features[np.asarray(held, dtype=np.int64)]
    )
    design_outer = np.concatenate(
        [np.ones((z_outer.shape[0], 1)), z_outer], axis=1
    )
    target = ctx.pool.scores[outer_array, 1] - ctx.pool.scores[outer_array, 0]
    coefficients = _weighted_ridge(
        design_outer, target, np.ones(len(outer)), RIDGE_ALPHA
    )
    design_held = np.concatenate(
        [np.ones((z_held.shape[0], 1)), z_held], axis=1
    )
    return design_held @ coefficients


def uplit_routerx(ctx: Context, outer: Sequence[int], held: Sequence[int]) -> np.ndarray:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    texts_outer = [ctx.texts[index] for index in outer]
    texts_held = [ctx.texts[index] for index in held]
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=60000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=120000)
    x_outer = hstack([word.fit_transform(texts_outer), char.fit_transform(texts_outer)]).tocsr()
    x_held = hstack([word.transform(texts_held), char.transform(texts_held)]).tocsr()
    outer_array = np.asarray(outer, dtype=np.int64)
    target = ctx.pool.scores[outer_array, 1] - ctx.pool.scores[outer_array, 0]
    model = Ridge(alpha=1.0)
    model.fit(x_outer, target)
    return model.predict(x_held)


def uplift_shin(ctx: Context, outer: Sequence[int], held: Sequence[int]) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression, Ridge

    outer_array = np.asarray(outer, dtype=np.int64)
    held_array = np.asarray(held, dtype=np.int64)
    z_outer, z_held = _standardize(
        ctx.features[outer_array], ctx.features[held_array]
    )
    theta_design_outer = np.concatenate([z_outer, ctx.hashed512[outer_array]], axis=1)
    theta_design_held = np.concatenate([z_held, ctx.hashed512[held_array]], axis=1)
    theta_target = ctx.pool.scores[outer_array, 0]
    theta_model = Ridge(alpha=RIDGE_ALPHA)
    theta_model.fit(theta_design_outer, theta_target)
    theta_outer = theta_model.predict(theta_design_outer).reshape(-1, 1)
    theta_held = theta_model.predict(theta_design_held).reshape(-1, 1)

    probabilities = []
    for column in (0, 1):
        scores_m = ctx.pool.scores[outer_array, column]
        median_m = float(np.median(scores_m))
        labels = (scores_m >= median_m).astype(np.int64)
        model = LogisticRegression(C=10.0, max_iter=1000)
        model.fit(theta_outer, labels)
        probabilities.append(model.predict_proba(theta_held)[:, 1])
    return probabilities[1] - probabilities[0]


def uplift_histgb(ctx: Context, outer: Sequence[int], held: Sequence[int]) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    outer_array = np.asarray(outer, dtype=np.int64)
    target = ctx.pool.scores[outer_array, 1] - ctx.pool.scores[outer_array, 0]
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=20260816,
    )
    model.fit(ctx.features[outer_array], target)
    return model.predict(ctx.features[np.asarray(held, dtype=np.int64)])


ARM_FITTERS = {
    BASELINE_ARM: uplift_incumbent,
    "routerx-tfidf-uplift": uplit_routerx,
    "shin-irt-ability": uplift_shin,
    "gbm-histgb-uplift": uplift_histgb,
}


def select_with_uplift(
    ctx: Context,
    indexes: Sequence[int],
    uplift: np.ndarray,
    tier: str,
) -> Tuple[Tuple[str, ...], float]:
    predictions = [
        (float(uplift[position]), ctx.guarded_costs[index])
        for position, index in enumerate(indexes)
    ]
    selected, ratio = select_fast_balanced(
        predictions,
        cap=ctx.tier_caps[tier],
        runaway_fraction=float(ctx.guard.value["runaway_fraction"]),
        max_upgrade_fraction=float(ctx.guard.value["max_upgrade_fraction"]),
    )
    return selected, ratio


def _batch_actual_ratio(
    ctx: Context,
    indexes: Sequence[int],
    models: Sequence[str],
) -> float:
    chosen_column = {"ax31-light": 0, "ax31": 1, "axk1-think": 2}
    numerator = math.fsum(
        float(ctx.pool.costs[index][chosen_column[model]])
        for index, model in zip(indexes, models)
    )
    denominator = math.fsum(float(ctx.pool.costs[index][0]) for index in indexes)
    return numerator / max(denominator, 1e-12)


def evaluate_seed(ctx: Context, seed: int) -> Mapping[str, Any]:
    pool = ctx.pool
    n_episodes = len(pool.episodes)
    folds = assign_balanced_group_folds(
        pool.group_keys, pool.families, folds=FOLDS, seed=seed
    )
    leaked = fold_leakage_count(pool.group_keys, folds)
    if leaked:
        raise ProtocolError(f"grouped fold leakage: {leaked}")

    chosen: dict[tuple[str, str], list[float]] = {
        (arm, tier): [] for arm in ARMS for tier in TIERS_EVAL
    }
    family_order: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []

    first_fold: Optional[dict[str, Any]] = None
    repeat_fold: Optional[dict[str, Any]] = None
    for fold in range(FOLDS):
        held = tuple(i for i, value in enumerate(folds) if value == fold)
        outer = [i for i, value in enumerate(folds) if value != fold]
        qualities = {arm: fitter(ctx, outer, held) for arm, fitter in ARM_FITTERS.items()}
        selections: dict[tuple[str, str], Tuple[str, ...]] = {}
        ratios: dict[tuple[str, str], float] = {}
        for arm in ARMS:
            for tier in TIERS_EVAL:
                models, ratio_value = select_with_uplift(
                    ctx, held, qualities[arm], tier
                )
                selections[(arm, tier)] = models
                ratios[(arm, tier)] = ratio_value
        if fold == 0:
            first_fold = {"selections": dict(selections), "parent": held}
            repeat_qualities = {
                arm: fitter(ctx, outer, held) for arm, fitter in ARM_FITTERS.items()
            }
            repeat_selections = {}
            for arm in ARMS:
                for tier in TIERS_EVAL:
                    repeat_selections[(arm, tier)], _ = select_with_uplift(
                        ctx, held, repeat_qualities[arm], tier
                    )
            repeat_fold = {"selections": repeat_selections, "parent": held}
        for arm in ARMS:
            for tier in TIERS_EVAL:
                for index, model in zip(held, selections[(arm, tier)]):
                    column = MODEL_COLUMN[model]
                    chosen[(arm, tier)].append(float(pool.scores[index, column]))
                ratio_rows.append(
                    {
                        "arm": arm,
                        "fold": int(fold),
                        "tier": tier,
                        "predicted_ratio": float(ratios[(arm, tier)]),
                    }
                )
        for position, index in enumerate(held):
            row: dict[str, Any] = {
                "episode_id": pool.episodes[index].episode_id,
                "family": ctx.families[index],
                "fold": int(fold),
            }
            for arm in CHALLENGER_ARMS:
                row[f"{arm}:fast"] = selections[(arm, "fast")][position]
                row[f"{arm}:balanced"] = selections[(arm, "balanced")][position]
                row[f"{BASELINE_ARM}:fast"] = selections[(BASELINE_ARM, "fast")][position]
                row[f"{BASELINE_ARM}:balanced"] = selections[(BASELINE_ARM, "balanced")][position]
            audit_rows.append(row)
            family_order.append(ctx.families[index])

    assert first_fold is not None and repeat_fold is not None
    deterministic = all(
        repeat_fold["selections"][key] == first_fold["selections"][key]
        for key in first_fold["selections"]
    )
    del first_fold, repeat_fold

    pooled_delta = {}
    tier_deltas = {}
    for arm in CHALLENGER_ARMS:
        delta_fast = math.fsum(chosen[(arm, "fast")]) / n_episodes - math.fsum(
            chosen[(BASELINE_ARM, "fast")]
        ) / n_episodes
        delta_balanced = math.fsum(chosen[(arm, "balanced")]) / n_episodes - math.fsum(
            chosen[(BASELINE_ARM, "balanced")]
        ) / n_episodes
        tier_deltas[arm] = {"fast": delta_fast, "balanced": delta_balanced}
        pooled_delta[arm] = (
            TIER_WEIGHTS["fast"] * delta_fast + TIER_WEIGHTS["balanced"] * delta_balanced
        )

    family_deltas: dict[str, Mapping[str, Any]] = {}
    for arm in CHALLENGER_ARMS:
        buckets: dict[str, list[tuple[float, float]]] = {}
        for position, family_value in enumerate(family_order):
            base_fast = chosen[(BASELINE_ARM, "fast")][position]
            base_balanced = chosen[(BASELINE_ARM, "balanced")][position]
            buckets.setdefault(family_value, []).append(
                (
                    chosen[(arm, "fast")][position] - base_fast,
                    chosen[(arm, "balanced")][position] - base_balanced,
                )
            )
        rows: dict[str, Any] = {}
        for name, values in sorted(buckets.items()):
            n_f = len(values)
            mean_fast = math.fsum(v[0] for v in values) / n_f
            mean_balanced = math.fsum(v[1] for v in values) / n_f
            contribution = (
                TIER_WEIGHTS["fast"] * mean_fast + TIER_WEIGHTS["balanced"] * mean_balanced
            ) * (n_f / n_episodes)
            rows[name] = {
                "delta_contribution": float(contribution),
                "n": n_f,
            }
        family_deltas[arm] = rows

    tvball_worst = {}
    for arm in CHALLENGER_ARMS:
        eligible = [
            float(row["delta_contribution"])
            for row in family_deltas[arm].values()
            if int(row["n"]) >= VIEW_MIN_N
        ]
        spread = min(eligible) - max(eligible)
        tvball_worst[arm] = float(pooled_delta[arm] + TVBALL_EPSILON * spread)

    return {
        "seed": int(seed),
        "pooled_delta": pooled_delta,
        "tier_deltas": tier_deltas,
        "tvball_worst": tvball_worst,
        "family_deltas": family_deltas,
        "ratio_rows": ratio_rows,
        "determinism_passed": bool(deterministic),
        "audit_rows": audit_rows,
    }


MODEL_COLUMN = {"ax31-light": 0, "ax31": 1, "axk1-think": 2}


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("challenger-heads output exists; refuse overwrite")

    ctx = Context.build()
    thresholds = dict(protocol["thresholds"])
    seeds = [int(seed) for seed in protocol["fresh_seeds"]]

    results: dict[str, Any] = {}
    audits: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        result = evaluate_seed(ctx, seed)
        audits[str(seed)] = result.pop("audit_rows")
        results[str(seed)] = result

    gate_block: dict[str, Any] = {"arms": {}}
    passing_arms: list[str] = []
    for arm in CHALLENGER_ARMS:
        deltas = [float(results[str(seed)]["pooled_delta"][arm]) for seed in seeds]
        tvballs = [float(results[str(seed)]["tvball_worst"][arm]) for seed in seeds]
        mean_delta = math.fsum(deltas) / len(deltas)
        worst_delta = min(deltas)
        arm_failures: list[str] = []

        safety_failures = []
        fast_limit = float(thresholds["budget_limits"]["fast_actual_ratio_max"])
        balanced_limit = float(thresholds["budget_limits"]["balanced_actual_ratio_max"])
        for seed in seeds:
            for row in results[str(seed)]["ratio_rows"]:
                limit = fast_limit if row["tier"] == "fast" else balanced_limit
                if row["arm"] == arm and float(row["predicted_ratio"]) >= limit:
                    safety_failures.append(
                        {"seed": seed, "fold": row["fold"], "tier": row["tier"],
                         "predicted_ratio": float(row["predicted_ratio"])}
                    )
        determinism_passed = all(
            bool(results[str(seed)]["determinism_passed"]) for seed in seeds
        )

        passed = bool(
            mean_delta >= float(thresholds["mean_delta_min"])
            and worst_delta > float(thresholds["worst_seed_delta_min_exclusive"])
            and min(tvballs) >= float(thresholds["tvball_worst_min"])
            and not safety_failures
            and determinism_passed
        )
        if not passed:
            if mean_delta < float(thresholds["mean_delta_min"]):
                arm_failures.append("mean_delta")
            if worst_delta <= float(thresholds["worst_seed_delta_min_exclusive"]):
                arm_failures.append("worst_seed_delta")
            if min(tvballs) < float(thresholds["tvball_worst_min"]):
                arm_failures.append("tvball")
            if safety_failures:
                arm_failures.append("budget_safety")
        else:
            passing_arms.append((mean_delta, arm))
        gate_block["arms"][arm] = {
            "mean_delta": mean_delta,
            "worst_delta": worst_delta,
            "delta_by_seed": {str(s): d for s, d in zip(seeds, deltas)},
            "tvball_worst_min": min(tvballs),
            "safety_failures": safety_failures,
            "failures": arm_failures,
            "passed": passed,
        }

    if passing_arms:
        passing_arms.sort(reverse=True)
        best_arm = passing_arms[0][1]
        decision = str(protocol["decisions"]["pass"])
        reason = f"Promotion window opens for {best_arm}."
        gate_block["window_arm"] = best_arm
    else:
        decision = str(protocol["decisions"]["fail"])
        reason = str(protocol["decision_reasons"]["fail"])

    audit_document = {
        "arms": list(ARMS),
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "rows": {seed: audits[seed] for seed in sorted(audits)},
    }

    report = {
        "audit": {
            "n_rows": sum(len(rows) for rows in audits.values()),
            "relative_path": AUDIT_RELATIVE,
            "sha256": sha256_text(canonical_json_text(audit_document)),
        },
        "decision": decision,
        "decision_reason": reason,
        "experiment": EXPERIMENT,
        "fold_seeds": seeds,
        "gate": gate_block,
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": {
            seed: {
                key: value
                for key, value in results[seed].items()
                if key != "audit_rows"
            }
            for seed in results
        },
        "thresholds": thresholds,
    }
    core = sort_mapping(
        {
            key: report[key]
            for key in (
                "audit",
                "decision",
                "decision_reason",
                "experiment",
                "fold_seeds",
                "gate",
                "protocol_sha256",
                "report_type",
                "schema_version",
                "thresholds",
            )
        }
    )
    encoded = json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    report["decision_core_sha256"] = sha256_text(encoded)

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
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("protocol is not a JSON object")
    digest = verify_protocol(payload, expected_protocol_sha256)
    report, _audit = assemble(
        payload, digest, output=output, audit_output=audit_output
    )
    return report
