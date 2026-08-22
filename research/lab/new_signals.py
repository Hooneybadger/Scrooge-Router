# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""New-signals — AST/numeric/choice extractor blocks inside the serving allocator.

Issue #21 guidance (objective change) is exhausted by the E1 chain, and
challenger-heads showed external model classes lose with existing
features. The one untested direction is genuinely new signal sources:
pure-stdlib extractor blocks appended to the incumbent 14-d structural
features. Everything else stays shipped/frozen; paired grouped OOF on
fresh seeds isolates whether new signals add value through the actual
serving path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router import family_guard_router
from ossp_router.cost_calibrated_router import structural_features
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


EXPERIMENT = "new-signals-v1"
REPORT_TYPE = "scrooge-new-signals-v1"
SCHEMA_VERSION = 1
BASELINE_ARM = "incumbent-refit"
CANDIDATE_ARMS: Tuple[str, ...] = (
    "signals-ast",
    "signals-numeric",
    "signals-choice",
    "signals-all",
)
ARMS: Tuple[str, ...] = (BASELINE_ARM,) + CANDIDATE_ARMS
BLOCK_ORDER: Mapping[str, int] = {
    "signals-ast": 0,
    "signals-numeric": 1,
    "signals-choice": 2,
    "signals-all": 3,
}
AUDIT_RELATIVE = "build/run-new-signals/episode-audit.json"
REPORT_RELATIVE = "build/run-new-signals/report.json"

FOLDS = 5
RIDGE_ALPHA = 100.0
TVBALL_EPSILON = 0.014204545454545449
VIEW_MIN_N = 20
TIER_WEIGHTS = {"fast": 0.4, "balanced": 0.3}
TIERS_EVAL: Tuple[str, ...] = ("fast", "balanced")
MODEL_COLUMN = {"ax31-light": 0, "ax31": 1, "axk1-think": 2}

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_DECIMAL = re.compile(r"\d+\.\d+")
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
_CURRENCY = re.compile(r"[$€£¥]|\b(?:USD|KRW|EUR|JPY)\b|원|달러")
_DIGIT_RUN = re.compile(r"\d+")
_QUESTION = re.compile(r"[?？]")
_CHOICE_LINE = re.compile(r"(?:^|\n)[ \t]*\(?(?:[A-Ea-e]|[1-5])[).:.][ \t]")

AST_BLOCK_DIM = 16
NUMERIC_BLOCK_DIM = 6
CHOICE_BLOCK_DIM = 4


def _log1p(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


def _code_text(text: str) -> str:
    fences = _FENCE.findall(text)
    if fences:
        return "\n".join(fences)
    return text


def extract_ast_block(text: str) -> Tuple[float, ...]:
    code = _code_text(text)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return (0.0,) * AST_BLOCK_DIM
    n_functions = n_classes = n_loops = n_if = n_asserts = n_try = 0
    n_comps = n_lambdas = n_calls = n_returns = n_binops = 0
    n_compares = n_strings = n_imports = 0
    for node in ast.walk(tree):
        name = type(node).__name__
        if name in ("FunctionDef", "AsyncFunctionDef"):
            n_functions += 1
        elif name == "ClassDef":
            n_classes += 1
        elif name in ("For", "While", "AsyncFor"):
            n_loops += 1
        elif name == "If":
            n_if += 1
        elif name == "Assert":
            n_asserts += 1
        elif name in ("Try", "TryStar"):
            n_try += 1
        elif name in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
            n_comps += 1
        elif name == "Lambda":
            n_lambdas += 1
        elif name == "Call":
            n_calls += 1
        elif name == "Return":
            n_returns += 1
        elif name == "BinOp":
            n_binops += 1
        elif name == "Compare":
            n_compares += 1
        elif name == "Constant" and isinstance(getattr(node, "value", None), str):
            n_strings += 1
        elif name in ("Import", "ImportFrom"):
            n_imports += 1

    def depth_of(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        return 1 + max((depth_of(child) for child in children), default=0)

    max_depth = depth_of(tree)
    return (
        1.0,
        _log1p(n_functions),
        _log1p(n_classes),
        _log1p(n_loops),
        _log1p(n_if),
        _log1p(n_asserts),
        _log1p(n_try),
        _log1p(n_comps),
        _log1p(n_lambdas),
        _log1p(n_calls),
        _log1p(n_returns),
        _log1p(n_binops),
        _log1p(n_compares),
        _log1p(n_strings),
        _log1p(n_imports),
        float(min(max_depth, 100)),
    )


def extract_numeric_block(text: str) -> Tuple[float, ...]:
    numbers = [float(match.group(0)) for match in _NUMBER.finditer(text)]
    digit_runs = [len(match.group(0)) for match in _DIGIT_RUN.finditer(text)]
    max_magnitude = max((abs(value) for value in numbers), default=0.0)
    mean_run = math.fsum(digit_runs) / len(digit_runs) if digit_runs else 0.0
    return (
        _log1p(len(numbers)),
        math.log10(max_magnitude + 1.0),
        _log1p(len(_DECIMAL.findall(text))),
        _log1p(len(_PERCENT.findall(text))),
        float(bool(_CURRENCY.search(text))),
        min(mean_run, 32.0),
    )


def extract_choice_block(text: str) -> Tuple[float, ...]:
    markers = _CHOICE_LINE.findall(text)
    questions = len(_QUESTION.findall(text))
    return (
        _log1p(len(markers)),
        _log1p(questions),
        float(len(markers) >= 2),
        _log1p(min(len(markers), 64)),
    )


BLOCK_EXTRACTORS: Mapping[str, Tuple[Any, ...]] = {
    "signals-ast": (extract_ast_block,),
    "signals-numeric": (extract_numeric_block,),
    "signals-choice": (extract_choice_block,),
    "signals-all": (extract_ast_block, extract_numeric_block, extract_choice_block),
}


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
    limits = thresholds["actual_ratio_limits"]
    if not (1.0 < float(limits["fast"]) < float(limits["balanced"])):
        raise ProtocolError("actual ratio limits are not ordered fast < balanced")
    return digest


@dataclass(frozen=True)
class Context:
    """Frozen serving context shared by every arm."""

    pool: Any
    policy: Any
    guard: family_guard_router.GuardedArtifact
    texts: Tuple[str, ...]
    features: np.ndarray
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
            families=tuple(
                family_guard_router.prompt_family(episode) for episode in pool.episodes
            ),
            guarded_costs=tuple(guarded_costs),
            tier_caps={
                tier: float(guard.value["predicted_caps"][tier])
                for tier in TIERS_EVAL
            },
        )

    def block_matrix(self, arm: str) -> np.ndarray:
        rows = []
        for text in self.texts:
            row: list[float] = []
            for extractor in BLOCK_EXTRACTORS[arm]:
                row.extend(extractor(text))
            rows.append(tuple(row))
        return np.asarray(rows, dtype=np.float64)


def _standardize(
    train: np.ndarray, apply_to: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train - mean) / scale, (apply_to - mean) / scale


def fit_uplift(
    ctx: Context, outer: Sequence[int], held: Sequence[int], arm: str
) -> np.ndarray:
    outer_array = np.asarray(outer, dtype=np.int64)
    held_array = np.asarray(held, dtype=np.int64)
    base_outer = ctx.features[outer_array]
    base_held = ctx.features[held_array]
    if arm == BASELINE_ARM:
        x_outer, x_held = base_outer, base_held
    else:
        block = ctx.block_matrix(arm)
        x_outer = np.concatenate([base_outer, block[outer_array]], axis=1)
        x_held = np.concatenate([base_held, block[held_array]], axis=1)
    z_outer, z_held = _standardize(x_outer, x_held)
    design_outer = np.concatenate(
        [np.ones((z_outer.shape[0], 1)), z_outer], axis=1
    )
    target = ctx.pool.scores[outer_array, 1] - ctx.pool.scores[outer_array, 0]
    coefficients = _weighted_ridge(
        design_outer, target, np.ones(len(outer)), RIDGE_ALPHA
    )
    design_held = np.concatenate([np.ones((z_held.shape[0], 1)), z_held], axis=1)
    return design_held @ coefficients


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
        qualities = {
            arm: fit_uplift(ctx, outer, held, arm) for arm in ARMS
        }
        selections: dict[tuple[str, str], Tuple[str, ...]] = {}
        predicted_ratios: dict[tuple[str, str], float] = {}
        actual_ratios: dict[tuple[str, str], float] = {}
        light_total = math.fsum(float(pool.costs[index][0]) for index in held)
        for arm in ARMS:
            for tier in TIERS_EVAL:
                models, predicted_ratio = select_with_uplift(
                    ctx, held, qualities[arm], tier
                )
                selections[(arm, tier)] = models
                predicted_ratios[(arm, tier)] = predicted_ratio
                numerator = math.fsum(
                    float(pool.costs[index][MODEL_COLUMN[model]])
                    for index, model in zip(held, models)
                )
                actual_ratios[(arm, tier)] = numerator / light_total
        if fold == 0:
            repeat_qualities = {
                arm: fit_uplift(ctx, outer, held, arm) for arm in ARMS
            }
            repeat_selections: dict[tuple[str, str], Tuple[str, ...]] = {}
            for arm in ARMS:
                for tier in TIERS_EVAL:
                    repeat_models, _ratio = select_with_uplift(
                        ctx, held, repeat_qualities[arm], tier
                    )
                    repeat_selections[(arm, tier)] = repeat_models
            first_fold = {"selections": dict(selections)}
            repeat_fold = {"selections": repeat_selections}
        for arm in ARMS:
            for tier in TIERS_EVAL:
                models = selections[(arm, tier)]
                for index, model in zip(held, models):
                    chosen[(arm, tier)].append(
                        float(pool.scores[index, MODEL_COLUMN[model]])
                    )
                ratio_rows.append(
                    {
                        "arm": arm,
                        "fold": int(fold),
                        "tier": tier,
                        "predicted_ratio": float(predicted_ratios[(arm, tier)]),
                        "actual_ratio": float(actual_ratios[(arm, tier)]),
                    }
                )
        for position, index in enumerate(held):
            row: dict[str, Any] = {
                "episode_id": pool.episodes[index].episode_id,
                "family": ctx.families[index],
                "fold": int(fold),
            }
            for arm in ARMS:
                row[f"{arm}:fast"] = selections[(arm, "fast")][position]
                row[f"{arm}:balanced"] = selections[(arm, "balanced")][position]
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
    for arm in CANDIDATE_ARMS:
        delta_fast = math.fsum(chosen[(arm, "fast")]) / n_episodes - math.fsum(
            chosen[(BASELINE_ARM, "fast")]
        ) / n_episodes
        delta_balanced = math.fsum(chosen[(arm, "balanced")]) / n_episodes - math.fsum(
            chosen[(BASELINE_ARM, "balanced")]
        ) / n_episodes
        tier_deltas[arm] = {"fast": delta_fast, "balanced": delta_balanced}
        pooled_delta[arm] = (
            TIER_WEIGHTS["fast"] * delta_fast
            + TIER_WEIGHTS["balanced"] * delta_balanced
        )

    family_deltas: dict[str, Mapping[str, Any]] = {}
    for arm in CANDIDATE_ARMS:
        buckets: dict[str, list[tuple[float, float]]] = {}
        for position, family_value in enumerate(family_order):
            buckets.setdefault(family_value, []).append(
                (
                    chosen[(arm, "fast")][position]
                    - chosen[(BASELINE_ARM, "fast")][position],
                    chosen[(arm, "balanced")][position]
                    - chosen[(BASELINE_ARM, "balanced")][position],
                )
            )
        rows: dict[str, Any] = {}
        for name, values in sorted(buckets.items()):
            n_f = len(values)
            contribution = (
                TIER_WEIGHTS["fast"] * math.fsum(v[0] for v in values)
                + TIER_WEIGHTS["balanced"] * math.fsum(v[1] for v in values)
            ) / n_episodes
            rows[name] = {"delta_contribution": float(contribution), "n": n_f}
        family_deltas[arm] = rows

    tvball_worst = {}
    for arm in CANDIDATE_ARMS:
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


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("new-signals output exists; refuse overwrite")

    ctx = Context.build()
    thresholds = dict(protocol["thresholds"])
    limits = thresholds["actual_ratio_limits"]
    seeds = [int(seed) for seed in protocol["fresh_seeds"]]

    results: dict[str, Any] = {}
    audits: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        result = evaluate_seed(ctx, seed)
        audits[str(seed)] = result.pop("audit_rows")
        results[str(seed)] = result

    gate_block: dict[str, Any] = {"arms": {}}
    ranked_arms: list[tuple[float, int, str]] = []
    determinism_passed = all(
        bool(results[str(seed)]["determinism_passed"]) for seed in seeds
    )
    gate_block["determinism_passed"] = bool(determinism_passed)

    for arm in CANDIDATE_ARMS:
        deltas = [float(results[str(seed)]["pooled_delta"][arm]) for seed in seeds]
        tvballs = [float(results[str(seed)]["tvball_worst"][arm]) for seed in seeds]
        mean_delta = math.fsum(deltas) / len(deltas)
        worst_delta = min(deltas)
        safety_failures = [
            {
                "seed": seed,
                "fold": row["fold"],
                "tier": row["tier"],
                "actual_ratio": float(row["actual_ratio"]),
            }
            for seed in seeds
            for row in results[str(seed)]["ratio_rows"]
            if row["arm"] == arm
            and float(row["actual_ratio"]) >= float(limits[row["tier"]])
        ]
        passed = bool(
            mean_delta >= float(thresholds["mean_delta_min"])
            and worst_delta > float(thresholds["worst_seed_delta_min_exclusive"])
            and min(tvballs) >= float(thresholds["tvball_worst_min"])
            and not safety_failures
            and determinism_passed
        )
        arm_failures: list[str] = []
        if mean_delta < float(thresholds["mean_delta_min"]):
            arm_failures.append("mean_delta")
        if worst_delta <= float(thresholds["worst_seed_delta_min_exclusive"]):
            arm_failures.append("worst_seed_delta")
        if min(tvballs) < float(thresholds["tvball_worst_min"]):
            arm_failures.append("tvball")
        if safety_failures:
            arm_failures.append("budget_safety")
        if passed:
            ranked_arms.append((mean_delta, BLOCK_ORDER[arm], arm))
        gate_block["arms"][arm] = {
            "mean_delta": mean_delta,
            "worst_delta": worst_delta,
            "delta_by_seed": {str(s): d for s, d in zip(seeds, deltas)},
            "tvball_worst_min": min(tvballs),
            "actual_ratio_worst": {
                tier: max(
                    (
                        float(row["actual_ratio"])
                        for seed in seeds
                        for row in results[str(seed)]["ratio_rows"]
                        if row["arm"] == arm and row["tier"] == tier
                    ),
                    default=0.0,
                )
                for tier in TIERS_EVAL
            },
            "safety_failures": safety_failures,
            "failures": arm_failures,
            "passed": passed,
        }

    if ranked_arms and determinism_passed:
        ranked_arms.sort(key=lambda item: (-item[0], item[1]))
        best_arm = ranked_arms[0][2]
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
            seed: {key: value for key, value in results[seed].items()}
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
                "seed_results",
                "thresholds",
            )
        }
    )
    encoded = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
