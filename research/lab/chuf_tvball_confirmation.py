# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""CHUF TV-ball confirmation protocol. Phase A seals the contract only.

Fresh seeds are derived before any confirmation score is seen. Validation
and epsilon use public inputs only. The run path loads outcomes later and
is not invoked by protocol tests.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from ossp_router.cost_calibrated_router import prompt_family
from ossp_router.protocol import TIERS, load_input
from research.lab.e1_objectives import (
    ALLOCATOR,
    GATE_VIEW_DROP,
    VIEW_MIN_N,
    allocate_all_tiers,
    canonical_json_text,
    score_decisions,
    sha256_text,
    stress_views,
    write_json_atomic,
)
from research.lab.e1b_quality_models import CHAMPION_ABS
from research.lab.e1c_regime_residual import relabel_folds
from research.lab.e1f_cost_conditioned_frontier import (
    AX31_POLICY,
    BASELINE_NAME,
    BETA_PRIOR_A,
    BETA_PRIOR_B,
    CANDIDATE_NAME,
    COST_EPS,
    COST_FEATURE_DEFINITION,
    EXPECTED_BASELINE_20260821,
    FAMILY_DEFINITION,
    FOLD_SEEDS as E1F_OLD_SEEDS,
    GATE_MEAN_DELTA,
    GATE_WORST_DELTA,
    K1_POLICY,
    MIN_CELL_GROUPS,
    N_COST_BINS,
    POSTERIOR_FORMULA,
    ax31_selections_match,
    binomial_counts,
    oof_chuf_heads,
    premium_parent_models,
)
from research.lab.modeling import sort_mapping
from research.lab.public_pool import (
    DEV_INPUTS,
    EXPECTED_DEV_INPUTS_SHA256,
    EXPECTED_DEV_OUTCOMES_SHA256,
    EXPECTED_N_DEV,
    EXPECTED_N_PUBLIC,
    EXPECTED_N_TRAIN,
    EXPECTED_TRAIN_INPUTS_SHA256,
    EXPECTED_TRAIN_OUTCOMES_SHA256,
    ROOT,
    TRAIN_INPUTS,
    sha256_path,
)
from research.lab.quality_heads import content_tie_keys


PROTOCOL_ID = "chuf-tvball-confirmation-v1"
PROTOCOL_RELATIVE = "research/protocols/chuf-tvball-confirmation.v1.json"
PROTOCOL_PATH = ROOT / PROTOCOL_RELATIVE
REPORT_TYPE = "scrooge-chuf-tvball-confirmation"
SCHEMA_VERSION = 1
SEED_PREFIX = "scrooge-chuf-tvball-confirmation-v1"
E1F_DECISION_CORE_SHA256 = (
    "f4cca0d425b47bda6e42be9b5c11b64e3cf9c57efd2810f11b22b1bd6051ba79"
)
E1F_AUDIT_SHA256 = (
    "63fa06c07db908448b2cfe4a6c24ccebbcc8ed2d82dd388d4b0444c6f2cbf0db"
)
E1F_SOURCE_RELATIVE = "research/lab/e1f_cost_conditioned_frontier.py"
E1F_SOURCE_PATH = ROOT / E1F_SOURCE_RELATIVE
OLD_SEEDS: Tuple[int, ...] = tuple(int(seed) for seed in E1F_OLD_SEEDS)
N_FRESH_SEEDS = 12
EXPECTED_EPSILON = 0.014204545454545449
EXPECTED_PROTOCOL_SHA256 = (
    "37fa19fe8ab20a90773ad9568074e73ec9e0721a29d22f5bf80c8f939494a04c"
)
TV_WORST_MIN = -0.003
EXPECTED_POLICY_SHA256 = (
    "7c892c423da5fa762e7e1a93b9fa071be51e259b65d2b63a5ba434c4342d7a8e"
)
REQUIRED_FAMILIES: Tuple[str, ...] = (
    "english_multiple_choice",
    "korean_multiple_choice",
    "korean_reasoning",
    "latex_math",
    "long_context",
    "other",
    "python_program",
    "rule_reasoning",
    "symbolic_math",
    "word_problem",
)
PASS_DECISION = "record-chuf-tvball-confirmation-pass-await-risk-phase"
FAIL_DECISION = "record-chuf-tvball-confirmation-fail-current-runtime"
OUT_RELATIVE = "build/confirm-chuf-tvball"
AUDIT_RELATIVE = "build/confirm-chuf-tvball/episode-audit.json"
E1F_REPORT_RELATIVE = "build/compare-e1f-cost-conditioned-frontier/report.json"
SEED_DERIVATION = (
    "digest_i = SHA256(UTF8(PREFIX) + NUL + bytes.fromhex(E1F_CORE) + "
    "i.to_bytes(4,'big')); seed_i = int.from_bytes(digest_i[:4],'big') "
    "& 0x7fffffff; i=0..11. Collision or old-seed overlap fails closed "
    "and must not skip to the next digest."
)
TV_FORMULA = (
    "tv_worst = official_delta + epsilon * "
    "(min_family_delta - max_family_delta)"
)
EPSILON_FORMULA = "0.5 * sum_f abs(p_train_f - p_dev_f)"


def derive_fresh_seeds(
    *,
    n: int = N_FRESH_SEEDS,
    core_sha: str = E1F_DECISION_CORE_SHA256,
    old_seeds: Sequence[int] = OLD_SEEDS,
    prefix: str = SEED_PREFIX,
) -> Tuple[int, ...]:
    """Result-independent seeds. Do not skip a colliding digest."""

    seeds: list[int] = []
    seen: set[int] = set()
    blocked = {int(seed) for seed in old_seeds}
    for index in range(int(n)):
        digest = hashlib.sha256(
            prefix.encode("utf-8")
            + b"\0"
            + bytes.fromhex(core_sha)
            + int(index).to_bytes(4, "big")
        ).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed in seen:
            raise RuntimeError(
                f"fresh seed collision at i={index}: {seed}; fail closed"
            )
        if seed in blocked:
            raise RuntimeError(
                f"fresh seed overlaps diagnostic old seed at i={index}: "
                f"{seed}; fail closed"
            )
        seen.add(seed)
        seeds.append(seed)
    if len(seeds) != int(n):
        raise RuntimeError("fresh seed count drifted")
    return tuple(seeds)


def e1f_source_sha256(path: Path = E1F_SOURCE_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture_snapshot() -> dict[str, Any]:
    """Live CHUF constants. Changing e1f rules must fail verification."""

    return sort_mapping(
        {
            "ax31_policy": AX31_POLICY,
            "baseline_name": BASELINE_NAME,
            "beta_prior_a": BETA_PRIOR_A,
            "beta_prior_b": BETA_PRIOR_B,
            "candidate_name": CANDIDATE_NAME,
            "champion_absolute": CHAMPION_ABS,
            "cost_eps": COST_EPS,
            "cost_feature_definition": COST_FEATURE_DEFINITION,
            "digitize_right": True,
            "expected_baseline_20260821": EXPECTED_BASELINE_20260821,
            "family_definition": FAMILY_DEFINITION,
            "gate_mean_delta": GATE_MEAN_DELTA,
            "gate_view_drop": GATE_VIEW_DROP,
            "gate_worst_delta": GATE_WORST_DELTA,
            "k1_policy": K1_POLICY,
            "min_cell_groups": MIN_CELL_GROUPS,
            "n_cost_bins": N_COST_BINS,
            "posterior_formula": dict(POSTERIOR_FORMULA),
            "quantile_points": [0.25, 0.50, 0.75],
            "view_min_n": VIEW_MIN_N,
        }
    )


def family_counts_from_inputs(
    train_path: Path = TRAIN_INPUTS,
    dev_path: Path = DEV_INPUTS,
) -> dict[str, Any]:
    """Input-only family counts. Outcomes are never opened."""

    train = load_input(train_path)
    dev = load_input(dev_path)
    train_counts: dict[str, int] = {name: 0 for name in REQUIRED_FAMILIES}
    dev_counts: dict[str, int] = {name: 0 for name in REQUIRED_FAMILIES}
    for episode in train.episodes:
        name = prompt_family(episode)
        if name not in train_counts:
            raise RuntimeError(f"unknown train family {name}")
        train_counts[name] += 1
    for episode in dev.episodes:
        name = prompt_family(episode)
        if name not in dev_counts:
            raise RuntimeError(f"unknown dev family {name}")
        dev_counts[name] += 1
    if sum(train_counts.values()) != EXPECTED_N_TRAIN:
        raise RuntimeError("train family counts drifted")
    if sum(dev_counts.values()) != EXPECTED_N_DEV:
        raise RuntimeError("dev family counts drifted")
    pool_counts = {
        name: int(train_counts[name] + dev_counts[name])
        for name in REQUIRED_FAMILIES
    }
    if any(pool_counts[name] < VIEW_MIN_N for name in REQUIRED_FAMILIES):
        raise RuntimeError("a required family has n<20 on the public pool")
    n_pool = EXPECTED_N_PUBLIC
    center = {
        name: float(pool_counts[name]) / float(n_pool) for name in REQUIRED_FAMILIES
    }
    return {
        "center": center,
        "dev": dict(dev_counts),
        "n_dev": EXPECTED_N_DEV,
        "n_pool": n_pool,
        "n_train": EXPECTED_N_TRAIN,
        "pool": pool_counts,
        "train": dict(train_counts),
    }


def epsilon_from_counts(counts: Mapping[str, Any]) -> float:
    n_train = float(counts["n_train"])
    n_dev = float(counts["n_dev"])
    total = 0.0
    for name in REQUIRED_FAMILIES:
        p_train = float(counts["train"][name]) / n_train
        p_dev = float(counts["dev"][name]) / n_dev
        total += abs(p_train - p_dev)
    return 0.5 * total


def epsilon_from_input_paths(
    train_path: Path = TRAIN_INPUTS,
    dev_path: Path = DEV_INPUTS,
) -> float:
    return epsilon_from_counts(family_counts_from_inputs(train_path, dev_path))


def assert_epsilon_pin(value: float) -> float:
    number = float(value)
    if number != EXPECTED_EPSILON:
        raise RuntimeError(
            f"TV epsilon drifted: {number!r} != {EXPECTED_EPSILON!r}"
        )
    return number


def tv_worst(
    official_delta: float,
    epsilon: float,
    family_deltas: Mapping[str, float],
) -> float:
    """Closed-form TV-ball worst. Family map must be n>=20 only."""

    if not family_deltas:
        raise RuntimeError("TV-ball family deltas are empty")
    values = [float(delta) for delta in family_deltas.values()]
    if any(value != value for value in values):
        raise RuntimeError("TV-ball family delta is NaN")
    return float(official_delta) + float(epsilon) * (min(values) - max(values))


def confirmation_gate(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fresh-seed gates only. Dirac per-family fails are diagnostic."""

    if len(seed_reports) != N_FRESH_SEEDS:
        raise RuntimeError(
            f"confirmation gate expects {N_FRESH_SEEDS} fresh seeds, "
            f"got {len(seed_reports)}"
        )
    seeds = [int(row["fold_seed"]) for row in seed_reports]
    if tuple(seeds) != derive_fresh_seeds():
        raise RuntimeError("confirmation seed list is not the sealed fresh list")
    if any(seed in OLD_SEEDS for seed in seeds):
        raise RuntimeError("diagnostic old seeds entered the confirmation gate")

    deltas = [float(row["delta"]) for row in seed_reports]
    qualities = [float(row["candidate_quality"]) for row in seed_reports]
    cap_fail: list[int] = []
    identity_fail: list[int] = []
    k1_fail: list[int] = []
    tv_fail: list[int] = []
    family_fail: list[int] = []
    dirac_diag: list[dict[str, Any]] = []
    for row in seed_reports:
        seed = int(row["fold_seed"])
        if not (
            bool(row["baseline_caps_ok"])
            and bool(row["candidate_caps_ok"])
            and bool(row["baseline_fold_caps_ok"])
            and bool(row["candidate_fold_caps_ok"])
        ):
            cap_fail.append(seed)
        if not bool(row["ax31_identical"]):
            identity_fail.append(seed)
        if not bool(row["k1_fast_balanced_zero"]):
            k1_fail.append(seed)
        tv_value = row.get("tv_worst")
        if tv_value is None or tv_value != tv_value:
            family_fail.append(seed)
        elif float(tv_value) < TV_WORST_MIN:
            tv_fail.append(seed)
        families = row.get("family_deltas") or {}
        missing = [
            name for name in REQUIRED_FAMILIES if name not in families
        ]
        if missing or any(
            families[name] != families[name] for name in families
        ):
            family_fail.append(seed)
        dirac = [
            name
            for name, delta in families.items()
            if float(delta) < -GATE_VIEW_DROP
        ]
        if dirac:
            dirac_diag.append({"failures": dirac, "seed": seed})

    mean_delta = float(sum(deltas) / len(deltas))
    worst_delta = float(min(deltas))
    mean_quality = float(sum(qualities) / len(qualities))
    passed = bool(
        not cap_fail
        and not identity_fail
        and not k1_fail
        and not tv_fail
        and not family_fail
        and mean_delta >= GATE_MEAN_DELTA
        and worst_delta >= GATE_WORST_DELTA
        and mean_quality >= CHAMPION_ABS
    )
    return {
        "cap_failures": cap_fail,
        "dirac_failures_diagnostic": dirac_diag,
        "family_failures": family_fail,
        "identity_failures": identity_fail,
        "k1_failures": k1_fail,
        "mean_absolute": mean_quality,
        "mean_delta": mean_delta,
        "passed": passed,
        "phase2_executed": False,
        "runtime_export": False,
        "thresholds": {
            "mean_absolute": CHAMPION_ABS,
            "mean_delta": GATE_MEAN_DELTA,
            "tv_worst_min": TV_WORST_MIN,
            "view_min_n": VIEW_MIN_N,
            "worst_delta": GATE_WORST_DELTA,
        },
        "tv_failures": tv_fail,
        "worst_absolute": float(min(qualities)),
        "worst_delta": worst_delta,
    }


def build_canonical_protocol() -> dict[str, Any]:
    """Seal the confirmation contract from live inputs and e1f constants."""

    counts = family_counts_from_inputs()
    epsilon = assert_epsilon_pin(epsilon_from_counts(counts))
    fresh = derive_fresh_seeds()
    pins = {
        "dev_inputs_sha256": EXPECTED_DEV_INPUTS_SHA256,
        "dev_outcomes_sha256": EXPECTED_DEV_OUTCOMES_SHA256,
        "n_dev": EXPECTED_N_DEV,
        "n_public": EXPECTED_N_PUBLIC,
        "n_train": EXPECTED_N_TRAIN,
        "policy_sha256": EXPECTED_POLICY_SHA256,
        "train_inputs_sha256": EXPECTED_TRAIN_INPUTS_SHA256,
        "train_outcomes_sha256": EXPECTED_TRAIN_OUTCOMES_SHA256,
    }
    if TRAIN_INPUTS.is_file():
        digest = sha256_path(TRAIN_INPUTS)
        if digest != EXPECTED_TRAIN_INPUTS_SHA256:
            raise RuntimeError("train inputs hash drifted while sealing protocol")
    if DEV_INPUTS.is_file():
        digest = sha256_path(DEV_INPUTS)
        if digest != EXPECTED_DEV_INPUTS_SHA256:
            raise RuntimeError("dev inputs hash drifted while sealing protocol")
    return sort_mapping(
        {
            "architecture": architecture_snapshot(),
            "candidate": CANDIDATE_NAME,
            "decisions": {
                "fail": FAIL_DECISION,
                "pass": PASS_DECISION,
            },
            "diagnostic_old_seeds": list(OLD_SEEDS),
            "e1f": {
                "audit_sha256": E1F_AUDIT_SHA256,
                "decision": "record-e1f-no-promote",
                "decision_core_sha256": E1F_DECISION_CORE_SHA256,
                "source_relative": E1F_SOURCE_RELATIVE,
            },
            "e1f_source_sha256": e1f_source_sha256(),
            "epsilon": epsilon,
            "epsilon_formula": EPSILON_FORMULA,
            "experiment": PROTOCOL_ID,
            "family_counts": counts,
            "family_definition": FAMILY_DEFINITION,
            "fresh_seeds": list(fresh),
            "n_fresh_seeds": N_FRESH_SEEDS,
            "output": {
                "audit_relative": AUDIT_RELATIVE,
                "e1f_report_forbidden": E1F_REPORT_RELATIVE,
                "report_relative": f"{OUT_RELATIVE}/report.json",
            },
            "pins": pins,
            "protocol_id": PROTOCOL_ID,
            "required_families": list(REQUIRED_FAMILIES),
            "retroactive_promote": False,
            "runtime_export": False,
            "schema_version": SCHEMA_VERSION,
            "seed_derivation": {
                "algorithm": SEED_DERIVATION,
                "core_sha256": E1F_DECISION_CORE_SHA256,
                "fail_closed_on_collision": True,
                "n": N_FRESH_SEEDS,
                "prefix": SEED_PREFIX,
                "skip_digest_on_collision": False,
            },
            "thresholds": {
                "mean_absolute": CHAMPION_ABS,
                "mean_delta": GATE_MEAN_DELTA,
                "tv_worst_min": TV_WORST_MIN,
                "view_min_n": VIEW_MIN_N,
                "worst_delta": GATE_WORST_DELTA,
            },
            "tv_ball": {
                "center": "pooled_family_proportion",
                "epsilon_source": "materialized_inputs_family_counts_only",
                "formula": TV_FORMULA,
                "n_ge": VIEW_MIN_N,
            },
        }
    )


def canonical_protocol_text(protocol: Mapping[str, Any]) -> str:
    return canonical_json_text(sort_mapping(dict(protocol)))


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_protocol_text(protocol))


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("protocol is not a JSON object")
    if "generated_at" in payload:
        raise RuntimeError("protocol must not contain generated_at")
    return payload


def assert_live_architecture(protocol: Mapping[str, Any]) -> None:
    live = architecture_snapshot()
    sealed = protocol.get("architecture")
    if sealed != live:
        raise RuntimeError("CHUF architecture snapshot drifted from e1f imports")
    if protocol.get("e1f_source_sha256") != e1f_source_sha256():
        raise RuntimeError("CHUF source hash drifted from the sealed snapshot")
    if protocol.get("candidate") != CANDIDATE_NAME:
        raise RuntimeError("confirmation candidate is not chuf-v1")
    if protocol.get("e1f", {}).get("decision_core_sha256") != E1F_DECISION_CORE_SHA256:
        raise RuntimeError("sealed E1F decision core drifted")
    if protocol.get("e1f", {}).get("audit_sha256") != E1F_AUDIT_SHA256:
        raise RuntimeError("sealed E1F audit sha drifted")


def verify_protocol(
    protocol: Mapping[str, Any],
    expected_sha256: str,
    *,
    train_path: Path = TRAIN_INPUTS,
    dev_path: Path = DEV_INPUTS,
) -> str:
    """Input-only protocol check. Does not open outcomes or fit CHUF."""

    digest = protocol_sha256(protocol)
    if digest != expected_sha256:
        raise RuntimeError(
            f"protocol sha mismatch: got {digest}, expected {expected_sha256}"
        )
    assert_live_architecture(protocol)
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    if fresh != derive_fresh_seeds():
        raise RuntimeError("sealed fresh seeds drifted from the derivation")
    if len(set(fresh)) != N_FRESH_SEEDS:
        raise RuntimeError("sealed fresh seeds are not unique")
    if set(fresh) & set(OLD_SEEDS):
        raise RuntimeError("sealed fresh seeds overlap diagnostic old seeds")
    if tuple(protocol["diagnostic_old_seeds"]) != OLD_SEEDS:
        raise RuntimeError("diagnostic old seeds drifted")
    if tuple(protocol["required_families"]) != REQUIRED_FAMILIES:
        raise RuntimeError("required family list drifted")
    counts = family_counts_from_inputs(train_path, dev_path)
    epsilon = assert_epsilon_pin(epsilon_from_counts(counts))
    if protocol["epsilon"] != epsilon:
        raise RuntimeError("sealed epsilon drifted from input-only TV")
    if protocol["family_counts"]["train"] != counts["train"]:
        raise RuntimeError("sealed train family counts drifted")
    if protocol["family_counts"]["dev"] != counts["dev"]:
        raise RuntimeError("sealed dev family counts drifted")
    return digest


def write_canonical_protocol(path: Path = PROTOCOL_PATH) -> Tuple[dict[str, Any], str]:
    protocol = build_canonical_protocol()
    write_json_atomic(path, protocol)
    return protocol, protocol_sha256(protocol)


def _caps_ok(scored: Mapping[str, Any]) -> bool:
    return all(bool(scored["tiers"][tier]["within_hard_cap"]) for tier in TIERS)


def _k1_fast_balanced_zero(scored: Mapping[str, Any]) -> bool:
    fast = int(scored["tiers"]["fast"]["model_counts"]["axk1-think"])
    balanced = int(scored["tiers"]["balanced"]["model_counts"]["axk1-think"])
    return fast == 0 and balanced == 0


def _evaluate_head(pool: Any, head: Any, tie_keys: Sequence[str]) -> dict[str, Any]:
    import numpy as np

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
            float(pool.costs[mask, 0].sum()),
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
        "fold_caps_ok": all(_caps_ok(row) for row in per_fold),
        "k1_fast_balanced_zero": _k1_fast_balanced_zero(pooled),
        "per_fold": per_fold,
        "pooled": pooled,
        "pooled_models": pooled_models,
    }


def _family_deltas(views: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in views:
        if row.get("kind") != "family":
            continue
        if int(row["n"]) < VIEW_MIN_N:
            continue
        delta = row.get("delta")
        if delta is None:
            raise RuntimeError(f"family {row['name']} delta is missing")
        number = float(delta)
        if number != number:
            raise RuntimeError(f"family {row['name']} delta is NaN")
        out[str(row["name"])] = number
    missing = [name for name in REQUIRED_FAMILIES if name not in out]
    if missing:
        raise RuntimeError(f"required families missing from n>=20 views: {missing}")
    return {name: out[name] for name in REQUIRED_FAMILIES}


def evaluate_fresh_seed(pool: Any, seed: int, epsilon: float) -> dict[str, Any]:
    """Later-call scoring path. Protocol tests must not invoke this."""

    current = relabel_folds(pool, int(seed))
    trials, successes, _labels = binomial_counts(current)
    baseline_head, candidate_head, _rows = oof_chuf_heads(
        current, n=trials, k=successes
    )
    ties = content_tie_keys(current.texts)
    baseline = _evaluate_head(current, baseline_head, ties)
    candidate = _evaluate_head(current, candidate_head, ties)
    identity = ax31_selections_match(
        baseline_head.pred_qa,
        candidate_head.pred_qa,
        current.costs,
        current.light_total,
        ties,
    )
    identity["fast_models"] = tuple(baseline["pooled_models"]["fast"]) == tuple(
        candidate["pooled_models"]["fast"]
    )
    identity["balanced_models"] = tuple(
        baseline["pooled_models"]["balanced"]
    ) == tuple(candidate["pooled_models"]["balanced"])
    identity["premium_parent_models"] = premium_parent_models(
        baseline_head.pred_qa, current.costs, current.light_total, ties
    ) == premium_parent_models(
        candidate_head.pred_qa, current.costs, current.light_total, ties
    )
    identity["all"] = all(identity.values())
    views = stress_views(
        current, baseline["pooled_models"], candidate["pooled_models"]
    )
    families = _family_deltas(views)
    baseline_quality = float(baseline["pooled"]["quality_weighted"])
    candidate_quality = float(candidate["pooled"]["quality_weighted"])
    delta = candidate_quality - baseline_quality
    worst = tv_worst(delta, epsilon, families)
    dirac = [
        name for name, value in families.items() if value < -GATE_VIEW_DROP
    ]
    return {
        "ax31_identical": bool(identity["all"]),
        "ax31_identical_to_baseline": identity,
        "baseline_caps_ok": _caps_ok(baseline["pooled"]),
        "baseline_fold_caps_ok": bool(baseline["fold_caps_ok"]),
        "baseline_quality": baseline_quality,
        "candidate_caps_ok": _caps_ok(candidate["pooled"]),
        "candidate_fold_caps_ok": bool(candidate["fold_caps_ok"]),
        "candidate_quality": candidate_quality,
        "delta": delta,
        "dirac_failures_diagnostic": dirac,
        "family_deltas": families,
        "fold_seed": int(seed),
        "k1_fast_balanced_zero": bool(
            baseline["k1_fast_balanced_zero"] and candidate["k1_fast_balanced_zero"]
        ),
        "tv_worst": worst,
        "views": views,
    }


def episode_audit_document(
    seed_pools: Mapping[int, Any],
    heads: Mapping[int, Tuple[Any, Any]],
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
                    "pred_qa": float(cand_head.pred_qa[index]),
                    "pred_qk": float(cand_head.pred_qk[index]),
                    "seed": int(seed),
                    "selected": {
                        BASELINE_NAME: {
                            tier: str(base_models[tier][index]) for tier in TIERS
                        },
                        CANDIDATE_NAME: {
                            tier: str(cand_models[tier][index]) for tier in TIERS
                        },
                    },
                }
            )
        seed_blocks[str(seed)] = {"n_rows": len(rows), "rows": rows}
    return {
        "experiment": PROTOCOL_ID,
        "prompt_text_included": False,
        "seeds": seed_blocks,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "allocator": report["allocator"],
            "audit": report["audit"],
            "candidate": report["candidate"],
            "confirmation_gate": report["confirmation_gate"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "e1f": report["e1f"],
            "epsilon": report["epsilon"],
            "experiment": report["experiment"],
            "fold_seeds": report["fold_seeds"],
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


def refuse_e1f_output_path(path: Path) -> None:
    text = path.resolve().as_posix()
    if "compare-e1f-cost-conditioned-frontier" in text:
        raise RuntimeError("confirmation must not write the E1F report path")


def run_confirmation(
    protocol: Mapping[str, Any],
    *,
    output: Path,
    audit_output: Path,
) -> dict[str, Any]:
    """Public confirmation. Later call only. Tests must not invoke this."""

    from research.lab.public_pool import load_public_pool

    refuse_e1f_output_path(output)
    refuse_e1f_output_path(audit_output)
    if output.exists() or audit_output.exists():
        raise RuntimeError("confirmation output exists; refuse overwrite")
    digest = protocol_sha256(protocol)
    verify_protocol(protocol, digest)
    epsilon = float(protocol["epsilon"])
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    pool = load_public_pool()
    seed_rows = []
    seed_payload = {}
    seed_pools = {}
    heads = {}
    for seed in fresh:
        current = relabel_folds(pool, int(seed))
        seed_pools[int(seed)] = current
        trials, successes, _labels = binomial_counts(current)
        base_head, cand_head, _fold_rows = oof_chuf_heads(
            current, n=trials, k=successes
        )
        heads[int(seed)] = (base_head, cand_head)
        row = evaluate_fresh_seed(current, int(seed), epsilon)
        seed_rows.append(row)
        seed_payload[str(seed)] = {
            "ax31_identical": row["ax31_identical"],
            "baseline_quality": row["baseline_quality"],
            "candidate_quality": row["candidate_quality"],
            "delta": row["delta"],
            "dirac_failures_diagnostic": row["dirac_failures_diagnostic"],
            "family_deltas": row["family_deltas"],
            "tv_worst": row["tv_worst"],
        }
    gate = confirmation_gate(seed_rows)
    if gate["passed"]:
        decision = PASS_DECISION
        reason = (
            "Fresh-seed TV-ball confirmation passed. This is not a runtime "
            "export and does not open predicted-cost Phase 2. Hand off to "
            "the risk phase only."
        )
    else:
        decision = FAIL_DECISION
        reason = (
            "Fresh-seed TV-ball confirmation failed. Keep the current "
            "runtime. Do not retune bins, posteriors, or seeds."
        )
    audit_document = episode_audit_document(seed_pools, heads)
    audit_sha = sha256_text(canonical_json_text(audit_document))
    report = {
        "allocator": dict(ALLOCATOR),
        "audit": {
            "n_rows": sum(
                block["n_rows"] for block in audit_document["seeds"].values()
            ),
            "relative_path": AUDIT_RELATIVE,
            "sha256": audit_sha,
        },
        "candidate": CANDIDATE_NAME,
        "confirmation_gate": gate,
        "decision": decision,
        "decision_reason": reason,
        "e1f": dict(protocol["e1f"]),
        "epsilon": epsilon,
        "experiment": PROTOCOL_ID,
        "fold_seeds": list(fresh),
        "protocol_sha256": digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_payload,
        "thresholds": dict(protocol["thresholds"]),
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    write_json_atomic(audit_output, audit_document)
    write_json_atomic(output, report)
    return report


def validation_function_names() -> Tuple[str, ...]:
    return (
        "assert_epsilon_pin",
        "assert_live_architecture",
        "architecture_snapshot",
        "build_canonical_protocol",
        "derive_fresh_seeds",
        "e1f_source_sha256",
        "epsilon_from_counts",
        "epsilon_from_input_paths",
        "family_counts_from_inputs",
        "load_protocol",
        "protocol_sha256",
        "tv_worst",
        "verify_protocol",
    )


def assert_validation_path_has_no_outcomes(source: str | None = None) -> None:
    """AST contract: protocol validation never reads outcomes or fits CHUF."""

    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden_calls = {
        "load_outcomes",
        "load_public_pool",
        "evaluate_fresh_seed",
        "oof_chuf_heads",
        "run_confirmation",
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in validation_function_names():
        node = functions[name]
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in forbidden_calls:
                raise RuntimeError(
                    f"{name} references forbidden {child.id} on the "
                    "protocol validation path"
                )
            if isinstance(child, ast.Attribute) and child.attr in forbidden_calls:
                raise RuntimeError(
                    f"{name} references forbidden {child.attr} on the "
                    "protocol validation path"
                )
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                lowered = child.value.lower()
                if "outcomes.json" in lowered or "/outcomes/" in lowered:
                    raise RuntimeError(
                        f"{name} embeds an outcomes path on the "
                        "validation path"
                    )


__all__ = (
    "AUDIT_RELATIVE",
    "E1F_AUDIT_SHA256",
    "E1F_DECISION_CORE_SHA256",
    "EXPECTED_EPSILON",
    "EXPECTED_PROTOCOL_SHA256",
    "FAIL_DECISION",
    "N_FRESH_SEEDS",
    "OLD_SEEDS",
    "OUT_RELATIVE",
    "PASS_DECISION",
    "PROTOCOL_PATH",
    "PROTOCOL_RELATIVE",
    "REQUIRED_FAMILIES",
    "SEED_PREFIX",
    "TV_WORST_MIN",
    "architecture_snapshot",
    "assert_validation_path_has_no_outcomes",
    "build_canonical_protocol",
    "canonical_protocol_text",
    "confirmation_gate",
    "derive_fresh_seeds",
    "e1f_source_sha256",
    "epsilon_from_input_paths",
    "load_protocol",
    "protocol_sha256",
    "run_confirmation",
    "tv_worst",
    "verify_protocol",
    "write_canonical_protocol",
)
