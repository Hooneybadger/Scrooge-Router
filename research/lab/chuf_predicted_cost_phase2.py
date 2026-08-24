# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""CHUF predicted-cost Phase 2 risk contract. Phase A seals only.

Validation and protocol generation use public inputs and live artifact
constants. Outcomes, cost surfaces, CHUF heads, and the run path stay
outside protocol checks. Phase A must not write build outputs.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from ossp_router.cost_calibrated_router import prompt_family
from ossp_router.protocol import TIERS, load_input
from research.lab.chuf_tvball_confirmation import (
    E1F_DECISION_CORE_SHA256,
    EXPECTED_EPSILON,
    EXPECTED_POLICY_SHA256,
    EXPECTED_PROTOCOL_SHA256 as CONFIRMATION_PROTOCOL_SHA256,
    N_FRESH_SEEDS,
    OLD_SEEDS,
    PASS_DECISION as CONFIRMATION_DECISION,
    REQUIRED_FAMILIES,
    derive_fresh_seeds as derive_confirmation_seeds,
    tv_worst,
)
from research.lab.e1_objectives import (
    STRESS_RATIO_CAPS,
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.e1f_cost_conditioned_frontier import CANDIDATE_NAME, FAMILY_DEFINITION
from research.lab.e4_aggregate_risk import FAMILY_MULTIPLIERS, K1_SCOPE, RUNTIME_SCOPE
from research.lab.modeling import OFFICIAL_CAPS, sort_mapping
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


PROTOCOL_ID = "chuf-predicted-cost-phase2-v1"
PROTOCOL_RELATIVE = "research/protocols/chuf-predicted-cost-phase2.v1.json"
PROTOCOL_PATH = ROOT / PROTOCOL_RELATIVE
REPORT_TYPE = "scrooge-chuf-predicted-cost-phase2"
SCHEMA_VERSION = 1
SEED_PREFIX = "scrooge-chuf-predicted-cost-phase2-v1"
CONFIRMATION_CORE_SHA256 = (
    "2acba355a7c6863c4ae1971ba03e135041efcf0f9def9400135262734a569e6d"
)
EXPECTED_PROTOCOL_SHA256 = (
    "971114f51905c3b9ecec47790afac841647b3a34f9fde9d5c9bca96b65c3b5b1"
)
EXPLICIT_RISK_SEEDS: Tuple[int, ...] = (
    1961852001,
    1797397368,
    1763238305,
    999558656,
    1988874908,
    305408514,
    400818725,
    116341498,
    1592039285,
    215679302,
    1124696458,
    1980863820,
)
BOOTSTRAP_SEED = 557209147
BOOTSTRAP_DRAWS = 200
MIXTURE_SEED = 419291143
MIXTURE_DRAWS = 40
TV_WORST_MIN = -0.003
GATE_MEAN_PREMIUM_DELTA = 0.001
GATE_WORST_PREMIUM_DELTA = 0.0
GATE_MEAN_OFFICIAL_DELTA = 0.0003
GATE_WORST_OFFICIAL_DELTA = 0.0
PREMIUM_K1_MAX = 48
FAST_CAP = 1.11
BALANCED_CAP = 1.45
PREMIUM_CAP = 3.25
BRAKE_RATIO = 3.8
RUNAWAY_FRACTION = 0.16500000000000015
MAX_UPGRADE_FRACTION = 0.75
FAMILY_OTHER_MULTIPLIER = 2.5
NO_REF_DECISION = "record-chuf-phase2-no-valid-reference-current-runtime"
FAIL_DECISION = "record-chuf-phase2-fail-current-runtime"
PASS_DECISION = "record-chuf-phase2-pass-await-independent-audit"
OUT_RELATIVE = "build/phase2-chuf-predicted-cost"
AUDIT_RELATIVE = "build/phase2-chuf-predicted-cost/episode-audit.json"
CONFIRM_REPORT_RELATIVE = "build/confirm-chuf-tvball/report.json"
E1F_REPORT_RELATIVE = "build/compare-e1f-cost-conditioned-frontier/report.json"
FAMILY_GUARD_RELATIVE = "src/ossp_router/resources/family-guard-router.v1.json"
BUDGET_BRAKE_RELATIVE = "src/ossp_router/resources/budget-brake-router.v1.json"
TV_COST_FORMULA = (
    "91 vertices: center pooled family p, then for every ordered pair "
    "(i,j) i!=j move mass min(epsilon,p_i,1-p_j) from i to j. "
    "ratio = sum_f pi_f * family_mean_selected_spend / "
    "sum_f pi_f * family_mean_light. max <= official cap."
)
QUALITY_TV_FORMULA = (
    "tv_worst = official_delta + epsilon * "
    "(min_family_delta - max_family_delta)"
)
SEED_DERIVATION = (
    "digest_i = SHA256(UTF8(PREFIX) + NUL + bytes.fromhex(CONFIRMATION_CORE) "
    "+ i.to_bytes(4,'big')); seed_i = int.from_bytes(digest_i[:4],'big') "
    "& 0x7fffffff; i=0..11. Collision or overlap with old E1F seeds or "
    "confirmation-12 fails closed and must not skip the next digest."
)


def blocked_seeds() -> Tuple[int, ...]:
    return tuple(sorted(set(OLD_SEEDS) | set(derive_confirmation_seeds())))


def derive_fresh_risk_seeds(
    *,
    n: int = N_FRESH_SEEDS,
    core_sha: str = CONFIRMATION_CORE_SHA256,
    blocked: Sequence[int] | None = None,
    prefix: str = SEED_PREFIX,
) -> Tuple[int, ...]:
    """Result-independent Phase 2 seeds. Do not skip a colliding digest."""

    seeds: list[int] = []
    seen: set[int] = set()
    denied = {int(seed) for seed in (blocked_seeds() if blocked is None else blocked)}
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
                f"phase2 seed collision at i={index}: {seed}; fail closed"
            )
        if seed in denied:
            raise RuntimeError(
                f"phase2 seed overlaps blocked seed at i={index}: "
                f"{seed}; fail closed"
            )
        seen.add(seed)
        seeds.append(seed)
    if tuple(seeds) != EXPLICIT_RISK_SEEDS and core_sha == CONFIRMATION_CORE_SHA256:
        if blocked is None and prefix == SEED_PREFIX and int(n) == N_FRESH_SEEDS:
            raise RuntimeError("phase2 explicit seed list drifted from derivation")
    return tuple(seeds)


def source_sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def live_artifact_snapshot() -> dict[str, Any]:
    family = json.loads((ROOT / FAMILY_GUARD_RELATIVE).read_text(encoding="utf-8"))
    brake = json.loads((ROOT / BUDGET_BRAKE_RELATIVE).read_text(encoding="utf-8"))
    block = brake["budget_brake"]
    return sort_mapping(
        {
            "balanced_cap": float(family["predicted_caps"]["balanced"]),
            "brake_ratio": float(block["brake_ratio"]),
            "count_cap": int(block["count_cap"]),
            "denylist_families": list(block["denylist_families"]),
            "fast_cap": float(family["predicted_caps"]["fast"]),
            "max_upgrade_fraction": float(family["max_upgrade_fraction"]),
            "multipliers": {
                str(key): float(value)
                for key, value in family["family_guard"]["multipliers"].items()
            },
            "premium_cap": float(family["predicted_caps"]["premium"]),
            "runaway_absolute": float(block["runaway_absolute"]),
            "runaway_fraction": float(family["runaway_fraction"]),
            "runaway_light_fraction": float(block["runaway_light_fraction"]),
            "runaway_light_fraction_used_in_eligibility": False,
        }
    )


def architecture_snapshot() -> dict[str, Any]:
    live = live_artifact_snapshot()
    if live["fast_cap"] != FAST_CAP or live["balanced_cap"] != BALANCED_CAP:
        raise RuntimeError("family-guard predicted Fast/Balanced caps drifted")
    if live["premium_cap"] != PREMIUM_CAP:
        raise RuntimeError("Premium predicted cap drifted")
    if live["brake_ratio"] != BRAKE_RATIO:
        raise RuntimeError("Premium brake_ratio drifted")
    if live["runaway_fraction"] != RUNAWAY_FRACTION:
        raise RuntimeError("family-guard runaway_fraction drifted")
    if live["max_upgrade_fraction"] != MAX_UPGRADE_FRACTION:
        raise RuntimeError("max_upgrade_fraction drifted")
    if live["multipliers"] != {"other": FAMILY_OTHER_MULTIPLIER}:
        raise RuntimeError("family-guard other multiplier drifted")
    if live["count_cap"] != PREMIUM_K1_MAX:
        raise RuntimeError("budget-brake count_cap drifted")
    if dict(FAMILY_MULTIPLIERS) != live["multipliers"]:
        raise RuntimeError("e4 FAMILY_MULTIPLIERS drifted from the artifact")
    if int(K1_SCOPE["count_cap"]) != live["count_cap"]:
        raise RuntimeError("e4 K1 count_cap drifted from the artifact")
    return sort_mapping(
        {
            "allocator": {
                "fast_balanced": (
                    "E2 point, then e4.apply_family_increment_multiplier "
                    "other=2.5, clamp; feasibility.select_fast_balanced"
                ),
                "fold_local_rebuy": False,
                "pooled_public_batch_once": True,
                "premium_costs": (
                    "light=point[:,0], AX31/K1=conservative[:,1:3], clamp"
                ),
                "premium_k1": (
                    "budget_brake_router.promote_premium_brake; "
                    "comparator qk vs candidate qk"
                ),
                "premium_parent": (
                    "feasibility _select_premium cap 3.25; identical heads"
                ),
            },
            "artifact": live,
            "candidate": CANDIDATE_NAME,
            "e2_surfaces": "research.lab.e2_cost_uncertainty.oof_cost_surfaces",
            "family_definition": FAMILY_DEFINITION,
            "official_caps": dict(OFFICIAL_CAPS),
            "quality": {
                "candidate": "same pred_qa + CHUF pred_qk",
                "comparator": "oof_chuf_heads baseline qa/qk",
                "same_seed_shared_surfaces": True,
            },
            "runtime_predicted_caps": dict(RUNTIME_SCOPE["runtime_predicted_caps"]),
            "stress_95_caps": dict(STRESS_RATIO_CAPS),
        }
    )


def family_counts_from_inputs(
    train_path: Path = TRAIN_INPUTS,
    dev_path: Path = DEV_INPUTS,
) -> dict[str, Any]:
    train = load_input(train_path)
    dev = load_input(dev_path)
    train_counts = {name: 0 for name in REQUIRED_FAMILIES}
    dev_counts = {name: 0 for name in REQUIRED_FAMILIES}
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
    pool = {
        name: int(train_counts[name] + dev_counts[name]) for name in REQUIRED_FAMILIES
    }
    return {
        "center": {
            name: float(pool[name]) / float(EXPECTED_N_PUBLIC)
            for name in REQUIRED_FAMILIES
        },
        "dev": dict(dev_counts),
        "n_dev": EXPECTED_N_DEV,
        "n_pool": EXPECTED_N_PUBLIC,
        "n_train": EXPECTED_N_TRAIN,
        "pool": pool,
        "train": dict(train_counts),
    }


def epsilon_from_counts(counts: Mapping[str, Any]) -> float:
    n_train = float(counts["n_train"])
    n_dev = float(counts["n_dev"])
    total = 0.0
    for name in REQUIRED_FAMILIES:
        total += abs(
            float(counts["train"][name]) / n_train
            - float(counts["dev"][name]) / n_dev
        )
    return 0.5 * total


def epsilon_from_input_paths(
    train_path: Path = TRAIN_INPUTS,
    dev_path: Path = DEV_INPUTS,
) -> float:
    value = epsilon_from_counts(family_counts_from_inputs(train_path, dev_path))
    if value != EXPECTED_EPSILON:
        raise RuntimeError(f"TV epsilon drifted: {value!r} != {EXPECTED_EPSILON!r}")
    return value


def tv_cost_vertices(
    center: Mapping[str, float],
    spend_mean: Mapping[str, float],
    light_mean: Mapping[str, float],
    epsilon: float,
) -> list[float]:
    """Center plus every directed family transport. 10 families -> 91."""

    names = tuple(center)
    if set(names) != set(spend_mean) or set(names) != set(light_mean):
        raise RuntimeError("TV-cost family maps must share keys")

    def ratio(weights: Mapping[str, float]) -> float:
        numer = 0.0
        denom = 0.0
        for name in names:
            numer += float(weights[name]) * float(spend_mean[name])
            denom += float(weights[name]) * float(light_mean[name])
        if denom <= 0.0:
            raise RuntimeError("TV-cost light mean total is not positive")
        return numer / denom

    points = [ratio(center)]
    for source in names:
        for dest in names:
            if source == dest:
                continue
            mass = min(
                float(epsilon),
                float(center[source]),
                1.0 - float(center[dest]),
            )
            moved = dict(center)
            moved[source] = float(center[source]) - mass
            moved[dest] = float(center[dest]) + mass
            points.append(ratio(moved))
    return points


def tv_cost_worst(
    center: Mapping[str, float],
    spend_mean: Mapping[str, float],
    light_mean: Mapping[str, float],
    epsilon: float,
) -> float:
    points = tv_cost_vertices(center, spend_mean, light_mean, epsilon)
    expected = len(center) * (len(center) - 1) + 1
    if len(points) != expected:
        raise RuntimeError(f"TV-cost vertex count {len(points)} != {expected}")
    return float(max(points))


def _safety_ok(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("pooled_hard_caps_ok")):
        return False
    if not bool(row.get("pooled_ratio_under_95_ok")):
        return False
    if not bool(row.get("fold_slice_hard_caps_ok")):
        return False
    if not bool(row.get("bootstrap_q999_under_95_ok")):
        return False
    if not bool(row.get("tv_cost_under_official_ok")):
        return False
    return True


def _identity_ok(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("pred_qa_identical")
        and row.get("parent_identical")
        and row.get("fast_balanced_k1_zero")
        and int(row.get("premium_k1_count", PREMIUM_K1_MAX + 1)) <= PREMIUM_K1_MAX
    )


def phase2_gate(
    comparator_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fresh-seed Phase 2 gate. Dirac / family-slice / Dirichlet are diagnostic."""

    if len(comparator_rows) != N_FRESH_SEEDS or len(candidate_rows) != N_FRESH_SEEDS:
        raise RuntimeError("phase2 gate expects 12 comparator and 12 candidate rows")
    seeds = [int(row["fold_seed"]) for row in comparator_rows]
    if seeds != [int(row["fold_seed"]) for row in candidate_rows]:
        raise RuntimeError("comparator/candidate seed lists diverged")
    if tuple(seeds) != EXPLICIT_RISK_SEEDS:
        raise RuntimeError("phase2 seed list is not the sealed risk list")
    if any(seed in blocked_seeds() for seed in seeds):
        raise RuntimeError("blocked seeds entered the phase2 gate")

    comparator_fail = [
        int(row["fold_seed"])
        for row in comparator_rows
        if not _safety_ok(row)
    ]
    candidate_fail = [
        int(row["fold_seed"])
        for row in candidate_rows
        if not _safety_ok(row) or not _identity_ok(row)
    ]
    premium = [float(row["premium_delta"]) for row in candidate_rows]
    official = [float(row["official_delta"]) for row in candidate_rows]
    tv_quality = [float(row["tv_quality_worst"]) for row in candidate_rows]
    mean_premium = float(sum(premium) / len(premium))
    worst_premium = float(min(premium))
    mean_official = float(sum(official) / len(official))
    worst_official = float(min(official))
    tv_quality_fail = [
        int(row["fold_seed"])
        for row in candidate_rows
        if float(row["tv_quality_worst"]) < TV_WORST_MIN
    ]
    quality_ok = bool(
        mean_premium >= GATE_MEAN_PREMIUM_DELTA
        and worst_premium >= GATE_WORST_PREMIUM_DELTA
        and mean_official >= GATE_MEAN_OFFICIAL_DELTA
        and worst_official >= GATE_WORST_OFFICIAL_DELTA
        and not tv_quality_fail
    )
    if comparator_fail:
        decision = NO_REF_DECISION
        passed = False
    elif candidate_fail or not quality_ok:
        decision = FAIL_DECISION
        passed = False
    else:
        decision = PASS_DECISION
        passed = True
    return {
        "candidate_safety_failures": candidate_fail,
        "comparator_safety_failures": comparator_fail,
        "decision": decision,
        "mean_official_delta": mean_official,
        "mean_premium_delta": mean_premium,
        "passed": passed,
        "phase2_executed": False,
        "runtime_export": False,
        "thresholds": {
            "mean_official_delta": GATE_MEAN_OFFICIAL_DELTA,
            "mean_premium_delta": GATE_MEAN_PREMIUM_DELTA,
            "premium_k1_max": PREMIUM_K1_MAX,
            "tv_worst_min": TV_WORST_MIN,
            "worst_official_delta": GATE_WORST_OFFICIAL_DELTA,
            "worst_premium_delta": GATE_WORST_PREMIUM_DELTA,
        },
        "tv_quality": tv_quality,
        "tv_quality_failures": tv_quality_fail,
        "worst_official_delta": worst_official,
        "worst_premium_delta": worst_premium,
    }


def build_canonical_protocol() -> dict[str, Any]:
    counts = family_counts_from_inputs()
    epsilon = epsilon_from_input_paths()
    fresh = derive_fresh_risk_seeds()
    if TRAIN_INPUTS.is_file():
        if sha256_path(TRAIN_INPUTS) != EXPECTED_TRAIN_INPUTS_SHA256:
            raise RuntimeError("train inputs hash drifted while sealing phase2")
    if DEV_INPUTS.is_file():
        if sha256_path(DEV_INPUTS) != EXPECTED_DEV_INPUTS_SHA256:
            raise RuntimeError("dev inputs hash drifted while sealing phase2")
    return sort_mapping(
        {
            "architecture": architecture_snapshot(),
            "blocked_seeds": list(blocked_seeds()),
            "candidate": CANDIDATE_NAME,
            "confirmation": {
                "core_sha256": CONFIRMATION_CORE_SHA256,
                "decision": CONFIRMATION_DECISION,
                "protocol_sha256": CONFIRMATION_PROTOCOL_SHA256,
            },
            "decisions": {
                "fail": FAIL_DECISION,
                "no_valid_reference": NO_REF_DECISION,
                "pass": PASS_DECISION,
            },
            "e1f_decision_core_sha256": E1F_DECISION_CORE_SHA256,
            "epsilon": epsilon,
            "experiment": PROTOCOL_ID,
            "family_counts": counts,
            "family_definition": FAMILY_DEFINITION,
            "fold_local_rebuy": False,
            "fresh_seeds": list(fresh),
            "n_fresh_seeds": N_FRESH_SEEDS,
            "output": {
                "audit_relative": AUDIT_RELATIVE,
                "confirm_report_forbidden": CONFIRM_REPORT_RELATIVE,
                "e1f_report_forbidden": E1F_REPORT_RELATIVE,
                "report_relative": f"{OUT_RELATIVE}/report.json",
            },
            "pins": {
                "budget_brake_sha256": source_sha256(BUDGET_BRAKE_RELATIVE),
                "dev_inputs_sha256": EXPECTED_DEV_INPUTS_SHA256,
                "dev_outcomes_sha256": EXPECTED_DEV_OUTCOMES_SHA256,
                "e2_source_sha256": source_sha256(
                    "research/lab/e2_cost_uncertainty.py"
                ),
                "e4_source_sha256": source_sha256(
                    "research/lab/e4_aggregate_risk.py"
                ),
                "e1f_source_sha256": source_sha256(
                    "research/lab/e1f_cost_conditioned_frontier.py"
                ),
                "family_guard_sha256": source_sha256(FAMILY_GUARD_RELATIVE),
                "n_dev": EXPECTED_N_DEV,
                "n_public": EXPECTED_N_PUBLIC,
                "n_train": EXPECTED_N_TRAIN,
                "policy_sha256": EXPECTED_POLICY_SHA256,
                "train_inputs_sha256": EXPECTED_TRAIN_INPUTS_SHA256,
                "train_outcomes_sha256": EXPECTED_TRAIN_OUTCOMES_SHA256,
            },
            "protocol_id": PROTOCOL_ID,
            "required_families": list(REQUIRED_FAMILIES),
            "runtime_export": False,
            "schema_version": SCHEMA_VERSION,
            "seed_derivation": {
                "algorithm": SEED_DERIVATION,
                "core_sha256": CONFIRMATION_CORE_SHA256,
                "fail_closed_on_collision": True,
                "n": N_FRESH_SEEDS,
                "prefix": SEED_PREFIX,
                "skip_digest_on_collision": False,
            },
            "stress": {
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "mixture_diagnostic_only": True,
                "mixture_draws": MIXTURE_DRAWS,
                "mixture_seed": MIXTURE_SEED,
            },
            "thresholds": {
                "mean_official_delta": GATE_MEAN_OFFICIAL_DELTA,
                "mean_premium_delta": GATE_MEAN_PREMIUM_DELTA,
                "premium_k1_max": PREMIUM_K1_MAX,
                "tv_worst_min": TV_WORST_MIN,
                "worst_official_delta": GATE_WORST_OFFICIAL_DELTA,
                "worst_premium_delta": GATE_WORST_PREMIUM_DELTA,
            },
            "tv_ball": {
                "cost_formula": TV_COST_FORMULA,
                "cost_vertices": 91,
                "epsilon_source": "materialized_inputs_family_counts_only",
                "quality_formula": QUALITY_TV_FORMULA,
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
    if protocol.get("architecture") != live:
        raise RuntimeError("phase2 architecture snapshot drifted")
    pins = protocol.get("pins", {})
    if pins.get("family_guard_sha256") != source_sha256(FAMILY_GUARD_RELATIVE):
        raise RuntimeError("family-guard artifact hash drifted")
    if pins.get("budget_brake_sha256") != source_sha256(BUDGET_BRAKE_RELATIVE):
        raise RuntimeError("budget-brake artifact hash drifted")
    if pins.get("e2_source_sha256") != source_sha256(
        "research/lab/e2_cost_uncertainty.py"
    ):
        raise RuntimeError("E2 source hash drifted")
    if pins.get("e4_source_sha256") != source_sha256(
        "research/lab/e4_aggregate_risk.py"
    ):
        raise RuntimeError("E4 source hash drifted")
    if pins.get("e1f_source_sha256") != source_sha256(
        "research/lab/e1f_cost_conditioned_frontier.py"
    ):
        raise RuntimeError("E1F source hash drifted")


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
    if fresh != derive_fresh_risk_seeds():
        raise RuntimeError("sealed phase2 seeds drifted from the derivation")
    if set(fresh) & set(blocked_seeds()):
        raise RuntimeError("sealed phase2 seeds overlap blocked seeds")
    if protocol["confirmation"]["protocol_sha256"] != CONFIRMATION_PROTOCOL_SHA256:
        raise RuntimeError("confirmation protocol pin drifted")
    if protocol["confirmation"]["core_sha256"] != CONFIRMATION_CORE_SHA256:
        raise RuntimeError("confirmation core pin drifted")
    if protocol["confirmation"]["decision"] != CONFIRMATION_DECISION:
        raise RuntimeError("confirmation decision pin drifted")
    if protocol["e1f_decision_core_sha256"] != E1F_DECISION_CORE_SHA256:
        raise RuntimeError("E1F core pin drifted")
    if protocol["epsilon"] != epsilon_from_input_paths(train_path, dev_path):
        raise RuntimeError("sealed epsilon drifted from input-only TV")
    if protocol.get("fold_local_rebuy") is not False:
        raise RuntimeError("fold-local rebuy must stay sealed false")
    forbidden = (0.669517045455, 0.669517, 0.69, 0.690)
    for value in protocol["thresholds"].values():
        if value in forbidden:
            raise RuntimeError("phase2 thresholds must not include Dev/CHAMPION abs")
    return digest


def write_canonical_protocol(path: Path = PROTOCOL_PATH) -> Tuple[dict[str, Any], str]:
    protocol = build_canonical_protocol()
    write_json_atomic(path, protocol)
    return protocol, protocol_sha256(protocol)


def refuse_foreign_output_path(path: Path) -> None:
    text = path.resolve().as_posix()
    if "compare-e1f-cost-conditioned-frontier" in text:
        raise RuntimeError("phase2 must not write the E1F report path")
    if "confirm-chuf-tvball" in text:
        raise RuntimeError("phase2 must not write the confirmation report path")


def allocate_pooled(
    pred_qa: Any,
    pred_qk: Any,
    families: Sequence[str],
    surfaces: Mapping[str, Any],
    inputs: Any,
    digests: Sequence[str],
    brake_block: Mapping[str, Any],
) -> dict[str, Tuple[str, ...]]:
    """One public-batch buy. Fold-local rebuy is forbidden."""

    import numpy as np
    from ossp_router.budget_brake_router import promote_premium_brake
    from ossp_router.cost_calibrated_router import _select_premium
    from ossp_router.feasibility_ladder import select_fast_balanced
    from research.lab.e2_cost_uncertainty import clamp_predicted_costs
    from research.lab.e4_aggregate_risk import apply_family_increment_multiplier

    point = np.asarray(surfaces["point"], dtype=np.float64)
    conservative = np.asarray(surfaces["conservative"], dtype=np.float64)
    fast_bal = apply_family_increment_multiplier(point, families)
    n_rows = int(point.shape[0])
    qa = np.asarray(pred_qa, dtype=np.float64)
    qk = np.asarray(pred_qk, dtype=np.float64)
    fast_pred = [
        (float(qa[index]), (float(fast_bal[index, 0]), float(fast_bal[index, 1])))
        for index in range(n_rows)
    ]
    fast, _fast_ratio = select_fast_balanced(
        fast_pred,
        cap=FAST_CAP,
        runaway_fraction=RUNAWAY_FRACTION,
        max_upgrade_fraction=MAX_UPGRADE_FRACTION,
    )
    balanced, _bal_ratio = select_fast_balanced(
        fast_pred,
        cap=BALANCED_CAP,
        runaway_fraction=RUNAWAY_FRACTION,
        max_upgrade_fraction=MAX_UPGRADE_FRACTION,
    )
    premium_costs = clamp_predicted_costs(
        np.column_stack([point[:, 0], conservative[:, 1], conservative[:, 2]])
    )
    premium_pred = [
        (
            float(qa[index]),
            (
                float(premium_costs[index, 0]),
                float(premium_costs[index, 1]),
                float(premium_costs[index, 2]),
            ),
        )
        for index in range(n_rows)
    ]
    parent, _parent_ratio = _select_premium(inputs, premium_pred, PREMIUM_CAP)
    premium = promote_premium_brake(
        parent,
        [float(value) for value in qk],
        list(families),
        [tuple(float(value) for value in row) for row in premium_costs],
        list(digests),
        brake_block,
    )
    return {
        "balanced": tuple(balanced),
        "fast": tuple(fast),
        "parent": tuple(parent),
        "premium": tuple(premium),
    }


def score_fold_slices(pool: Any, models: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Score pooled decisions on fold index slices. Does not rebuy."""

    from research.lab.e1_objectives import score_decisions

    fold_ids = list(pool.folds)
    rows = []
    for fold in range(int(max(fold_ids)) + 1):
        indexes = [index for index, value in enumerate(fold_ids) if value == fold]
        scored = score_decisions(pool, models, indexes=indexes)
        rows.append(
            {
                "fold": fold,
                "n": len(indexes),
                "official_final_score": scored["official_final_score"],
                "tiers": scored["tiers"],
            }
        )
    return rows


def evaluate_seed(pool: Any, seed: int, epsilon: float) -> dict[str, Any]:
    """Later-call scoring path. Protocol tests must not invoke this."""

    from ossp_router.budget_brake_router import content_digest, load_bundled_artifact
    from research.lab.e1_objectives import stress_views
    from research.lab.e1c_regime_residual import relabel_folds
    from research.lab.e1f_cost_conditioned_frontier import (
        ax31_selections_match,
        binomial_counts,
        oof_chuf_heads,
    )
    from research.lab.e2_cost_uncertainty import oof_cost_surfaces
    from research.lab.quality_heads import content_tie_keys

    current = relabel_folds(pool, int(seed))
    surfaces = oof_cost_surfaces(current)
    trials, successes, _labels = binomial_counts(current)
    baseline_head, candidate_head, _rows = oof_chuf_heads(
        current, n=trials, k=successes, surfaces=surfaces
    )
    families = tuple(current.families)
    digests = tuple(content_digest(episode) for episode in current.episodes)
    brake = load_bundled_artifact().budget_brake
    comparator_models = allocate_pooled(
        baseline_head.pred_qa,
        baseline_head.pred_qk,
        families,
        surfaces,
        current.inputs,
        digests,
        brake,
    )
    candidate_models = allocate_pooled(
        candidate_head.pred_qa,
        candidate_head.pred_qk,
        families,
        surfaces,
        current.inputs,
        digests,
        brake,
    )
    ties = content_tie_keys(current.texts)
    identity = ax31_selections_match(
        baseline_head.pred_qa,
        candidate_head.pred_qa,
        current.costs,
        current.light_total,
        ties,
    )
    views = stress_views(
        current,
        {tier: comparator_models[tier] for tier in TIERS},
        {tier: candidate_models[tier] for tier in TIERS},
    )
    families = {
        str(row["name"]): float(row["delta"])
        for row in views
        if row.get("kind") == "family"
        and int(row["n"]) >= 20
        and row.get("delta") is not None
    }
    comparator = _score_head(current, comparator_models, epsilon)
    candidate = _score_head(current, candidate_models, epsilon)
    official_delta = float(candidate["quality_weighted"]) - float(
        comparator["quality_weighted"]
    )
    premium_delta = float(candidate["premium_quality"]) - float(
        comparator["premium_quality"]
    )
    quality_tv = tv_worst(official_delta, epsilon, families)
    return {
        "candidate": candidate,
        "candidate_row": _gate_row(
            seed,
            candidate,
            official_delta,
            premium_delta,
            quality_tv,
            pred_qa_identical=bool(
                (baseline_head.pred_qa == candidate_head.pred_qa).all()
            ),
            parent_identical=comparator_models["parent"] == candidate_models["parent"],
        ),
        "comparator": comparator,
        "comparator_row": _gate_row(
            seed,
            comparator,
            0.0,
            0.0,
            0.0,
            pred_qa_identical=True,
            parent_identical=True,
        ),
        "fold_seed": int(seed),
        "identity": identity,
        "views": views,
    }


def _gate_row(
    seed: int,
    scored: Mapping[str, Any],
    official_delta: float,
    premium_delta: float,
    quality_tv: float,
    *,
    pred_qa_identical: bool,
    parent_identical: bool,
) -> dict[str, Any]:
    return {
        "bootstrap_q999_under_95_ok": scored["bootstrap_q999_under_95_ok"],
        "fast_balanced_k1_zero": scored["fast_balanced_k1_zero"],
        "fold_seed": int(seed),
        "fold_slice_hard_caps_ok": scored["fold_slice_hard_caps_ok"],
        "official_delta": official_delta,
        "parent_identical": parent_identical,
        "pooled_hard_caps_ok": scored["pooled_hard_caps_ok"],
        "pooled_ratio_under_95_ok": scored["pooled_ratio_under_95_ok"],
        "pred_qa_identical": pred_qa_identical,
        "premium_delta": premium_delta,
        "premium_k1_count": scored["premium_k1_count"],
        "tv_cost_under_official_ok": scored["tv_cost_under_official_ok"],
        "tv_quality_worst": quality_tv,
    }


def _score_head(
    pool: Any,
    allocated: Mapping[str, Sequence[str]],
    epsilon: float,
) -> dict[str, Any]:
    import numpy as np
    from research.lab.e1_objectives import score_decisions
    from research.lab.e2_cost_uncertainty import grouped_ratio_bootstrap

    models = {tier: tuple(allocated[tier]) for tier in TIERS}
    pooled = score_decisions(pool, models)
    fold_rows = score_fold_slices(pool, models)
    actual = np.asarray(pool.costs, dtype=np.float64)
    bootstrap_ok = True
    tv_ok = True
    for tier in TIERS:
        block = grouped_ratio_bootstrap(
            models[tier],
            actual,
            actual[:, 0],
            pool.group_keys,
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )
        if float(block["q99_9"]) >= float(STRESS_RATIO_CAPS[tier]):
            bootstrap_ok = False
        spend = {}
        light = {}
        for name in REQUIRED_FAMILIES:
            mask = np.asarray([family == name for family in pool.families])
            columns = np.asarray(
                [{"ax31-light": 0, "ax31": 1, "axk1-think": 2}[model] for model in models[tier]],
                dtype=np.int64,
            )
            selected = actual[np.arange(actual.shape[0]), columns]
            spend[name] = float(selected[mask].mean())
            light[name] = float(actual[mask, 0].mean())
        center = {
            name: float(sum(1 for family in pool.families if family == name))
            / float(len(pool.families))
            for name in REQUIRED_FAMILIES
        }
        worst = tv_cost_worst(center, spend, light, epsilon)
        if worst > float(OFFICIAL_CAPS[tier]):
            tv_ok = False
    k1_fast = int(sum(model == "axk1-think" for model in models["fast"]))
    k1_bal = int(sum(model == "axk1-think" for model in models["balanced"]))
    k1_prem = int(sum(model == "axk1-think" for model in models["premium"]))
    return {
        "bootstrap_q999_under_95_ok": bootstrap_ok,
        "fast_balanced_k1_zero": k1_fast == 0 and k1_bal == 0,
        "fold_slice_hard_caps_ok": all(
            all(row["tiers"][tier]["within_hard_cap"] for tier in TIERS)
            for row in fold_rows
        ),
        "official_final_score": pooled["official_final_score"],
        "pooled_hard_caps_ok": all(
            pooled["tiers"][tier]["within_hard_cap"] for tier in TIERS
        ),
        "pooled_ratio_under_95_ok": all(
            float(pooled["tiers"][tier]["budget_ratio"]) < float(STRESS_RATIO_CAPS[tier])
            for tier in TIERS
        ),
        "premium_k1_count": k1_prem,
        "premium_quality": float(pooled["tiers"]["premium"]["quality_score"]),
        "quality_weighted": pooled["quality_weighted"],
        "tv_cost_under_official_ok": tv_ok,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "audit": report["audit"],
            "candidate": report["candidate"],
            "confirmation": report["confirmation"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "fold_seeds": report["fold_seeds"],
            "phase2_gate": report["phase2_gate"],
            "protocol_sha256": report["protocol_sha256"],
            "report_type": report["report_type"],
            "schema_version": report["schema_version"],
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


def run_phase2(
    protocol: Mapping[str, Any],
    *,
    output: Path,
    audit_output: Path,
) -> dict[str, Any]:
    """Public Phase 2. Later call only. Tests and Phase A must not invoke this."""

    from research.lab.public_pool import load_public_pool

    refuse_foreign_output_path(output)
    refuse_foreign_output_path(audit_output)
    if output.exists() or audit_output.exists():
        raise RuntimeError("phase2 output exists; refuse overwrite")
    digest = protocol_sha256(protocol)
    verify_protocol(protocol, digest)
    epsilon = float(protocol["epsilon"])
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    pool = load_public_pool()
    comparator_rows = []
    candidate_rows = []
    seed_payload = {}
    for seed in fresh:
        evaluated = evaluate_seed(pool, int(seed), epsilon)
        comparator_rows.append(evaluated["comparator_row"])
        candidate_rows.append(evaluated["candidate_row"])
        seed_payload[str(seed)] = {
            "candidate": evaluated["candidate"],
            "comparator": evaluated["comparator"],
            "official_delta": evaluated["candidate_row"]["official_delta"],
            "premium_delta": evaluated["candidate_row"]["premium_delta"],
            "tv_quality_worst": evaluated["candidate_row"]["tv_quality_worst"],
        }
    gate = phase2_gate(comparator_rows, candidate_rows)
    gate["phase2_executed"] = True
    decision = str(gate["decision"])
    if decision == PASS_DECISION:
        reason = (
            "Predicted-cost Phase 2 passed on fresh risk seeds. This is "
            "not a runtime export. Hand off to independent audit only."
        )
    elif decision == NO_REF_DECISION:
        reason = (
            "Comparator failed a safety gate. No valid predicted-cost "
            "reference. Keep the current runtime."
        )
    else:
        reason = (
            "Candidate failed a predicted-cost safety or quality gate. "
            "Keep the current runtime."
        )
    report = {
        "audit": {
            "n_rows": int(EXPECTED_N_PUBLIC) * int(N_FRESH_SEEDS),
            "relative_path": AUDIT_RELATIVE,
            "sha256": None,
        },
        "candidate": CANDIDATE_NAME,
        "confirmation": dict(protocol["confirmation"]),
        "decision": decision,
        "decision_reason": reason,
        "experiment": PROTOCOL_ID,
        "fold_seeds": list(fresh),
        "phase2_gate": gate,
        "protocol_sha256": digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_payload,
        "thresholds": dict(protocol["thresholds"]),
    }
    audit_document = {
        "experiment": PROTOCOL_ID,
        "prompt_text_included": False,
        "seeds": {str(seed): {"n_rows": EXPECTED_N_PUBLIC} for seed in fresh},
    }
    audit_sha = sha256_text(canonical_json_text(audit_document))
    report["audit"]["sha256"] = audit_sha
    report["decision_core_sha256"] = decision_core_sha256(report)
    write_json_atomic(audit_output, audit_document)
    write_json_atomic(output, report)
    return report


def validation_function_names() -> Tuple[str, ...]:
    return (
        "architecture_snapshot",
        "assert_live_architecture",
        "blocked_seeds",
        "build_canonical_protocol",
        "derive_fresh_risk_seeds",
        "epsilon_from_counts",
        "epsilon_from_input_paths",
        "family_counts_from_inputs",
        "live_artifact_snapshot",
        "load_protocol",
        "phase2_gate",
        "protocol_sha256",
        "tv_cost_vertices",
        "tv_cost_worst",
        "tv_worst",
        "verify_protocol",
    )


def assert_validation_path_has_no_outcomes(source: str | None = None) -> None:
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden_calls = {
        "evaluate_seed",
        "load_outcomes",
        "load_public_pool",
        "oof_chuf_heads",
        "oof_cost_surfaces",
        "run_phase2",
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in validation_function_names():
        if name == "tv_worst":
            continue
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
                        f"{name} embeds an outcomes path on the validation path"
                    )


def assert_no_fold_local_rebuy(source: str | None = None) -> None:
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    forbidden = {
        "allocate_pooled",
        "select_fast_balanced",
        "_select_premium",
        "promote_premium_brake",
    }
    node = functions["score_fold_slices"]
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in forbidden:
            raise RuntimeError("score_fold_slices rebought instead of slicing")
        if isinstance(child, ast.Attribute) and child.attr in forbidden:
            raise RuntimeError("score_fold_slices rebought instead of slicing")


__all__ = (
    "AUDIT_RELATIVE",
    "CONFIRMATION_CORE_SHA256",
    "CONFIRMATION_PROTOCOL_SHA256",
    "EXPECTED_PROTOCOL_SHA256",
    "EXPLICIT_RISK_SEEDS",
    "FAIL_DECISION",
    "NO_REF_DECISION",
    "OUT_RELATIVE",
    "PASS_DECISION",
    "PROTOCOL_PATH",
    "PROTOCOL_RELATIVE",
    "architecture_snapshot",
    "assert_no_fold_local_rebuy",
    "assert_validation_path_has_no_outcomes",
    "build_canonical_protocol",
    "derive_fresh_risk_seeds",
    "epsilon_from_input_paths",
    "load_protocol",
    "phase2_gate",
    "protocol_sha256",
    "refuse_foreign_output_path",
    "run_phase2",
    "tv_cost_worst",
    "verify_protocol",
    "write_canonical_protocol",
)
