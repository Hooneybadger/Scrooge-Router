# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""CHUF frozen-runtime fidelity protocol. Phase A seals the contract only.

Comparator pins reproduce current Train/Dev ``budget_brake_router``
scores. Candidate keeps frozen cost/parent/guard/brake and replaces
Premium qK with split-local CHUF OOF. Validation never opens outcomes.
"""

from __future__ import annotations

import ast
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from ossp_router.protocol import TIERS, load_input
from research.lab.chuf_predicted_cost_phase2 import (
    BALANCED_CAP,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONFIRMATION_CORE_SHA256,
    CONFIRMATION_PROTOCOL_SHA256,
    EXPLICIT_RISK_SEEDS,
    EXPECTED_PROTOCOL_SHA256 as PHASE2_PROTOCOL_SHA256,
    FAMILY_OTHER_MULTIPLIER,
    FAST_CAP,
    PREMIUM_CAP,
    PREMIUM_K1_MAX,
    RUNAWAY_FRACTION,
    live_artifact_snapshot,
    source_sha256,
    tv_cost_worst,
)
from research.lab.chuf_tvball_confirmation import (
    E1F_DECISION_CORE_SHA256,
    EXPECTED_POLICY_SHA256,
    N_FRESH_SEEDS,
    OLD_SEEDS,
    PASS_DECISION as CONFIRMATION_DECISION,
    REQUIRED_FAMILIES,
    derive_fresh_seeds as derive_confirmation_seeds,
    epsilon_from_input_paths,
    family_counts_from_inputs,
    tv_worst,
)
from research.lab.e1_objectives import (
    STRESS_RATIO_CAPS,
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.e1f_cost_conditioned_frontier import CANDIDATE_NAME, FAMILY_DEFINITION
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


PROTOCOL_ID = "chuf-frozen-runtime-fidelity-v1"
PROTOCOL_RELATIVE = "research/protocols/chuf-frozen-runtime-fidelity.v1.json"
PROTOCOL_PATH = ROOT / PROTOCOL_RELATIVE
REPORT_TYPE = "scrooge-chuf-frozen-runtime-fidelity"
SCHEMA_VERSION = 1
SEED_PREFIX = "scrooge-chuf-frozen-runtime-fidelity-v1"
PHASE2_CORE_SHA256 = (
    "b84cc866d24fa36b974abfe44bbe7dbfd581c7555465fc4a60f635767a8e7edd"
)
EXPECTED_PROTOCOL_SHA256 = (
    "5ace9752612f477338beeeb862a388c3f198257aa5dc38d603574e72d26cca6b"
)
EXPLICIT_FIDELITY_SEEDS: Tuple[int, ...] = (
    1043203741,
    1783423358,
    511394098,
    1329561813,
    1860797546,
    2146174231,
    1250738729,
    1773845641,
    578057546,
    1452987153,
    1985170374,
    1303320403,
)
TV_WORST_MIN = -0.003
TRAIN_DELTA_NUM = 3
TRAIN_DELTA_DEN = 17600
DEV_DELTA_NUM = 3
DEV_DELTA_DEN = 8800
WEIGHTED_DELTA_NUM = 3
WEIGHTED_DELTA_DEN = 13200
TRAIN_OFFICIAL_DELTA_MIN = TRAIN_DELTA_NUM / TRAIN_DELTA_DEN
DEV_OFFICIAL_DELTA_MIN = DEV_DELTA_NUM / DEV_DELTA_DEN
WEIGHTED_OFFICIAL_DELTA_MIN = WEIGHTED_DELTA_NUM / WEIGHTED_DELTA_DEN
SPLITS: Tuple[str, ...] = ("train", "dev")
NO_REF_DECISION = "record-chuf-frozen-runtime-fidelity-no-valid-reference-current-runtime"
FAIL_DECISION = "record-chuf-frozen-runtime-fidelity-fail-current-runtime"
PASS_DECISION = "record-chuf-frozen-runtime-fidelity-pass-await-independent-audit"
OUT_RELATIVE = "build/frozen-runtime-fidelity"
AUDIT_RELATIVE = "build/frozen-runtime-fidelity/episode-audit.json"
PHASE2_REPORT_RELATIVE = "build/phase2-chuf-predicted-cost/report.json"
CONFIRM_REPORT_RELATIVE = "build/confirm-chuf-tvball/report.json"
E1F_REPORT_RELATIVE = "build/compare-e1f-cost-conditioned-frontier/report.json"
FAMILY_GUARD_RELATIVE = "src/ossp_router/resources/family-guard-router.v1.json"
BUDGET_BRAKE_RELATIVE = "src/ossp_router/resources/budget-brake-router.v1.json"
COMPARATOR_PINS = {
    "dev": {
        "n_k1": 16,
        "official_final_score": "0.669517045455",
        "ratios": {
            "balanced": "1.396000996251",
            "fast": "1.093011852072",
            "premium": "2.160755720509",
        },
        "tier_quality": {
            "balanced": "0.674431818182",
            "fast": "0.643181818182",
            "premium": "0.699715909091",
        },
    },
    "note": (
        "Reproduction of current budget_brake_router.make_submission "
        "official score/ratio/K1. These are not quality-gate thresholds."
    ),
    "train": {
        "n_k1": 29,
        "official_final_score": "0.658636363636",
        "ratios": {
            "balanced": "1.36997673036",
            "fast": "1.085284264625",
            "premium": "2.167834666971",
        },
        "tier_quality": {
            "balanced": "0.668181818182",
            "fast": "0.628125",
            "premium": "0.689772727273",
        },
    },
}
SEED_DERIVATION = (
    "digest_i = SHA256(UTF8(PREFIX) + NUL + bytes.fromhex(PHASE2_CORE) + "
    "i.to_bytes(4,'big')); seed_i = int.from_bytes(digest_i[:4],'big') "
    "& 0x7fffffff; i=0..11. Collision or overlap with old E1F, "
    "confirmation-12, or phase2-12 fails closed."
)


def blocked_seeds() -> Tuple[int, ...]:
    return tuple(
        sorted(
            set(OLD_SEEDS)
            | set(derive_confirmation_seeds())
            | set(EXPLICIT_RISK_SEEDS)
        )
    )


def derive_fresh_fidelity_seeds(
    *,
    n: int = N_FRESH_SEEDS,
    core_sha: str = PHASE2_CORE_SHA256,
    blocked: Sequence[int] | None = None,
    prefix: str = SEED_PREFIX,
) -> Tuple[int, ...]:
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
                f"fidelity seed collision at i={index}: {seed}; fail closed"
            )
        if seed in denied:
            raise RuntimeError(
                f"fidelity seed overlaps blocked seed at i={index}: "
                f"{seed}; fail closed"
            )
        seen.add(seed)
        seeds.append(seed)
    if (
        blocked is None
        and prefix == SEED_PREFIX
        and int(n) == N_FRESH_SEEDS
        and core_sha == PHASE2_CORE_SHA256
        and tuple(seeds) != EXPLICIT_FIDELITY_SEEDS
    ):
        raise RuntimeError("fidelity explicit seed list drifted from derivation")
    return tuple(seeds)


def architecture_snapshot() -> dict[str, Any]:
    live = live_artifact_snapshot()
    if live["fast_cap"] != FAST_CAP or live["balanced_cap"] != BALANCED_CAP:
        raise RuntimeError("family-guard predicted caps drifted")
    if live["runaway_fraction"] != RUNAWAY_FRACTION:
        raise RuntimeError("runaway_fraction drifted")
    if live["multipliers"] != {"other": FAMILY_OTHER_MULTIPLIER}:
        raise RuntimeError("other multiplier drifted")
    if live["count_cap"] != PREMIUM_K1_MAX:
        raise RuntimeError("count_cap drifted")
    return sort_mapping(
        {
            "allocator": {
                "chuf_r_frozen_refit": False,
                "e2_surfaces_in_allocator": False,
                "fast_balanced_parent": (
                    "budget_brake_router.make_submission / family_guard; "
                    "identical to comparator"
                ),
                "fold_local_rebuy": False,
                "pooled_public_batch": False,
                "premium_qk": "split-local oof_chuf_heads pred_qk only",
                "split_local_batch": True,
            },
            "artifact": live,
            "candidate": CANDIDATE_NAME,
            "family_definition": FAMILY_DEFINITION,
            "official_caps": dict(OFFICIAL_CAPS),
            "predicted_caps": {
                "balanced": BALANCED_CAP,
                "fast": FAST_CAP,
                "premium": PREMIUM_CAP,
            },
            "stress_95_caps": dict(STRESS_RATIO_CAPS),
        }
    )


def quality_thresholds() -> dict[str, Any]:
    return {
        "dev_official_delta": DEV_OFFICIAL_DELTA_MIN,
        "dev_official_delta_fraction": [DEV_DELTA_NUM, DEV_DELTA_DEN],
        "mean_only_exemption": False,
        "premium_k1_max": PREMIUM_K1_MAX,
        "train_official_delta": TRAIN_OFFICIAL_DELTA_MIN,
        "train_official_delta_fraction": [TRAIN_DELTA_NUM, TRAIN_DELTA_DEN],
        "tv_worst_min": TV_WORST_MIN,
        "weighted_official_delta": WEIGHTED_OFFICIAL_DELTA_MIN,
        "weighted_official_delta_fraction": [WEIGHTED_DELTA_NUM, WEIGHTED_DELTA_DEN],
    }


def weighted_official_delta(train_delta: float, dev_delta: float) -> float:
    return (
        float(EXPECTED_N_TRAIN) * float(train_delta)
        + float(EXPECTED_N_DEV) * float(dev_delta)
    ) / float(EXPECTED_N_PUBLIC)


def _safety_ok(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("pooled_hard_caps_ok")
        and row.get("pooled_ratio_under_95_ok")
        and row.get("bootstrap_q999_under_95_ok")
        and row.get("tv_cost_under_official_ok")
    )


def _identity_ok(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("fast_identical")
        and row.get("balanced_identical")
        and row.get("parent_identical")
        and row.get("fast_balanced_k1_zero")
        and int(row.get("premium_k1_count", PREMIUM_K1_MAX + 1)) <= PREMIUM_K1_MAX
    )


def _quality_ok(row: Mapping[str, Any]) -> bool:
    train_delta = float(row["train_official_delta"])
    dev_delta = float(row["dev_official_delta"])
    weighted = weighted_official_delta(train_delta, dev_delta)
    return bool(
        train_delta >= TRAIN_OFFICIAL_DELTA_MIN
        and dev_delta >= DEV_OFFICIAL_DELTA_MIN
        and weighted >= WEIGHTED_OFFICIAL_DELTA_MIN
        and float(row["train_tv_quality_worst"]) >= TV_WORST_MIN
        and float(row["dev_tv_quality_worst"]) >= TV_WORST_MIN
    )


def fidelity_gate(
    comparator: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(candidate_rows) != N_FRESH_SEEDS:
        raise RuntimeError("fidelity gate expects 12 candidate seed rows")
    seeds = [int(row["fold_seed"]) for row in candidate_rows]
    if tuple(seeds) != EXPLICIT_FIDELITY_SEEDS:
        raise RuntimeError("fidelity seed list is not the sealed list")
    if any(seed in blocked_seeds() for seed in seeds):
        raise RuntimeError("blocked seeds entered the fidelity gate")
    comparator_fail = not (
        bool(comparator.get("pins_reproduced"))
        and all(_safety_ok(comparator[split]) for split in SPLITS)
    )
    quality_fail = [
        int(row["fold_seed"]) for row in candidate_rows if not _quality_ok(row)
    ]
    safety_fail = [
        int(row["fold_seed"])
        for row in candidate_rows
        if not all(_safety_ok(row[split]) for split in SPLITS) or not _identity_ok(row)
    ]
    if comparator_fail:
        decision = NO_REF_DECISION
        passed = False
    elif safety_fail or quality_fail:
        decision = FAIL_DECISION
        passed = False
    else:
        decision = PASS_DECISION
        passed = True
    return {
        "candidate_quality_failures": quality_fail,
        "candidate_safety_failures": safety_fail,
        "comparator_valid": not comparator_fail,
        "decision": decision,
        "mean_only_exemption": False,
        "passed": passed,
        "runtime_export": False,
        "thresholds": quality_thresholds(),
    }


def build_canonical_protocol() -> dict[str, Any]:
    counts = family_counts_from_inputs()
    epsilon = epsilon_from_input_paths()
    fresh = derive_fresh_fidelity_seeds()
    if TRAIN_INPUTS.is_file() and sha256_path(TRAIN_INPUTS) != EXPECTED_TRAIN_INPUTS_SHA256:
        raise RuntimeError("train inputs hash drifted while sealing fidelity")
    if DEV_INPUTS.is_file() and sha256_path(DEV_INPUTS) != EXPECTED_DEV_INPUTS_SHA256:
        raise RuntimeError("dev inputs hash drifted while sealing fidelity")
    return sort_mapping(
        {
            "architecture": architecture_snapshot(),
            "blocked_seeds": list(blocked_seeds()),
            "candidate": CANDIDATE_NAME,
            "chuf_r_frozen_refit": False,
            "comparator_reproduction": sort_mapping(COMPARATOR_PINS),
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
            "e2_surfaces_in_allocator": False,
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
                "phase2_report_forbidden": PHASE2_REPORT_RELATIVE,
                "report_relative": f"{OUT_RELATIVE}/report.json",
            },
            "phase2": {
                "core_sha256": PHASE2_CORE_SHA256,
                "protocol_sha256": PHASE2_PROTOCOL_SHA256,
            },
            "pins": {
                "budget_brake_sha256": source_sha256(BUDGET_BRAKE_RELATIVE),
                "dev_inputs_sha256": EXPECTED_DEV_INPUTS_SHA256,
                "dev_outcomes_sha256": EXPECTED_DEV_OUTCOMES_SHA256,
                "e1f_source_sha256": source_sha256(
                    "research/lab/e1f_cost_conditioned_frontier.py"
                ),
                "e2_source_sha256": source_sha256(
                    "research/lab/e2_cost_uncertainty.py"
                ),
                "e4_source_sha256": source_sha256(
                    "research/lab/e4_aggregate_risk.py"
                ),
                "family_guard_sha256": source_sha256(FAMILY_GUARD_RELATIVE),
                "n_dev": EXPECTED_N_DEV,
                "n_public": EXPECTED_N_PUBLIC,
                "n_train": EXPECTED_N_TRAIN,
                "policy_sha256": EXPECTED_POLICY_SHA256,
                "train_inputs_sha256": EXPECTED_TRAIN_INPUTS_SHA256,
                "train_outcomes_sha256": EXPECTED_TRAIN_OUTCOMES_SHA256,
            },
            "pooled_public_batch": False,
            "protocol_id": PROTOCOL_ID,
            "required_families": list(REQUIRED_FAMILIES),
            "runtime_export": False,
            "schema_version": SCHEMA_VERSION,
            "seed_derivation": {
                "algorithm": SEED_DERIVATION,
                "core_sha256": PHASE2_CORE_SHA256,
                "fail_closed_on_collision": True,
                "n": N_FRESH_SEEDS,
                "prefix": SEED_PREFIX,
                "skip_digest_on_collision": False,
            },
            "split_local_batch": True,
            "stress": {
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "fold_slice_diagnostic_only": True,
                "mixture_diagnostic_only": True,
            },
            "thresholds": quality_thresholds(),
            "tv_ball": {
                "cost_vertices": 91,
                "epsilon_source": "materialized_inputs_family_counts_only",
                "quality_formula": (
                    "tv_worst = official_delta + epsilon * "
                    "(min_family_delta - max_family_delta)"
                ),
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
    if protocol.get("architecture") != architecture_snapshot():
        raise RuntimeError("fidelity architecture snapshot drifted")
    if protocol.get("e2_surfaces_in_allocator") is not False:
        raise RuntimeError("E2 surfaces must stay out of the allocator")
    if protocol.get("chuf_r_frozen_refit") is not False:
        raise RuntimeError("CHUF r frozen-refit must stay false")
    if protocol.get("split_local_batch") is not True:
        raise RuntimeError("fidelity must stay split-local")
    if protocol.get("pooled_public_batch") is not False:
        raise RuntimeError("pooled public batch must stay false")
    if protocol.get("fold_local_rebuy") is not False:
        raise RuntimeError("fold-local rebuy must stay false")
    pins = protocol.get("pins", {})
    if pins.get("family_guard_sha256") != source_sha256(FAMILY_GUARD_RELATIVE):
        raise RuntimeError("family-guard hash drifted")
    if pins.get("budget_brake_sha256") != source_sha256(BUDGET_BRAKE_RELATIVE):
        raise RuntimeError("budget-brake hash drifted")


def verify_protocol(
    protocol: Mapping[str, Any],
    expected_sha256: str,
    *,
    train_path: Path = TRAIN_INPUTS,
    dev_path: Path = DEV_INPUTS,
) -> str:
    digest = protocol_sha256(protocol)
    if digest != expected_sha256:
        raise RuntimeError(
            f"protocol sha mismatch: got {digest}, expected {expected_sha256}"
        )
    assert_live_architecture(protocol)
    if tuple(int(seed) for seed in protocol["fresh_seeds"]) != derive_fresh_fidelity_seeds():
        raise RuntimeError("sealed fidelity seeds drifted")
    if set(protocol["fresh_seeds"]) & set(blocked_seeds()):
        raise RuntimeError("sealed fidelity seeds overlap blocked seeds")
    if protocol["phase2"]["core_sha256"] != PHASE2_CORE_SHA256:
        raise RuntimeError("phase2 core pin drifted")
    if protocol["phase2"]["protocol_sha256"] != PHASE2_PROTOCOL_SHA256:
        raise RuntimeError("phase2 protocol pin drifted")
    if protocol["epsilon"] != epsilon_from_input_paths(train_path, dev_path):
        raise RuntimeError("sealed epsilon drifted")
    thresholds = protocol["thresholds"]
    if thresholds["train_official_delta"] != 3 / 17600:
        raise RuntimeError("train official delta threshold drifted")
    if thresholds["dev_official_delta"] != 3 / 8800:
        raise RuntimeError("dev official delta threshold drifted")
    if thresholds["weighted_official_delta"] != 3 / 13200:
        raise RuntimeError("weighted official delta threshold drifted")
    encoded = json.dumps(thresholds, sort_keys=True)
    for banned in ("0.669517", "0.658636", "0.69"):
        if banned in encoded:
            raise RuntimeError("quality thresholds contain a comparator/abs pin")
    repro = protocol["comparator_reproduction"]
    if "0.669517045455" not in json.dumps(repro):
        raise RuntimeError("comparator Dev official pin missing")
    if "0.658636363636" not in json.dumps(repro):
        raise RuntimeError("comparator Train official pin missing")
    return digest


def write_canonical_protocol(path: Path = PROTOCOL_PATH) -> Tuple[dict[str, Any], str]:
    protocol = build_canonical_protocol()
    write_json_atomic(path, protocol)
    return protocol, protocol_sha256(protocol)


def refuse_foreign_output_path(path: Path) -> None:
    text = path.resolve().as_posix()
    if "compare-e1f-cost-conditioned-frontier" in text:
        raise RuntimeError("fidelity must not write the E1F report path")
    if "confirm-chuf-tvball" in text:
        raise RuntimeError("fidelity must not write the confirmation report path")
    if "phase2-chuf-predicted-cost" in text:
        raise RuntimeError("fidelity must not write the phase2 report path")


def pins_reproduced(split: str, official: Mapping[str, Any], n_k1: int) -> bool:
    pin = COMPARATOR_PINS[split]
    if Decimal(str(official["final_score"])) != Decimal(pin["official_final_score"]):
        return False
    if int(n_k1) != int(pin["n_k1"]):
        return False
    for tier in TIERS:
        observed = Decimal(str(official["tiers"][tier]["budget_ratio"]))
        if observed != Decimal(pin["ratios"][tier]):
            return False
    return True


def allocate_frozen_runtime(
    inputs: Any,
    artifact: Any,
    quality: Sequence[float] | None = None,
) -> dict[str, Tuple[str, ...]]:
    """Frozen cost/parent/guard/brake. Optional Premium qK override only."""

    from ossp_router.budget_brake_router import (
        content_digest,
        make_submission,
        premium_prediction_row,
        select_premium_with_brake,
    )
    from ossp_router.cost_calibrated_router import prompt_family as family_of
    from ossp_router.feasibility_ladder import _select_premium
    from ossp_router.protocol import load_bundled_policy

    policy = load_bundled_policy()
    fast = make_submission(inputs, policy, artifact, "fast")
    balanced = make_submission(inputs, policy, artifact, "balanced")
    premium_rows = tuple(
        premium_prediction_row(episode, policy, artifact) for episode in inputs.episodes
    )
    parent, _ratio = _select_premium(
        inputs,
        [(row[0], row[1]) for row in premium_rows],
        float(artifact.value["predicted_caps"]["premium"]),
    )
    if quality is None:
        premium = make_submission(inputs, policy, artifact, "premium")
        premium_models = tuple(
            decision.model_id for decision in premium.submission.decisions
        )
    else:
        families = tuple(family_of(episode) for episode in inputs.episodes)
        digests = tuple(content_digest(episode) for episode in inputs.episodes)
        premium_models = tuple(
            select_premium_with_brake(
                inputs,
                policy,
                artifact,
                premium_rows,
                quality=quality,
                families=families,
                digests=digests,
            )
        )
    return {
        "balanced": tuple(
            decision.model_id for decision in balanced.submission.decisions
        ),
        "fast": tuple(decision.model_id for decision in fast.submission.decisions),
        "parent": tuple(parent),
        "premium": premium_models,
    }


def load_split_pool(split: str, *, fold_seed: int) -> Any:
    """Split-local public pool. Run path only."""

    from ossp_router.protocol import load_bundled_policy, load_outcomes
    from research.lab.grouped_crossfit import (
        FOLDS,
        assign_balanced_group_folds,
        families_of,
        fold_balance,
        fold_leakage_count,
        group_episodes,
        language_view,
        length_view,
    )
    from research.lab.prompt_features import episode_text_of
    from research.lab.public_pool import DEV_OUTCOMES, TRAIN_OUTCOMES, PublicPool
    from research.lab.validation import public_arrays

    if split == "train":
        input_path, outcome_path = TRAIN_INPUTS, TRAIN_OUTCOMES
        expected_n = EXPECTED_N_TRAIN
        expected_in = EXPECTED_TRAIN_INPUTS_SHA256
        expected_out = EXPECTED_TRAIN_OUTCOMES_SHA256
    elif split == "dev":
        input_path, outcome_path = DEV_INPUTS, DEV_OUTCOMES
        expected_n = EXPECTED_N_DEV
        expected_in = EXPECTED_DEV_INPUTS_SHA256
        expected_out = EXPECTED_DEV_OUTCOMES_SHA256
    else:
        raise RuntimeError(f"unknown split {split!r}")
    if sha256_path(input_path) != expected_in:
        raise RuntimeError(f"{split} inputs hash drifted")
    if sha256_path(outcome_path) != expected_out:
        raise RuntimeError(f"{split} outcomes hash drifted")
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcome_path)
    if len(inputs.episodes) != expected_n:
        raise RuntimeError(f"{split} episode count drifted")
    policy = load_bundled_policy()
    arrays = public_arrays(inputs, outcomes, policy)
    texts = tuple(episode_text_of(episode) for episode in inputs.episodes)
    families = families_of(inputs.episodes)
    grouping = group_episodes(inputs.episodes)
    fold_ids = assign_balanced_group_folds(
        grouping.group_keys, families, folds=FOLDS, seed=int(fold_seed)
    )
    leaked = fold_leakage_count(grouping.group_keys, fold_ids)
    if leaked:
        raise RuntimeError(f"{split} grouped fold leakage: {leaked}")
    return PublicPool(
        episodes=inputs.episodes,
        texts=texts,
        families=families,
        languages=tuple(language_view(text) for text in texts),
        length_views=tuple(length_view(text) for text in texts),
        group_keys=grouping.group_keys,
        exact_keys=grouping.exact_keys,
        template_keys=grouping.template_keys,
        folds=fold_ids,
        scores=arrays.scores.astype("float64"),
        costs=arrays.costs.astype("float64"),
        light_total=float(arrays.costs[:, 0].sum()),
        identity={
            "fold_seed": int(fold_seed),
            "folds": FOLDS,
            "n_episodes": expected_n,
            "split": split,
        },
        grouping={
            "n_groups": grouping.n_groups,
            "n_singleton_groups": grouping.n_singleton_groups,
        },
        fold_table=fold_balance(grouping.group_keys, fold_ids, families),
        inputs=inputs,
        outcomes=outcomes,
        policy=policy,
        split_labels=(split,) * expected_n,
    )


def score_split_allocation(
    pool: Any,
    allocated: Mapping[str, Sequence[str]],
    epsilon: float,
) -> dict[str, Any]:
    import numpy as np
    from research.lab.e1_objectives import score_decisions
    from research.lab.e2_cost_uncertainty import grouped_ratio_bootstrap

    models = {tier: tuple(allocated[tier]) for tier in TIERS}
    pooled = score_decisions(pool, models)
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
        columns = np.asarray(
            [{"ax31-light": 0, "ax31": 1, "axk1-think": 2}[model] for model in models[tier]],
            dtype=np.int64,
        )
        selected = actual[np.arange(actual.shape[0]), columns]
        for name in REQUIRED_FAMILIES:
            mask = np.asarray([family == name for family in pool.families])
            spend[name] = float(selected[mask].mean())
            light[name] = float(actual[mask, 0].mean())
        center = {
            name: float(sum(1 for family in pool.families if family == name))
            / float(len(pool.families))
            for name in REQUIRED_FAMILIES
        }
        if tv_cost_worst(center, spend, light, epsilon) > float(OFFICIAL_CAPS[tier]):
            tv_ok = False
    k1_fast = int(sum(model == "axk1-think" for model in models["fast"]))
    k1_bal = int(sum(model == "axk1-think" for model in models["balanced"]))
    k1_prem = int(sum(model == "axk1-think" for model in models["premium"]))
    return {
        "bootstrap_q999_under_95_ok": bootstrap_ok,
        "fast_balanced_k1_zero": k1_fast == 0 and k1_bal == 0,
        "fold_slice_hard_caps_ok": None,
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


def _family_official_deltas(
    pool: Any,
    comparator: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    from research.lab.e1_objectives import score_decisions

    deltas: dict[str, float] = {}
    for name in REQUIRED_FAMILIES:
        indexes = [index for index, family in enumerate(pool.families) if family == name]
        if len(indexes) < 20:
            continue
        base = score_decisions(pool, comparator, indexes=indexes)
        cand = score_decisions(pool, candidate, indexes=indexes)
        deltas[name] = float(cand["official_final_score"]) - float(
            base["official_final_score"]
        )
    return deltas


def evaluate_seed(
    train_pool: Any,
    dev_pool: Any,
    comparator_models: Mapping[str, Mapping[str, Sequence[str]]],
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    """Later-call path. Protocol tests must not invoke this."""

    from ossp_router.budget_brake_router import load_bundled_artifact
    from research.lab.e1c_regime_residual import relabel_folds
    from research.lab.e1f_cost_conditioned_frontier import binomial_counts, oof_chuf_heads
    from research.lab.e2_cost_uncertainty import oof_cost_surfaces

    artifact = load_bundled_artifact()
    scored: dict[str, Any] = {}
    for name, pool in (("train", train_pool), ("dev", dev_pool)):
        current = relabel_folds(pool, int(seed))
        surfaces = oof_cost_surfaces(current)
        trials, successes, _labels = binomial_counts(current)
        _base, candidate, _fold = oof_chuf_heads(
            current, n=trials, k=successes, surfaces=surfaces
        )
        allocated = allocate_frozen_runtime(
            current.inputs, artifact, quality=candidate.pred_qk
        )
        safety = score_split_allocation(current, allocated, epsilon)
        family_deltas = _family_official_deltas(
            current, comparator_models[name], allocated
        )
        official_delta = float(safety["official_final_score"]) - float(
            score_split_allocation(current, comparator_models[name], epsilon)[
                "official_final_score"
            ]
        )
        scored[name] = {
            "allocated": allocated,
            "official_delta": official_delta,
            "safety": safety,
            "tv_quality_worst": tv_worst(official_delta, epsilon, family_deltas),
        }
    train = scored["train"]
    dev = scored["dev"]
    comparator_train = comparator_models["train"]
    comparator_dev = comparator_models["dev"]
    return {
        "dev": dev["safety"],
        "dev_official_delta": dev["official_delta"],
        "dev_tv_quality_worst": dev["tv_quality_worst"],
        "balanced_identical": train["allocated"]["balanced"] == comparator_train["balanced"]
        and dev["allocated"]["balanced"] == comparator_dev["balanced"],
        "fast_balanced_k1_zero": train["safety"]["fast_balanced_k1_zero"]
        and dev["safety"]["fast_balanced_k1_zero"],
        "fast_identical": train["allocated"]["fast"] == comparator_train["fast"]
        and dev["allocated"]["fast"] == comparator_dev["fast"],
        "fold_seed": int(seed),
        "parent_identical": train["allocated"]["parent"] == comparator_train["parent"]
        and dev["allocated"]["parent"] == comparator_dev["parent"],
        "premium_k1_count": max(
            int(train["safety"]["premium_k1_count"]),
            int(dev["safety"]["premium_k1_count"]),
        ),
        "train": train["safety"],
        "train_official_delta": train["official_delta"],
        "train_tv_quality_worst": train["tv_quality_worst"],
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "audit": report["audit"],
            "candidate": report["candidate"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "fidelity_gate": report["fidelity_gate"],
            "fold_seeds": report["fold_seeds"],
            "phase2": report["phase2"],
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


def run_fidelity(
    protocol: Mapping[str, Any],
    *,
    output: Path,
    audit_output: Path,
) -> dict[str, Any]:
    """Public fidelity. Later call only. Phase A must not invoke this."""

    from ossp_router.budget_brake_router import load_bundled_artifact
    from ossp_router.protocol import load_bundled_policy
    from research.lab.modeling import official_score

    refuse_foreign_output_path(output)
    refuse_foreign_output_path(audit_output)
    if output.exists() or audit_output.exists():
        raise RuntimeError("fidelity output exists; refuse overwrite")
    digest = protocol_sha256(protocol)
    verify_protocol(protocol, digest)
    epsilon = float(protocol["epsilon"])
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    artifact = load_bundled_artifact()
    policy = load_bundled_policy()
    train_pool = load_split_pool("train", fold_seed=fresh[0])
    dev_pool = load_split_pool("dev", fold_seed=fresh[0])
    comparator_models = {
        "train": allocate_frozen_runtime(train_pool.inputs, artifact),
        "dev": allocate_frozen_runtime(dev_pool.inputs, artifact),
    }
    reproduced = True
    comparator_safety: dict[str, Any] = {}
    for name, pool in (("train", train_pool), ("dev", dev_pool)):
        models = comparator_models[name]
        official = official_score(pool.inputs, pool.outcomes, policy, models)
        n_k1 = int(sum(model == "axk1-think" for model in models["premium"]))
        reproduced = reproduced and pins_reproduced(name, official, n_k1)
        comparator_safety[name] = score_split_allocation(pool, models, epsilon)
    comparator = {
        "pins_reproduced": reproduced,
        **comparator_safety,
    }
    candidate_rows = []
    seed_payload = {}
    for seed in fresh:
        row = evaluate_seed(train_pool, dev_pool, comparator_models, int(seed), epsilon)
        candidate_rows.append(row)
        seed_payload[str(seed)] = {
            "dev_official_delta": row["dev_official_delta"],
            "train_official_delta": row["train_official_delta"],
            "weighted_official_delta": weighted_official_delta(
                row["train_official_delta"], row["dev_official_delta"]
            ),
        }
    gate = fidelity_gate(comparator, candidate_rows)
    decision = str(gate["decision"])
    if decision == PASS_DECISION:
        reason = (
            "Frozen-runtime fidelity passed on fresh seeds. This is not "
            "a runtime export. Hand off to independent audit only."
        )
    elif decision == NO_REF_DECISION:
        reason = (
            "Comparator failed pin reproduction or current-runtime safety. "
            "No valid frozen-runtime reference. Keep the current runtime."
        )
    else:
        reason = (
            "Candidate failed a frozen-runtime safety, identity, or "
            "quality gate. Keep the current runtime."
        )
    report = {
        "audit": {
            "n_rows": (int(EXPECTED_N_TRAIN) + int(EXPECTED_N_DEV)) * int(N_FRESH_SEEDS),
            "relative_path": AUDIT_RELATIVE,
            "sha256": None,
        },
        "candidate": CANDIDATE_NAME,
        "decision": decision,
        "decision_reason": reason,
        "experiment": PROTOCOL_ID,
        "fidelity_gate": gate,
        "fold_seeds": list(fresh),
        "phase2": dict(protocol["phase2"]),
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
        "seeds": {
            str(seed): {"n_dev": EXPECTED_N_DEV, "n_train": EXPECTED_N_TRAIN}
            for seed in fresh
        },
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
        "derive_fresh_fidelity_seeds",
        "fidelity_gate",
        "load_protocol",
        "pins_reproduced",
        "protocol_sha256",
        "quality_thresholds",
        "verify_protocol",
        "weighted_official_delta",
    )


def assert_validation_path_has_no_outcomes(source: str | None = None) -> None:
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden = {
        "evaluate_seed",
        "load_outcomes",
        "load_public_pool",
        "load_split_pool",
        "oof_chuf_heads",
        "oof_cost_surfaces",
        "run_fidelity",
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in validation_function_names():
        node = functions[name]
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in forbidden:
                raise RuntimeError(f"{name} references forbidden {child.id}")
            if isinstance(child, ast.Attribute) and child.attr in forbidden:
                raise RuntimeError(f"{name} references forbidden {child.attr}")


def assert_allocator_has_no_e2_surfaces(source: str | None = None) -> None:
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    forbidden = {"oof_cost_surfaces", "oof_chuf_heads"}
    for child in ast.walk(functions["allocate_frozen_runtime"]):
        if isinstance(child, ast.Name) and child.id in forbidden:
            raise RuntimeError("allocator uses E2/CHUF surfaces")
        if isinstance(child, ast.Attribute) and child.attr in forbidden:
            raise RuntimeError("allocator uses E2/CHUF surfaces")


__all__ = (
    "COMPARATOR_PINS",
    "DEV_OFFICIAL_DELTA_MIN",
    "EXPLICIT_FIDELITY_SEEDS",
    "EXPECTED_PROTOCOL_SHA256",
    "FAIL_DECISION",
    "NO_REF_DECISION",
    "OUT_RELATIVE",
    "PASS_DECISION",
    "PHASE2_CORE_SHA256",
    "PROTOCOL_PATH",
    "TRAIN_OFFICIAL_DELTA_MIN",
    "WEIGHTED_OFFICIAL_DELTA_MIN",
    "architecture_snapshot",
    "assert_allocator_has_no_e2_surfaces",
    "assert_validation_path_has_no_outcomes",
    "blocked_seeds",
    "build_canonical_protocol",
    "derive_fresh_fidelity_seeds",
    "fidelity_gate",
    "load_protocol",
    "protocol_sha256",
    "quality_thresholds",
    "refuse_foreign_output_path",
    "run_fidelity",
    "verify_protocol",
    "weighted_official_delta",
    "write_canonical_protocol",
)
