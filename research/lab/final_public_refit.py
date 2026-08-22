# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Final-public-refit — brake-forest refit on the full public pool.

The organizer rules allow fitting on all public Train+Dev outcomes, and
every honest-score leader ships heads fitted that way. This module grows
the frozen ExtraTrees recipe's training set from 1,760 to 2,640 episodes
and changes nothing else: cost heads, caps, guard multipliers and every
brake constant stay bundled, so Fast/Balanced remain bit-identical and
Premium budget safety is inherited structurally. Acceptance gates are
sealed in ``research/protocols/final-public-refit.v1.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from ossp_router import budget_brake_router
from ossp_router.protocol import TIERS
from research.lab.e5_brake_conditioned import (
    K1_MODEL,
    MODEL_INDEX,
    ProtocolError,
    fit_arms,
)
from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.modeling import official_score, sort_mapping
from research.lab.public_pool import load_public_pool, subset_inputs, subset_outcomes


EXPERIMENT = "final-public-refit-v1"
REPORT_TYPE = "scrooge-final-public-refit-v1"
SCHEMA_VERSION = 1
CANDIDATE_RELATIVE = "build/run-final-public-refit/budget-brake-router.full-public.json"
REPORT_RELATIVE = "build/run-final-public-refit/report.json"
AUDIT_RELATIVE = "build/run-final-public-refit/episode-audit.json"
DEV_PIN_FINAL_SCORE = 0.669517045455
ACTUAL_RATIO_LIMIT = 3.8
PREDICTED_RATIO_LIMIT = 3.25


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(dict(protocol)))


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
    match = re.search(
        r"<\s*(\d+(?:\.\d+)?)", str(protocol["gates"]["g4_actual_budget"])
    )
    if match is None or float(match.group(1)) != ACTUAL_RATIO_LIMIT:
        raise ProtocolError("actual ratio gate text drifted")
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
    return digest


def build_candidate(harness) -> Mapping[str, Any]:
    """Refit the brake forest on every public episode; copy the rest."""

    from research.export.brake_artifact import _tree_payload

    indexes = list(range(len(harness.pool.episodes)))
    fit = fit_arms(harness, indexes)
    trees = [_tree_payload(estimator) for estimator in fit.forest.estimators_]
    if not trees or any(not tree["left"] for tree in trees):
        raise ProtocolError("refitted forest is empty")
    candidate = json.loads(json.dumps(harness.brake.value))
    candidate["budget_brake"]["forest"] = {"n_trees": len(trees), "trees": trees}
    candidate["provenance"] = {
        "cost_artifact_sha256": harness.brake.value["provenance"][
            "cost_artifact_sha256"
        ],
        "uplift_artifact_sha256": harness.brake.value["provenance"][
            "uplift_artifact_sha256"
        ],
        "dev_data_used": True,
        "export_note": (
            "Full-public final fit of the budget-brake ExtraTrees quality "
            "head only: identical hyperparameters, target, weights and "
            "features as the Train-fitted export, grown from 1,760 to "
            "2,640 episodes. Cost heads, ladder recalibration and caps, "
            "family-guard multipliers and every brake constant are "
            "byte-copied from the bundled artifact, so Fast/Balanced stay "
            "bit-identical and Premium predicted-budget safety is "
            "inherited."
        ),
    }
    return candidate


def _replay(
    harness,
    artifact: budget_brake_router.BrakeArtifact,
    indexes: Sequence[int],
    *,
    label: str,
    premium_only: bool = False,
) -> Mapping[str, Any]:
    batch = subset_inputs(harness.pool.inputs, indexes)
    outcomes = subset_outcomes(harness.pool.inputs, harness.pool.outcomes, indexes)
    tiers = ("premium",) if premium_only else TIERS
    plans = {
        tier: budget_brake_router.make_submission(
            batch, harness.policy, artifact, tier
        )
        for tier in tiers
    }
    premium_models = tuple(
        decision.model_id for decision in plans["premium"].submission.decisions
    )
    numerator = math.fsum(
        float(harness.pool.costs[index, MODEL_INDEX[model]])
        for index, model in zip(indexes, premium_models)
    )
    denominator = math.fsum(float(harness.pool.costs[index, 0]) for index in indexes)
    row: dict[str, Any] = {
        "batch": label,
        "n": len(indexes),
        "premium_predicted_ratio": float(plans["premium"].predicted_budget_ratio),
        "premium_actual_ratio": numerator / max(denominator, 1e-12),
        "premium_n_k1": int(sum(model == K1_MODEL for model in premium_models)),
    }
    if not premium_only:
        official = official_score(batch, outcomes, harness.policy, {
            tier: plans[tier].submission for tier in tiers
        })
        row["official_final_score"] = float(official["final_score"])
        row["tiers"] = {
            tier: {
                "budget_passed": bool(official["tiers"][tier]["budget_passed"]),
                "budget_ratio": float(official["tiers"][tier]["budget_ratio"]),
                "quality_score": float(official["tiers"][tier]["quality_score"]),
            }
            for tier in tiers
        }
        row["fast_balanced_models"] = [
            tuple(
                decision.model_id for decision in plans[tier].submission.decisions
            )
            for tier in ("fast", "balanced")
        ]
    return row


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
    candidate_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists() or candidate_output.exists():
        raise ProtocolError("final-refit output exists; refuse overwrite")

    pool = load_public_pool()
    harness = _harness(pool)
    gates: dict[str, Any] = {}
    failures: list[str] = []

    candidate_value = build_candidate(harness)
    try:
        candidate = budget_brake_router.load_artifact_mapping(candidate_value)
        gates["g1_contract_load"] = True
    except Exception as error:  # noqa: BLE001 - contract gate reports any failure
        gates["g1_contract_load"] = False
        failures.append(f"g1: {error}")
        candidate = None  # type: ignore[assignment]
    if candidate is None:
        raise ProtocolError("candidate artifact failed contract validation")

    train_indexes = [
        index
        for index, label in enumerate(pool.split_labels)
        if label == "train"
    ]
    dev_indexes = [
        index
        for index, label in enumerate(pool.split_labels)
        if label == "dev"
    ]

    current_replay = {
        "train": _replay(harness, harness.brake, train_indexes, label="train"),
        "dev": _replay(harness, harness.brake, dev_indexes, label="dev"),
    }
    candidate_replay = {
        "train": _replay(harness, candidate, train_indexes, label="train"),
        "dev": _replay(harness, candidate, dev_indexes, label="dev"),
    }

    first_dev = _replay(harness, candidate, dev_indexes, label="dev-repeat-1", premium_only=True)
    second_dev = _replay(harness, candidate, dev_indexes, label="dev-repeat-2", premium_only=True)
    first_models = _replay_tuple(candidate, harness, dev_indexes)
    second_models = _replay_tuple(candidate, harness, dev_indexes)
    gates["g2_determinism"] = bool(
        first_models == second_models
        and first_dev["premium_actual_ratio"] == second_dev["premium_actual_ratio"]
        and first_dev["premium_n_k1"] == second_dev["premium_n_k1"]
        and first_dev["premium_predicted_ratio"] == second_dev["premium_predicted_ratio"]
    )
    if not gates["g2_determinism"]:
        failures.append("g2: dev repeat drifted")

    batches = [current_replay["train"], current_replay["dev"],
               candidate_replay["train"], candidate_replay["dev"]]
    family_batches: list[Mapping[str, Any]] = []
    for family_name in sorted(set(harness.families)):
        indexes = [
            index
            for index, value in enumerate(harness.families)
            if value == family_name
        ]
        old_row = _replay(harness, harness.brake, indexes, label=f"family:{family_name}", premium_only=True)
        new_row = _replay(harness, candidate, indexes, label=f"family:{family_name}", premium_only=True)
        family_batches.append({"old": old_row, "new": new_row})
    gates["g4_batch_rows"] = family_batches

    predicted_ok = all(
        row["premium_predicted_ratio"] <= PREDICTED_RATIO_LIMIT
        for row in batches
    ) and all(
        row["old"]["premium_predicted_ratio"] <= PREDICTED_RATIO_LIMIT
        and row["new"]["premium_predicted_ratio"] <= PREDICTED_RATIO_LIMIT
        for row in family_batches
    )
    gates["g3_predicted_brake"] = bool(predicted_ok)
    if not predicted_ok:
        failures.append("g3: predicted ratio exceeded the brake")

    actual_values = [row["premium_actual_ratio"] for row in batches]
    worst_family = max(
        max(row["old"]["premium_actual_ratio"], row["new"]["premium_actual_ratio"])
        for row in family_batches
    )
    gates["g4_actual_budget"] = bool(
        max(actual_values) < ACTUAL_RATIO_LIMIT and worst_family < ACTUAL_RATIO_LIMIT
    )
    gates["g4_worst_family_ratio"] = float(worst_family)
    if not gates["g4_actual_budget"]:
        failures.append("g4: actual ratio reached the stress limit")

    identity_ok = all(
        candidate_replay[split]["fast_balanced_models"]
        == current_replay[split]["fast_balanced_models"]
        for split in ("train", "dev")
    )
    gates["fast_balanced_identity"] = bool(identity_ok)
    if not identity_ok:
        failures.append("identity: fast/balanced drifted")

    dev_final = float(candidate_replay["dev"]["official_final_score"])
    gates["g5_dev_floor"] = bool(dev_final >= DEV_PIN_FINAL_SCORE)
    gates["g5_observed_dev_final"] = dev_final
    if not gates["g5_dev_floor"]:
        failures.append(f"g5: dev replay {dev_final!r} below the pin")

    passed = all(
        gates[key]
        for key in (
            "g1_contract_load",
            "g2_determinism",
            "g3_predicted_brake",
            "g4_actual_budget",
            "g5_dev_floor",
            "fast_balanced_identity",
        )
    )
    decision = str(protocol["decisions"]["pass" if passed else "fail"])
    reason = str(protocol["decision_reasons"]["pass" if passed else "fail"])

    candidate_text = canonical_json_text(sort_mapping(dict(candidate_value)))
    candidate_digest = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
    current_path = Path(
        __file__).resolve().parents[2] / "src/ossp_router/resources/budget-brake-router.v1.json"
    current_digest = hashlib.sha256(current_path.read_bytes()).hexdigest()

    audit_document = {
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "batches": {
            "family_batches": [
                {"view": row["old"]["batch"], "n": row["old"]["n"]}
                for row in family_batches
            ],
            "replays": ["train", "dev"],
        },
    }

    report = {
        "audit": {
            "relative_path": AUDIT_RELATIVE,
            "sha256": sha256_text(canonical_json_text(audit_document)),
        },
        "candidate": {
            "artifact_sha256": candidate_digest,
            "bytes": len(candidate_text.encode("utf-8")),
            "n_trees": int(candidate_value["budget_brake"]["forest"]["n_trees"]),
            "n_nodes": sum(
                len(tree["left"]) for tree in candidate_value["budget_brake"]["forest"]["trees"]
            ),
            "output_relative": CANDIDATE_RELATIVE,
        },
        "candidate_replays": {
            "train": candidate_replay["train"],
            "dev": candidate_replay["dev"],
        },
        "current": {
            "artifact_sha256": current_digest,
            "dev_replay": current_replay["dev"],
            "train_replay": current_replay["train"],
        },
        "decision": decision,
        "decision_reason": reason,
        "experiment": EXPERIMENT,
        "failures": failures,
        "gates": gates,
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
    }
    core = sort_mapping(
        {
            key: report[key]
            for key in (
                "audit",
                "candidate",
                "candidate_replays",
                "current",
                "decision",
                "decision_reason",
                "experiment",
                "failures",
                "gates",
                "protocol_sha256",
                "report_type",
                "schema_version",
            )
        }
    )
    encoded = json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    report["decision_core_sha256"] = sha256_text(encoded)

    write_json_atomic(audit_output, audit_document)
    write_json_atomic(candidate_output, candidate_value)
    write_json_atomic(output, report)
    return report, audit_document


def _replay_tuple(artifact, harness, indexes: Sequence[int]) -> Tuple[str, ...]:
    """Independent premium replay used by determinism spot checks."""

    batch = subset_inputs(harness.pool.inputs, indexes)
    plan = budget_brake_router.make_submission(batch, harness.policy, artifact, "premium")
    return tuple(decision.model_id for decision in plan.submission.decisions)


def _harness(pool):
    from research.lab.e5_brake_conditioned import Harness

    return Harness.build(pool)


def run_from_protocol(
    protocol_path: Path,
    expected_protocol_sha256: str,
    *,
    output: Path,
    audit_output: Path,
    candidate_output: Path,
) -> Mapping[str, Any]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("protocol is not a JSON object")
    digest = verify_protocol(payload, expected_protocol_sha256)
    report, _audit = assemble(
        payload,
        digest,
        output=output,
        audit_output=audit_output,
        candidate_output=candidate_output,
    )
    return report
