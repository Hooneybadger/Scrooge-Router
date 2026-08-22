# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E7 — apply the shipped residual family guard to Premium parent allocation.

Fast/Balanced and every brake constant stay frozen. The candidate only
reprices residual AX31 increments with the existing family-guard
multiplier before the two-action Premium parent runs. Brake spend still
uses unguarded predicted costs. This is a safety patch for the audit
finding that a residual-only Premium batch realizes 4.38.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ossp_router.protocol import TIERS, load_input, load_outcomes, policy_sha256
from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.modeling import sort_mapping
from research.lab.public_pool import (
    DEV_INPUTS,
    DEV_OUTCOMES,
    EXPECTED_DEV_INPUTS_SHA256,
    EXPECTED_DEV_OUTCOMES_SHA256,
    EXPECTED_N_DEV,
    EXPECTED_N_TRAIN,
    EXPECTED_TRAIN_INPUTS_SHA256,
    EXPECTED_TRAIN_OUTCOMES_SHA256,
    TRAIN_INPUTS,
    TRAIN_OUTCOMES,
    sha256_path,
)
from research.lab.serving_replica import (
    INFLATION,
    NEAR_FRAC,
    OFFICIAL_CAPS,
    PINNED_DEV_FINAL_SCORE,
    PIN_TOLERANCE,
    ProtocolError,
    RESIDUAL_FAMILY,
    ServingReplica,
    SplitReplica,
    composition_views,
    family_views,
    json_float,
    model_counts,
    official_tier_block,
    score_models,
)


EXPERIMENT = "e7-premium-residual-guard-v1"
REPORT_TYPE = "scrooge-e7-premium-residual-guard-v1"
SCHEMA_VERSION = 1
BASELINE_ARM = "shipped"
CANDIDATE_ARM = "premium-residual-guard"
ARMS: Tuple[str, ...] = (BASELINE_ARM, CANDIDATE_ARM)
AUDIT_RELATIVE = "build/run-e7-premium-residual-guard/episode-audit.json"
REPORT_RELATIVE = "build/run-e7-premium-residual-guard/report.json"


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
    if protocol["arms"]["baseline"] != BASELINE_ARM:
        raise ProtocolError("baseline arm drifted")
    if protocol["arms"]["candidate"] != CANDIDATE_ARM:
        raise ProtocolError("candidate arm drifted")
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


def _guard_parent(arm: str) -> bool:
    if arm == BASELINE_ARM:
        return False
    if arm == CANDIDATE_ARM:
        return True
    raise ProtocolError(f"unknown arm {arm}")


def evaluate_arm_on_split(
    replica: ServingReplica,
    split: SplitReplica,
    arm: str,
) -> dict[str, Any]:
    full = list(range(split.n))
    guard = _guard_parent(arm)
    selections = replica.allocate_all(
        split, full, caps=replica.shipped_caps, guard_parent=guard
    )
    official = replica.official(split, selections)
    views = composition_views(split)
    stress: dict[str, Any] = {}
    premium_view_failures = []
    residual_premium = None
    for name, indexes in views.items():
        view_sel = replica.allocate_all(
            split, indexes, caps=replica.shipped_caps, guard_parent=guard
        )
        stress[name] = {}
        for tier in TIERS:
            scored = score_models(split.scores, split.costs, indexes, view_sel[tier])
            inflated = float(scored["actual_ratio"]) * INFLATION
            cap = float(OFFICIAL_CAPS[tier])
            ruin = bool(float(scored["actual_ratio"]) > cap + 1e-15)
            ruin_inflated = bool(inflated > cap + 1e-15)
            stress[name][tier] = {
                "actual_ratio": scored["actual_ratio"],
                "counts": scored["counts"],
                "inflated_ratio": json_float(inflated),
                "n": scored["n"],
                "quality": scored["quality"],
                "ruin": ruin,
                "ruin_inflated": ruin_inflated,
            }
            if tier == "premium" and (ruin or ruin_inflated):
                premium_view_failures.append(
                    {
                        "actual_ratio": scored["actual_ratio"],
                        "inflated_ratio": json_float(inflated),
                        "view": name,
                    }
                )
        if name == f"family:{RESIDUAL_FAMILY}":
            residual_premium = dict(stress[name]["premium"])
    family_rows = {}
    for family, indexes in family_views(split.families).items():
        models = [selections["premium"][index] for index in indexes]
        family_rows[family] = score_models(
            split.scores, split.costs, indexes, models
        )
    return {
        "counts": {tier: model_counts(selections[tier]) for tier in TIERS},
        "families_premium": family_rows,
        "final_score": float(official["final_score"]),
        "official": {tier: official_tier_block(official, tier) for tier in TIERS},
        "premium_view_failures": premium_view_failures,
        "residual_fraction": json_float(split.residual_frac),
        "residual_premium": residual_premium,
        "selections": {tier: list(selections[tier]) for tier in TIERS},
        "stress": stress,
    }


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("e7 premium-residual-guard output exists; refuse overwrite")

    replica = ServingReplica.load()
    identity = {
        "dev_inputs_sha256": sha256_path(DEV_INPUTS),
        "dev_outcomes_sha256": sha256_path(DEV_OUTCOMES),
        "n_dev": EXPECTED_N_DEV,
        "n_train": EXPECTED_N_TRAIN,
        "policy_sha256": policy_sha256(replica.policy),
        "train_inputs_sha256": sha256_path(TRAIN_INPUTS),
        "train_outcomes_sha256": sha256_path(TRAIN_OUTCOMES),
    }
    expected_pins = {
        "dev_inputs_sha256": EXPECTED_DEV_INPUTS_SHA256,
        "dev_outcomes_sha256": EXPECTED_DEV_OUTCOMES_SHA256,
        "train_inputs_sha256": EXPECTED_TRAIN_INPUTS_SHA256,
        "train_outcomes_sha256": EXPECTED_TRAIN_OUTCOMES_SHA256,
    }
    for key, expected in expected_pins.items():
        if identity[key] != expected:
            raise ProtocolError(f"materialized {key} drifted")
    verify_protocol(protocol, protocol_digest, pool_identity=identity)

    train = replica.build_split("train", load_input(TRAIN_INPUTS), load_outcomes(TRAIN_OUTCOMES))
    dev = replica.build_split("dev", load_input(DEV_INPUTS), load_outcomes(DEV_OUTCOMES))
    if train.n != EXPECTED_N_TRAIN or dev.n != EXPECTED_N_DEV:
        raise ProtocolError("split n drifted")

    shipped_runtime = {tier: replica.runtime_models(dev, tier) for tier in TIERS}
    shipped_replica = replica.allocate_all(
        dev, list(range(dev.n)), caps=replica.shipped_caps, guard_parent=False
    )
    fidelity = all(
        tuple(shipped_runtime[tier]) == tuple(shipped_replica[tier]) for tier in TIERS
    )
    pin_official = replica.official(dev, shipped_runtime)
    pin = {
        "final_score": float(pin_official["final_score"]),
        "matched": bool(
            abs(float(pin_official["final_score"]) - PINNED_DEV_FINAL_SCORE)
            <= PIN_TOLERANCE
        ),
        "pinned_final_score": PINNED_DEV_FINAL_SCORE,
        "replica_fidelity": bool(fidelity),
    }
    if not pin["matched"] or not fidelity:
        raise ProtocolError("shipped Dev pin or replica fidelity failed")

    results: dict[str, dict[str, Any]] = {}
    selections: dict[str, dict[str, dict[str, list[str]]]] = {}
    for arm in ARMS:
        train_row = evaluate_arm_on_split(replica, train, arm)
        dev_row = evaluate_arm_on_split(replica, dev, arm)
        selections[arm] = {
            "dev": dev_row.pop("selections"),
            "train": train_row.pop("selections"),
        }
        results[arm] = {"dev": dev_row, "train": train_row}

    thresholds = protocol["thresholds"]
    baseline_train = float(results[BASELINE_ARM]["train"]["final_score"])
    baseline_dev = float(results[BASELINE_ARM]["dev"]["final_score"])
    cand_train = float(results[CANDIDATE_ARM]["train"]["final_score"])
    cand_dev = float(results[CANDIDATE_ARM]["dev"]["final_score"])
    train_delta = cand_train - baseline_train
    dev_delta = cand_dev - baseline_dev

    fast_bal_identical = all(
        selections[CANDIDATE_ARM][label][tier]
        == selections[BASELINE_ARM][label][tier]
        for label in ("train", "dev")
        for tier in ("fast", "balanced")
    )

    official_fail = [
        {"split": label, "tier": tier, "budget_ratio": row["budget_ratio"]}
        for label in ("train", "dev")
        for tier, row in results[CANDIDATE_ARM][label]["official"].items()
        if (not row["budget_passed"])
        or float(row["budget_ratio"]) >= NEAR_FRAC * float(OFFICIAL_CAPS[tier]) - 1e-15
    ]
    premium_view_fail = list(results[CANDIDATE_ARM]["train"]["premium_view_failures"]) + list(
        results[CANDIDATE_ARM]["dev"]["premium_view_failures"]
    )

    residual_rows = []
    residual_ok = True
    residual_improved = True
    actual_max = float(thresholds["residual_premium_actual_max"])
    inflated_max = float(thresholds["residual_premium_inflated_max"])
    for label in ("train", "dev"):
        base = results[BASELINE_ARM][label]["residual_premium"]
        cand = results[CANDIDATE_ARM][label]["residual_premium"]
        residual_rows.append({"baseline": base, "candidate": cand, "split": label})
        if cand is None or base is None:
            residual_ok = False
            residual_improved = False
            continue
        if float(cand["actual_ratio"]) > actual_max + 1e-15:
            residual_ok = False
        if float(cand["inflated_ratio"]) > inflated_max + 1e-15:
            residual_ok = False
        if float(cand["actual_ratio"]) >= float(base["actual_ratio"]) - 1e-15:
            residual_improved = False

    failures = []
    if not fast_bal_identical:
        failures.append("fast_balanced_identity")
    if official_fail:
        failures.append("official_safety")
    if premium_view_fail:
        failures.append("premium_view_safety")
    if not residual_ok:
        failures.append("residual_premium_cap")
    if not residual_improved:
        failures.append("residual_premium_must_improve")
    if dev_delta <= float(thresholds["dev_delta_min_exclusive"]):
        failures.append("dev_veto")

    passed = not failures
    decision = str(protocol["decisions"]["pass" if passed else "fail"])
    reason = str(protocol["decision_reasons"]["pass" if passed else "fail"])

    gate = {
        "dev_delta": json_float(dev_delta),
        "dev_score": json_float(cand_dev),
        "failures": failures,
        "fast_balanced_identical": bool(fast_bal_identical),
        "official_failures": official_fail,
        "passed": passed,
        "pin": pin,
        "premium_view_failures": premium_view_fail,
        "residual_improved": bool(residual_improved),
        "residual_ok": bool(residual_ok),
        "residual_rows": residual_rows,
        "train_delta": json_float(train_delta),
        "train_score": json_float(cand_train),
        "window_arm": CANDIDATE_ARM if passed else None,
    }

    audit_document = {
        "arms": list(ARMS),
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "rows": {
            label: [
                {
                    "episode_id": split.inputs.episodes[index].episode_id,
                    "family": split.families[index],
                    **{
                        arm: {
                            tier: selections[arm][label][tier][index] for tier in TIERS
                        }
                        for arm in ARMS
                    },
                }
                for index in range(split.n)
            ]
            for label, split in (("train", train), ("dev", dev))
        },
    }
    report = {
        "audit": {
            "n_rows": int(EXPECTED_N_TRAIN + EXPECTED_N_DEV),
            "relative_path": AUDIT_RELATIVE,
            "sha256": sha256_text(canonical_json_text(audit_document)),
        },
        "decision": decision,
        "decision_reason": reason,
        "experiment": EXPERIMENT,
        "gate": gate,
        "identity": identity,
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "results": results,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "thresholds": dict(thresholds),
    }
    core = sort_mapping(
        {
            key: report[key]
            for key in (
                "audit",
                "decision",
                "decision_reason",
                "experiment",
                "gate",
                "identity",
                "protocol_sha256",
                "report_type",
                "results",
                "schema_version",
                "thresholds",
            )
        }
    )
    report["decision_core_sha256"] = sha256_text(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
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
