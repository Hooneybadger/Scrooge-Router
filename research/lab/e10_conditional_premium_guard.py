# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E10 — Premium parent/denylist only when residual fraction is high.

Public mixed batches sit near residual 0.10. The grid is 0.25 / 0.50 /
0.75, the remaining thirds of (0, 1] after that public mix, not a fit
to 4.38. Below the threshold the arm is shipped. Fast/Balanced stay
shipped. No new multiplier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

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
    residual_fraction,
    score_models,
)


EXPERIMENT = "e10-conditional-premium-guard-v1"
REPORT_TYPE = "scrooge-e10-conditional-premium-guard-v1"
SCHEMA_VERSION = 1
BASELINE_ARM = "shipped"
CANDIDATE_ARMS: Tuple[str, ...] = (
    "cond-parent-0.75",
    "cond-parent-0.50",
    "cond-parent-0.25",
    "cond-parent-denylist-0.75",
    "cond-parent-denylist-0.50",
    "cond-parent-denylist-0.25",
)
AUDIT_RELATIVE = "build/run-e10-conditional-premium-guard/episode-audit.json"
REPORT_RELATIVE = "build/run-e10-conditional-premium-guard/report.json"


class ArmKnobs(NamedTuple):
    threshold: Optional[float]
    guard_parent: bool
    denylist_other: bool


ARM_KNOBS: Mapping[str, ArmKnobs] = {
    BASELINE_ARM: ArmKnobs(None, False, False),
    "cond-parent-0.75": ArmKnobs(0.75, True, False),
    "cond-parent-0.50": ArmKnobs(0.50, True, False),
    "cond-parent-0.25": ArmKnobs(0.25, True, False),
    "cond-parent-denylist-0.75": ArmKnobs(0.75, True, True),
    "cond-parent-denylist-0.50": ArmKnobs(0.50, True, True),
    "cond-parent-denylist-0.25": ArmKnobs(0.25, True, True),
}
ARMS: Tuple[str, ...] = (BASELINE_ARM,) + CANDIDATE_ARMS


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
    if protocol.get("experiment") != EXPERIMENT or protocol.get("protocol_id") != EXPERIMENT:
        raise ProtocolError("protocol experiment id drifted")
    if protocol["arms"]["baseline"] != BASELINE_ARM:
        raise ProtocolError("baseline arm drifted")
    if tuple(protocol["arms"]["candidates"]) != CANDIDATE_ARMS:
        raise ProtocolError("sealed candidate arm list drifted")
    sealed = protocol["arms"]["knobs"]
    if set(sealed) != set(CANDIDATE_ARMS):
        raise ProtocolError("sealed knob keys drifted")
    for arm in CANDIDATE_ARMS:
        row = sealed[arm]
        expected = ARM_KNOBS[arm]
        if abs(float(row["threshold"]) - float(expected.threshold)) > 1e-15:
            raise ProtocolError(f"sealed threshold for {arm} drifted")
        if bool(row["guard_parent"]) != expected.guard_parent:
            raise ProtocolError(f"sealed guard_parent for {arm} drifted")
        if bool(row["denylist_other"]) != expected.denylist_other:
            raise ProtocolError(f"sealed denylist_other for {arm} drifted")
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


def arm_knobs(arm: str) -> ArmKnobs:
    if arm not in ARM_KNOBS:
        raise ProtocolError(f"unknown arm {arm}")
    return ARM_KNOBS[arm]


def allocate_arm(
    replica: ServingReplica,
    split: SplitReplica,
    indexes: Sequence[int],
    arm: str,
) -> dict[str, Tuple[str, ...]]:
    knobs = arm_knobs(arm)
    fraction = residual_fraction([split.families[index] for index in indexes])
    active = knobs.threshold is not None and fraction + 1e-15 >= float(knobs.threshold)
    extra = (RESIDUAL_FAMILY,) if active and knobs.denylist_other else ()
    return replica.allocate_all(
        split,
        indexes,
        caps=replica.shipped_caps,
        guard_parent=bool(active and knobs.guard_parent),
        denylist_extra=extra,
    )


def evaluate_arm_on_split(
    replica: ServingReplica, split: SplitReplica, arm: str
) -> dict[str, Any]:
    full = list(range(split.n))
    selections = allocate_arm(replica, split, full, arm)
    official = replica.official(split, selections)
    views = composition_views(split)
    stress: dict[str, Any] = {}
    premium_view_failures = []
    residual_premium = None
    knobs = arm_knobs(arm)
    for name, indexes in views.items():
        view_sel = allocate_arm(replica, split, indexes, arm)
        stress[name] = {
            "residual_fraction": json_float(
                residual_fraction([split.families[index] for index in indexes])
            )
        }
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
        family_rows[family] = score_models(split.scores, split.costs, indexes, models)
    return {
        "counts": {tier: model_counts(selections[tier]) for tier in TIERS},
        "families_premium": family_rows,
        "final_score": float(official["final_score"]),
        "knobs": {
            "denylist_other": knobs.denylist_other,
            "guard_parent": knobs.guard_parent,
            "threshold": None if knobs.threshold is None else json_float(knobs.threshold),
        },
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
        raise ProtocolError("e10 conditional-premium-guard output exists; refuse overwrite")
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
    for key, expected in (
        ("dev_inputs_sha256", EXPECTED_DEV_INPUTS_SHA256),
        ("dev_outcomes_sha256", EXPECTED_DEV_OUTCOMES_SHA256),
        ("train_inputs_sha256", EXPECTED_TRAIN_INPUTS_SHA256),
        ("train_outcomes_sha256", EXPECTED_TRAIN_OUTCOMES_SHA256),
    ):
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
            abs(float(pin_official["final_score"]) - PINNED_DEV_FINAL_SCORE) <= PIN_TOLERANCE
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
    actual_max = float(thresholds["residual_premium_actual_max"])
    inflated_max = float(thresholds["residual_premium_inflated_max"])
    gate: dict[str, Any] = {"arms": {}, "pin": pin}
    ranked: list[str] = []
    for arm in CANDIDATE_ARMS:
        train_score = float(results[arm]["train"]["final_score"])
        dev_score = float(results[arm]["dev"]["final_score"])
        train_delta = train_score - baseline_train
        dev_delta = dev_score - baseline_dev
        official_identical = all(
            selections[arm][label][tier] == selections[BASELINE_ARM][label][tier]
            for label in ("train", "dev")
            for tier in TIERS
        )
        fast_bal_identical = all(
            selections[arm][label][tier] == selections[BASELINE_ARM][label][tier]
            for label in ("train", "dev")
            for tier in ("fast", "balanced")
        )
        official_fail = [
            {"split": label, "tier": tier, "budget_ratio": row["budget_ratio"]}
            for label in ("train", "dev")
            for tier, row in results[arm][label]["official"].items()
            if (not row["budget_passed"])
            or float(row["budget_ratio"]) >= NEAR_FRAC * float(OFFICIAL_CAPS[tier]) - 1e-15
        ]
        premium_view_fail = list(results[arm]["train"]["premium_view_failures"]) + list(
            results[arm]["dev"]["premium_view_failures"]
        )
        residual_rows = []
        residual_ok = True
        residual_improved = True
        for label in ("train", "dev"):
            base = results[BASELINE_ARM][label]["residual_premium"]
            cand = results[arm][label]["residual_premium"]
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
        if not official_identical:
            failures.append("official_identity")
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
        gate["arms"][arm] = {
            "dev_delta": json_float(dev_delta),
            "dev_score": json_float(dev_score),
            "failures": failures,
            "fast_balanced_identical": bool(fast_bal_identical),
            "knobs": dict(results[arm]["train"]["knobs"]),
            "official_failures": official_fail,
            "official_identical": bool(official_identical),
            "passed": passed,
            "premium_view_failures": premium_view_fail,
            "residual_improved": bool(residual_improved),
            "residual_ok": bool(residual_ok),
            "residual_rows": residual_rows,
            "train_delta": json_float(train_delta),
            "train_score": json_float(train_score),
        }
        if passed:
            ranked.append(arm)
    if ranked:
        winner = ranked[0]
        decision = str(protocol["decisions"]["pass"])
        reason = f"Promotion window opens for {winner}."
        gate["window_arm"] = winner
    else:
        decision = str(protocol["decisions"]["fail"])
        reason = str(protocol["decision_reasons"]["fail"])
        gate["window_arm"] = None
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
                        arm: {tier: selections[arm][label][tier][index] for tier in TIERS}
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
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
    report, _audit = assemble(payload, digest, output=output, audit_output=audit_output)
    return report
