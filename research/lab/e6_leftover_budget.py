# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E6 — leftover-budget recovery on the frozen serving path.

Quality heads stay shipped. The only knobs are Fast/Balanced predicted
caps (composition-conditioned or static) and the Premium brake ratio.
Runaway, family-guard multipliers, count cap and the ExtraTrees forest
stay frozen. Train official score selects; Dev is a safety/regression veto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

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
    ServingReplica,
    SplitReplica,
    TIER_WEIGHTS,
    TVBALL_EPSILON,
    VIEW_MIN_N,
    composition_views,
    conditioned_balanced_cap,
    conditioned_fast_cap,
    family_views,
    json_float,
    model_counts,
    official_tier_block,
    residual_fraction,
    score_models,
    tvball_worst,
)


EXPERIMENT = "e6-leftover-budget-v1"
REPORT_TYPE = "scrooge-e6-leftover-budget-v1"
SCHEMA_VERSION = 1
BASELINE_ARM = "shipped"
CANDIDATE_ARMS: Tuple[str, ...] = (
    "cond-residual-fast",
    "cond-residual-both",
    "static-fast-1.13",
    "premium-brake-3.40",
    "premium-brake-3.55",
)
ARMS: Tuple[str, ...] = (BASELINE_ARM,) + CANDIDATE_ARMS
AUDIT_RELATIVE = "build/run-e6-leftover-budget/episode-audit.json"
REPORT_RELATIVE = "build/run-e6-leftover-budget/report.json"


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
    if tuple(protocol["arms"]["candidates"]) != CANDIDATE_ARMS:
        raise ProtocolError("sealed candidate arm list drifted")
    if protocol["arms"]["baseline"] != BASELINE_ARM:
        raise ProtocolError("baseline arm drifted")
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


def arm_knobs(
    replica: ServingReplica,
    arm: str,
    families: Sequence[str],
) -> Tuple[dict[str, float], float]:
    caps = dict(replica.shipped_caps)
    brake = replica.shipped_brake_ratio
    fraction = residual_fraction(families)
    if arm == BASELINE_ARM:
        return caps, brake
    if arm == "cond-residual-fast":
        caps["fast"] = conditioned_fast_cap(fraction)
        return caps, brake
    if arm == "cond-residual-both":
        caps["fast"] = conditioned_fast_cap(fraction)
        caps["balanced"] = conditioned_balanced_cap(fraction)
        return caps, brake
    if arm == "static-fast-1.13":
        caps["fast"] = 1.13
        return caps, brake
    if arm == "premium-brake-3.40":
        return caps, 3.40
    if arm == "premium-brake-3.55":
        return caps, 3.55
    raise ProtocolError(f"unknown arm {arm}")


def evaluate_arm_on_split(
    replica: ServingReplica,
    split: SplitReplica,
    arm: str,
) -> dict[str, Any]:
    full = list(range(split.n))
    caps, brake = arm_knobs(replica, arm, split.families)
    selections = replica.allocate_all(split, full, caps=caps, brake_ratio=brake)
    official = replica.official(split, selections)
    views = composition_views(split)
    stress: dict[str, Any] = {}
    safety_failures = []
    for name, indexes in views.items():
        view_families = [split.families[index] for index in indexes]
        view_caps, view_brake = arm_knobs(replica, arm, view_families)
        view_sel = replica.allocate_all(
            split, indexes, caps=view_caps, brake_ratio=view_brake
        )
        stress[name] = {"caps": dict(view_caps), "brake_ratio": float(view_brake)}
        for tier in TIERS:
            scored = score_models(split.scores, split.costs, indexes, view_sel[tier])
            inflated = float(scored["actual_ratio"]) * INFLATION
            cap = float(OFFICIAL_CAPS[tier])
            ruin = bool(float(scored["actual_ratio"]) > cap + 1e-15)
            ruin_inflated = bool(inflated > cap + 1e-15)
            stress[name][tier] = {
                "actual_ratio": scored["actual_ratio"],
                "inflated_ratio": json_float(inflated),
                "n": scored["n"],
                "quality": scored["quality"],
                "ruin": ruin,
                "ruin_inflated": ruin_inflated,
            }
            if ruin or ruin_inflated:
                safety_failures.append(
                    {
                        "actual_ratio": scored["actual_ratio"],
                        "inflated_ratio": json_float(inflated),
                        "tier": tier,
                        "view": name,
                    }
                )
    family_deltas = []
    family_rows = {}
    for family, indexes in family_views(split.families).items():
        qualities = {}
        for tier in TIERS:
            models = [selections[tier][index] for index in indexes]
            qualities[tier] = float(
                score_models(split.scores, split.costs, indexes, models)["quality"]
            )
        weighted = (
            TIER_WEIGHTS["fast"] * qualities["fast"]
            + TIER_WEIGHTS["balanced"] * qualities["balanced"]
            + TIER_WEIGHTS["premium"] * qualities["premium"]
        )
        family_rows[family] = {
            "n": int(len(indexes)),
            "weighted_quality": json_float(weighted),
        }
        family_deltas.append(weighted)
    return {
        "caps": dict(caps),
        "brake_ratio": float(brake),
        "counts": {tier: model_counts(selections[tier]) for tier in TIERS},
        "families": family_rows,
        "final_score": float(official["final_score"]),
        "official": {tier: official_tier_block(official, tier) for tier in TIERS},
        "residual_fraction": json_float(split.residual_frac),
        "safety_failures": safety_failures,
        "selections": {tier: list(selections[tier]) for tier in TIERS},
        "stress": stress,
        "weighted_quality_by_family": family_rows,
    }


def family_delta_tvball(
    baseline_families: Mapping[str, Mapping[str, Any]],
    candidate_families: Mapping[str, Mapping[str, Any]],
    pooled_delta: float,
) -> float:
    deltas = []
    for family, row in candidate_families.items():
        if family not in baseline_families:
            continue
        if int(row["n"]) < VIEW_MIN_N:
            continue
        deltas.append(
            float(row["weighted_quality"])
            - float(baseline_families[family]["weighted_quality"])
        )
    return tvball_worst(pooled_delta, deltas)


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("e6 leftover-budget output exists; refuse overwrite")

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

    shipped_runtime = {
        tier: replica.runtime_models(dev, tier) for tier in TIERS
    }
    shipped_replica = replica.allocate_all(
        dev, list(range(dev.n)), caps=replica.shipped_caps
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

    baseline_train = float(results[BASELINE_ARM]["train"]["final_score"])
    baseline_dev = float(results[BASELINE_ARM]["dev"]["final_score"])
    thresholds = protocol["thresholds"]
    gate: dict[str, Any] = {"arms": {}, "pin": pin}
    ranked: list[tuple[float, str]] = []

    for arm in CANDIDATE_ARMS:
        train_score = float(results[arm]["train"]["final_score"])
        dev_score = float(results[arm]["dev"]["final_score"])
        train_delta = train_score - baseline_train
        dev_delta = dev_score - baseline_dev
        tvball = family_delta_tvball(
            results[BASELINE_ARM]["train"]["families"],
            results[arm]["train"]["families"],
            train_delta,
        )
        official_fail = [
            {"split": label, "tier": tier, "budget_ratio": row["budget_ratio"]}
            for label in ("train", "dev")
            for tier, row in results[arm][label]["official"].items()
            if (not row["budget_passed"])
            or float(row["budget_ratio"]) >= NEAR_FRAC * float(OFFICIAL_CAPS[tier]) - 1e-15
        ]
        view_fail = list(results[arm]["train"]["safety_failures"]) + list(
            results[arm]["dev"]["safety_failures"]
        )
        failures = []
        if train_delta < float(thresholds["train_delta_min"]):
            failures.append("train_delta")
        if dev_delta <= float(thresholds["dev_delta_min_exclusive"]):
            failures.append("dev_veto")
        if tvball < float(thresholds["tvball_worst_min"]):
            failures.append("tvball")
        if official_fail:
            failures.append("official_safety")
        if view_fail:
            failures.append("view_safety")
        passed = not failures
        gate["arms"][arm] = {
            "dev_delta": json_float(dev_delta),
            "dev_score": json_float(dev_score),
            "failures": failures,
            "official_failures": official_fail,
            "passed": passed,
            "train_delta": json_float(train_delta),
            "train_score": json_float(train_score),
            "tvball_worst": json_float(tvball),
            "view_failures": view_fail,
        }
        if passed:
            ranked.append((train_delta, arm))

    if ranked:
        ranked.sort(key=lambda item: (-item[0], CANDIDATE_ARMS.index(item[1])))
        winner = ranked[0][1]
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
        "epsilon": TVBALL_EPSILON,
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
    import json

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
    import json

    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("protocol is not a JSON object")
    digest = verify_protocol(payload, expected_protocol_sha256)
    report, _audit = assemble(
        payload, digest, output=output, audit_output=audit_output
    )
    return report
