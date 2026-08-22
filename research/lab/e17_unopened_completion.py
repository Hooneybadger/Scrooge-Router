# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E17 — named arms that were registered and then left unopened.

E13/E14 refused Fast 1.08. E14 refused official-binding 0.25 and left
three-family batches as a later leftover. E10 underbind 0.73 was not
invented. E9-always plus the shipped Fast switches was never stacked.
The original Fast grid hole 1.10 was never run with always-on E9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

from ossp_router import budget_brake_router
from ossp_router.feasibility_ladder import select_fast_balanced
from ossp_router.protocol import TIERS, load_input, load_outcomes, policy_sha256
from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.e15_unified_guard import reprice_fast_balanced
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
    RESIDUAL_FAMILY,
    SHIPPED_FAST_CAP,
    ProtocolError,
    ServingReplica,
    SplitReplica,
    composition_views,
    json_float,
    max_family_fraction,
    model_counts,
    official_tier_block,
    pair_family_views,
    residual_fraction,
    score_models,
    top2_family_fraction,
    top3_family_fraction,
    triple_family_views,
)


EXPERIMENT = "e17-unopened-completion-v1"
REPORT_TYPE = "scrooge-e17-unopened-completion-v1"
SCHEMA_VERSION = 1
BASELINE_ARM = "shipped"
CANDIDATE_ARMS: Tuple[str, ...] = (
    "e9-keep-e14",
    "e9-keep-e13-e14",
    "e9-keep-e13",
    "e9-keep-e14-0.50",
    "cond-fast-1.08-0.75",
    "cond-fast-1.08-0.50",
    "cond-fast-1.08-0.25",
    "cond-top2-1.08-0.75",
    "cond-top2-1.08-0.50",
    "cond-top2-1.07-0.25",
    "cond-top3-1.07-0.75",
    "cond-top3-1.07-0.50",
    "cond-top3-1.08-0.75",
    "cond-top3-1.05-0.75",
    "leftover-e10-0.73",
    "e9-fast-1.08-word-1.50",
    "e9-fast-1.08-word-2.00",
    "unify-e9-fast-1.10",
)
LEFTOVER_FAST_FAMILIES: Tuple[str, ...] = (
    "english_multiple_choice",
    "word_problem",
)
LOWEST_TOP3_THRESHOLD = 0.50
SHIPPED_E13_THRESHOLD = 0.75
SHIPPED_E13_CAP = float(budget_brake_router.CONDITIONAL_FAST_CAP)


class ArmKnobs(NamedTuple):
    fast_mode: str
    fast_cap: Optional[float]
    threshold: Optional[float]
    unify_premium: bool
    e10_threshold: Optional[float]
    leftover_fast_mult: Optional[float]


ARM_KNOBS: Mapping[str, ArmKnobs] = {
    BASELINE_ARM: ArmKnobs("shipped", None, None, False, None, None),
    "e9-keep-e14": ArmKnobs("e14-only", 1.07, 0.75, True, None, None),
    "e9-keep-e13-e14": ArmKnobs("e13-e14", 1.07, 0.75, True, None, None),
    "e9-keep-e13": ArmKnobs("e13-only", 1.07, 0.75, True, None, None),
    "e9-keep-e14-0.50": ArmKnobs("e14-only", 1.07, 0.50, True, None, None),
    "cond-fast-1.08-0.75": ArmKnobs("e13", 1.08, 0.75, False, None, None),
    "cond-fast-1.08-0.50": ArmKnobs("e13", 1.08, 0.50, False, None, None),
    "cond-fast-1.08-0.25": ArmKnobs("e13", 1.08, 0.25, False, None, None),
    "cond-top2-1.08-0.75": ArmKnobs("e14", 1.08, 0.75, False, None, None),
    "cond-top2-1.08-0.50": ArmKnobs("e14", 1.08, 0.50, False, None, None),
    "cond-top2-1.07-0.25": ArmKnobs("e14", 1.07, 0.25, False, None, None),
    "cond-top3-1.07-0.75": ArmKnobs("top3", 1.07, 0.75, False, None, None),
    "cond-top3-1.07-0.50": ArmKnobs("top3", 1.07, 0.50, False, None, None),
    "cond-top3-1.08-0.75": ArmKnobs("top3", 1.08, 0.75, False, None, None),
    "cond-top3-1.05-0.75": ArmKnobs("top3", 1.05, 0.75, False, None, None),
    "leftover-e10-0.73": ArmKnobs("shipped", None, None, False, 0.73, None),
    "e9-fast-1.08-word-1.50": ArmKnobs("global", 1.08, None, True, None, 1.50),
    "e9-fast-1.08-word-2.00": ArmKnobs("global", 1.08, None, True, None, 2.00),
    "unify-e9-fast-1.10": ArmKnobs("global", 1.10, None, True, None, None),
}
ARMS: Tuple[str, ...] = (BASELINE_ARM,) + CANDIDATE_ARMS
AUDIT_RELATIVE = "build/run-e17-unopened-completion/episode-audit.json"
REPORT_RELATIVE = "build/run-e17-unopened-completion/report.json"
TOP3_MODES = frozenset({"top3"})


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(dict(protocol)))


def _optional_float(left: Any, right: Optional[float], *, name: str, arm: str) -> None:
    if right is None:
        if left is not None:
            raise ProtocolError(f"sealed {name} for {arm} drifted")
        return
    if left is None or abs(float(left) - float(right)) > 1e-15:
        raise ProtocolError(f"sealed {name} for {arm} drifted")


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
    if tuple(protocol["arms"]["leftover_fast_families"]) != LEFTOVER_FAST_FAMILIES:
        raise ProtocolError("leftover Fast families drifted")
    sealed = protocol["arms"]["knobs"]
    if set(sealed) != set(CANDIDATE_ARMS):
        raise ProtocolError("sealed knob keys drifted")
    for arm in CANDIDATE_ARMS:
        row = sealed[arm]
        expected = ARM_KNOBS[arm]
        if str(row["fast_mode"]) != expected.fast_mode:
            raise ProtocolError(f"sealed fast_mode for {arm} drifted")
        _optional_float(row.get("fast_cap"), expected.fast_cap, name="fast_cap", arm=arm)
        _optional_float(row.get("threshold"), expected.threshold, name="threshold", arm=arm)
        _optional_float(
            row.get("e10_threshold"), expected.e10_threshold, name="e10_threshold", arm=arm
        )
        _optional_float(
            row.get("leftover_fast_mult"),
            expected.leftover_fast_mult,
            name="leftover_fast_mult",
            arm=arm,
        )
        if bool(row["unify_premium"]) != expected.unify_premium:
            raise ProtocolError(f"sealed unify_premium for {arm} drifted")
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


def leftover_fast_extras(arm: str) -> Mapping[str, float]:
    knobs = arm_knobs(arm)
    if knobs.leftover_fast_mult is None:
        return {}
    return {family: float(knobs.leftover_fast_mult) for family in LEFTOVER_FAST_FAMILIES}


def shipped_e13_cap(families: Sequence[str]) -> float:
    if max_family_fraction(families) + 1e-15 >= SHIPPED_E13_THRESHOLD:
        return SHIPPED_E13_CAP
    return float(SHIPPED_FAST_CAP)


def resolve_fast_cap(families: Sequence[str], arm: str) -> float:
    knobs = arm_knobs(arm)
    mode = knobs.fast_mode
    max_frac = max_family_fraction(families)
    top2 = top2_family_fraction(families)
    top3 = top3_family_fraction(families)
    if mode == "shipped" or mode == "e13-only":
        return shipped_e13_cap(families)
    if mode == "global":
        if knobs.fast_cap is None:
            raise ProtocolError(f"{arm} is missing a Fast cap")
        return float(knobs.fast_cap)
    if mode == "e13":
        if knobs.fast_cap is None or knobs.threshold is None:
            raise ProtocolError(f"{arm} is missing e13 knobs")
        if max_frac + 1e-15 >= float(knobs.threshold):
            return float(knobs.fast_cap)
        return float(SHIPPED_FAST_CAP)
    if mode == "e14":
        if knobs.fast_cap is None or knobs.threshold is None:
            raise ProtocolError(f"{arm} is missing e14 knobs")
        if top2 + 1e-15 >= float(knobs.threshold):
            return float(knobs.fast_cap)
        return shipped_e13_cap(families)
    if mode == "e14-only":
        if knobs.fast_cap is None or knobs.threshold is None:
            raise ProtocolError(f"{arm} is missing e14-only knobs")
        if top2 + 1e-15 >= float(knobs.threshold):
            return float(knobs.fast_cap)
        return float(SHIPPED_FAST_CAP)
    if mode == "e13-e14":
        if top2 + 1e-15 >= SHIPPED_E13_THRESHOLD:
            return SHIPPED_E13_CAP
        return float(SHIPPED_FAST_CAP)
    if mode == "top3":
        if knobs.fast_cap is None or knobs.threshold is None:
            raise ProtocolError(f"{arm} is missing top3 knobs")
        if top3 + 1e-15 >= float(knobs.threshold):
            return float(knobs.fast_cap)
        return shipped_e13_cap(families)
    raise ProtocolError(f"unknown fast_mode {mode}")


def arm_caps(
    replica: ServingReplica,
    families: Sequence[str],
    arm: str,
) -> dict[str, float]:
    caps = dict(replica.shipped_caps)
    caps["fast"] = resolve_fast_cap(families, arm)
    return caps


def premium_guard_active(families: Sequence[str], arm: str) -> bool:
    knobs = arm_knobs(arm)
    if knobs.unify_premium:
        return True
    threshold = (
        float(knobs.e10_threshold)
        if knobs.e10_threshold is not None
        else float(budget_brake_router.CONDITIONAL_PREMIUM_RESIDUAL_THRESHOLD)
    )
    return residual_fraction(families) + 1e-15 >= threshold


def allocate_arm(
    replica: ServingReplica,
    split: SplitReplica,
    indexes: Sequence[int],
    arm: str,
) -> dict[str, Tuple[str, ...]]:
    families = [split.families[index] for index in indexes]
    caps = arm_caps(replica, families, arm)
    extras = leftover_fast_extras(arm)
    active = premium_guard_active(families, arm)
    if extras:
        predictions = reprice_fast_balanced(split, indexes, extras)
        selected: dict[str, Tuple[str, ...]] = {}
        for tier, cap in (("fast", caps["fast"]), ("balanced", caps["balanced"])):
            chosen, _ratio = select_fast_balanced(
                predictions,
                cap=float(cap),
                runaway_fraction=replica.runaway_fraction,
                max_upgrade_fraction=replica.max_upgrade_fraction,
            )
            selected[tier] = chosen
        selected["premium"] = replica.allocate_premium(
            split,
            indexes,
            guard_parent=True,
            denylist_extra=(RESIDUAL_FAMILY,),
        )
        return selected
    return replica.allocate_all(
        split,
        indexes,
        caps=caps,
        guard_parent=active,
        denylist_extra=(RESIDUAL_FAMILY,) if active else (),
    )


def binding_views(split: SplitReplica) -> dict[str, Tuple[int, ...]]:
    views = dict(composition_views(split))
    views.update(pair_family_views(split.families, split.digests))
    return views


def _score_tier(split: SplitReplica, indexes: Sequence[int], models: Sequence[str]) -> dict[str, Any]:
    scored = score_models(split.scores, split.costs, indexes, models)
    inflated = float(scored["actual_ratio"]) * INFLATION
    official_cap = float(OFFICIAL_CAPS["fast"])
    return {
        "actual_ratio": scored["actual_ratio"],
        "counts": scored["counts"],
        "inflated_ratio": json_float(inflated),
        "n": scored["n"],
        "quality": scored["quality"],
        "ruin": bool(float(scored["actual_ratio"]) > official_cap + 1e-15),
        "ruin_inflated": bool(inflated > official_cap + 1e-15),
    }


def evaluate_arm_on_split(
    replica: ServingReplica, split: SplitReplica, arm: str
) -> dict[str, Any]:
    full = list(range(split.n))
    selections = allocate_arm(replica, split, full, arm)
    official = replica.official(split, selections)
    views = binding_views(split)
    stress: dict[str, Any] = {}
    fast_view_failures = []
    premium_residual_failures = []
    knobs = arm_knobs(arm)
    for name, indexes in views.items():
        view_sel = allocate_arm(replica, split, indexes, arm)
        families = [split.families[index] for index in indexes]
        caps = arm_caps(replica, families, arm)
        stress[name] = {
            "caps": dict(caps),
            "max_family_fraction": json_float(max_family_fraction(families)),
            "residual_fraction": json_float(residual_fraction(families)),
            "top2_family_fraction": json_float(top2_family_fraction(families)),
            "top3_family_fraction": json_float(top3_family_fraction(families)),
        }
        for tier in TIERS:
            scored = score_models(split.scores, split.costs, indexes, view_sel[tier])
            inflated = float(scored["actual_ratio"]) * INFLATION
            official_cap = float(OFFICIAL_CAPS[tier])
            ruin = bool(float(scored["actual_ratio"]) > official_cap + 1e-15)
            ruin_inflated = bool(inflated > official_cap + 1e-15)
            stress[name][tier] = {
                "actual_ratio": scored["actual_ratio"],
                "counts": scored["counts"],
                "inflated_ratio": json_float(inflated),
                "n": scored["n"],
                "quality": scored["quality"],
                "ruin": ruin,
                "ruin_inflated": ruin_inflated,
            }
            if tier == "fast" and (ruin or ruin_inflated):
                fast_view_failures.append(
                    {
                        "actual_ratio": scored["actual_ratio"],
                        "inflated_ratio": json_float(inflated),
                        "max_family_fraction": json_float(max_family_fraction(families)),
                        "top2_family_fraction": json_float(top2_family_fraction(families)),
                        "top3_family_fraction": json_float(top3_family_fraction(families)),
                        "view": name,
                    }
                )
            if (
                tier == "premium"
                and name == f"family:{RESIDUAL_FAMILY}"
                and (ruin or ruin_inflated)
            ):
                premium_residual_failures.append(
                    {
                        "actual_ratio": scored["actual_ratio"],
                        "inflated_ratio": json_float(inflated),
                        "view": name,
                    }
                )
    triple_failures = []
    for name, indexes in triple_family_views(split.families, split.digests).items():
        view_sel = allocate_arm(replica, split, indexes, arm)
        families = [split.families[index] for index in indexes]
        scored = _score_tier(split, indexes, view_sel["fast"])
        if scored["ruin"] or scored["ruin_inflated"]:
            triple_failures.append(
                {
                    "actual_ratio": scored["actual_ratio"],
                    "inflated_ratio": scored["inflated_ratio"],
                    "max_family_fraction": json_float(max_family_fraction(families)),
                    "top2_family_fraction": json_float(top2_family_fraction(families)),
                    "top3_family_fraction": json_float(top3_family_fraction(families)),
                    "view": name,
                }
            )
    residual_indexes = [
        index for index, family in enumerate(split.families) if family == RESIDUAL_FAMILY
    ]
    residual_models = [selections["premium"][index] for index in residual_indexes]
    residual_official = score_models(
        split.scores, split.costs, residual_indexes, residual_models
    )
    residual_name = f"family:{RESIDUAL_FAMILY}"
    residual_view = None
    if residual_name in stress:
        residual_view = {
            "actual_ratio": stress[residual_name]["premium"]["actual_ratio"],
            "inflated_ratio": stress[residual_name]["premium"]["inflated_ratio"],
            "n": stress[residual_name]["premium"]["n"],
        }
    return {
        "caps": dict(arm_caps(replica, split.families, arm)),
        "counts": {tier: model_counts(selections[tier]) for tier in TIERS},
        "fast_view_failures": fast_view_failures,
        "final_score": float(official["final_score"]),
        "knobs": {
            "e10_threshold": None
            if knobs.e10_threshold is None
            else json_float(knobs.e10_threshold),
            "fast_cap": None if knobs.fast_cap is None else json_float(knobs.fast_cap),
            "fast_mode": knobs.fast_mode,
            "leftover_fast_mult": None
            if knobs.leftover_fast_mult is None
            else json_float(knobs.leftover_fast_mult),
            "threshold": None if knobs.threshold is None else json_float(knobs.threshold),
            "unify_premium": knobs.unify_premium,
        },
        "official": {tier: official_tier_block(official, tier) for tier in TIERS},
        "official_residual_subgroup": {
            "actual_ratio": residual_official["actual_ratio"],
            "counts": residual_official["counts"],
            "inflated_ratio": json_float(float(residual_official["actual_ratio"]) * INFLATION),
            "n": residual_official["n"],
        },
        "premium_residual_failures": premium_residual_failures,
        "residual_fraction": json_float(split.residual_frac),
        "residual_only_premium": residual_view,
        "selections": {tier: list(selections[tier]) for tier in TIERS},
        "stress": stress,
        "top3_family_fraction": json_float(top3_family_fraction(split.families)),
        "triple_view_failures": triple_failures,
    }


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("e17 unopened-completion output exists; refuse overwrite")
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
    for label, split in (("train", train), ("dev", dev)):
        top3 = top3_family_fraction(split.families)
        if top3 + 1e-15 >= LOWEST_TOP3_THRESHOLD:
            raise ProtocolError(
                f"official {label} top-3 {top3:.6f} reached the lowest top-3 threshold"
            )
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
        official_fail = [
            {"split": label, "tier": tier, "budget_ratio": row["budget_ratio"]}
            for label in ("train", "dev")
            for tier, row in results[arm][label]["official"].items()
            if (not row["budget_passed"])
            or float(row["budget_ratio"]) >= NEAR_FRAC * float(OFFICIAL_CAPS[tier]) - 1e-15
        ]
        fast_view_fail = list(results[arm]["train"]["fast_view_failures"]) + list(
            results[arm]["dev"]["fast_view_failures"]
        )
        residual_fail = list(results[arm]["train"]["premium_residual_failures"]) + list(
            results[arm]["dev"]["premium_residual_failures"]
        )
        triple_fail = list(results[arm]["train"]["triple_view_failures"]) + list(
            results[arm]["dev"]["triple_view_failures"]
        )
        failures = []
        if official_fail:
            failures.append("official_safety")
        if fast_view_fail:
            failures.append("fast_view_safety")
        if residual_fail:
            failures.append("residual_premium_cap")
        if arm_knobs(arm).fast_mode in TOP3_MODES and triple_fail:
            failures.append("triple_view_safety")
        if dev_delta <= float(thresholds["dev_delta_min_exclusive"]):
            failures.append("dev_veto")
        passed = not failures
        gate["arms"][arm] = {
            "dev_delta": json_float(dev_delta),
            "dev_score": json_float(dev_score),
            "failures": failures,
            "fast_view_failures": fast_view_fail,
            "knobs": dict(results[arm]["train"]["knobs"]),
            "n_triple_failures": len(triple_fail),
            "official_failures": official_fail,
            "official_identical": bool(official_identical),
            "official_residual_subgroup": {
                label: results[arm][label]["official_residual_subgroup"]
                for label in ("train", "dev")
            },
            "passed": passed,
            "premium_residual_failures": residual_fail,
            "residual_only_premium": {
                label: results[arm][label]["residual_only_premium"]
                for label in ("train", "dev")
            },
            "train_delta": json_float(train_delta),
            "train_score": json_float(train_score),
            "triple_view_failures": triple_fail,
        }
        if passed:
            ranked.append(arm)
    if ranked:
        winner = ranked[0]
        decision = str(protocol["decisions"]["pass"])
        reason = f"Unopened-completion window opens for {winner}."
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
