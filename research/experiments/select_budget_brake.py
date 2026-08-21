# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the frozen budget-brake selection on Train and report Dev veto.

A fixed promotion count does not bound spend: the same N expensive K1 rows
can push an inflated Premium ratio over the official cap. The overlay keeps
the family-guard Fast and Balanced paths, then promotes parent ax31 rows
in predicted Q_K order while the batch predicted Premium ratio stays at
or under 3.25. Constants are frozen. Dev is scored once and has a veto,
not a vote.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ossp_router import budget_brake_router, family_guard_router
from ossp_router.protocol import MODEL_IDS, TIERS, load_input, load_outcomes
from research.lab.modeling import OFFICIAL_CAPS, load_train, official_score
from research.lab.validation import public_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "select-budget-brake"
DEV_INPUTS = ROOT / "data" / "materialized" / "dev" / "inputs.json"
DEV_OUTCOMES = ROOT / "data" / "dev" / "outcomes.json"
_AX31 = MODEL_IDS[1]
_K1 = MODEL_IDS[2]

LOCKED = {
    "brake_ratio": 3.25,
    "count_cap": 48,
    "denylist_families": [
        "korean_reasoning",
        "python_program",
        "rule_reasoning",
    ],
    "runaway_absolute": 0.17152750745633214,
    "runaway_light_fraction": 0.02,
    "train_full_pred_light": 8.576375372816607,
    "count_rule_worst_adaptive_inflated": {
        "16": 3.5370,
        "24": 3.7252,
        "32": 3.7792,
        "48": 3.8928,
        "64": 4.5011,
    },
}


def _models(plan: Any) -> tuple[str, ...]:
    return tuple(decision.model_id for decision in plan.submission.decisions)


def _promotion_record(
    parent_models: Sequence[str],
    selected_models: Sequence[str],
    scores: np.ndarray,
) -> dict[str, Any]:
    wins = 0
    ties = 0
    losses = 0
    n_k1 = 0
    for index, (parent, chosen) in enumerate(zip(parent_models, selected_models)):
        if chosen == _K1:
            n_k1 += 1
        if parent == chosen:
            continue
        if parent != _AX31 or chosen != _K1:
            raise RuntimeError("overlay changed a row that is not ax31 to axk1-think")
        left = float(scores[index, 1])
        right = float(scores[index, 2])
        if right > left:
            wins += 1
        elif right == left:
            ties += 1
        else:
            losses += 1
    return {
        "n_k1": int(n_k1),
        "n_promoted": int(wins + ties + losses),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
    }


def _tier_view(official: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for tier in TIERS:
        item = official["tiers"][tier]
        rows[tier] = {
            "budget_passed": bool(item["budget_passed"]),
            "near_budget": bool(item["near_budget"]),
            "quality_score": item["quality_score"],
            "budget_ratio": item["budget_ratio"],
            "tier_score": item["tier_score"],
        }
    return {
        "final_score": official["final_score"],
        "tiers": rows,
    }


def _score_router(
    label: str,
    inputs: Any,
    outcomes: Any,
    policy: Any,
    scores: np.ndarray,
    *,
    parent_artifact: family_guard_router.GuardedArtifact,
    brake_artifact: budget_brake_router.BrakeArtifact,
) -> dict[str, Any]:
    parent_plans = {
        tier: family_guard_router.make_submission(inputs, policy, parent_artifact, tier)
        for tier in TIERS
    }
    brake_plans = {
        tier: budget_brake_router.make_submission(inputs, policy, brake_artifact, tier)
        for tier in TIERS
    }
    for tier in ("fast", "balanced"):
        if _models(parent_plans[tier]) != _models(brake_plans[tier]):
            raise RuntimeError(f"{label} {tier} drifted from family_guard_router")
    parent_official = official_score(
        inputs,
        outcomes,
        policy,
        {tier: parent_plans[tier].submission for tier in TIERS},
    )
    brake_official = official_score(
        inputs,
        outcomes,
        policy,
        {tier: brake_plans[tier].submission for tier in TIERS},
    )
    promotion = _promotion_record(
        _models(parent_plans["premium"]),
        _models(brake_plans["premium"]),
        scores,
    )
    return {
        "parent": _tier_view(parent_official),
        "brake": _tier_view(brake_official),
        "promotion": promotion,
    }


def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading", flush=True)
    bundle = load_train(None)
    policy = bundle.policy
    parent_artifact = family_guard_router.load_bundled_artifact()
    brake_artifact = budget_brake_router.load_bundled_artifact()
    block = brake_artifact.budget_brake
    for key in (
        "brake_ratio",
        "count_cap",
        "runaway_absolute",
        "runaway_light_fraction",
        "train_full_pred_light",
    ):
        if float(block[key]) != float(LOCKED[key]):
            raise RuntimeError(f"frozen {key} drifted: {block[key]!r}")
    if list(block["denylist_families"]) != LOCKED["denylist_families"]:
        raise RuntimeError("frozen denylist drifted")

    print("\n=== Stage A: Train evidence ===", flush=True)
    train = _score_router(
        "train",
        bundle.inputs,
        bundle.outcomes,
        policy,
        bundle.scores,
        parent_artifact=parent_artifact,
        brake_artifact=brake_artifact,
    )
    print(
        f"train parent={train['parent']['final_score']} "
        f"brake={train['brake']['final_score']} "
        f"n_k1={train['promotion']['n_k1']} "
        f"promoted={train['promotion']['n_promoted']} "
        f"win/tie/loss="
        f"{train['promotion']['wins']}/{train['promotion']['ties']}/"
        f"{train['promotion']['losses']}",
        flush=True,
    )
    for tier in TIERS:
        row = train["brake"]["tiers"][tier]
        print(
            f"  {tier:9} budget_passed={row['budget_passed']} "
            f"near_budget={row['near_budget']} "
            f"quality={row['quality_score']} "
            f"budget_ratio={row['budget_ratio']}",
            flush=True,
        )

    print("\n=== Stage B: Dev veto (once) ===", flush=True)
    dev_inputs = load_input(DEV_INPUTS)
    dev_outcomes = load_outcomes(DEV_OUTCOMES)
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)
    dev = _score_router(
        "dev",
        dev_inputs,
        dev_outcomes,
        policy,
        np.asarray(arrays.scores),
        parent_artifact=parent_artifact,
        brake_artifact=brake_artifact,
    )
    print(
        f"dev parent={dev['parent']['final_score']} "
        f"brake={dev['brake']['final_score']} "
        f"n_k1={dev['promotion']['n_k1']} "
        f"promoted={dev['promotion']['n_promoted']} "
        f"win/tie/loss="
        f"{dev['promotion']['wins']}/{dev['promotion']['ties']}/"
        f"{dev['promotion']['losses']}",
        flush=True,
    )
    for tier in TIERS:
        row = dev["brake"]["tiers"][tier]
        print(
            f"  {tier:9} budget_passed={row['budget_passed']} "
            f"near_budget={row['near_budget']} "
            f"quality={row['quality_score']} "
            f"budget_ratio={row['budget_ratio']}",
            flush=True,
        )

    budget_ok = all(
        bool(split["brake"]["tiers"][tier]["budget_passed"])
        and (not bool(split["brake"]["tiers"][tier]["near_budget"]))
        for split in (train, dev)
        for tier in TIERS
    )
    veto_ok = (
        budget_ok
        and float(dev["brake"]["final_score"]) > float(dev["parent"]["final_score"])
    )
    print(
        f"\nDev veto budget_ok={budget_ok} score_up={veto_ok} "
        f"count_rule_n64_over_cap={LOCKED['count_rule_worst_adaptive_inflated']['64']}",
        flush=True,
    )

    report = {
        "experiment": "select budget brake",
        "decision": (
            "record-budget-brake-promote-predicted-ratio"
            if veto_ok
            else "record-budget-brake-retain-family-guard"
        ),
        "accepted": bool(veto_ok),
        "protocol": {
            "stage_a": "Train official score of the frozen overlay versus the family guard",
            "stage_b": "Dev scored once; veto if a tier fails budget or the score falls",
            "locked": LOCKED,
        },
        "train": train,
        "dev": dev,
        "official_caps": {tier: float(OFFICIAL_CAPS[tier]) for tier in TIERS},
        "timing_s": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("DONE", report["decision"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
