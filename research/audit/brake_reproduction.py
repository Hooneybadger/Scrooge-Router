# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the submitted budget-brake router on Train and Dev and score it officially."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from ossp_router import budget_brake_router, family_guard_router
from ossp_router.protocol import MODEL_IDS, TIERS, load_input, load_outcomes
from research.lab.modeling import load_train, official_score


ROOT = Path(__file__).resolve().parents[2]
DEV_INPUTS = ROOT / "data" / "materialized" / "dev" / "inputs.json"
DEV_OUTCOMES = ROOT / "data" / "dev" / "outcomes.json"
DEV_FINAL_SCORE = Decimal("0.670710227273")
DEV_N_K1 = 32
DEV_RATIOS = {
    "fast": Decimal("1.093011852072"),
    "balanced": Decimal("1.396000996251"),
    "premium": Decimal("2.315836178068"),
}
_K1 = MODEL_IDS[2]


def _models(plan: Any) -> tuple[str, ...]:
    return tuple(decision.model_id for decision in plan.submission.decisions)


def _gate(name: str, ok: bool) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    return bool(ok)


def _score_split(
    label: str,
    inputs: Any,
    outcomes: Any,
    policy: Any,
    artifact: budget_brake_router.BrakeArtifact,
    parent: family_guard_router.GuardedArtifact,
) -> dict[str, Any]:
    parent_fast = _models(family_guard_router.make_submission(inputs, policy, parent, "fast"))
    parent_balanced = _models(
        family_guard_router.make_submission(inputs, policy, parent, "balanced")
    )
    fast = budget_brake_router.make_submission(inputs, policy, artifact, "fast")
    balanced = budget_brake_router.make_submission(inputs, policy, artifact, "balanced")
    premium = budget_brake_router.make_submission(inputs, policy, artifact, "premium")
    fast_models = _models(fast)
    balanced_models = _models(balanced)
    premium_models = _models(premium)
    if fast_models != parent_fast:
        raise RuntimeError(f"{label} Fast decisions drifted from family_guard_router")
    if balanced_models != parent_balanced:
        raise RuntimeError(f"{label} Balanced decisions drifted from family_guard_router")
    official = official_score(
        inputs,
        outcomes,
        policy,
        {
            "fast": fast.submission,
            "balanced": balanced.submission,
            "premium": premium.submission,
        },
    )
    n_k1 = sum(1 for model_id in premium_models if model_id == _K1)
    tiers = {}
    for tier in TIERS:
        row = official["tiers"][tier]
        tiers[tier] = {
            "budget_passed": bool(row["budget_passed"]),
            "near_budget": bool(row["near_budget"]),
            "quality_score": row["quality_score"],
            "budget_ratio": row["budget_ratio"],
        }
    return {
        "final_score": official["final_score"],
        "n_k1": int(n_k1),
        "tiers": tiers,
    }


def main() -> int:
    bundle = load_train(None)
    policy = bundle.policy
    artifact = budget_brake_router.load_bundled_artifact()
    parent = family_guard_router.load_bundled_artifact()
    dev_inputs = load_input(DEV_INPUTS)
    dev_outcomes = load_outcomes(DEV_OUTCOMES)
    print(
        "loaded budget-brake artifact "
        f"type={artifact.value['artifact_type']} "
        f"n_trees={artifact.budget_brake['forest']['n_trees']}"
    )

    train = _score_split("train", bundle.inputs, bundle.outcomes, policy, artifact, parent)
    dev = _score_split("dev", dev_inputs, dev_outcomes, policy, artifact, parent)

    ok = True
    for label, row in (("train", train), ("dev", dev)):
        print(
            f"{label}: official_final_score={row['final_score']} n_k1={row['n_k1']}"
        )
        for tier in TIERS:
            item = row["tiers"][tier]
            print(
                f"  {tier:9} budget_passed={item['budget_passed']} "
                f"near_budget={item['near_budget']} "
                f"quality={item['quality_score']} "
                f"budget_ratio={item['budget_ratio']}"
            )

    ok = _gate(
        "Dev official final_score is 0.670710227273",
        Decimal(str(dev["final_score"])) == DEV_FINAL_SCORE,
    ) and ok
    ok = _gate("Dev n_k1 is 32", int(dev["n_k1"]) == DEV_N_K1) and ok
    ratio_ok = True
    for tier, expected in DEV_RATIOS.items():
        ratio_ok = ratio_ok and Decimal(str(dev["tiers"][tier]["budget_ratio"])) == expected
    ok = _gate(
        "Dev Fast/Balanced/Premium realized ratios match the locked values",
        ratio_ok,
    ) and ok
    budget_ok = True
    for label, row in (("train", train), ("dev", dev)):
        for tier in TIERS:
            item = row["tiers"][tier]
            budget_ok = (
                budget_ok
                and bool(item["budget_passed"])
                and (not bool(item["near_budget"]))
            )
    ok = _gate(
        "every tier budget_passed true and near_budget false on both splits",
        budget_ok,
    ) and ok
    ok = _gate("Fast/Balanced identical to family_guard_router", True) and ok
    print("ALL_MATCH", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
