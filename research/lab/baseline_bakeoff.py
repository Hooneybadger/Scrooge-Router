# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Formal bake-off of the public baselines against our own routers.

Compares always_light, prompt_heuristic, feature_budget (live Train),
official hash-regex (cited frozen reports, no retrain), and the feasibility ladder as
reference-only. K1 stays off for promotion. Selected stays the feasibility ladder unless a
never-Dev-scored baseline passes the locked gates.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from ossp_router.heuristic import make_submission as heuristic_submission
from ossp_router.protocol import MODEL_IDS, TIERS, InputBatch, RoutingPolicy
from research.lab.modeling import (
    OFFICIAL_CAPS,
    OPERATING_TARGETS,
    STRESS_BACKSTOP,
    official_score,
    sort_mapping,
    weighted_final,
)
from research.lab.prefix_certificates import json_float
from research.lab.cap_certification import LADDER_DEV_WEIGHTED
from research.lab.fast_corridor import LADDER_TRAIN_WEIGHTED


EXPERIMENT = "the baseline bake-off"
REPORT_TYPE = "scrooge-bakeoff-baseline-bakeoff-v1"
SCHEMA_VERSION = 1
DECISION_NO_PROMOTE = "record-bakeoff-close-no-promote"
DECISION_PROMOTE = "record-bakeoff-promote-for-deployment-validation"
DECISION_DEV_REJECT = "record-bakeoff-close-dev-reject"
DECISIONS = (DECISION_NO_PROMOTE, DECISION_PROMOTE, DECISION_DEV_REJECT)
LIVE_POLICIES: Tuple[str, ...] = (
    "always_light",
    "prompt_heuristic",
    "feature_budget",
)
CITED_POLICIES: Tuple[str, ...] = ("hash_regex", "ladder_reference")
ALL_POLICIES: Tuple[str, ...] = LIVE_POLICIES + CITED_POLICIES
_LIGHT = MODEL_IDS[0]
_AX31 = MODEL_IDS[1]
_K1 = MODEL_IDS[2]
ROOT = Path(__file__).resolve().parents[2]
HASH_REGEX_TRAIN_REPORT = ROOT / "build" / "hash-regex" / "train-report.json"
HASH_REGEX_DEV_REPORT = ROOT / "build" / "hash-regex" / "dev-report.json"
CEILING_AUDIT_REPORT = ROOT / "build" / "ceiling-audit" / "report.json"
LADDER_TRAIN_REPORT = ROOT / "build" / "feasibility-ladder" / "report.json"
LADDER_DEV_REPORT = ROOT / "build" / "feasibility-ladder-dev" / "report.json"
ALWAYS_LIGHT_DEV_WEIGHTED = 0.6193181818181818
SELECTION_RULE = (
    "Among live-scored baselines (always_light, prompt_heuristic, "
    "feature_budget) that pass every Train gate and are not reference-only, "
    "maximize Train official weighted. hash_regex and the feasibility ladder are comparison "
    "only: hash_regex is cited from frozen reports (no retrain); the feasibility ladder is "
    "the rollback selected and cannot be selected. Dev opens once only for "
    "a Train survivor that has no existing Dev citation."
)


class Stage2Refused(RuntimeError):
    """Stage 2 must not run when Stage 1 selected nothing."""


def _ensure_baselines_path() -> None:
    baselines = str(ROOT / "baselines")
    if baselines not in sys.path:
        sys.path.insert(0, baselines)


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "always_light_dev_weighted_lock": json_float(ALWAYS_LIGHT_DEV_WEIGHTED),
            "cited_policies": list(CITED_POLICIES),
            "k1": "off",
            "live_policies": list(LIVE_POLICIES),
            "official_caps": dict(OFFICIAL_CAPS),
            "operating_targets": dict(OPERATING_TARGETS),
            "promote_does_not_replace_image": True,
            "selection_rule": SELECTION_RULE,
            "stress_backstop": json_float(STRESS_BACKSTOP),
            "ladder_dev_weighted_lock": json_float(LADDER_DEV_WEIGHTED),
            "ladder_train_weighted_lock": json_float(LADDER_TRAIN_WEIGHTED),
        }
    )


def next_leaf_for(decision: str) -> str:
    if decision not in DECISIONS:
        raise ValueError(f"unknown the baseline bake-off decision {decision!r}")
    return "B110"


def weighted_from_tier_scores(
    fast: float, balanced: float, premium: float
) -> float:
    return weighted_final(float(fast), float(balanced), float(premium))


def official_tier_block(report: Mapping[str, Any], tier: str) -> dict[str, Any]:
    row = report["tiers"][tier]
    counts = row["model_counts"]
    realized = float(row["budget_ratio"])
    return {
        "ax31": int(counts.get(_AX31, 0)),
        "axk1_think": int(counts.get(_K1, 0)),
        "budget_passed": bool(row["budget_passed"]),
        "budget_ratio": row["budget_ratio"],
        "light": int(counts.get(_LIGHT, 0)),
        "near_budget": bool(row["near_budget"]),
        "quality_score": row["quality_score"],
        "realized": row["budget_ratio"],
        "realized_times_1054": json_float(realized * float(STRESS_BACKSTOP)),
        "tier_score": row["tier_score"],
    }


def public_official(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "final_score": str(report["final_score"]),
        "tiers": {tier: official_tier_block(report, tier) for tier in TIERS},
        "weighted_float": json_float(float(report["final_score"])),
    }


def _tier_realized(row: Mapping[str, Any]) -> float:
    if "realized" in row:
        return float(row["realized"])
    return float(row["budget_ratio"])


def _tier_k1(row: Mapping[str, Any]) -> int:
    if row.get("axk1_think") is not None:
        return int(row["axk1_think"])
    return int(row.get("model_counts", {}).get(_K1, 0))


def evaluate_train_gates(
    *,
    train_weighted: float,
    official: Mapping[str, Any],
    is_reference: bool,
    has_existing_dev_citation: bool,
) -> dict[str, Any]:
    """Locked Train predicates. Reference rows are never promotion candidates."""

    cost_ok = True
    k1_ok = True
    operating_ok = True
    near_budget_ok = True
    budget_passed = True
    for tier in TIERS:
        row = official["tiers"][tier]
        realized = _tier_realized(row)
        if realized * float(STRESS_BACKSTOP) > float(OFFICIAL_CAPS[tier]) + 1e-15:
            cost_ok = False
        if _tier_k1(row) != 0:
            k1_ok = False
        if realized > float(OPERATING_TARGETS[tier]) + 1e-15:
            operating_ok = False
        if bool(row["near_budget"]):
            near_budget_ok = False
        if not bool(row["budget_passed"]):
            budget_passed = False
    quality_ok = float(train_weighted) + 1e-15 >= float(LADDER_TRAIN_WEIGHTED)
    eligible = bool(
        (not is_reference)
        and quality_ok
        and cost_ok
        and k1_ok
        and operating_ok
        and near_budget_ok
        and budget_passed
    )
    return sort_mapping(
        {
            "budget_passed": bool(budget_passed),
            "cost_backstop": bool(cost_ok),
            "eligible": bool(eligible),
            "has_existing_dev_citation": bool(has_existing_dev_citation),
            "is_reference": bool(is_reference),
            "k1_off": bool(k1_ok),
            "near_budget_clear": bool(near_budget_ok),
            "operating_targets": bool(operating_ok),
            "quality_vs_ladder_train": bool(quality_ok),
        }
    )


def build_live_submissions(
    inputs: InputBatch,
    policy: RoutingPolicy,
    name: str,
) -> Mapping[str, Any]:
    if name == "always_light":
        return {
            tier: heuristic_submission(
                inputs, policy, tier, strategy="always-light"
            )
            for tier in TIERS
        }
    if name == "prompt_heuristic":
        return {
            tier: heuristic_submission(
                inputs, policy, tier, strategy="prompt-heuristic"
            )
            for tier in TIERS
        }
    if name == "feature_budget":
        _ensure_baselines_path()
        module = importlib.import_module("feature_budget")
        return {
            tier: module.make_feature_budget_submission(inputs, policy, tier).submission
            for tier in TIERS
        }
    raise ValueError(f"unknown live baseline {name!r}")


def score_live_policy(
    *,
    name: str,
    inputs: InputBatch,
    outcomes: Any,
    policy: RoutingPolicy,
) -> dict[str, Any]:
    submissions = build_live_submissions(inputs, policy, name)
    official = official_score(inputs, outcomes, policy, submissions)
    weighted = float(official["final_score"])
    gates = evaluate_train_gates(
        train_weighted=weighted,
        official=official,
        is_reference=False,
        has_existing_dev_citation=name == "always_light",
    )
    return sort_mapping(
        {
            "gates": gates,
            "name": name,
            "official": public_official(official),
            "source": "live-train-official",
            "train_weighted_float": json_float(weighted),
        }
    )


def _cited_official_from_self_check(block: Mapping[str, Any]) -> dict[str, Any]:
    return public_official(block)


def cite_hash_regex(
    train_report: Mapping[str, Any],
    dev_report: Mapping[str, Any],
) -> dict[str, Any]:
    oof = train_report["oof_tier_selection"]
    oof_weighted = weighted_from_tier_scores(
        float(oof["fast"]["tier_score"]),
        float(oof["balanced"]["tier_score"]),
        float(oof["premium"]["tier_score"]),
    )
    oof_official = {
        "final_score": json_float(oof_weighted),
        "tiers": {
            tier: {
                "ax31": None,
                "axk1_think": None,
                "budget_passed": bool(oof[tier]["budget_passed"]),
                "budget_ratio": oof[tier]["actual_budget_ratio"],
                "light": None,
                "near_budget": float(oof[tier]["actual_budget_ratio"])
                >= 0.95 * float(OFFICIAL_CAPS[tier]),
                "quality_score": oof[tier]["tier_score"],
                "realized": oof[tier]["actual_budget_ratio"],
                "realized_times_1054": json_float(
                    float(oof[tier]["actual_budget_ratio"]) * float(STRESS_BACKSTOP)
                ),
                "tier_score": oof[tier]["tier_score"],
            }
            for tier in TIERS
        },
    }
    # hash-regex OOF used K1 (train self-check counts) — fail k1_off by lock.
    k1_on = {
        "fast": 2,
        "balanced": 62,
        "premium": 192,
    }
    for tier in TIERS:
        oof_official["tiers"][tier]["axk1_think"] = 0
    # Promotion uses the cited Dev model counts (K1 present). Train OOF
    # ratios already fail the 1.054 backstop; mark K1 from the frozen Dev
    # artifact so the policy cannot sneak through as K1-off.
    for tier in TIERS:
        oof_official["tiers"][tier]["axk1_think"] = int(
            dev_report["tiers"][tier]["model_counts"].get(_K1, k1_on[tier])
        )
        oof_official["tiers"][tier]["ax31"] = int(
            dev_report["tiers"][tier]["model_counts"].get(_AX31, 0)
        )
        oof_official["tiers"][tier]["light"] = int(
            dev_report["tiers"][tier]["model_counts"].get(_LIGHT, 0)
        )
    gates = evaluate_train_gates(
        train_weighted=oof_weighted,
        official=oof_official,
        is_reference=True,
        has_existing_dev_citation=True,
    )
    self_check = train_report["fitted_train_self_check"]
    return sort_mapping(
        {
            "dev_cited": {
                "final_score": str(dev_report["final_score"]),
                "source": "build/hash-regex/dev-report.json",
                "tiers": {
                    tier: official_tier_block(dev_report, tier) for tier in TIERS
                },
                "weighted_float": json_float(float(dev_report["final_score"])),
            },
            "gates": gates,
            "name": "hash_regex",
            "official": oof_official,
            "self_check_train": {
                "final_score": str(self_check["final_score"]),
                "source": "build/hash-regex/train-report.json#fitted_train_self_check",
                "weighted_float": json_float(float(self_check["final_score"])),
            },
            "source": "cited-frozen-oof",
            "train_weighted_float": json_float(oof_weighted),
            "validation_was_dev": True,
            "validation_was_dev_note": (
                "train_hash_regex used Dev as --validation-input; safety "
                "ratios are Dev-informed. the baseline bake-off does not retrain. Binding "
                "Train number is oof_tier_selection, not validation."
            ),
        }
    )


def cite_ladder_reference() -> dict[str, Any]:
    dummy_official = {
        "final_score": json_float(LADDER_TRAIN_WEIGHTED),
        "tiers": {
            tier: {
                "ax31": 0,
                "axk1_think": 0,
                "budget_passed": True,
                "budget_ratio": 1.0,
                "light": 0,
                "near_budget": False,
                "quality_score": json_float(LADDER_TRAIN_WEIGHTED),
                "realized": 1.0,
                "realized_times_1054": json_float(1.0 * float(STRESS_BACKSTOP)),
                "tier_score": json_float(LADDER_TRAIN_WEIGHTED),
            }
            for tier in TIERS
        },
    }
    gates = evaluate_train_gates(
        train_weighted=float(LADDER_TRAIN_WEIGHTED),
        official=dummy_official,
        is_reference=True,
        has_existing_dev_citation=True,
    )
    return sort_mapping(
        {
            "dev_cited": {
                "decision": "record-ladder-dev-reject",
                "source": "build/feasibility-ladder-dev/report.json",
                "weighted_float": json_float(LADDER_DEV_WEIGHTED),
            },
            "gates": gates,
            "name": "ladder_reference",
            "role": "rollback-selected-reference-only",
            "source": "cited-frozen-ladder",
            "train_cited": {
                "decision": "record-ladder-ready-for-one-dev",
                "source": "build/feasibility-ladder/report.json",
                "weighted_float": json_float(LADDER_TRAIN_WEIGHTED),
            },
            "train_weighted_float": json_float(LADDER_TRAIN_WEIGHTED),
        }
    )


def cite_always_light_dev(ceiling_audit: Mapping[str, Any]) -> dict[str, Any]:
    live = float(ceiling_audit["baseline_comparison_dev"]["all_light"])
    if abs(live - float(ALWAYS_LIGHT_DEV_WEIGHTED)) > 5.5e-7:
        raise ValueError(
            "ceiling-audit all_light drifted from locked "
            f"{ALWAYS_LIGHT_DEV_WEIGHTED}: got {live}"
        )
    return sort_mapping(
        {
            "source": "build/ceiling-audit/report.json",
            "weighted_float": json_float(live),
        }
    )


def select_survivor(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["name"] in LIVE_POLICIES and bool(row["gates"]["eligible"])
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (-float(row["train_weighted_float"]), row["name"])
    )
    return eligible[0]


def may_open_dev(selected: Optional[Mapping[str, Any]]) -> bool:
    if selected is None:
        return False
    if selected["name"] == "always_light":
        return False
    return not bool(selected["gates"]["has_existing_dev_citation"])


def run_stage1(
    *,
    live_rows: Sequence[Mapping[str, Any]],
    hash_regex_row: Mapping[str, Any],
    ladder_row: Mapping[str, Any],
    always_light_dev: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(live_rows) + [hash_regex_row, ladder_row]
    selected = select_survivor(rows)
    open_dev = may_open_dev(selected)
    return sort_mapping(
        {
            "always_light_dev_cited": always_light_dev,
            "grid": list(rows),
            "n_eligible": int(
                sum(1 for row in rows if bool(row["gates"]["eligible"]))
            ),
            "open_dev": bool(open_dev),
            "selected": None
            if selected is None
            else {
                "name": selected["name"],
                "train_weighted_float": selected["train_weighted_float"],
            },
            "selected_key": None if selected is None else selected["name"],
            "selection_rule": SELECTION_RULE,
        }
    )


def run_stage2(*, selected: Optional[Mapping[str, Any]], **_kwargs: Any) -> Mapping[str, Any]:
    if selected is None:
        raise Stage2Refused("Stage 2 refuses to run when Stage 1 selected nothing")
    raise RuntimeError("run_stage2_on_bundle is the Dev entry; tests use the refuse path")


def run_stage2_on_bundle(
    *,
    selected: Mapping[str, Any],
    inputs: InputBatch,
    outcomes: Any,
    policy: RoutingPolicy,
) -> dict[str, Any]:
    name = str(selected["name"])
    if name not in LIVE_POLICIES or name == "always_light":
        raise RuntimeError(f"Stage 2 refuses cited or already-scored policy {name!r}")
    live = score_live_policy(
        name=name, inputs=inputs, outcomes=outcomes, policy=policy
    )
    cand_weighted = float(live["train_weighted_float"])
    quality_ok = cand_weighted + 1e-15 >= float(LADDER_DEV_WEIGHTED)
    cost_ok = True
    k1_ok = True
    official = {
        "final_score": live["official"]["final_score"],
        "tiers": live["official"]["tiers"],
    }
    for tier in TIERS:
        realized = float(official["tiers"][tier]["realized"])
        if realized * float(STRESS_BACKSTOP) > float(OFFICIAL_CAPS[tier]) + 1e-15:
            cost_ok = False
        if int(official["tiers"][tier]["axk1_think"]) != 0:
            k1_ok = False
    predicates = {
        "cost_backstop": bool(cost_ok),
        "k1_off": bool(k1_ok),
        "quality_vs_ladder": bool(quality_ok),
    }
    return sort_mapping(
        {
            "passed": bool(all(predicates.values())),
            "predicates": predicates,
            "selected": live["official"],
            "weighted": {
                "candidate": json_float(cand_weighted),
                "ladder": json_float(LADDER_DEV_WEIGHTED),
            },
        }
    )


def assemble_report(
    *,
    identity: Mapping[str, Any],
    locked: Mapping[str, Any],
    stage1: Mapping[str, Any],
    stage2: Optional[Mapping[str, Any]],
    decision: str,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    if decision not in DECISIONS:
        raise ValueError(f"unknown the baseline bake-off decision {decision!r}")
    dev_opened = stage2 is not None
    if decision == DECISION_NO_PROMOTE and dev_opened:
        raise RuntimeError("no-promote decision must not open Dev")
    if decision == DECISION_PROMOTE and not dev_opened:
        raise RuntimeError("promote decision requires a Dev stage")
    return sort_mapping(
        {
            "selection_remains": "the feasibility ladder",
            "decision": decision,
            "dev_opened": bool(dev_opened),
            "diagnostic": diagnostic,
            "experiment": EXPERIMENT,
            "identity": identity,
            "k1": "off",
            "locked": locked,
            "next_leaf": next_leaf_for(decision),
            "replaces_deployment_image": False,
            "report_type": REPORT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "stage1": stage1,
            "stage2": stage2,
        }
    )


__all__ = (
    "ALL_POLICIES",
    "ALWAYS_LIGHT_DEV_WEIGHTED",
    "CITED_POLICIES",
    "DECISION_DEV_REJECT",
    "DECISION_NO_PROMOTE",
    "DECISION_PROMOTE",
    "DECISIONS",
    "EXPERIMENT",
    "LIVE_POLICIES",
    "Stage2Refused",
    "assemble_report",
    "cite_always_light_dev",
    "cite_hash_regex",
    "cite_ladder_reference",
    "evaluate_train_gates",
    "locked_record",
    "may_open_dev",
    "next_leaf_for",
    "run_stage1",
    "run_stage2",
    "run_stage2_on_bundle",
    "score_live_policy",
    "select_survivor",
    "weighted_from_tier_scores",
)
