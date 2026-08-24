# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E27 — confirm Premium brake_ratio 3.80 without conformal rollback.

E25 passed, but the bound clipped to 1.0 and rolled nothing. This
protocol scores the stdlib knob that actually fired and asks whether
it should enter the runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router import budget_brake_router
from ossp_router.feasibility_ladder import _select_premium_configured
from ossp_router.protocol import TIERS, load_input, load_outcomes, policy_sha256
from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.e4_aggregate_risk import conformal_bound, rollback_until_bound
from research.lab.e5_brake_conditioned import FOLDS, ProtocolError, derive_fresh_seeds
from research.lab.grouped_crossfit import (
    assign_balanced_group_folds,
    fold_leakage_count,
    group_episodes,
)
from research.lab.modeling import paired_group_bootstrap, sort_mapping
from research.lab.public_pool import (
    DEV_INPUTS,
    DEV_OUTCOMES,
    EXPECTED_N_DEV,
    EXPECTED_N_TRAIN,
    TRAIN_INPUTS,
    TRAIN_OUTCOMES,
    sha256_path,
    subset_inputs,
)
from research.lab.serving_replica import (
    INFLATION,
    K1,
    MODEL_INDEX,
    NEAR_FRAC,
    OFFICIAL_CAPS,
    PINNED_DEV_FINAL_SCORE,
    PIN_TOLERANCE,
    RESIDUAL_FAMILY,
    ServingReplica,
    SplitReplica,
    TIER_WEIGHTS,
    family_views,
    json_float,
    official_tier_block,
    score_models,
)


EXPERIMENT = "e27-premium-brake-380-v1"
REPORT_TYPE = "scrooge-e27-premium-brake-380-v1"
SCHEMA_VERSION = 1
BASELINE_NAME = "shipped"
PRIMARY_NAME = "premium-brake-3.80"
IDENTITY_NAME = "premium-conformal-rollback"
AUDIT_RELATIVE = "build/run-e27-premium-brake-380/episode-audit.json"
REPORT_RELATIVE = "build/run-e27-premium-brake-380/report.json"
BUY_BRAKE = 3.8
SETTLE_CAP = 3.8
BOOTSTRAP_DRAWS = 10000
E26_DECISION_CORE = (
    "84c2075d35bff7cc5c1dc1297bf03e50ce76c4b4b2f2e6cce6983f290279d914"
)
EXPECTED_PROTOCOL_SHA256 = (
    "2a570a61bcaba15c9778e73344ee73087475b2d3264535d0b96fadd7bcde7730"
)


def _predicted_matrix(split: SplitReplica, indexes: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [split.premium_rows[index][1] for index in indexes], dtype=np.float64
    )


def _spend(models: Sequence[str], costs: np.ndarray) -> float:
    return float(
        math.fsum(
            float(costs[row, MODEL_INDEX[model]])
            for row, model in enumerate(models)
        )
    )


def allocate_premium_buy(
    replica: ServingReplica,
    split: SplitReplica,
    indexes: Sequence[int],
    *,
    brake_ratio: float,
) -> Tuple[str, ...]:
    families = tuple(split.families[index] for index in indexes)
    active = budget_brake_router.premium_residual_composition_guard(families)
    extra = (RESIDUAL_FAMILY,) if active else ()
    return replica.allocate_premium(
        split,
        indexes,
        brake_ratio=float(brake_ratio),
        guard_parent=bool(active),
        denylist_extra=extra,
    )


def premium_parent(
    replica: ServingReplica, split: SplitReplica, indexes: Sequence[int]
) -> Tuple[str, ...]:
    families = tuple(split.families[index] for index in indexes)
    active = budget_brake_router.premium_residual_composition_guard(families)
    batch = subset_inputs(split.inputs, indexes)
    raw_rows = tuple(split.premium_rows[index] for index in indexes)
    if active:
        parent_rows = tuple(
            (
                row[0],
                budget_brake_router.guard_premium_parent_costs(
                    split.inputs.episodes[index],
                    row[1],
                    replica.brake,
                ),
            )
            for index, row in zip(indexes, raw_rows)
        )
    else:
        parent_rows = raw_rows
    parent, _ratio = _select_premium_configured(
        batch,
        parent_rows,
        float(replica.shipped_caps["premium"]),
        replica.brake.family_guard.base,
    )
    return parent


def apply_conformal_rollback(
    split: SplitReplica,
    indexes: Sequence[int],
    parent: Sequence[str],
    bought: Sequence[str],
    bound: float,
    settle_cap: float,
) -> Tuple[Tuple[str, ...], int]:
    pred = _predicted_matrix(split, indexes)
    n_rows = len(indexes)
    chosen = np.asarray([model == K1 for model in bought], dtype=bool)
    current = np.empty(n_rows, dtype=np.float64)
    increment = np.zeros(n_rows, dtype=np.float64)
    uplift = np.asarray(
        [split.premium_quality[index] for index in indexes], dtype=np.float64
    )
    for row, parent_model in enumerate(parent):
        current[row] = float(pred[row, MODEL_INDEX[parent_model]])
        if parent_model == "ax31":
            increment[row] = max(
                float(pred[row, MODEL_INDEX[K1]]) - current[row], 0.0
            )
    pred_light = float(pred[:, 0].sum())
    selected, n_rolled = rollback_until_bound(
        chosen,
        uplift,
        increment,
        current,
        pred_light,
        float(settle_cap),
        float(bound),
        tuple(split.digests[index] for index in indexes),
    )
    models = tuple(
        K1 if bool(selected[row]) else parent[row] for row in range(n_rows)
    )
    return models, int(n_rolled)


def allocate_premium_conformal(
    replica: ServingReplica,
    split: SplitReplica,
    indexes: Sequence[int],
    bound: float,
) -> Tuple[Tuple[str, ...], int]:
    parent = premium_parent(replica, split, indexes)
    bought = allocate_premium_buy(
        replica, split, indexes, brake_ratio=BUY_BRAKE
    )
    return apply_conformal_rollback(
        split, indexes, parent, bought, bound, SETTLE_CAP
    )


def buy_ratio(
    replica: ServingReplica, split: SplitReplica, indexes: Sequence[int]
) -> float:
    models = allocate_premium_buy(
        replica, split, indexes, brake_ratio=BUY_BRAKE
    )
    pred = _predicted_matrix(split, indexes)
    pred_spend = _spend(models, pred)
    actual_spend = math.fsum(
        float(split.costs[index, MODEL_INDEX[model]])
        for index, model in zip(indexes, models)
    )
    if pred_spend <= 0.0:
        raise ProtocolError("predicted selected spend is not positive")
    return float(actual_spend / pred_spend)


def fit_train_bound(
    replica: ServingReplica, train: SplitReplica, seed: int
) -> Mapping[str, Any]:
    folds = assign_balanced_group_folds(
        train.digests, train.families, folds=FOLDS, seed=int(seed)
    )
    leaked = fold_leakage_count(train.digests, folds)
    if leaked:
        raise ProtocolError(f"grouped fold leakage: {leaked}")
    ratios = []
    for fold in range(FOLDS):
        held = [index for index, value in enumerate(folds) if value == fold]
        if not held:
            continue
        ratios.append(buy_ratio(replica, train, held))
    fitted = conformal_bound(ratios)
    return {
        "bound": float(fitted["bound"]),
        "n_ratios": int(fitted["n"]),
        "q99": fitted["q99"],
        "ratios": [float(value) for value in fitted["ratios"]],
        "seed": int(seed),
    }


def evaluate_shipped(
    replica: ServingReplica, split: SplitReplica
) -> dict[str, Any]:
    selections = {tier: replica.runtime_models(split, tier) for tier in TIERS}
    official = replica.official(split, selections)
    premium_scored = score_models(
        split.scores, split.costs, list(range(split.n)), selections["premium"]
    )
    return {
        "final_score": float(official["final_score"]),
        "n_rolled": 0,
        "official": {
            tier: official_tier_block(official, tier) for tier in TIERS
        },
        "premium_k1": int(sum(model == K1 for model in selections["premium"])),
        "premium_quality": float(premium_scored["quality"]),
        "premium_ratio": float(premium_scored["actual_ratio"]),
        "selections": {tier: list(models) for tier, models in selections.items()},
    }


def evaluate_conformal(
    replica: ServingReplica, split: SplitReplica, bound: float
) -> dict[str, Any]:
    shipped = {tier: replica.runtime_models(split, tier) for tier in TIERS}
    premium, n_rolled = allocate_premium_conformal(
        replica, split, list(range(split.n)), bound
    )
    selections = {
        "fast": shipped["fast"],
        "balanced": shipped["balanced"],
        "premium": premium,
    }
    official = replica.official(split, selections)
    premium_scored = score_models(
        split.scores, split.costs, list(range(split.n)), premium
    )
    return {
        "final_score": float(official["final_score"]),
        "n_rolled": int(n_rolled),
        "official": {
            tier: official_tier_block(official, tier) for tier in TIERS
        },
        "premium_k1": int(sum(model == K1 for model in premium)),
        "premium_quality": float(premium_scored["quality"]),
        "premium_ratio": float(premium_scored["actual_ratio"]),
        "selections": {tier: list(models) for tier, models in selections.items()},
    }


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
    if protocol["arms"]["baseline"]["name"] != BASELINE_NAME:
        raise ProtocolError("baseline arm drifted")
    if protocol["arms"]["primary"]["name"] != PRIMARY_NAME:
        raise ProtocolError("primary arm drifted")
    if protocol["arms"]["identity_conformal"]["name"] != IDENTITY_NAME:
        raise ProtocolError("identity arm drifted")
    if str(protocol["arms"]["identity_conformal"].get("selection_use")) != "none":
        raise ProtocolError("conformal identity must not select")
    derivation = protocol["seed_derivation"]
    if str(derivation["core_sha256"]) != E26_DECISION_CORE:
        raise ProtocolError("e27 core must be the e26 decision core")
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    expected_seeds = derive_fresh_seeds(
        str(derivation["prefix"]),
        str(derivation["core_sha256"]),
        int(derivation["n"]),
        [int(value) for value in derivation["forbidden_previous_seeds"]],
    )
    if fresh != expected_seeds or len(set(fresh)) != len(fresh):
        raise ProtocolError("sealed fresh seeds drifted from the derivation")
    thresholds = protocol["thresholds"]
    if abs(float(thresholds["buy_brake_ratio"]) - BUY_BRAKE) > 1e-15:
        raise ProtocolError("buy_brake_ratio drifted")
    if int(thresholds["bootstrap_draws"]) != BOOTSTRAP_DRAWS:
        raise ProtocolError("bootstrap_draws drifted")
    if int(protocol.get("n_bootstrap_draws", -1)) != BOOTSTRAP_DRAWS:
        raise ProtocolError("n_bootstrap_draws drifted")
    if str(thresholds.get("full_batch_family_selection_use")) != "none":
        raise ProtocolError("full-batch family ratios must not be kill gates")
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


def predicted_ratio(split: SplitReplica, models: Sequence[str]) -> float:
    if len(models) != split.n:
        raise ProtocolError("predicted ratio length mismatch")
    spend = math.fsum(
        float(split.premium_rows[index][1][MODEL_INDEX[model]])
        for index, model in enumerate(models)
    )
    light = math.fsum(float(split.premium_rows[index][1][0]) for index in range(split.n))
    if light <= 0.0:
        raise ProtocolError("predicted light sum is not positive")
    return float(spend / light)


def residual_isolated_buy(
    replica: ServingReplica, split: SplitReplica
) -> Optional[dict[str, Any]]:
    views = family_views(split.families)
    name = f"family:{RESIDUAL_FAMILY}"
    if name not in views:
        return None
    indexes = views[name]
    models = allocate_premium_buy(
        replica, split, indexes, brake_ratio=BUY_BRAKE
    )
    scored = score_models(split.scores, split.costs, indexes, models)
    return {
        "actual_ratio": float(scored["actual_ratio"]),
        "inflated_ratio": float(scored["actual_ratio"]) * INFLATION,
        "n": int(scored["n"]),
        "quality": float(scored["quality"]),
        "k1": int(sum(model == K1 for model in models)),
    }


def full_batch_family_rows(
    split: SplitReplica, models: Sequence[str]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, indexes in family_views(split.families).items():
        subset = [models[index] for index in indexes]
        scored = score_models(split.scores, split.costs, indexes, subset)
        rows[name] = {
            "actual_ratio": float(scored["actual_ratio"]),
            "inflated_ratio": float(scored["actual_ratio"]) * INFLATION,
            "k1": int(sum(model == K1 for model in subset)),
            "n": int(scored["n"]),
            "quality": float(scored["quality"]),
        }
    return rows


def evaluate_brake(
    replica: ServingReplica, split: SplitReplica
) -> dict[str, Any]:
    shipped = {tier: replica.runtime_models(split, tier) for tier in TIERS}
    premium = allocate_premium_buy(
        replica, split, list(range(split.n)), brake_ratio=BUY_BRAKE
    )
    selections = {
        "fast": shipped["fast"],
        "balanced": shipped["balanced"],
        "premium": premium,
    }
    official = replica.official(split, selections)
    premium_scored = score_models(
        split.scores, split.costs, list(range(split.n)), premium
    )
    return {
        "final_score": float(official["final_score"]),
        "n_rolled": 0,
        "official": {tier: official_tier_block(official, tier) for tier in TIERS},
        "predicted_ratio": predicted_ratio(split, premium),
        "premium_k1": int(sum(model == K1 for model in premium)),
        "premium_quality": float(premium_scored["quality"]),
        "premium_ratio": float(premium_scored["actual_ratio"]),
        "residual_isolated": residual_isolated_buy(replica, split),
        "selections": {tier: list(models) for tier, models in selections.items()},
    }


def _without_selections(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "selections"}


def official_failures(label: str, block: Mapping[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for tier, row in block["official"].items():
        cap = float(OFFICIAL_CAPS[tier])
        if (not row["budget_passed"]) or float(row["budget_ratio"]) >= (
            NEAR_FRAC * cap - 1e-15
        ):
            failed.append(
                {
                    "split": label,
                    "tier": tier,
                    "budget_ratio": float(row["budget_ratio"]),
                }
            )
    return failed


def residual_over_cap(row: Optional[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> bool:
    if row is None:
        return True
    return (
        float(row["actual_ratio"])
        > float(thresholds["residual_premium_actual_max"]) + 1e-15
        or float(row["inflated_ratio"])
        > float(thresholds["residual_premium_inflated_max"]) + 1e-15
    )


def episode_official_gains(
    split: SplitReplica, base_premium: Sequence[str], cand_premium: Sequence[str]
) -> np.ndarray:
    weight = float(TIER_WEIGHTS["premium"])
    gains = np.empty(split.n, dtype=np.float64)
    for index in range(split.n):
        gains[index] = weight * (
            float(split.scores[index, MODEL_INDEX[cand_premium[index]]])
            - float(split.scores[index, MODEL_INDEX[base_premium[index]]])
        )
    return gains


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "audit": report["audit"],
            "candidate_primary": report["candidate_primary"],
            "constants": report["constants"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "fold_seeds": report["fold_seeds"],
            "gate": report["gate"],
            "pin_dev_replay": report["pin_dev_replay"],
            "protocol_sha256": report["protocol_sha256"],
            "report_type": report["report_type"],
            "schema_version": report["schema_version"],
            "seed_results": report["seed_results"],
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


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("e27 output exists; refuse overwrite")
    if "src/" in str(output) or "src/" in str(audit_output):
        raise ProtocolError("e27 must not write under src/")

    replica = ServingReplica.load()
    identity_pins = {
        "dev_inputs_sha256": sha256_path(DEV_INPUTS),
        "dev_outcomes_sha256": sha256_path(DEV_OUTCOMES),
        "policy_sha256": policy_sha256(replica.policy),
        "train_inputs_sha256": sha256_path(TRAIN_INPUTS),
        "train_outcomes_sha256": sha256_path(TRAIN_OUTCOMES),
    }
    verify_protocol(protocol, protocol_digest, pool_identity=identity_pins)
    train = replica.build_split(
        "train", load_input(TRAIN_INPUTS), load_outcomes(TRAIN_OUTCOMES)
    )
    dev = replica.build_split(
        "dev", load_input(DEV_INPUTS), load_outcomes(DEV_OUTCOMES)
    )
    if train.n != EXPECTED_N_TRAIN or dev.n != EXPECTED_N_DEV:
        raise ProtocolError("split n drifted")

    shipped_dev = {tier: replica.runtime_models(dev, tier) for tier in TIERS}
    pin_official = replica.official(dev, shipped_dev)
    pin = {
        "final_score": float(pin_official["final_score"]),
        "matched": bool(
            abs(float(pin_official["final_score"]) - PINNED_DEV_FINAL_SCORE)
            <= PIN_TOLERANCE
        ),
        "pinned_final_score": PINNED_DEV_FINAL_SCORE,
    }
    if not pin["matched"]:
        raise ProtocolError("shipped Dev pin failed")

    thresholds = dict(protocol["thresholds"])
    seeds = [int(seed) for seed in protocol["fresh_seeds"]]
    base_train = evaluate_shipped(replica, train)
    base_dev = evaluate_shipped(replica, dev)
    cand_train = evaluate_brake(replica, train)
    cand_dev = evaluate_brake(replica, dev)

    fast_bal_ok = all(
        cand_train["selections"][tier] == base_train["selections"][tier]
        and cand_dev["selections"][tier] == base_dev["selections"][tier]
        for tier in ("fast", "balanced")
    )
    train_q_delta = float(
        cand_train["premium_quality"] - base_train["premium_quality"]
    )
    dev_delta = float(cand_dev["final_score"] - base_dev["final_score"])
    official_fail = official_failures("train", cand_train) + official_failures(
        "dev", cand_dev
    )

    base_train["predicted_ratio"] = predicted_ratio(train, base_train["selections"]["premium"])
    base_dev["predicted_ratio"] = predicted_ratio(dev, base_dev["selections"]["premium"])

    full_batch = {
        "train": {
            "baseline": full_batch_family_rows(train, base_train["selections"]["premium"]),
            "primary": full_batch_family_rows(train, cand_train["selections"]["premium"]),
        },
        "dev": {
            "baseline": full_batch_family_rows(dev, base_dev["selections"]["premium"]),
            "primary": full_batch_family_rows(dev, cand_dev["selections"]["premium"]),
        },
    }

    audit_rows = []
    for index, (shipped_model, cand_model) in enumerate(
        zip(base_dev["selections"]["premium"], cand_dev["selections"]["premium"])
    ):
        if shipped_model == cand_model:
            continue
        audit_rows.append(
            {
                "episode_id": dev.inputs.episodes[index].episode_id,
                "family": dev.families[index],
                "from": shipped_model,
                "to": cand_model,
            }
        )

    gains = episode_official_gains(
        dev, base_dev["selections"]["premium"], cand_dev["selections"]["premium"]
    )
    group_keys = group_episodes(dev.inputs.episodes).group_keys
    if len(group_keys) != dev.n:
        raise ProtocolError("dev group keys drifted")

    seed_results: dict[str, Any] = {}
    safety_failures: list[dict[str, Any]] = []
    bootstrap_q25: list[float] = []
    for seed in seeds:
        fitted = fit_train_bound(replica, train, seed)
        conf_train = evaluate_conformal(replica, train, float(fitted["bound"]))
        conf_dev = evaluate_conformal(replica, dev, float(fitted["bound"]))
        identical = bool(
            conf_train["selections"]["premium"] == cand_train["selections"]["premium"]
            and conf_dev["selections"]["premium"] == cand_dev["selections"]["premium"]
            and int(conf_train["n_rolled"]) == 0
            and int(conf_dev["n_rolled"]) == 0
        )
        boot = paired_group_bootstrap(
            gains, group_keys, draws=BOOTSTRAP_DRAWS, seed=int(seed)
        )
        bootstrap_q25.append(float(boot["q2_5"]))
        seed_fail = []
        if not identical:
            seed_fail.append("conformal_identity")
        if float(boot["q2_5"]) <= float(thresholds["bootstrap_q2_5_min_exclusive"]):
            seed_fail.append("bootstrap_q2_5")
        if seed_fail:
            safety_failures.append({"seed": seed, "failures": list(seed_fail)})
        seed_results[str(seed)] = {
            "bound": dict(fitted),
            "bootstrap": boot,
            "conformal_identical": identical,
            "conformal_n_rolled": {
                "dev": int(conf_dev["n_rolled"]),
                "train": int(conf_train["n_rolled"]),
            },
            "failures": list(seed_fail),
        }

    deterministic_fail = []
    if not fast_bal_ok:
        deterministic_fail.append("fast_balanced_identity")
    if official_fail:
        deterministic_fail.append("official_safety")
    if residual_over_cap(cand_train["residual_isolated"], thresholds) or residual_over_cap(
        cand_dev["residual_isolated"], thresholds
    ):
        deterministic_fail.append("residual_isolated_cap")
    if train_q_delta <= float(thresholds["train_premium_quality_delta_min_exclusive"]):
        deterministic_fail.append("train_premium_quality")
    if dev_delta <= float(thresholds["dev_delta_min_exclusive"]):
        deterministic_fail.append("dev_official")
    for label, block in (("train", cand_train), ("dev", cand_dev)):
        if float(block["predicted_ratio"]) > float(thresholds["predicted_ratio_max"]) + 1e-12:
            deterministic_fail.append(f"{label}_predicted_ratio")
        if float(block["premium_ratio"]) >= float(thresholds["actual_ratio_max"]) - 1e-15:
            deterministic_fail.append(f"{label}_actual_ratio")
    if cand_dev["premium_k1"] <= base_dev["premium_k1"]:
        deterministic_fail.append("dev_k1_did_not_increase")
    if deterministic_fail:
        safety_failures.append({"seed": None, "failures": list(deterministic_fail)})

    gates_ok = (not safety_failures) and bool(pin["matched"])
    decision = str(protocol["decisions"]["pass" if gates_ok else "fail"])
    reason = str(protocol["decision_reasons"]["pass" if gates_ok else "fail"])

    other_key = f"family:{RESIDUAL_FAMILY}"
    other_share = None
    if other_key in full_batch["dev"]["baseline"] and abs(dev_delta) > 1e-18:
        n_other = int(full_batch["dev"]["primary"][other_key]["n"])
        dq_other = float(
            full_batch["dev"]["primary"][other_key]["quality"]
            - full_batch["dev"]["baseline"][other_key]["quality"]
        )
        other_share = json_float(
            (n_other / float(dev.n)) * dq_other * float(TIER_WEIGHTS["premium"]) / dev_delta
        )

    audit_document = {
        "experiment": EXPERIMENT,
        "n_changed_dev": len(audit_rows),
        "prompt_text_included": False,
        "rows": audit_rows,
    }
    report = {
        "audit": {
            "n_rows": len(audit_rows),
            "relative_path": AUDIT_RELATIVE,
            "sha256": sha256_text(canonical_json_text(audit_document)),
        },
        "candidate_primary": PRIMARY_NAME,
        "constants": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "buy_brake_ratio": BUY_BRAKE,
            "inflation": INFLATION,
            "parent_predicted_cap": 3.25,
        },
        "decision": decision,
        "decision_reason": reason,
        "deltas": {
            "dev_official": json_float(dev_delta),
            "dev_other_official_share_evidence_only": other_share,
            "train_premium_quality": json_float(train_q_delta),
        },
        "experiment": EXPERIMENT,
        "fold_seeds": seeds,
        "full_batch_family_evidence_only": full_batch,
        "gate": {
            "bootstrap_q2_5_min": json_float(min(bootstrap_q25)),
            "conformal_identical_all_seeds": all(
                bool(block["conformal_identical"]) for block in seed_results.values()
            ),
            "dev_delta": json_float(dev_delta),
            "fast_balanced_identical": bool(fast_bal_ok),
            "official_failures": official_fail,
            "passed": bool(gates_ok),
            "pin_matched": bool(pin["matched"]),
            "safety_failures": safety_failures,
            "train_premium_quality_delta": json_float(train_q_delta),
        },
        "pin_dev_replay": pin,
        "primary": {
            "dev": _without_selections(cand_dev),
            "train": _without_selections(cand_train),
        },
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_results,
        "shipped": {
            "dev": _without_selections(base_dev),
            "train": _without_selections(base_train),
        },
        "thresholds": thresholds,
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
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
