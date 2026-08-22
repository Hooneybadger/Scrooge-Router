# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Champion robustness audit of the shipped budget-brake runtime.

This is a diagnostic, not a promotion experiment. It applies the public
red-team questions (H1–H4) to Scrooge itself, then measures serving-path
safety, leftover-budget binding, oracle gap, and replica fidelity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from ossp_router.protocol import TIERS, load_input, load_outcomes, policy_sha256
from research.diagnostics.oracle_gap import oracle_at_budget
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
    AX31,
    INFLATION,
    K1,
    LIGHT,
    NEAR_FRAC,
    OFFICIAL_CAPS,
    PINNED_DEV_FINAL_SCORE,
    PIN_TOLERANCE,
    ProtocolError,
    RESIDUAL_FAMILY,
    ServingReplica,
    SplitReplica,
    TIER_WEIGHTS,
    TVBALL_EPSILON,
    VIEW_MIN_N,
    composition_views,
    json_float,
    model_counts,
    official_tier_block,
    pearson,
    score_models,
    tvball_worst,
    weighted_quality,
)


EXPERIMENT = "champion-robustness-audit-v1"
REPORT_TYPE = "scrooge-champion-robustness-audit-v1"
SCHEMA_VERSION = 1
AUDIT_RELATIVE = "build/run-champion-robustness-audit/episode-audit.json"
REPORT_RELATIVE = "build/run-champion-robustness-audit/report.json"
BINDING_AXES = ("pin", "safety", "replica_fidelity")
CONDITIONAL_AXES = ("h1_ranking", "h2_family", "leftover")


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
    if protocol.get("promotion") is not False:
        raise ProtocolError("audit protocol must set promotion=false")
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


def _quartile_mean(values: Sequence[float], quartile: int) -> Optional[float]:
    if len(values) < 4:
        return None
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    start = (quartile * n) // 4
    stop = ((quartile + 1) * n) // 4
    chunk = ordered[start:stop] or ordered[start:]
    return float(math.fsum(chunk) / float(len(chunk)))


def _axis(status: str, **payload: Any) -> dict[str, Any]:
    block = {"status": status}
    block.update(payload)
    return block


def ranking_block(
    split: SplitReplica,
    models: Sequence[str],
    *,
    predicted: Sequence[float],
    realized: Sequence[float],
    selected_model: str,
) -> dict[str, Any]:
    selected = [index for index, model in enumerate(models) if model == selected_model]
    leftover = [index for index, model in enumerate(models) if model != selected_model]
    selected_real = [float(realized[index]) for index in selected]
    leftover_real = [float(realized[index]) for index in leftover]
    selected_mean = (
        float(math.fsum(selected_real) / float(len(selected_real)))
        if selected_real
        else 0.0
    )
    leftover_mean = (
        float(math.fsum(leftover_real) / float(len(leftover_real)))
        if leftover_real
        else 0.0
    )
    ties = int(sum(abs(float(value)) < 1e-9 for value in realized))
    return {
        "leftover_mean_realized": json_float(leftover_mean),
        "n_leftover": int(len(leftover)),
        "n_selected": int(len(selected)),
        "n_tie_realized": ties,
        "pearson_pred_vs_realized": json_float(pearson(predicted, realized)),
        "selected_mean_realized": json_float(selected_mean),
        "selected_minus_leftover": json_float(selected_mean - leftover_mean),
    }


def family_quality_block(
    split: SplitReplica, models: Sequence[str]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    inversions = []
    family_deltas = []
    for family, indexes in sorted(family_views_local(split.families).items()):
        chosen = score_models(split.scores, split.costs, indexes, [models[i] for i in indexes])
        light = float(split.scores[list(indexes), 0].mean())
        delta = float(chosen["quality"]) - light
        row = {
            "actual_ratio": chosen["actual_ratio"],
            "always_light_quality": json_float(light),
            "delta_vs_always_light": json_float(delta),
            "n": int(len(indexes)),
            "quality": chosen["quality"],
        }
        rows[family] = row
        family_deltas.append(delta)
        if delta < 0.0:
            inversions.append(
                {"delta": json_float(delta), "family": family, "n": int(len(indexes))}
            )
    pooled = float(
        score_models(
            split.scores, split.costs, list(range(split.n)), models
        )["quality"]
        - float(split.scores[:, 0].mean())
    )
    return {
        "families": rows,
        "inversions": inversions,
        "pooled_delta_vs_always_light": json_float(pooled),
        "tvball_worst_vs_always_light": json_float(tvball_worst(pooled, family_deltas)),
    }


def family_views_local(families: Sequence[str]) -> dict[str, Tuple[int, ...]]:
    buckets: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        buckets.setdefault(family, []).append(index)
    return {
        name: tuple(indexes)
        for name, indexes in sorted(buckets.items())
        if len(indexes) >= VIEW_MIN_N
    }


def leftover_fast_balanced(
    split: SplitReplica, selections: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    uplift = split.uplift()
    out: dict[str, Any] = {}
    for tier in ("fast", "balanced"):
        models = selections[tier]
        unused = [
            index
            for index, model in enumerate(models)
            if model == LIGHT and float(uplift[index]) > 0.0
        ]
        used = [index for index, model in enumerate(models) if model == AX31]
        unused_real = [float(split.realized_delta31()[index]) for index in unused]
        used_real = [float(split.realized_delta31()[index]) for index in used]
        out[tier] = {
            "n_positive_pred_left_on_light": int(len(unused)),
            "n_upgraded": int(len(used)),
            "unused_mean_realized_delta31": json_float(
                math.fsum(unused_real) / float(len(unused_real)) if unused_real else 0.0
            ),
            "upgraded_mean_realized_delta31": json_float(
                math.fsum(used_real) / float(len(used_real)) if used_real else 0.0
            ),
        }
    return out


def leftover_premium(
    split: SplitReplica,
    models: Sequence[str],
    *,
    denylist: Sequence[str],
    quality: Sequence[float],
) -> dict[str, Any]:
    denied = set(denylist)
    selected = []
    eligible_unbought = []
    blocked_quality = 0
    blocked_denylist = 0
    blocked_parent = 0
    for index, model in enumerate(models):
        parent_ax31 = model in (AX31, K1)
        if not parent_ax31:
            blocked_parent += 1
            continue
        if split.families[index] in denied:
            blocked_denylist += 1
            continue
        if float(quality[index]) <= 0.0:
            blocked_quality += 1
            continue
        if model == K1:
            selected.append(index)
        else:
            eligible_unbought.append(index)
    selected_delta = [float(split.realized_deltak()[index]) for index in selected]
    leftover_delta = [float(split.realized_deltak()[index]) for index in eligible_unbought]
    selected_cost = [
        float(split.costs[index, 2] - split.costs[index, 1]) for index in selected
    ]
    return {
        "blocked_denylist": int(blocked_denylist),
        "blocked_non_parent": int(blocked_parent),
        "blocked_nonpositive_pred": int(blocked_quality),
        "n_eligible_unbought": int(len(eligible_unbought)),
        "n_k1": int(len(selected)),
        "selected_cost_q1_mean": _quartile_mean(selected_cost, 0),
        "selected_cost_q4_mean": _quartile_mean(selected_cost, 3),
        "selected_mean_deltak": json_float(
            math.fsum(selected_delta) / float(len(selected_delta)) if selected_delta else 0.0
        ),
        "unbought_mean_deltak": json_float(
            math.fsum(leftover_delta) / float(len(leftover_delta)) if leftover_delta else 0.0
        ),
    }


def leftover_block_with_replica(
    replica: ServingReplica,
    split: SplitReplica,
    selections: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    block = leftover_fast_balanced(split, selections)
    denylist = tuple(replica.brake.budget_brake["denylist_families"])
    block["premium"] = leftover_premium(
        split,
        selections["premium"],
        denylist=denylist,
        quality=split.premium_quality,
    )
    return block


def evaluate_split(replica: ServingReplica, split: SplitReplica) -> dict[str, Any]:
    indexes = list(range(split.n))
    selections = replica.allocate_all(
        split,
        indexes,
        caps=replica.shipped_caps,
        brake_ratio=replica.shipped_brake_ratio,
    )
    runtime = {tier: replica.runtime_models(split, tier) for tier in TIERS}
    fidelity = {
        tier: {
            "matched": bool(tuple(runtime[tier]) == tuple(selections[tier])),
            "n_mismatch": int(
                sum(
                    left != right
                    for left, right in zip(runtime[tier], selections[tier])
                )
            ),
        }
        for tier in TIERS
    }
    official = replica.official(split, runtime)
    repeat = replica.allocate_all(
        split,
        indexes,
        caps=replica.shipped_caps,
        brake_ratio=replica.shipped_brake_ratio,
    )
    determinism = all(tuple(repeat[tier]) == tuple(selections[tier]) for tier in TIERS)

    h1 = {}
    for tier in ("fast", "balanced"):
        h1[tier] = ranking_block(
            split,
            selections[tier],
            predicted=split.uplift().tolist(),
            realized=split.realized_delta31().tolist(),
            selected_model=AX31,
        )
    h1["premium_k1"] = ranking_block(
        split,
        selections["premium"],
        predicted=list(split.premium_quality),
        realized=split.realized_deltak().tolist(),
        selected_model=K1,
    )

    h2 = {tier: family_quality_block(split, selections[tier]) for tier in TIERS}
    family_weighted: list[tuple[str, float]] = []
    for family, family_indexes in family_views_local(split.families).items():
        deltas = []
        for tier, weight in TIER_WEIGHTS.items():
            models = [selections[tier][index] for index in family_indexes]
            quality = score_models(
                split.scores, split.costs, family_indexes, models
            )["quality"]
            light = float(split.scores[list(family_indexes), 0].mean())
            deltas.append(weight * (float(quality) - light))
        family_weighted.append((family, float(math.fsum(deltas))))
    official_vs_light = float(official["final_score"]) - weighted_quality(
        {tier: float(split.scores[:, 0].mean()) for tier in TIERS}
    )
    h2["weighted"] = {
        "inversions": [
            {"delta": json_float(delta), "family": name}
            for name, delta in family_weighted
            if delta < 0.0
        ],
        "pooled_delta_vs_always_light": json_float(official_vs_light),
        "tvball_worst_vs_always_light": json_float(
            tvball_worst(official_vs_light, [delta for _name, delta in family_weighted])
        ),
    }

    views = composition_views(split)
    stress: dict[str, Any] = {}
    safety_failures = []
    for name, view_index in views.items():
        if len(view_index) < 2:
            continue
        view_sel = replica.allocate_all(
            split,
            view_index,
            caps=replica.shipped_caps,
            brake_ratio=replica.shipped_brake_ratio,
        )
        stress[name] = {}
        for tier in TIERS:
            scored = score_models(
                split.scores, split.costs, view_index, view_sel[tier]
            )
            inflated = float(scored["actual_ratio"]) * INFLATION
            cap = float(OFFICIAL_CAPS[tier])
            ruin = bool(float(scored["actual_ratio"]) > cap + 1e-15)
            ruin_inflated = bool(inflated > cap + 1e-15)
            near = bool(float(scored["actual_ratio"]) >= NEAR_FRAC * cap - 1e-15)
            stress[name][tier] = {
                "actual_ratio": scored["actual_ratio"],
                "counts": scored["counts"],
                "inflated_ratio": json_float(inflated),
                "n": scored["n"],
                "near_budget": near,
                "quality": scored["quality"],
                "ruin": ruin,
                "ruin_inflated": ruin_inflated,
            }
            binding = name == "full" or name.startswith("family:") or name.startswith(
                "digest-"
            )
            if binding and (ruin or ruin_inflated):
                safety_failures.append(
                    {
                        "actual_ratio": scored["actual_ratio"],
                        "inflated_ratio": json_float(inflated),
                        "tier": tier,
                        "view": name,
                    }
                )

    leftover = leftover_block_with_replica(replica, split, selections)
    oracles = {}
    for tier in TIERS:
        realized = float(
            score_models(
                split.scores, split.costs, indexes, selections[tier]
            )["actual_ratio"]
        )
        same_q, same_n = oracle_at_budget(split.scores, split.costs, realized)
        limit = min(
            float(OFFICIAL_CAPS[tier]) / INFLATION,
            NEAR_FRAC * float(OFFICIAL_CAPS[tier]),
        )
        limit_q, limit_n = oracle_at_budget(split.scores, split.costs, limit)
        router_q = float(
            score_models(
                split.scores, split.costs, indexes, selections[tier]
            )["quality"]
        )
        oracles[tier] = {
            "gap_same_budget": json_float(same_q - router_q),
            "operating_limit": json_float(limit),
            "oracle_limit_quality": json_float(limit_q),
            "oracle_limit_upgrades": int(limit_n),
            "oracle_same_quality": json_float(same_q),
            "oracle_same_upgrades": int(same_n),
            "realized_ratio": json_float(realized),
            "router_quality": json_float(router_q),
        }

    return {
        "determinism_passed": bool(determinism),
        "family_mix": {
            name: int(sum(family == name for family in split.families))
            for name in sorted(set(split.families))
        },
        "fidelity": fidelity,
        "final_score": float(official["final_score"]),
        "h1": h1,
        "h2": h2,
        "leftover": leftover,
        "n": int(split.n),
        "official": {tier: official_tier_block(official, tier) for tier in TIERS},
        "oracle": oracles,
        "replica_counts": {
            tier: model_counts(selections[tier]) for tier in TIERS
        },
        "residual_fraction": json_float(split.residual_frac),
        "runtime_counts": {tier: model_counts(runtime[tier]) for tier in TIERS},
        "safety_failures": safety_failures,
        "selections": {tier: list(selections[tier]) for tier in TIERS},
        "stress": stress,
    }


def decide_axes(
    protocol: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = protocol["thresholds"]
    dev = splits["dev"]
    train = splits["train"]
    pin_matched = bool(
        abs(float(dev["final_score"]) - PINNED_DEV_FINAL_SCORE) <= PIN_TOLERANCE
    )
    pin = _axis(
        "pass" if pin_matched else "fail",
        dev_final=float(dev["final_score"]),
        matched=pin_matched,
        pinned=PINNED_DEV_FINAL_SCORE,
    )

    fidelity_ok = all(
        bool(splits[label]["fidelity"][tier]["matched"])
        and bool(splits[label]["determinism_passed"])
        for label in ("train", "dev")
        for tier in TIERS
    )
    replica = _axis(
        "pass" if fidelity_ok else "fail",
        train=train["fidelity"],
        dev=dev["fidelity"],
        determinism={
            "train": bool(train["determinism_passed"]),
            "dev": bool(dev["determinism_passed"]),
        },
    )

    safety_failures = list(train["safety_failures"]) + list(dev["safety_failures"])
    official_fail = [
        {"split": label, "tier": tier, "budget_ratio": row["budget_ratio"]}
        for label, block in splits.items()
        for tier, row in block["official"].items()
        if (not row["budget_passed"])
        or float(row["budget_ratio"]) >= NEAR_FRAC * float(OFFICIAL_CAPS[tier]) - 1e-15
    ]
    safety_ok = not safety_failures and not official_fail
    safety = _axis(
        "pass" if safety_ok else "fail",
        official_failures=official_fail,
        view_failures=safety_failures,
    )

    h1_gaps = [
        float(splits[label]["h1"][tier]["selected_minus_leftover"])
        for label in ("train", "dev")
        for tier in ("fast", "balanced")
    ]
    h1_pearson = [
        float(splits[label]["h1"][tier]["pearson_pred_vs_realized"])
        for label in ("train", "dev")
        for tier in ("fast", "balanced")
    ]
    h1_weak = bool(
        min(h1_gaps) < float(thresholds["h1_selected_gap_min"])
        or min(h1_pearson) < float(thresholds["h1_pearson_min"])
    )
    h1 = _axis(
        "conditional" if h1_weak else "pass",
        min_pearson=json_float(min(h1_pearson)),
        min_selected_gap=json_float(min(h1_gaps)),
    )

    inversions = [
        {"split": label, **row}
        for label in ("train", "dev")
        for row in splits[label]["h2"]["weighted"]["inversions"]
        if float(row["delta"]) < -float(thresholds["h2_family_drop_max"])
    ]
    tvball = min(
        float(splits[label]["h2"]["weighted"]["tvball_worst_vs_always_light"])
        for label in ("train", "dev")
    )
    if tvball < float(thresholds["h2_tvball_min"]):
        h2_status = "fail"
    elif inversions:
        h2_status = "conditional"
    else:
        h2_status = "pass"
    h2 = _axis(
        h2_status,
        hard_inversions=inversions,
        tvball_worst=json_float(tvball),
    )

    h3_rows = []
    h3_cheap = False
    for label in ("train", "dev"):
        prem = splits[label]["leftover"]["premium"]
        q1 = prem["selected_cost_q1_mean"]
        q4 = prem["selected_cost_q4_mean"]
        selected = float(prem["selected_mean_deltak"])
        unbought = float(prem["unbought_mean_deltak"])
        row = {
            "n_k1": prem["n_k1"],
            "selected_mean_deltak": selected,
            "split": label,
            "unbought_mean_deltak": unbought,
        }
        h3_rows.append(row)
        if prem["n_k1"] and q1 is not None and q4 is not None and q1 < q4:
            # Cheap-half selected ΔK much worse than expensive half is the H3 risk.
            if selected + 1e-12 < unbought:
                h3_cheap = True
    h3 = _axis(
        "conditional" if h3_cheap else "pass",
        rows=h3_rows,
    )

    train_final = float(train["final_score"])
    dev_final = float(dev["final_score"])
    h4 = _axis(
        "observational",
        dev_minus_train=json_float(dev_final - train_final),
        note="Train-fitted heads; this is split shift, not grouped OOF.",
        residual_fraction={
            "dev": float(dev["residual_fraction"]),
            "train": float(train["residual_fraction"]),
        },
        tv_radius=TVBALL_EPSILON,
    )

    leftover_unused = max(
        int(splits[label]["leftover"][tier]["n_positive_pred_left_on_light"])
        for label in ("train", "dev")
        for tier in ("fast", "balanced")
    )
    leftover_status = (
        "conditional"
        if leftover_unused >= int(thresholds["leftover_unused_min"])
        else "pass"
    )
    leftover = _axis(
        leftover_status,
        max_positive_pred_left_on_light=leftover_unused,
        premium={
            label: splits[label]["leftover"]["premium"] for label in ("train", "dev")
        },
    )

    oracle_gaps = [
        float(splits[label]["oracle"][tier]["gap_same_budget"])
        for label in ("train", "dev")
        for tier in TIERS
    ]
    oracle = _axis(
        "observational",
        max_same_budget_gap=json_float(max(oracle_gaps)),
        mean_same_budget_gap=json_float(math.fsum(oracle_gaps) / float(len(oracle_gaps))),
    )

    axes = {
        "h1_ranking": h1,
        "h2_family": h2,
        "h3_k1_tail": h3,
        "h4_split_shift": h4,
        "leftover": leftover,
        "oracle": oracle,
        "pin": pin,
        "replica_fidelity": replica,
        "safety": safety,
    }
    if any(axes[name]["status"] == "fail" for name in BINDING_AXES) or h2_status == "fail":
        overall = "fragile"
    elif any(axes[name]["status"] == "conditional" for name in CONDITIONAL_AXES + ("h3_k1_tail",)):
        overall = "conditionally_robust"
    else:
        overall = "robust"
    return {"axes": axes, "overall": overall}


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("champion robustness audit output exists; refuse overwrite")

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

    train_inputs = load_input(TRAIN_INPUTS)
    train_outcomes = load_outcomes(TRAIN_OUTCOMES)
    dev_inputs = load_input(DEV_INPUTS)
    dev_outcomes = load_outcomes(DEV_OUTCOMES)
    if len(train_inputs.episodes) != EXPECTED_N_TRAIN:
        raise ProtocolError("train n drifted")
    if len(dev_inputs.episodes) != EXPECTED_N_DEV:
        raise ProtocolError("dev n drifted")

    train = replica.build_split("train", train_inputs, train_outcomes)
    dev = replica.build_split("dev", dev_inputs, dev_outcomes)
    split_results = {
        "dev": evaluate_split(replica, dev),
        "train": evaluate_split(replica, train),
    }
    selections = {
        label: split_results[label].pop("selections") for label in ("train", "dev")
    }
    verdict = decide_axes(protocol, split_results)
    overall = str(verdict["overall"])
    decision = str(protocol["decisions"][overall])
    reason = str(protocol["decision_reasons"][overall])

    audit_document = {
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "rows": {
            label: [
                {
                    "episode_id": episode.episode_id,
                    "family": families[index],
                    **{tier: selections[label][tier][index] for tier in TIERS},
                }
                for index, episode in enumerate(inputs.episodes)
            ]
            for label, inputs, families in (
                ("train", train_inputs, train.families),
                ("dev", dev_inputs, dev.families),
            )
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
        "identity": identity,
        "promotion": False,
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "residual_family": RESIDUAL_FAMILY,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "shipped": {
            "balanced_cap": replica.shipped_caps["balanced"],
            "brake_ratio": replica.shipped_brake_ratio,
            "count_cap": replica.shipped_count_cap,
            "fast_cap": replica.shipped_caps["fast"],
            "max_upgrade_fraction": replica.max_upgrade_fraction,
            "premium_cap": replica.shipped_caps["premium"],
            "runaway_fraction": replica.runaway_fraction,
        },
        "splits": split_results,
        "thresholds": dict(protocol["thresholds"]),
        "verdict": verdict,
    }
    core = sort_mapping(
        {
            key: report[key]
            for key in (
                "audit",
                "decision",
                "decision_reason",
                "experiment",
                "identity",
                "promotion",
                "protocol_sha256",
                "report_type",
                "schema_version",
                "shipped",
                "splits",
                "thresholds",
                "verdict",
            )
        }
    )
    encoded = json_dumps_core(core)
    report["decision_core_sha256"] = sha256_text(encoded)
    write_json_atomic(audit_output, audit_document)
    write_json_atomic(output, report)
    return report, audit_document


def json_dumps_core(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
