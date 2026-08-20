# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Feasibility-ladder router: base heads with a drift-adjusted Fast budget.

Standard library only. The cost and uplift heads come from
cost_calibrated_router unchanged. What differs is how the Fast budget is
committed: instead of spending up to a single static cap, candidates are
admitted along a ladder of feasibility checks, so a mis-priced tail cannot
drag the realized ratio past the tier cap.

Premium keeps the base allocator.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .cost_calibrated_router import (
    _premium_prediction,
    _quality_prediction,
    _select_ax31,
    _select_premium,
    _token_predictions,
    structural_features,
)
from .heuristic import episode_text, write_submission_atomic
from .protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)


ARTIFACT_RESOURCE = "feasibility-ladder.v1.json"
ARTIFACT_TYPE = "scrooge-feasibility-ladder-v1"
SCHEMA_VERSION = 1
FINITE_COMPARE = 1e-12
K1_DISABLED_QUALITY = -1_000_000.0
REQUIRED_FIELDS = {
    "artifact_type",
    "cost",
    "k1",
    "k1_enabled",
    "max_upgrade_fraction",
    "policy_id",
    "policy_sha256",
    "predicted_caps",
    "premium_overlay",
    "provenance",
    "quality",
    "recalibration",
    "runaway_fraction",
    "safe_tier_caps",
    "schema_version",
    "selected_policy",
    "tier_kappa_q999",
}


@dataclass(frozen=True)
class LadderArtifact:
    """Frozen artifact holding the base heads and the ladder recalibration."""

    value: Mapping[str, Any]


@dataclass(frozen=True)
class RoutingPlan:
    """A finished submission with the ratio it was predicted to spend."""

    submission: Submission
    predicted_budget_ratio: float
    predicted_cap: float


def _load_resource_text() -> str:
    return resources.read_text(
        "ossp_router.resources", ARTIFACT_RESOURCE, encoding="utf-8"
    )


def _require_recalibration(value: Mapping[str, Any]) -> Mapping[str, Any]:
    recal = value.get("recalibration")
    if not isinstance(recal, Mapping):
        raise ProtocolError("recalibration block is missing")
    edges = recal.get("edges")
    factors = recal.get("factors")
    if not isinstance(edges, list) or not isinstance(factors, list):
        raise ProtocolError("recalibration edges or factors are invalid")
    if len(edges) != 9 or len(factors) != 10:
        raise ProtocolError("recalibration must be 10 equal-count bins")
    if recal.get("dev_data_used") is not False:
        raise ProtocolError("artifact must record that Dev was unused")
    return recal


def load_artifact_mapping(value: Any) -> LadderArtifact:
    """Validate a decoded artifact and wrap it."""

    if not isinstance(value, dict):
        raise ProtocolError("router artifact must be an object")
    if set(value) != REQUIRED_FIELDS:
        raise ProtocolError("router artifact fields are invalid")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["artifact_type"] != ARTIFACT_TYPE
        or not isinstance(value["k1_enabled"], bool)
    ):
        raise ProtocolError("unsupported router artifact")
    if set(value["predicted_caps"]) != set(TIERS):
        raise ProtocolError("predicted caps are incomplete")
    overlay = value["premium_overlay"]
    if (
        not isinstance(overlay, dict)
        or overlay.get("tier") != "premium"
        or overlay.get("group_method") != "exact-content-sha256-v1"
    ):
        raise ProtocolError("Premium overlay is invalid")
    _require_recalibration(value)
    k1 = value["k1"]
    if not isinstance(k1, dict) or k1.get("scope") != "premium-only":
        raise ProtocolError("k1 configuration is invalid")
    return LadderArtifact(value)


def load_bundled_artifact() -> LadderArtifact:
    """Load and validate the artifact shipped inside the package."""

    return load_artifact_mapping(json.loads(_load_resource_text()))


def load_artifact_file(path: Path) -> LadderArtifact:
    """Load and validate an artifact from disk."""

    return load_artifact_mapping(json.loads(path.read_text(encoding="utf-8")))


def _digitize_right(value: float, edges: Sequence[float]) -> int:
    """Match numpy.digitize(value, edges, right=True)."""

    index = 0
    for edge in edges:
        if value <= float(edge):
            break
        index += 1
    return index


def apply_rank_recal(
    predicted_incremental: float, edges: Sequence[float], factors: Sequence[float]
) -> float:
    """Scale a predicted increment by the factor of the bin it lands in."""

    bin_index = _digitize_right(predicted_incremental, edges)
    if bin_index < 0:
        bin_index = 0
    last = len(factors) - 1
    if bin_index > last:
        bin_index = last
    recalibrated = predicted_incremental * float(factors[bin_index])
    if not math.isfinite(recalibrated):
        raise ProtocolError("recalibration produced a non-finite cost")
    return recalibrated


def apply_cost_recal(
    costs: Tuple[float, float], artifact: LadderArtifact
) -> Tuple[float, float]:
    """Recalibrate the ax31 increment and leave the light cost alone."""

    recal = artifact.value["recalibration"]
    light, ax31 = costs
    increment = ax31 - light
    recalibrated = apply_rank_recal(increment, recal["edges"], recal["factors"])
    return (light, light + recalibrated)


def apply_runaway_guard(
    costs: Sequence[Tuple[float, float]], fraction: float
) -> Tuple[Tuple[float, float], ...]:
    """Clamp increments that alone would eat too much of the batch."""

    light_total = math.fsum(pair[0] for pair in costs)
    if (not math.isfinite(light_total)) or light_total <= 0.0:
        raise ProtocolError("light denominator is not positive")
    threshold = float(fraction) * light_total
    guarded = []
    for light, ax31 in costs:
        if (ax31 - light) > threshold:
            guarded.append((light, light))
        else:
            guarded.append((light, ax31))
    return tuple(guarded)


def apply_upgrade_count_cap(
    selected: Sequence[str],
    costs: Sequence[Tuple[float, float]],
    fraction: float,
) -> Tuple[str, ...]:
    """Drop the weakest upgrades until they fit the allowed share."""

    chosen = list(selected)
    maximum = int(math.floor(float(fraction) * len(chosen)))
    ax31 = [index for index, model_id in enumerate(chosen) if model_id == MODEL_IDS[1]]
    extra = len(ax31) - maximum
    if extra <= 0:
        return tuple(chosen)
    increments = [costs[index][1] - costs[index][0] for index in ax31]
    order = sorted(range(len(ax31)), key=lambda key: (-increments[key], key))
    for key in order[:extra]:
        chosen[ax31[key]] = MODEL_IDS[0]
    return tuple(chosen)


def spend_ratio(
    costs: Sequence[Tuple[float, float]], selected: Sequence[str]
) -> float:
    """Realized cost of a selection over the all-light denominator."""

    light_total = math.fsum(pair[0] for pair in costs)
    if light_total <= 0.0:
        raise ProtocolError("light denominator is not positive")
    selected_total = math.fsum(
        costs[index][1] if model_id == MODEL_IDS[1] else costs[index][0]
        for index, model_id in enumerate(selected)
    )
    return selected_total / light_total


def _k1_is_active(artifact: LadderArtifact) -> bool:
    value = artifact.value
    if value.get("k1_enabled") is not True:
        return False
    quality = value.get("k1", {}).get("quality")
    return isinstance(quality, Mapping)


def _k1_uplift(episode: Any, artifact: LadderArtifact) -> float:
    if not _k1_is_active(artifact):
        return K1_DISABLED_QUALITY
    return _quality_prediction(
        episode,
        {"quality": artifact.value["k1"]["quality"]},
        structural_features(episode),
    )


def _select_premium_configured(
    inputs: InputBatch,
    predictions: Sequence[Tuple[float, Tuple[float, float, float]]],
    cap_ratio: float,
    artifact: LadderArtifact,
) -> Tuple[Tuple[str, ...], float]:
    """Base Premium allocator; K1 quality stays an artifact flag, not a rewrite."""

    if not _k1_is_active(artifact):
        return _select_premium(inputs, predictions, cap_ratio)
    raised = tuple(
        (uplift, costs, _k1_uplift(episode, artifact))
        for episode, (uplift, costs) in zip(inputs.episodes, predictions)
    )
    return _select_premium_with_k1(inputs, raised, cap_ratio)


def _select_premium_with_k1(
    inputs: InputBatch,
    predictions: Sequence[Tuple[float, Tuple[float, float, float], float]],
    cap_ratio: float,
) -> Tuple[Tuple[str, ...], float]:
    grouped: dict[str, list[int]] = {}
    for index, episode in enumerate(inputs.episodes):
        digest = hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest()
        grouped.setdefault(digest, []).append(index)
    names = sorted(grouped)
    rows = [grouped[name] for name in names]
    quality = [
        (
            0.0,
            math.fsum(predictions[index][0] for index in indexes),
            math.fsum(predictions[index][2] for index in indexes),
        )
        for indexes in rows
    ]
    costs = [
        tuple(
            math.fsum(predictions[index][1][model] for index in indexes)
            for model in range(3)
        )
        for indexes in rows
    ]
    states = [0] * len(names)
    versions = [0] * len(names)
    light_total = math.fsum(group_costs[0] for group_costs in costs)
    budget = cap_ratio * light_total
    total_cost = light_total
    queue: list[Tuple[float, float, str, int, int, int, float, int]] = []

    def push_upgrades(group_index: int) -> None:
        source = states[group_index]
        for target in range(1, 3):
            if target == source:
                continue
            incremental = costs[group_index][target] - costs[group_index][source]
            gain = quality[group_index][target] - quality[group_index][source]
            if incremental <= 0.0 or gain <= 0.0:
                continue
            heapq.heappush(
                queue,
                (
                    -(gain / incremental),
                    -gain,
                    names[group_index],
                    target,
                    versions[group_index],
                    group_index,
                    incremental,
                    source,
                ),
            )

    for group_index in range(len(names)):
        push_upgrades(group_index)
    while queue:
        (
            _density,
            _gain,
            _name,
            target,
            version,
            group_index,
            incremental,
            source,
        ) = heapq.heappop(queue)
        if version != versions[group_index] or states[group_index] != source:
            continue
        if total_cost + incremental > budget + 1e-12:
            continue
        states[group_index] = target
        versions[group_index] += 1
        total_cost += incremental
        push_upgrades(group_index)

    selected = [MODEL_IDS[0]] * len(predictions)
    for group_index, indexes in enumerate(rows):
        for index in indexes:
            selected[index] = MODEL_IDS[states[group_index]]
    return tuple(selected), total_cost / light_total


def _fast_balanced_head(
    episode: Any, policy: RoutingPolicy, artifact: LadderArtifact
) -> Tuple[float, Tuple[float, float]]:
    """Same numbers as cost_calibrated_router.predict_episode; structural features once.

    Speed-only. Does not change caps, bins, guards, or any decision rule.
    """

    value = artifact.value
    structural = structural_features(episode)
    uplift = _quality_prediction(episode, value, structural)
    tokens = _token_predictions(episode, value, structural)
    costs = []
    for model_index, model_id in enumerate(MODEL_IDS[:2]):
        rates = policy.models[model_id]
        costs.append(
            float(rates.fixed_cost)
            + tokens[2 * model_index]
            * float(rates.input_token_rate)
            / float(policy.token_unit)
            + tokens[2 * model_index + 1]
            * float(rates.output_token_rate)
            / float(policy.token_unit)
        )
    costs[0] = max(costs[0], 1e-15)
    costs[1] = max(costs[1], costs[0] * 1.05)
    return uplift, (costs[0], costs[1])


def predict_fast_balanced_row(
    episode: Any, policy: RoutingPolicy, artifact: LadderArtifact
) -> Tuple[float, Tuple[float, float]]:
    """Head prediction with the cost recalibration applied."""

    uplift, costs = _fast_balanced_head(episode, policy, artifact)
    return uplift, apply_cost_recal(costs, artifact)


def select_fast_balanced(
    predictions: Sequence[Tuple[float, Tuple[float, float]]],
    *,
    cap: float,
    runaway_fraction: float,
    max_upgrade_fraction: float,
) -> Tuple[Tuple[str, ...], float]:
    """Buy upgrades by uplift per unit cost until the cap is reached."""

    guarded = apply_runaway_guard(
        tuple(costs for _uplift, costs in predictions), runaway_fraction
    )
    ranked = tuple(
        (predictions[index][0], guarded[index]) for index in range(len(predictions))
    )
    selected, _ratio = _select_ax31(ranked, cap)
    selected = apply_upgrade_count_cap(selected, guarded, max_upgrade_fraction)
    ratio = spend_ratio(guarded, selected)
    if ratio > float(cap) + FINITE_COMPARE:
        selected = tuple(MODEL_IDS[0] for _ in selected)
        ratio = 1.0
    return selected, ratio


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: LadderArtifact,
    tier: str,
) -> RoutingPlan:
    """Choose one model per episode so the tier stays inside its budget."""

    if tier not in TIERS:
        raise ProtocolError(f"unknown tier: {tier}")
    value = artifact.value
    if value["policy_id"] != policy.policy_id or value[
        "policy_sha256"
    ] != policy_sha256(policy):
        raise ProtocolError("router artifact policy mismatch")
    predicted_cap = float(value["predicted_caps"][tier])
    if tier == "premium":
        predictions = tuple(
            _premium_prediction(episode, policy, artifact)
            for episode in inputs.episodes
        )
        selected, ratio = _select_premium_configured(
            inputs, predictions, predicted_cap, artifact
        )
        if not _k1_is_active(artifact) and any(
            model_id == MODEL_IDS[2] for model_id in selected
        ):
            raise ProtocolError("K1 is disabled but a Premium decision used it")
    else:
        predictions = tuple(
            predict_fast_balanced_row(episode, policy, artifact)
            for episode in inputs.episodes
        )
        selected, ratio = select_fast_balanced(
            predictions,
            cap=predicted_cap,
            runaway_fraction=float(value["runaway_fraction"]),
            max_upgrade_fraction=float(value["max_upgrade_fraction"]),
        )
        if any(model_id == MODEL_IDS[2] for model_id in selected):
            raise ProtocolError("Fast/Balanced selected K1")
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )
    return RoutingPlan(
        parse_submission(submission_to_dict(submission)), ratio, predicted_cap
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--artifact", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point."""

    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        artifact = (
            load_artifact_file(args.artifact)
            if args.artifact
            else load_bundled_artifact()
        )
        plan = make_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"OK: {args.tier} ladder submission "
        f"(predicted ratio {plan.predicted_budget_ratio:.6f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
