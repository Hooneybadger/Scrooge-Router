# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Premium overlay that promotes ax31 to axk1-think under a predicted-budget brake.

Fast and Balanced stay on the family-guard path so those decisions stay
bit-identical. Premium keeps the ladder's two-action allocation, then
promotes in predicted quality order while the batch predicted Premium
ratio stays under the frozen brake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from . import family_guard_router, feasibility_ladder
from .cost_calibrated_router import _premium_prediction, structural_features
from .heuristic import episode_text, write_submission_atomic
from .protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
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


ARTIFACT_RESOURCE = "budget-brake-router.v1.json"
ARTIFACT_TYPE = "scrooge-budget-brake-router-v1"
SCHEMA_VERSION = 1
BRAKE_FIELD = "budget_brake"
REQUIRED_FIELDS = set(family_guard_router.REQUIRED_FIELDS) | {BRAKE_FIELD}
REQUIRED_BRAKE_FIELDS = {
    "brake_ratio",
    "clip",
    "count_cap",
    "denylist_families",
    "enabled",
    "feature_signature",
    "forest",
    "runaway_absolute",
    "runaway_light_fraction",
    "train_full_pred_light",
}
FEATURE_SIGNATURE = "ossp_router.cost_calibrated_router.structural_features/14"
FINITE_COMPARE = 1e-12
_AX31 = MODEL_IDS[1]
_K1 = MODEL_IDS[2]


_FlatNode = Tuple[int, float, int, int, float]
_FlatTree = Tuple[_FlatNode, ...]


@dataclass(frozen=True)
class BrakeArtifact:
    value: Mapping[str, Any]
    family_guard: family_guard_router.GuardedArtifact
    budget_brake: Mapping[str, Any]
    forest: Tuple[_FlatTree, ...]
    n_trees_f: float


@dataclass(frozen=True)
class BrakePlan:
    submission: Submission
    predicted_budget_ratio: float
    predicted_cap: float
    premium_rows: Tuple[Tuple[float, Tuple[float, float, float]], ...] = ()


def _load_resource_text() -> str:
    return resources.read_text(
        "ossp_router.resources", ARTIFACT_RESOURCE, encoding="utf-8"
    )


def content_digest(episode: Episode) -> str:
    return hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest()


def _require_same_length(tree: Mapping[str, Any], keys: Sequence[str]) -> int:
    lengths = []
    for key in keys:
        values = tree.get(key)
        if not isinstance(values, list) or not values:
            raise ProtocolError("budget brake tree arrays are invalid")
        lengths.append(len(values))
    if len(set(lengths)) != 1:
        raise ProtocolError("budget brake tree arrays disagree in length")
    return lengths[0]


def _require_budget_brake(value: Mapping[str, Any]) -> dict[str, Any]:
    block = value.get(BRAKE_FIELD)
    if not isinstance(block, Mapping):
        raise ProtocolError("budget brake block is missing")
    if set(block) != REQUIRED_BRAKE_FIELDS:
        raise ProtocolError("budget brake block fields are invalid")
    if block.get("enabled") is not True:
        raise ProtocolError("budget brake block is disabled")
    try:
        brake = float(block["brake_ratio"])
        count_cap = int(block["count_cap"])
        runaway_absolute = float(block["runaway_absolute"])
        runaway_frac = float(block["runaway_light_fraction"])
        train_light = float(block["train_full_pred_light"])
    except (TypeError, ValueError) as error:
        raise ProtocolError("budget brake numeric fields are invalid") from error
    if not math.isfinite(brake) or not (1.0 < brake <= 4.0):
        raise ProtocolError("budget brake brake_ratio is outside (1.0, 4.0]")
    if count_cap < 0:
        raise ProtocolError("budget brake count_cap is negative")
    if not math.isfinite(runaway_absolute) or not math.isfinite(runaway_frac):
        raise ProtocolError("budget brake runaway fields are invalid")
    if not math.isfinite(train_light) or train_light <= 0.0:
        raise ProtocolError("budget brake train_full_pred_light is invalid")
    denylist = block.get("denylist_families")
    if not isinstance(denylist, list) or any(
        not isinstance(name, str) or not name for name in denylist
    ):
        raise ProtocolError("budget brake denylist_families are invalid")
    clip = block.get("clip")
    if (
        not isinstance(clip, list)
        or len(clip) != 2
        or not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in clip)
    ):
        raise ProtocolError("budget brake clip is invalid")
    if block.get("feature_signature") != FEATURE_SIGNATURE:
        raise ProtocolError("budget brake feature_signature is invalid")
    forest = block.get("forest")
    if not isinstance(forest, Mapping):
        raise ProtocolError("budget brake forest is missing")
    trees = forest.get("trees")
    n_trees = forest.get("n_trees")
    if not isinstance(trees, list) or not trees:
        raise ProtocolError("budget brake forest is empty")
    if not isinstance(n_trees, int) or n_trees != len(trees):
        raise ProtocolError("budget brake forest n_trees is invalid")
    keys = ("left", "right", "feature", "threshold", "value")
    for tree in trees:
        if not isinstance(tree, Mapping) or set(tree) != set(keys):
            raise ProtocolError("budget brake tree fields are invalid")
        _require_same_length(tree, keys)
        left = tree["left"]
        if not any(int(node) == -1 for node in left):
            raise ProtocolError("budget brake tree has no leaf")
    return {
        "brake_ratio": brake,
        "clip": (float(clip[0]), float(clip[1])),
        "count_cap": count_cap,
        "denylist_families": tuple(str(name) for name in denylist),
        "enabled": True,
        "feature_signature": FEATURE_SIGNATURE,
        "forest": forest,
        "runaway_absolute": runaway_absolute,
        "runaway_light_fraction": runaway_frac,
        "train_full_pred_light": train_light,
    }


def _family_guard_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    mapping = {key: value[key] for key in value if key != BRAKE_FIELD}
    mapping["artifact_type"] = family_guard_router.ARTIFACT_TYPE
    return mapping


def _flatten_forest(trees: Sequence[Mapping[str, Any]]) -> Tuple[_FlatTree, ...]:
    forest = []
    for tree in trees:
        forest.append(
            tuple(
                (
                    int(tree["feature"][index]),
                    float(tree["threshold"][index]),
                    int(tree["left"][index]),
                    int(tree["right"][index]),
                    float(tree["value"][index]),
                )
                for index in range(len(tree["left"]))
            )
        )
    return tuple(forest)


def load_artifact_mapping(value: Any) -> BrakeArtifact:
    if not isinstance(value, dict):
        raise ProtocolError("budget brake artifact must be an object")
    if set(value) != REQUIRED_FIELDS:
        raise ProtocolError("budget brake artifact fields are invalid")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("artifact_type") != ARTIFACT_TYPE:
        raise ProtocolError("unsupported budget brake artifact")
    budget_brake = _require_budget_brake(value)
    family_guard = family_guard_router.load_artifact_mapping(_family_guard_mapping(value))
    trees = budget_brake["forest"]["trees"]
    return BrakeArtifact(
        value,
        family_guard,
        budget_brake,
        _flatten_forest(trees),
        float(len(trees)),
    )


def load_bundled_artifact() -> BrakeArtifact:
    return load_artifact_mapping(json.loads(_load_resource_text()))


def load_artifact_file(path: Path) -> BrakeArtifact:
    return load_artifact_mapping(json.loads(path.read_text(encoding="utf-8")))


def _walk_tree(row: Sequence[float], tree: Mapping[str, Any]) -> float:
    left = tree["left"]
    right = tree["right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    value = tree["value"]
    node = 0
    while int(left[node]) != -1:
        if row[int(feature[node])] <= float(threshold[node]):
            node = int(left[node])
        else:
            node = int(right[node])
    return float(value[node])


def _walk_flat(row: Sequence[float], tree: _FlatTree) -> float:
    feature, threshold, left, right, value = tree[0]
    while left != -1:
        if row[feature] <= threshold:
            feature, threshold, left, right, value = tree[left]
        else:
            feature, threshold, left, right, value = tree[right]
    return value


def predict_quality_features(row: Sequence[float], artifact: BrakeArtifact) -> float:
    total = 0.0
    for tree in artifact.forest:
        total += _walk_flat(row, tree)
    mean = total / artifact.n_trees_f
    low, high = artifact.budget_brake["clip"]
    if mean < low:
        return float(low)
    if mean > high:
        return float(high)
    return mean


def predict_quality(episode: Episode, artifact: BrakeArtifact) -> float:
    return predict_quality_features(structural_features(episode), artifact)


def eligible_promotion_indices(
    parent_models: Sequence[str],
    families: Sequence[str],
    premium_costs: Sequence[Sequence[float]],
    block: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Rows that can still be promoted; predicted quality is not required to decide this set."""

    if not (len(parent_models) == len(families) == len(premium_costs)):
        raise ProtocolError("budget brake promotion arrays must align")
    denylist = set(block["denylist_families"])
    runaway = float(block["runaway_absolute"])
    eligible = []
    for index in range(len(parent_models)):
        if parent_models[index] != _AX31:
            continue
        if families[index] in denylist:
            continue
        increment = float(premium_costs[index][2] - premium_costs[index][1])
        if increment > runaway:
            continue
        eligible.append(index)
    return tuple(eligible)


def premium_prediction_row(
    episode: Episode, policy: RoutingPolicy, artifact: BrakeArtifact
) -> Tuple[float, Tuple[float, float, float]]:
    """Same Premium row the ladder two-action allocator already builds."""

    return _premium_prediction(episode, policy, artifact.family_guard.base)


def predicted_premium_spend(
    models: Sequence[str], costs: Sequence[Sequence[float]]
) -> float:
    if len(models) != len(costs):
        raise ProtocolError("budget brake premium costs must align with models")
    return math.fsum(
        float(costs[index][MODEL_IDS.index(model_id)])
        for index, model_id in enumerate(models)
    )


def predicted_premium_ratio(
    models: Sequence[str], costs: Sequence[Sequence[float]]
) -> float:
    light = math.fsum(float(row[0]) for row in costs)
    if light <= 0.0:
        raise ProtocolError("budget brake predicted light sum is not positive")
    return predicted_premium_spend(models, costs) / light


def promote_premium_brake(
    parent_models: Sequence[str],
    quality: Sequence[float],
    families: Sequence[str],
    premium_costs: Sequence[Sequence[float]],
    digests: Sequence[str],
    block: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Promote parent AX31 rows to K1 while predicted spend stays under the brake."""

    n_batch = len(parent_models)
    if not (
        n_batch
        == len(quality)
        == len(families)
        == len(premium_costs)
        == len(digests)
    ):
        raise ProtocolError("budget brake promotion arrays must align")
    denylist = set(block["denylist_families"])
    runaway = float(block["runaway_absolute"])
    count_cap = int(block["count_cap"])
    eligible = []
    for index in range(n_batch):
        if parent_models[index] != _AX31:
            continue
        if float(quality[index]) <= 0.0:
            continue
        if families[index] in denylist:
            continue
        increment = float(premium_costs[index][2] - premium_costs[index][1])
        if increment > runaway:
            continue
        eligible.append(index)
    eligible.sort(key=lambda index: (-float(quality[index]), digests[index]))
    pred_light_sum = math.fsum(float(row[0]) for row in premium_costs)
    if pred_light_sum <= 0.0:
        raise ProtocolError("budget brake predicted light sum is not positive")
    budget = float(block["brake_ratio"]) * pred_light_sum
    selected = list(parent_models)
    spend = predicted_premium_spend(selected, premium_costs)
    taken = 0
    for index in eligible:
        increment = float(premium_costs[index][2] - premium_costs[index][1])
        if increment <= 0.0:
            continue
        if spend + increment > budget + FINITE_COMPARE:
            continue
        selected[index] = _K1
        spend += increment
        taken += 1
        if taken >= count_cap:
            break
    return tuple(selected)


def select_premium_with_brake(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: BrakeArtifact,
    premium_rows: Sequence[Tuple[float, Tuple[float, float, float]]],
    *,
    quality: Optional[Sequence[float]] = None,
    families: Optional[Sequence[str]] = None,
    digests: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """Ladder two-action Premium allocation, then the frozen brake loop."""

    if len(premium_rows) != len(inputs.episodes):
        raise ProtocolError("budget brake premium rows must align with the batch")
    predicted_cap = float(artifact.value["predicted_caps"]["premium"])
    parent, _ratio = feasibility_ladder._select_premium_configured(
        inputs, premium_rows, predicted_cap, artifact.family_guard.base
    )
    episodes = inputs.episodes
    if families is None:
        families = tuple(family_guard_router.prompt_family(episode) for episode in episodes)
    costs = tuple(row[1] for row in premium_rows)
    if quality is None:
        eligible = eligible_promotion_indices(
            parent, families, costs, artifact.budget_brake
        )
        scored = [0.0] * len(episodes)
        if digests is None:
            digest_rows = [""] * len(episodes)
            for index in eligible:
                episode = episodes[index]
                scored[index] = predict_quality_features(
                    structural_features(episode), artifact
                )
                digest_rows[index] = content_digest(episode)
            digests = digest_rows
        else:
            for index in eligible:
                scored[index] = predict_quality_features(
                    structural_features(episodes[index]), artifact
                )
        quality = scored
    elif digests is None:
        digests = tuple(content_digest(episode) for episode in episodes)
    return promote_premium_brake(
        parent, quality, families, costs, digests, artifact.budget_brake
    )


def _plan_from_models(
    inputs: InputBatch,
    policy: RoutingPolicy,
    tier: str,
    selected: Sequence[str],
    ratio: float,
    predicted_cap: float,
    premium_rows: Tuple[Tuple[float, Tuple[float, float, float]], ...] = (),
) -> BrakePlan:
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
    return BrakePlan(
        parse_submission(submission_to_dict(submission)),
        ratio,
        predicted_cap,
        premium_rows,
    )


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: BrakeArtifact,
    tier: str,
) -> BrakePlan:
    if tier not in TIERS:
        raise ProtocolError(f"unknown tier: {tier}")
    value = artifact.value
    if value["policy_id"] != policy.policy_id or value["policy_sha256"] != policy_sha256(
        policy
    ):
        raise ProtocolError("budget brake artifact policy mismatch")
    if tier != "premium":
        plan = family_guard_router.make_submission(
            inputs, policy, artifact.family_guard, tier
        )
        if any(decision.model_id == _K1 for decision in plan.submission.decisions):
            raise ProtocolError("budget brake Fast/Balanced selected K1")
        return BrakePlan(
            plan.submission, plan.predicted_budget_ratio, plan.predicted_cap
        )
    premium_rows = tuple(
        premium_prediction_row(episode, policy, artifact)
        for episode in inputs.episodes
    )
    selected = select_premium_with_brake(inputs, policy, artifact, premium_rows)
    costs = tuple(row[1] for row in premium_rows)
    ratio = predicted_premium_ratio(selected, costs)
    return _plan_from_models(
        inputs,
        policy,
        tier,
        selected,
        ratio,
        float(value["predicted_caps"][tier]),
        premium_rows,
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
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        artifact = (
            load_artifact_file(args.artifact)
            if args.artifact
            else load_bundled_artifact()
        )
        inputs = load_input(args.input)
        plan = make_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
