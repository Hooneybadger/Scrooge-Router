# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Submitted router: ladder selection plus a content-only family cost guard.

The cost head under-prices ax31 for prompts it has not really learned. The
residual bucket, the prompts that match none of the known shapes, costs
roughly two and a half times its predicted ax31 increment on public Dev.
That single bucket is what forces a low Fast cap, because one static cap
has to hold margin for the worst bucket everywhere.

This router keeps the ladder's heads, calibration and allocator untouched
and only inflates the accounting cost of unfamiliar buckets before the
allocator spends. Familiar prompts therefore get a higher cap without the
unfamiliar ones riding along. Scoring still uses the real cost, so a wrong
guess is caught by the same budget checks as any other policy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from . import feasibility_ladder
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
from .small_batch import THRESHOLD, effective_cap


ARTIFACT_RESOURCE = "family-guard-router.v1.json"
ARTIFACT_TYPE = "scrooge-family-guard-router-v1"
BASE_ARTIFACT_TYPE = feasibility_ladder.ARTIFACT_TYPE
SCHEMA_VERSION = 1
GUARD_FIELD = "family_guard"
REQUIRED_FIELDS = set(feasibility_ladder.REQUIRED_FIELDS) | {GUARD_FIELD}
MULTIPLIER_CLIP = (1.0, 3.0)
RESIDUAL_FAMILY = "other"

_KOREAN = re.compile(r"[가-힣]")
_CHOICE = re.compile(r"(?:^|\n)\s*(?:[A-D][.)]|\([a-e]\))\s", re.IGNORECASE)
_WORD_PROBLEM = re.compile(
    r"\b(?:how many|how much|how long|how far|total|each|costs?|average|"
    r"percent|percentage|left over|altogether)\b",
    re.IGNORECASE,
)


def prompt_family_text(text: str) -> str:
    """Coarse content-only bucket. Mirrors the frozen validation classifier."""

    lowered = text.casefold()
    korean = bool(_KOREAN.search(text))
    if "def f(" in lowered and "assert f(" in lowered:
        return "python_program"
    if len(text) >= 4_000:
        return "long_context"
    if korean and "question:" in lowered and _CHOICE.search(text):
        return "korean_multiple_choice"
    if korean:
        return "korean_reasoning"
    if "question:" in lowered and _CHOICE.search(text):
        return "english_multiple_choice"
    if "question:" in lowered:
        return "rule_reasoning"
    if _CHOICE.search(text):
        return "english_multiple_choice"
    if _WORD_PROBLEM.search(text) and not any(
        marker in text for marker in ("$", "\\[", "**")
    ):
        return "word_problem"
    if any(marker in text for marker in ("$", "\\[", "\\frac", "\\begin")):
        return "latex_math"
    if any(
        marker in lowered
        for marker in ("calculate", "solve ", "let ", "wrt", "divide", "factor")
    ):
        return "symbolic_math"
    return RESIDUAL_FAMILY


def prompt_family(episode: Episode) -> str:
    """Coarse content-only bucket for the prompt."""

    return prompt_family_text(episode_text(episode))


@dataclass(frozen=True)
class GuardedArtifact:
    """Ladder artifact plus the per-family accounting multipliers."""

    value: Mapping[str, Any]
    base: feasibility_ladder.LadderArtifact
    multipliers: Mapping[str, float]


def _load_resource_text() -> str:
    return resources.read_text(
        "ossp_router.resources", ARTIFACT_RESOURCE, encoding="utf-8"
    )


def _require_guard(value: Mapping[str, Any]) -> dict[str, float]:
    guard = value.get(GUARD_FIELD)
    if not isinstance(guard, Mapping):
        raise ProtocolError("family guard block is missing")
    if guard.get("scope") != "fast-and-balanced accounting cost only":
        raise ProtocolError("family guard scope is invalid")
    raw = guard.get("multipliers")
    if not isinstance(raw, Mapping) or not raw:
        raise ProtocolError("family guard multipliers are invalid")
    low, high = MULTIPLIER_CLIP
    out: dict[str, float] = {}
    for family, multiplier in raw.items():
        if not isinstance(family, str):
            raise ProtocolError("family guard key is invalid")
        number = float(multiplier)
        if not low <= number <= high:
            raise ProtocolError("family multiplier is outside the clip")
        out[family] = number
    return out


def load_artifact_mapping(value: Any) -> GuardedArtifact:
    """Validate a decoded artifact and wrap it."""

    if not isinstance(value, dict):
        raise ProtocolError("router artifact must be an object")
    if set(value) != REQUIRED_FIELDS:
        raise ProtocolError("router artifact fields are invalid")
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_type"] != ARTIFACT_TYPE:
        raise ProtocolError("unsupported router artifact")
    multipliers = _require_guard(value)
    base_value = {key: value[key] for key in feasibility_ladder.REQUIRED_FIELDS}
    base_value["artifact_type"] = BASE_ARTIFACT_TYPE
    base = feasibility_ladder.load_artifact_mapping(base_value)
    return GuardedArtifact(value, base, multipliers)


def load_bundled_artifact() -> GuardedArtifact:
    """Load and validate the artifact shipped inside the package."""

    return load_artifact_mapping(json.loads(_load_resource_text()))


def load_artifact_file(path: Path) -> GuardedArtifact:
    """Load and validate an artifact from disk."""

    return load_artifact_mapping(json.loads(path.read_text(encoding="utf-8")))


def guard_multiplier(episode: Episode, artifact: GuardedArtifact) -> float:
    """Accounting multiplier for the prompt's family, 1.0 when unguarded."""

    return float(artifact.multipliers.get(prompt_family(episode), 1.0))


def guarded_prediction(
    episode: Episode, policy: RoutingPolicy, artifact: GuardedArtifact
) -> Tuple[float, Tuple[float, float]]:
    """The ladder row with the ax31 increment repriced for unfamiliar buckets."""

    uplift, (light, ax31) = feasibility_ladder.predict_fast_balanced_row(
        episode, policy, artifact.base
    )
    multiplier = guard_multiplier(episode, artifact)
    if multiplier > 1.0:
        ax31 = light + max(ax31 - light, 0.0) * multiplier
    return uplift, (light, ax31)


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: GuardedArtifact,
    tier: str,
) -> feasibility_ladder.RoutingPlan:
    """Choose one model per episode so the tier stays inside its budget."""

    if tier not in TIERS:
        raise ProtocolError(f"unknown tier: {tier}")
    value = artifact.value
    if value["policy_id"] != policy.policy_id or value["policy_sha256"] != policy_sha256(
        policy
    ):
        raise ProtocolError("router artifact policy mismatch")
    predicted_cap = float(value["predicted_caps"][tier])
    if len(inputs.episodes) < THRESHOLD:
        predicted_cap = effective_cap(
            predicted_cap,
            float(policy.tiers[tier].budget_multiplier),
            len(inputs.episodes),
        )
    if tier == "premium":
        return feasibility_ladder.make_submission(inputs, policy, artifact.base, tier)
    predictions = tuple(
        guarded_prediction(episode, policy, artifact) for episode in inputs.episodes
    )
    selected, ratio = feasibility_ladder.select_fast_balanced(
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
    return feasibility_ladder.RoutingPlan(
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
