# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Base router: hashed prompt features with calibrated cost and uplift heads.

Standard library only. A frozen artifact carries the fitted ridge
coefficients, the calibration and the per-tier caps; this module turns a
prompt into features, predicts what each model would cost and how much
accuracy it would buy, and spends the tier budget on the best ratio first.

The two routers layered above reuse these heads unchanged and differ only
in how they commit the budget.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import sys
import zlib
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .heuristic import episode_text, extract_features, write_submission_atomic
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

try:
    from ._fnvfast import hashed_features as _native_hashed_features
    from ._fnvfast import stable_hash as _native_stable_hash
except ImportError:  # Built into the container image only; source runs pure Python.
    _native_hashed_features = None
    _native_stable_hash = None


ARTIFACT_RESOURCE = "cost-calibrated-router.v1.json"
_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_UINT64_MASK = (1 << 64) - 1
_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_KOREAN = re.compile(r"[가-힣]")
_CHOICE = re.compile(r"(?:^|\n)\s*(?:[A-D][.)]|\([a-e]\))\s", re.IGNORECASE)
_WORD_PROBLEM = re.compile(
    r"\b(?:how many|how much|how long|how far|total|each|costs?|average|"
    r"percent|percentage|left over|altogether)\b",
    re.IGNORECASE,
)
_FORMAL_REASONING = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|"
    r"증명|유도|정리|보조정리|반례|귀납)\b",
    re.IGNORECASE,
)
_PROGRAM_ANALYSIS = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|"
    r"시간\s*복잡도|공간\s*복잡도|예외|스택\s*추적)\b",
    re.IGNORECASE,
)
_MULTI_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_SIMPLE_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|"
    r"요약|바꾸|번역|나열|추출)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CalibratedArtifact:
    """Frozen coefficients, calibration and per-tier caps for this router."""

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


def load_bundled_artifact() -> CalibratedArtifact:
    """Load and validate the artifact shipped inside the package."""

    value = json.loads(_load_resource_text())
    if not isinstance(value, dict):
        raise ProtocolError("router artifact must be an object")
    required = {
        "schema_version",
        "artifact_type",
        "selected_policy",
        "k1_enabled",
        "safe_tier_caps",
        "tier_kappa_q999",
        "policy_id",
        "policy_sha256",
        "quality",
        "cost",
        "premium_overlay",
        "provenance",
    }
    if set(value) != required:
        raise ProtocolError("router artifact fields are invalid")
    if (
        value["schema_version"] != 2
        or value["artifact_type"] != "scrooge-cost-calibrated-router-v1"
        or value["k1_enabled"] is not False
    ):
        raise ProtocolError("unsupported router artifact")
    if set(value["safe_tier_caps"]) != set(TIERS) or set(
        value["tier_kappa_q999"]
    ) != set(TIERS):
        raise ProtocolError("tier calibration is incomplete")
    overlay = value["premium_overlay"]
    if (
        not isinstance(overlay, dict)
        or set(overlay)
        != {
            "tier",
            "group_method",
            "kappa_q999",
            "safe_cap",
            "minimum_model_cost_step_ratio",
            "quality",
            "cost",
        }
        or overlay["tier"] != "premium"
        or overlay["group_method"] != "exact-content-sha256-v1"
        or float(overlay["kappa_q999"]) < 1.0
        or float(overlay["minimum_model_cost_step_ratio"]) < 1.0
        or len(overlay["cost"]["token_upper_bounds"]) != 6
        or set(overlay["cost"]["residual_upper"]) != set(MODEL_IDS[1:])
    ):
        raise ProtocolError("Premium overlay is invalid")
    return CalibratedArtifact(value)


@lru_cache(maxsize=262_144)
def _stable_hash(value: str) -> int:
    if _native_stable_hash is not None:
        return int(_native_stable_hash(value))
    digest = _FNV_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def _normalized_tokens(text: str) -> Tuple[str, ...]:
    result = []
    for token in _TOKEN.findall(text):
        normalized = token.casefold()
        result.append("<number>" if normalized.isdecimal() else normalized)
    return tuple(result)


def structural_features(episode: Episode) -> Tuple[float, ...]:
    """Length, shape and formatting features taken from the prompt text."""

    features = extract_features(episode)
    text = episode_text(episode)
    return (
        math.log1p(features.character_count),
        math.log1p(features.word_count),
        math.log1p(features.sentence_count),
        math.log1p(features.message_count),
        features.hangul_ratio,
        math.log1p(features.code_marker_count),
        math.log1p(features.math_marker_count),
        features.numeric_density,
        float(features.long_context),
        math.log1p(features.reasoning_marker_count),
        float(bool(_FORMAL_REASONING.search(text))),
        float(bool(_PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(_MULTI_CONSTRAINT.findall(text))),
        float(bool(_SIMPLE_TRANSFORM.search(text))),
    )


def hashed_features(
    episode: Episode, bins: int, algorithm: str = "fnv1a64"
) -> Tuple[float, ...]:
    """Hash the prompt tokens into a fixed number of bins."""

    if bins <= 0 or bins & (bins - 1):
        raise ValueError("hash bins must be a positive power of two")
    if algorithm not in ("fnv1a64", "crc32", "blake2b64"):
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    if algorithm == "fnv1a64" and _native_hashed_features is not None:
        return tuple(_native_hashed_features(episode_text(episode), bins))
    values = [0.0] * bins
    tokens = _normalized_tokens(episode_text(episode))
    terms = (
        *(f"w1:{token}" for token in tokens),
        *(f"w2:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:])),
    )
    for term in terms:
        encoded = term.encode("utf-8")
        if algorithm == "fnv1a64":
            digest = _stable_hash(term)
            sign_bit = 1 << 63
        elif algorithm == "crc32":
            digest = zlib.crc32(encoded)
            sign_bit = 1 << 31
        else:
            digest = int.from_bytes(
                hashlib.blake2b(encoded, digest_size=8).digest(), "little"
            )
            sign_bit = 1 << 63
        values[digest & (bins - 1)] += -1.0 if digest & sign_bit else 1.0
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm:
        values = [value / norm for value in values]
    return tuple(values)


def prompt_family(episode: Episode) -> str:
    """Coarse content-only bucket for the prompt."""

    text = episode_text(episode)
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
    return "other"


def _quality_prediction(
    episode: Episode,
    artifact: Mapping[str, Any],
    raw: Optional[Tuple[float, ...]] = None,
) -> float:
    quality = artifact["quality"]
    raw = structural_features(episode) if raw is None else raw
    scale = quality["scale"]
    coefficients = quality["coefficients"]
    return max(
        -1.0,
        min(
            1.0,
            float(quality["intercept"][0])
            + math.fsum(
                value / float(current_scale) * float(coefficient)
                for value, current_scale, coefficient in zip(raw, scale, coefficients)
            ),
        ),
    )


def _token_predictions(
    episode: Episode,
    artifact: Mapping[str, Any],
    structural: Optional[Tuple[float, ...]] = None,
) -> Tuple[float, ...]:
    cost = artifact["cost"]
    hash_bins = int(cost["hash_bins"])
    raw = structural_features(episode) if structural is None else structural
    if hash_bins:
        raw += hashed_features(
            episode,
            hash_bins,
            str(cost.get("hash_algorithm", "fnv1a64")),
        )
    standardized = tuple(
        (value - float(mean)) / float(scale)
        for value, mean, scale in zip(raw, cost["feature_mean"], cost["feature_scale"])
    )
    transformed = [float(value) for value in cost["target_intercept"]]
    for feature, row in zip(standardized, cost["coefficients"]):
        for target_index, coefficient in enumerate(row):
            transformed[target_index] += feature * float(coefficient)
    predictions = [
        math.expm1(min(math.log1p(10_000_000.0), max(0.0, value)))
        for value in transformed
    ]
    smearing = cost["smearing_factors"]
    factors = smearing.get(prompt_family(episode), smearing["__global__"])
    return tuple(
        max(0.0, (prediction + 1.0) * float(factor) - 1.0)
        for prediction, factor in zip(predictions, factors)
    )


def predict_episode(
    episode: Episode, policy: RoutingPolicy, artifact: CalibratedArtifact
) -> Tuple[float, Tuple[float, float]]:
    """Predict the uplift of ax31 over ax31-light and what each costs."""

    value = artifact.value
    uplift = _quality_prediction(episode, value)
    tokens = _token_predictions(episode, value)
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


def _premium_prediction(
    episode: Episode, policy: RoutingPolicy, artifact: CalibratedArtifact
) -> Tuple[float, Tuple[float, float, float]]:
    """Return the OOD-shrunk uplift and the conservatively inflated costs."""

    value = artifact.value
    overlay = value["premium_overlay"]
    structural = structural_features(episode)
    uplift = _quality_prediction(episode, value, structural)
    ood = overlay["quality"]["ood"]
    distance = math.sqrt(
        math.fsum(
            ((feature - float(mean)) / float(scale)) ** 2
            for feature, mean, scale in zip(
                structural, ood["feature_mean"], ood["feature_scale"]
            )
        )
        / len(structural)
    )
    weight = min(1.0, float(ood["threshold"]) / max(distance, 1e-12))
    if uplift > 0.0:
        uplift *= weight

    bounds = overlay["cost"]["token_upper_bounds"]
    tokens = tuple(
        min(max(token, 0.0), float(bound))
        for token, bound in zip(_token_predictions(episode, value, structural), bounds)
    )
    point_costs = []
    for model_index, model_id in enumerate(MODEL_IDS):
        rates = policy.models[model_id]
        point_costs.append(
            max(
                1e-15,
                float(rates.fixed_cost)
                + tokens[2 * model_index]
                * float(rates.input_token_rate)
                / float(policy.token_unit)
                + tokens[2 * model_index + 1]
                * float(rates.output_token_rate)
                / float(policy.token_unit),
            )
        )
    step = float(overlay["minimum_model_cost_step_ratio"])
    point_costs[1] = max(point_costs[1], point_costs[0] * step)
    point_costs[2] = max(point_costs[2], point_costs[1] * step)
    residual = overlay["cost"]["residual_upper"]
    kappa = float(overlay["kappa_q999"])
    safe_costs = [point_costs[0]]
    for model_index, model_id in enumerate(MODEL_IDS[1:], start=1):
        increment = max(
            0.0,
            point_costs[model_index] - point_costs[0] + float(residual[model_id]),
        )
        safe_costs.append(point_costs[0] + kappa * increment)
    safe_costs[1] = max(safe_costs[1], safe_costs[0] * step)
    safe_costs[2] = max(safe_costs[2], safe_costs[1] * step)
    return uplift, (safe_costs[0], safe_costs[1], safe_costs[2])


def _select_ax31(
    predictions: Sequence[Tuple[float, Tuple[float, float]]], cap_ratio: float
) -> Tuple[Tuple[str, ...], float]:
    light_total = math.fsum(costs[0] for _uplift, costs in predictions)
    cap = cap_ratio * light_total
    selected = [MODEL_IDS[0]] * len(predictions)
    ratio_groups: dict[float, list[Tuple[int, float]]] = {}
    for index, (uplift, costs) in enumerate(predictions):
        incremental_cost = costs[1] - costs[0]
        if uplift > 0.0 and incremental_cost > 0.0:
            ratio = uplift * light_total / incremental_cost
            ratio_groups.setdefault(ratio, []).append((index, incremental_cost))
    used = 0.0
    for ratio in sorted(ratio_groups, reverse=True):
        group = ratio_groups[ratio]
        group_cost = math.fsum(cost for _index, cost in group)
        if math.fsum((light_total, used, group_cost)) > cap:
            break
        for index, _cost in group:
            selected[index] = MODEL_IDS[1]
        used = math.fsum((used, group_cost))
    total = math.fsum((light_total, used))
    return tuple(selected), total / light_total


def _select_premium(
    inputs: InputBatch,
    predictions: Sequence[Tuple[float, Tuple[float, float, float]]],
    cap_ratio: float,
) -> Tuple[Tuple[str, ...], float]:
    """Match the revalidated indivisible exact-content group allocator."""

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
            -1_000_000.0 * len(indexes),
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


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: CalibratedArtifact,
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
    if tier == "premium" and "premium_overlay" in value:
        overlay = value["premium_overlay"]
        predicted_cap = float(overlay["safe_cap"])
        predictions = tuple(
            _premium_prediction(episode, policy, artifact)
            for episode in inputs.episodes
        )
        selected, ratio = _select_premium(inputs, predictions, predicted_cap)
    else:
        kappa = float(value["tier_kappa_q999"][tier])
        safe_cap = float(value["safe_tier_caps"][tier])
        official = float(policy.tiers[tier].budget_multiplier)
        predicted_cap = min(
            1.0 + (safe_cap - 1.0) / kappa,
            1.0 + (0.98 * official - 1.0) / kappa,
        )
        predictions = tuple(
            predict_episode(episode, policy, artifact) for episode in inputs.episodes
        )
        selected, ratio = _select_ax31(predictions, predicted_cap)
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point."""

    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        plan = make_submission(inputs, policy, load_bundled_artifact(), args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"OK: {args.tier} final submission "
        f"(predicted ratio {plan.predicted_budget_ratio:.6f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
