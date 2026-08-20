# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic helpers shared by the experiments: splits, folds, families."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from ossp_router.heuristic import episode_text
from ossp_router.protocol import MODEL_IDS, Episode, InputBatch, OutcomeBatch, RoutingPolicy


SAFE_TIER_CAPS = {
    "fast": Decimal("1.15"),
    "balanced": Decimal("1.75"),
    "premium": Decimal("3.25"),
}
_TOKEN = re.compile(r"[A-Za-zÀ-ɏ가-힣]+|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?![\w])")
_SPACE = re.compile(r"\s+")
_KOREAN = re.compile(r"[가-힣]")
_CHOICE = re.compile(r"(?:^|\n)\s*(?:[A-D][.)]|\([a-e]\))\s", re.IGNORECASE)
_WORD_PROBLEM = re.compile(
    r"\b(?:how many|how much|how long|how far|total|each|costs?|average|"
    r"percent|percentage|left over|altogether)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublicArrays:
    """Dense public outcome matrices aligned to input episode order."""

    scores: np.ndarray
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    costs: np.ndarray
    generations: np.ndarray


@dataclass(frozen=True)
class OraclePoint:
    """An exact public-data quality/cost point and its model allocation."""

    quality: float
    cost_ratio: float
    model_counts: Mapping[str, int]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def prompt_family(episode: Episode) -> str:
    """Return a coarse, content-only prompt family used only for validation."""

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
    if _WORD_PROBLEM.search(text) and not any(marker in text for marker in ("$", "\\[", "**")):
        return "word_problem"
    if any(marker in text for marker in ("$", "\\[", "\\frac", "\\begin")):
        return "latex_math"
    if any(marker in lowered for marker in ("calculate", "solve ", "let ", "wrt", "divide", "factor")):
        return "symbolic_math"
    return "other"


def normalized_template(episode: Episode) -> str:
    """Canonicalize superficial variation without using metadata or source IDs."""

    text = unicodedata.normalize("NFKC", episode_text(episode)).casefold()
    text = _NUMBER.sub("<number>", text)
    text = re.sub(r"\b[0-9a-f]{12,}\b", "<hex>", text)
    return _SPACE.sub(" ", text).strip()


def _shingles(episode: Episode) -> frozenset[str]:
    text = normalized_template(episode)
    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    if len(tokens) > 1_200:
        tokens = tokens[:500] + tokens[-500:]
    if len(tokens) < 3:
        return frozenset(tokens or ("<empty>",))
    return frozenset(
        "\x1f".join(tokens[index : index + 3])
        for index in range(len(tokens) - 2)
    )


def _simhash(shingles: Iterable[str]) -> int:
    votes = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            votes[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, vote in enumerate(votes):
        if vote >= 0:
            result |= 1 << bit
    return result


def prompt_group_keys(
    episodes: Sequence[Episode], *, maximum_hamming_distance: int = 8
) -> Tuple[str, ...]:
    """Cluster exact templates and near-duplicate prompts using content only."""

    if maximum_hamming_distance < 0 or maximum_hamming_distance > 64:
        raise ValueError("maximum_hamming_distance must be between 0 and 64")
    count = len(episodes)
    union = _DisjointSet(count)
    templates: Dict[str, int] = {}
    by_family: Dict[str, list[int]] = defaultdict(list)
    signatures = []
    digests = []
    for index, episode in enumerate(episodes):
        template = normalized_template(episode)
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        digests.append(digest)
        if template in templates:
            union.union(index, templates[template])
        else:
            templates[template] = index
        signatures.append(_simhash(_shingles(episode)))
        by_family[prompt_family(episode)].append(index)

    # Public data is small. Pairwise comparison within coarse content families is
    # clearer and more deterministic than an approximate nearest-neighbour index.
    for indexes in by_family.values():
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1 :]:
                if (signatures[left] ^ signatures[right]).bit_count() <= maximum_hamming_distance:
                    union.union(left, right)

    members: Dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        members[union.find(index)].append(index)
    group_digest = {
        root: min(digests[index] for index in indexes)
        for root, indexes in members.items()
    }
    return tuple(group_digest[union.find(index)] for index in range(count))


def assign_group_folds(
    episodes: Sequence[Episode], *, folds: int, seed: int
) -> Tuple[int, ...]:
    """Greedily balance content groups and prompt families across folds."""

    if folds < 2 or folds > len(episodes):
        raise ValueError("folds must be between 2 and the number of episodes")
    group_keys = prompt_group_keys(episodes)
    members: Dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        members[key].append(index)
    families = [prompt_family(episode) for episode in episodes]
    group_rows = []
    for key, indexes in members.items():
        family_counts = Counter(families[index] for index in indexes)
        tie = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
        group_rows.append((key, indexes, family_counts, tie))
    group_rows.sort(key=lambda row: (-len(row[1]), row[3]))

    fold_sizes = [0] * folds
    fold_families = [Counter() for _ in range(folds)]
    assignments = [-1] * len(episodes)
    for _key, indexes, family_counts, _tie in group_rows:
        chosen = min(
            range(folds),
            key=lambda fold: (
                fold_sizes[fold],
                sum(
                    fold_families[fold][family] * amount
                    for family, amount in family_counts.items()
                ),
                fold,
            ),
        )
        for index in indexes:
            assignments[index] = chosen
        fold_sizes[chosen] += len(indexes)
        fold_families[chosen].update(family_counts)
    return tuple(assignments)


def public_arrays(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy
) -> PublicArrays:
    """Validate and align the complete public outcome matrix."""

    if (
        inputs.schema_version != outcomes.schema_version
        or inputs.challenge_id != outcomes.challenge_id
        or inputs.split != outcomes.split
    ):
        raise ValueError("input and outcome metadata do not match")
    index = {
        (outcome.episode_id, outcome.model_id): outcome
        for outcome in outcomes.outcomes
    }
    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    }
    if set(index) != expected:
        raise ValueError("outcomes do not form a complete episode/model matrix")
    shape = (len(inputs.episodes), len(MODEL_IDS))
    scores = np.empty(shape, dtype=np.float64)
    input_tokens = np.empty(shape, dtype=np.float64)
    output_tokens = np.empty(shape, dtype=np.float64)
    generations = np.empty(shape, dtype=np.int64)
    costs = np.empty(shape, dtype=np.float64)
    unit = Decimal(policy.token_unit)
    for row, episode in enumerate(inputs.episodes):
        for column, model_id in enumerate(MODEL_IDS):
            outcome = index[(episode.episode_id, model_id)]
            rates = policy.models[model_id]
            cost = (
                rates.fixed_cost
                + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
                + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
            )
            scores[row, column] = float(outcome.score)
            input_tokens[row, column] = outcome.input_tokens
            output_tokens[row, column] = outcome.output_tokens
            generations[row, column] = outcome.num_generations
            costs[row, column] = float(cost)
    return PublicArrays(scores, input_tokens, output_tokens, costs, generations)


def _nanocredit_costs(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy
) -> np.ndarray:
    arrays = public_arrays(inputs, outcomes, policy)
    result = np.rint(arrays.costs * 1_000_000_000).astype(np.int64)
    if not np.allclose(result / 1_000_000_000, arrays.costs, rtol=0, atol=1e-15):
        raise ValueError("public costs cannot be represented as integer nanocredits")
    return result


def exact_oracle_points(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    caps: Sequence[Decimal],
) -> Mapping[str, OraclePoint]:
    """Solve every requested public MCKP cap exactly with one dynamic program."""

    arrays = public_arrays(inputs, outcomes, policy)
    score_units = np.rint(arrays.scores * 4).astype(np.int16)
    if not np.allclose(score_units / 4, arrays.scores, rtol=0, atol=1e-12):
        raise ValueError("exact oracle requires public scores in quarter-point units")
    costs = _nanocredit_costs(inputs, outcomes, policy)
    rows = len(inputs.episodes)
    maximum_points = 4 * rows
    infinity = np.iinfo(np.int64).max // 4
    previous = np.full(maximum_points + 1, infinity, dtype=np.int64)
    previous[0] = 0
    choices = np.full((rows, maximum_points + 1), -1, dtype=np.int8)
    reachable = 0
    for row in range(rows):
        current = np.full(maximum_points + 1, infinity, dtype=np.int64)
        row_choice = choices[row]
        next_reachable = reachable + 4
        for model_index in range(len(MODEL_IDS)):
            points = int(score_units[row, model_index])
            source = previous[: reachable + 1]
            candidate = source + costs[row, model_index]
            target = current[points : points + reachable + 1]
            better = candidate < target
            target[better] = candidate[better]
            row_choice[points : points + reachable + 1][better] = model_index
        previous = current
        reachable = next_reachable

    light_total = int(costs[:, 0].sum())
    results: Dict[str, OraclePoint] = {}
    for cap in caps:
        with localcontext() as context:
            context.prec = 80
            budget = int(
                (Decimal(light_total) * cap).to_integral_value(rounding=ROUND_FLOOR)
            )
        feasible = np.flatnonzero(previous <= budget)
        if not len(feasible):
            raise RuntimeError(f"no feasible oracle allocation for cap {cap}")
        points = int(feasible[-1])
        total_cost = int(previous[points])
        counts = Counter()
        cursor = points
        for row in range(rows - 1, -1, -1):
            model_index = int(choices[row, cursor])
            if model_index < 0:
                raise RuntimeError("oracle backtracking failed")
            counts[MODEL_IDS[model_index]] += 1
            cursor -= int(score_units[row, model_index])
        results[str(cap)] = OraclePoint(
            quality=points / (4 * rows),
            cost_ratio=total_cost / light_total,
            model_counts={model_id: counts[model_id] for model_id in MODEL_IDS},
        )
    return results


def quantile_higher(values: np.ndarray, probability: float) -> float:
    """Return an observed upper quantile with deterministic finite-sample behavior."""

    if not 0 <= probability <= 1 or values.size == 0:
        raise ValueError("quantile requires non-empty values and probability in [0, 1]")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[max(0, index)])
