# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic grouped cross-fit foundation for the public Train+Dev pool.

Group keys are content-only. Episode IDs and input order are never features
and never enter the group identifier. Exact canonical-prompt duplicates are
always united. Near-duplicates use character 5-gram Jaccard >= 0.90 after a
length-ratio block that is a necessary condition for that threshold, so the
block introduces no false negatives for the stated rule.

Template grouping uses the same number/hex canonicalization as
``research.lab.validation.normalized_template``. Source IDs are not present
on the public Episode schema and are not used.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from ossp_router.cost_calibrated_router import prompt_family
from ossp_router.heuristic import episode_text
from ossp_router.protocol import Episode


GROUPING_VERSION = "grouped-crossfit.v1"
FOLD_SEED = 20260821
FOLDS = 5
JACCARD_THRESHOLD = 0.90
CHAR_NGRAM = 5
# Jaccard >= t requires |A|/|B| >= t when |A| <= |B|.
LENGTH_RATIO_MIN = JACCARD_THRESHOLD

_NUMBER = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?![\w])")
_SPACE = re.compile(r"\s+")
_HANGUL = re.compile(r"[가-힣]")


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


def canonical_prompt(episode: Episode) -> str:
    """Exact routing-time prompt. Used for exact-duplicate grouping."""

    return episode_text(episode)


def content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_template_text(text: str) -> str:
    """Number/hex-canonical template. Content only; no IDs."""

    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = _NUMBER.sub("<number>", folded)
    folded = re.sub(r"\b[0-9a-f]{12,}\b", "<hex>", folded)
    return _SPACE.sub(" ", folded).strip()


def near_duplicate_text(text: str) -> str:
    """Canonical form for character n-grams: NFKC, casefold, collapsed space."""

    folded = unicodedata.normalize("NFKC", text).casefold()
    return _SPACE.sub(" ", folded).strip()


def char_ngrams(text: str, width: int = CHAR_NGRAM) -> frozenset[str]:
    if width < 1:
        raise ValueError("n-gram width must be >= 1")
    if len(text) < width:
        return frozenset((text or "<empty>",))
    return frozenset(text[index : index + width] for index in range(len(text) - width + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def language_view(text: str) -> str:
    return "korean" if _HANGUL.search(text) else "non_korean"


def length_view(text: str) -> str:
    count = len(text)
    if count < 120:
        return "len_lt_120"
    if count < 400:
        return "len_120_399"
    if count < 2000:
        return "len_400_1999"
    if count < 8000:
        return "len_2000_7999"
    return "len_ge_8000"


@dataclass(frozen=True)
class GroupingResult:
    """Order-invariant group labels and the diagnostics needed for the report."""

    group_keys: Tuple[str, ...]
    exact_keys: Tuple[str, ...]
    template_keys: Tuple[str, ...]
    n_episodes: int
    n_exact_groups: int
    n_template_groups: int
    n_groups: int
    n_near_duplicate_unions: int
    n_jaccard_comparisons: int
    n_singleton_groups: int
    largest_group: int
    group_size_histogram: Mapping[int, int]
    blocking: Mapping[str, object]


def _length_feasible(left_size: int, right_size: int) -> bool:
    smaller = min(left_size, right_size)
    larger = max(left_size, right_size)
    if larger == 0:
        return True
    return smaller / larger >= LENGTH_RATIO_MIN


def group_episodes(episodes: Sequence[Episode]) -> GroupingResult:
    """Assign content groups. Result is invariant to episode order."""

    count = len(episodes)
    if count == 0:
        return GroupingResult(
            group_keys=(),
            exact_keys=(),
            template_keys=(),
            n_episodes=0,
            n_exact_groups=0,
            n_template_groups=0,
            n_groups=0,
            n_near_duplicate_unions=0,
            n_jaccard_comparisons=0,
            n_singleton_groups=0,
            largest_group=0,
            group_size_histogram={},
            blocking={},
        )

    union = _DisjointSet(count)
    texts = [canonical_prompt(episode) for episode in episodes]
    exact_keys = [content_digest(text) for text in texts]
    template_keys = [content_digest(normalized_template_text(text)) for text in texts]
    near_texts = [near_duplicate_text(text) for text in texts]

    by_exact: Dict[str, int] = {}
    for index, key in enumerate(exact_keys):
        previous = by_exact.get(key)
        if previous is None:
            by_exact[key] = index
        else:
            union.union(index, previous)

    by_template: Dict[str, int] = {}
    for index, key in enumerate(template_keys):
        previous = by_template.get(key)
        if previous is None:
            by_template[key] = index
        else:
            union.union(index, previous)

    unique_near: Dict[str, int] = {}
    unique_order: list[tuple[str, int, int]] = []
    for index, near in enumerate(near_texts):
        previous = unique_near.get(near)
        if previous is not None:
            union.union(index, previous)
            continue
        unique_near[near] = index
        unique_order.append((content_digest(near), len(char_ngrams(near)), index))
    grams_by_index = {
        index: char_ngrams(near_texts[index]) for _digest, _size, index in unique_order
    }
    unique_order.sort(key=lambda row: (row[1], row[0]))

    n_comparisons = 0
    n_hits = 0
    for offset, (_left_digest, left_size, left_index) in enumerate(unique_order):
        for _right_digest, right_size, right_index in unique_order[offset + 1 :]:
            if not _length_feasible(left_size, right_size):
                break
            n_comparisons += 1
            score = jaccard(grams_by_index[left_index], grams_by_index[right_index])
            if score >= JACCARD_THRESHOLD:
                n_hits += 1
                union.union(left_index, right_index)

    members: Dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        members[union.find(index)].append(index)
    group_digest = {
        root: min(exact_keys[index] for index in indexes)
        for root, indexes in members.items()
    }
    group_keys = tuple(group_digest[union.find(index)] for index in range(count))
    sizes = Counter(len(indexes) for indexes in members.values())
    return GroupingResult(
        group_keys=group_keys,
        exact_keys=tuple(exact_keys),
        template_keys=tuple(template_keys),
        n_episodes=count,
        n_exact_groups=len(by_exact),
        n_template_groups=len(by_template),
        n_groups=len(members),
        n_near_duplicate_unions=int(n_hits),
        n_jaccard_comparisons=int(n_comparisons),
        n_singleton_groups=int(sizes.get(1, 0)),
        largest_group=max(len(indexes) for indexes in members.values()),
        group_size_histogram={int(size): int(amount) for size, amount in sorted(sizes.items())},
        blocking={
            "char_ngram": CHAR_NGRAM,
            "false_negative_note": (
                "Length-ratio blocking is a necessary condition for Jaccard "
                f">= {JACCARD_THRESHOLD}: if the smaller 5-gram set is below "
                f"{LENGTH_RATIO_MIN} of the larger, Jaccard cannot reach the "
                "threshold. Within a feasible window every unique near-dup "
                "canonical is compared exactly. Residual false negatives are "
                "only texts whose NFKC/casefold/whitespace form falls below "
                "0.90 after that canonicalization, or that differ only in "
                "ways the template rule does not capture."
            ),
            "jaccard_threshold": JACCARD_THRESHOLD,
            "length_ratio_min": LENGTH_RATIO_MIN,
            "method": "exact-sha + template-sha + length-feasible exact Jaccard",
            "source_grouping": False,
            "source_grouping_note": (
                "Public Episode records have no source/template provenance "
                "field. Source IDs are not inferred from episode_id."
            ),
            "template_grouping": True,
            "version": GROUPING_VERSION,
        },
    )


def assign_balanced_group_folds(
    group_keys: Sequence[str],
    families: Sequence[str],
    *,
    folds: int = FOLDS,
    seed: int = FOLD_SEED,
) -> Tuple[int, ...]:
    """Greedy size-and-family balance over groups. Order-invariant."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(group_keys) != len(families):
        raise ValueError("group_keys and families must align")
    if folds > len(group_keys):
        raise ValueError("folds must not exceed the number of episodes")

    members: Dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        members[key].append(index)
    group_rows = []
    for key, indexes in members.items():
        family_counts = Counter(families[index] for index in indexes)
        tie = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
        group_rows.append((key, indexes, family_counts, tie))
    group_rows.sort(key=lambda row: (-len(row[1]), row[3], row[0]))

    fold_sizes = [0] * folds
    fold_families = [Counter() for _ in range(folds)]
    assignments = [-1] * len(group_keys)
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
    if any(value < 0 for value in assignments):
        raise RuntimeError("fold assignment missed an episode")
    return tuple(assignments)


def fold_leakage_count(group_keys: Sequence[str], folds: Sequence[int]) -> int:
    """Return how many unique groups appear in more than one fold."""

    seen: Dict[str, set[int]] = defaultdict(set)
    for key, fold in zip(group_keys, folds):
        seen[key].add(int(fold))
    return sum(1 for assigned in seen.values() if len(assigned) > 1)


def fold_balance(
    group_keys: Sequence[str],
    folds: Sequence[int],
    families: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    n_folds = 1 + max(folds) if folds else 0
    for fold in range(n_folds):
        indexes = [index for index, value in enumerate(folds) if value == fold]
        family_counts = Counter(families[index] for index in indexes)
        group_count = len({group_keys[index] for index in indexes})
        rows.append(
            {
                "family_counts": {name: int(amount) for name, amount in sorted(family_counts.items())},
                "fold": fold,
                "n_episodes": len(indexes),
                "n_groups": group_count,
            }
        )
    return rows


def families_of(episodes: Sequence[Episode]) -> Tuple[str, ...]:
    return tuple(prompt_family(episode) for episode in episodes)


__all__ = (
    "CHAR_NGRAM",
    "FOLD_SEED",
    "FOLDS",
    "GROUPING_VERSION",
    "GroupingResult",
    "JACCARD_THRESHOLD",
    "assign_balanced_group_folds",
    "canonical_prompt",
    "char_ngrams",
    "content_digest",
    "families_of",
    "fold_balance",
    "fold_leakage_count",
    "group_episodes",
    "jaccard",
    "language_view",
    "length_view",
    "near_duplicate_text",
    "normalized_template_text",
)
