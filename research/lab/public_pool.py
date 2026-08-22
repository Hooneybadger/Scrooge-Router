# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Load the public Train+Dev pool for grouped cross-fitting.

Unlike ``research.lab.modeling.load_train``, this loader is allowed to open
Dev. Candidate selection still uses grouped OOF only; a full-public refit is
out of scope for E1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ossp_router.protocol import (
    MODEL_IDS,
    Episode,
    InputBatch,
    Outcome,
    OutcomeBatch,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)
from research.lab.grouped_crossfit import (
    FOLD_SEED,
    FOLDS,
    GROUPING_VERSION,
    assign_balanced_group_folds,
    families_of,
    fold_balance,
    fold_leakage_count,
    group_episodes,
    language_view,
    length_view,
)
from research.lab.prompt_features import episode_text_of
from research.lab.validation import public_arrays


ROOT = Path(__file__).resolve().parents[2]
TRAIN_INPUTS = ROOT / "data" / "materialized" / "train" / "inputs.json"
TRAIN_OUTCOMES = ROOT / "data" / "train" / "outcomes.json"
DEV_INPUTS = ROOT / "data" / "materialized" / "dev" / "inputs.json"
DEV_OUTCOMES = ROOT / "data" / "dev" / "outcomes.json"

EXPECTED_TRAIN_INPUTS_SHA256 = (
    "029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4"
)
EXPECTED_TRAIN_OUTCOMES_SHA256 = (
    "97a5a787086b3e1d9fa9c7945518543540e527ea248df4a4760de581b612a4ba"
)
EXPECTED_DEV_INPUTS_SHA256 = (
    "5920f9ea9e3da147aa546659054feb08afb7e11a0e4db6967b293ff79b759abc"
)
EXPECTED_DEV_OUTCOMES_SHA256 = (
    "acb7c5ed522c4e1b65e9ab14b3fe9458fcba32eb3d9de8d3f53e24b8904d2e66"
)
EXPECTED_N_TRAIN = 1760
EXPECTED_N_DEV = 880
EXPECTED_N_PUBLIC = EXPECTED_N_TRAIN + EXPECTED_N_DEV
PUBLIC_SPLIT = "public"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combine_inputs(train: InputBatch, dev: InputBatch) -> InputBatch:
    if train.schema_version != dev.schema_version:
        raise ValueError("train/dev schema_version mismatch")
    if train.challenge_id != dev.challenge_id:
        raise ValueError("train/dev challenge_id mismatch")
    train_ids = {episode.episode_id for episode in train.episodes}
    overlap = train_ids.intersection(episode.episode_id for episode in dev.episodes)
    if overlap:
        raise ValueError(f"train/dev episode_id overlap: {sorted(overlap)[:8]}")
    return InputBatch(
        schema_version=train.schema_version,
        challenge_id=train.challenge_id,
        split=PUBLIC_SPLIT,
        episodes=train.episodes + dev.episodes,
    )


def _combine_outcomes(train: OutcomeBatch, dev: OutcomeBatch) -> OutcomeBatch:
    if train.schema_version != dev.schema_version:
        raise ValueError("train/dev outcome schema_version mismatch")
    if train.challenge_id != dev.challenge_id:
        raise ValueError("train/dev outcome challenge_id mismatch")
    return OutcomeBatch(
        schema_version=train.schema_version,
        challenge_id=train.challenge_id,
        split=PUBLIC_SPLIT,
        outcomes=train.outcomes + dev.outcomes,
    )


def subset_inputs(inputs: InputBatch, indexes: Sequence[int]) -> InputBatch:
    episodes = tuple(inputs.episodes[index] for index in indexes)
    return InputBatch(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        split=inputs.split,
        episodes=episodes,
    )


def subset_outcomes(
    inputs: InputBatch, outcomes: OutcomeBatch, indexes: Sequence[int]
) -> OutcomeBatch:
    wanted = {inputs.episodes[index].episode_id for index in indexes}
    selected: list[Outcome] = [
        outcome for outcome in outcomes.outcomes if outcome.episode_id in wanted
    ]
    return OutcomeBatch(
        schema_version=outcomes.schema_version,
        challenge_id=outcomes.challenge_id,
        split=outcomes.split,
        outcomes=tuple(selected),
    )


@dataclass(frozen=True)
class PublicPool:
    """Train+Dev episodes aligned with public scores, costs, and CV groups."""

    episodes: Tuple[Episode, ...]
    texts: Tuple[str, ...]
    families: Tuple[str, ...]
    languages: Tuple[str, ...]
    length_views: Tuple[str, ...]
    group_keys: Tuple[str, ...]
    exact_keys: Tuple[str, ...]
    template_keys: Tuple[str, ...]
    folds: Tuple[int, ...]
    scores: np.ndarray
    costs: np.ndarray
    light_total: float
    identity: Mapping[str, Any]
    grouping: Mapping[str, Any]
    fold_table: Sequence[Mapping[str, Any]]
    inputs: InputBatch
    outcomes: OutcomeBatch
    policy: RoutingPolicy
    split_labels: Tuple[str, ...]


def load_public_pool(
    *,
    folds: int = FOLDS,
    fold_seed: int = FOLD_SEED,
    policy: RoutingPolicy | None = None,
) -> PublicPool:
    """Load and pin the 2,640 public episodes, then assign grouped folds."""

    if MODEL_IDS != ("ax31-light", "ax31", "axk1-think"):
        raise RuntimeError(f"MODEL_IDS drifted: {MODEL_IDS!r}")

    paths = {
        "dev_inputs": DEV_INPUTS,
        "dev_outcomes": DEV_OUTCOMES,
        "train_inputs": TRAIN_INPUTS,
        "train_outcomes": TRAIN_OUTCOMES,
    }
    digests = {name: sha256_path(path) for name, path in paths.items()}
    expected = {
        "dev_inputs": EXPECTED_DEV_INPUTS_SHA256,
        "dev_outcomes": EXPECTED_DEV_OUTCOMES_SHA256,
        "train_inputs": EXPECTED_TRAIN_INPUTS_SHA256,
        "train_outcomes": EXPECTED_TRAIN_OUTCOMES_SHA256,
    }
    for name, digest in digests.items():
        if digest != expected[name]:
            raise ValueError(f"{name}-hash-mismatch: got {digest}, expected {expected[name]}")

    train_inputs = load_input(TRAIN_INPUTS)
    train_outcomes = load_outcomes(TRAIN_OUTCOMES)
    dev_inputs = load_input(DEV_INPUTS)
    dev_outcomes = load_outcomes(DEV_OUTCOMES)
    if len(train_inputs.episodes) != EXPECTED_N_TRAIN:
        raise ValueError(f"expected {EXPECTED_N_TRAIN} train episodes")
    if len(dev_inputs.episodes) != EXPECTED_N_DEV:
        raise ValueError(f"expected {EXPECTED_N_DEV} dev episodes")
    if train_inputs.split != "train" or train_outcomes.split != "train":
        raise ValueError("train split labels are not train")
    if dev_inputs.split != "dev" or dev_outcomes.split != "dev":
        raise ValueError("dev split labels are not dev")

    bundled = policy or load_bundled_policy()
    inputs = _combine_inputs(train_inputs, dev_inputs)
    outcomes = _combine_outcomes(train_outcomes, dev_outcomes)
    arrays = public_arrays(inputs, outcomes, bundled)
    episodes = inputs.episodes
    texts = tuple(episode_text_of(episode) for episode in episodes)
    families = families_of(episodes)
    grouping = group_episodes(episodes)
    fold_ids = assign_balanced_group_folds(
        grouping.group_keys, families, folds=folds, seed=fold_seed
    )
    leaked = fold_leakage_count(grouping.group_keys, fold_ids)
    if leaked:
        raise RuntimeError(f"grouped fold leakage: {leaked} groups split across folds")

    identity = {
        "dev_inputs_sha256": digests["dev_inputs"],
        "dev_outcomes_sha256": digests["dev_outcomes"],
        "fold_seed": int(fold_seed),
        "folds": int(folds),
        "grouping_version": GROUPING_VERSION,
        "model_ids": list(MODEL_IDS),
        "n_dev": EXPECTED_N_DEV,
        "n_episodes": len(episodes),
        "n_train": EXPECTED_N_TRAIN,
        "policy_sha256": policy_sha256(bundled),
        "split": PUBLIC_SPLIT,
        "train_inputs_sha256": digests["train_inputs"],
        "train_outcomes_sha256": digests["train_outcomes"],
    }
    grouping_record = {
        "blocking": dict(grouping.blocking),
        "group_size_histogram": dict(grouping.group_size_histogram),
        "largest_group": grouping.largest_group,
        "n_exact_groups": grouping.n_exact_groups,
        "n_groups": grouping.n_groups,
        "n_jaccard_comparisons": grouping.n_jaccard_comparisons,
        "n_near_duplicate_unions": grouping.n_near_duplicate_unions,
        "n_singleton_groups": grouping.n_singleton_groups,
        "n_template_groups": grouping.n_template_groups,
    }
    return PublicPool(
        episodes=episodes,
        texts=texts,
        families=families,
        languages=tuple(language_view(text) for text in texts),
        length_views=tuple(length_view(text) for text in texts),
        group_keys=grouping.group_keys,
        exact_keys=grouping.exact_keys,
        template_keys=grouping.template_keys,
        folds=fold_ids,
        scores=np.asarray(arrays.scores, dtype=np.float64),
        costs=np.asarray(arrays.costs, dtype=np.float64),
        light_total=float(arrays.costs[:, 0].sum()),
        identity=identity,
        grouping=grouping_record,
        fold_table=fold_balance(grouping.group_keys, fold_ids, families),
        inputs=inputs,
        outcomes=outcomes,
        policy=bundled,
        split_labels=("train",) * EXPECTED_N_TRAIN + ("dev",) * EXPECTED_N_DEV,
    )


__all__ = (
    "EXPECTED_DEV_INPUTS_SHA256",
    "EXPECTED_DEV_OUTCOMES_SHA256",
    "EXPECTED_N_PUBLIC",
    "EXPECTED_TRAIN_INPUTS_SHA256",
    "EXPECTED_TRAIN_OUTCOMES_SHA256",
    "PUBLIC_SPLIT",
    "PublicPool",
    "load_public_pool",
    "sha256_path",
    "subset_inputs",
    "subset_outcomes",
)
