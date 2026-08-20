# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the modeling foundation shared foundation: features, folds, ridge, recalibration, scoring."""

from __future__ import annotations

import hashlib
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from research.lab.prompt_features import (
    ALLOWED_HASH_BINS,
    FEATURE_VERSION,
    STRUCTURAL_FEATURE_NAMES,
    episode_text_of,
    feature_row,
    feature_signature,
)
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    OutcomeBatch,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)
from ossp_router.scoring import score_submissions
from research.lab.validation import (
    assign_group_folds,
    prompt_family,
    prompt_group_keys,
    public_arrays,
    quantile_higher,
)


EXPERIMENT = "the modeling foundation"
REPORT_TYPE = "scrooge-modeling-foundation-v1"
SCHEMA_VERSION = 1
DECISION = "record-modeling-foundation"
FOLD_SEED = 2026082202
BOOTSTRAP_SEED = 2026082203
FOLDS = 5
HASH_BINS: Tuple[int, ...] = ALLOWED_HASH_BINS
TIER_WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
OFFICIAL_CAPS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
OPERATING_TARGETS = {"fast": 1.15, "balanced": 1.80, "premium": 3.40}
STRESS_BACKSTOP = 1.054
N_BINS = 10
FACTOR_CLIP = (0.5, 6.0)
FINITE_COMPARE = 1e-12
EXPECTED_MODEL_IDS = ("ax31-light", "ax31", "axk1-think")
EXPECTED_INPUTS_SHA256 = (
    "029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4"
)
EXPECTED_OUTCOMES_SHA256 = (
    "97a5a787086b3e1d9fa9c7945518543540e527ea248df4a4760de581b612a4ba"
)
SINGLE_MODEL = {
    "ax31-light": {"quality": 0.597301, "cost_ratio": 1.0},
    "ax31": {"quality": 0.678551, "cost_ratio": 2.155},
    "axk1-think": {"quality": 0.811648, "cost_ratio": 23.150},
}

# Intercept policy: feature_matrix prepends a column of ones at index 0.
# ridge_fit treats column 0 as an unpenalized intercept and does not add
# alpha to that diagonal entry. Downstream heads must keep this layout.
INTERCEPT_POLICY = (
    "column 0 is an unpenalized intercept of 1.0; "
    "columns 1..S are STRUCTURAL_FEATURE_NAMES; "
    "columns S+1..S+bins are the dense signed-hash buckets"
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_INPUTS = ROOT / "data" / "materialized" / "train" / "inputs.json"
DEFAULT_TRAIN_OUTCOMES = ROOT / "data" / "train" / "outcomes.json"


class ProtocolError(ValueError):
    """Closed-form / recalibration contract failure."""


@dataclass(frozen=True)
class TrainBundle:
    """Pinned Train split plus the public outcome matrices."""

    episodes: Tuple[Episode, ...]
    texts: Tuple[str, ...]
    families: Tuple[str, ...]
    group_keys: Tuple[str, ...]
    scores: np.ndarray
    costs: np.ndarray
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    light_total: float
    identity: Mapping[str, Any]
    inputs: InputBatch
    outcomes: OutcomeBatch
    policy: RoutingPolicy


@dataclass(frozen=True)
class RankRecal:
    """Equal-count rank isotonic recalibration (the feasibility ladder/the rank recalibration study contract)."""

    edges: np.ndarray
    raw_factors: np.ndarray
    pav_factors: np.ndarray
    clipped_factors: np.ndarray

    def apply(self, pred_inc: np.ndarray) -> np.ndarray:
        predicted = np.asarray(pred_inc, dtype=np.float64).reshape(-1)
        index = np.digitize(predicted, self.edges, right=True)
        last = int(self.clipped_factors.size) - 1
        index = np.clip(index, 0, last)
        return predicted * self.clipped_factors[index]


def sort_mapping(value: Any) -> Any:
    """Recursively sort mappings so emitted records are byte-stable."""

    if isinstance(value, Mapping):
        return {key: sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [sort_mapping(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def reject_dev_reference(path: Path) -> Path:
    """Fail closed if a public Dev data path is ever constructed."""

    text = os.path.normpath(os.fspath(path)).replace("\\", "/")
    lowered = text.lower()
    if "data/materialized/dev" in lowered or "data/dev/" in lowered or lowered.endswith("data/dev"):
        raise ValueError(f"the modeling foundation forbids constructing a Dev data path: {path}")
    parts = Path(text).parts
    for index, part in enumerate(parts):
        if part != "dev":
            continue
        parent = parts[index - 1] if index else ""
        grandparent = parts[index - 2] if index >= 2 else ""
        if parent == "data" or (parent == "materialized" and grandparent == "data"):
            raise ValueError(f"the modeling foundation forbids constructing a Dev data path: {path}")
    return path


def sha256_file(path: Path) -> str:
    reject_dev_reference(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def load_train(policy_path: Optional[Union[str, Path]] = None) -> TrainBundle:
    """Load the pinned Train split and fail closed on SHA-256 mismatch.

    Model columns follow ``ossp_router.protocol.MODEL_IDS``, which is
    asserted to be ``('ax31-light', 'ax31', 'axk1-think')``.
    """

    if MODEL_IDS != EXPECTED_MODEL_IDS:
        raise RuntimeError(
            "MODEL_IDS drifted from the the modeling foundation lock "
            f"{EXPECTED_MODEL_IDS!r}; got {MODEL_IDS!r}"
        )
    inputs_path = reject_dev_reference(DEFAULT_TRAIN_INPUTS)
    outcomes_path = reject_dev_reference(DEFAULT_TRAIN_OUTCOMES)
    inputs_digest = sha256_file(inputs_path)
    outcomes_digest = sha256_file(outcomes_path)
    if inputs_digest != EXPECTED_INPUTS_SHA256:
        raise ValueError(
            "train-inputs-hash-mismatch: "
            f"got {inputs_digest}, expected {EXPECTED_INPUTS_SHA256}"
        )
    if outcomes_digest != EXPECTED_OUTCOMES_SHA256:
        raise ValueError(
            "train-outcomes-hash-mismatch: "
            f"got {outcomes_digest}, expected {EXPECTED_OUTCOMES_SHA256}"
        )
    if policy_path is None:
        policy = load_bundled_policy()
        policy_source = "bundled"
        policy_file_sha256 = None
    else:
        resolved = reject_dev_reference(Path(policy_path))
        policy = load_policy(resolved)
        policy_source = os.fspath(resolved)
        policy_file_sha256 = sha256_file(resolved)
    inputs = load_input(inputs_path)
    outcomes = load_outcomes(outcomes_path)
    if inputs.split != "train" or outcomes.split != "train":
        raise ValueError(
            f"the modeling foundation is train-only; got inputs.split={inputs.split!r} "
            f"outcomes.split={outcomes.split!r}"
        )
    arrays = public_arrays(inputs, outcomes, policy)
    episodes = inputs.episodes
    texts = tuple(episode_text_of(episode) for episode in episodes)
    families = tuple(prompt_family(episode) for episode in episodes)
    group_keys = prompt_group_keys(episodes)
    light_total = float(arrays.costs[:, 0].sum())
    identity = {
        "feature_version": FEATURE_VERSION,
        "model_ids": list(MODEL_IDS),
        "n_episodes": len(episodes),
        "policy_file_sha256": policy_file_sha256,
        "policy_sha256": policy_sha256(policy),
        "policy_source": policy_source,
        "split": inputs.split,
        "train_inputs_sha256": inputs_digest,
        "train_inputs_sha256_expected": EXPECTED_INPUTS_SHA256,
        "train_outcomes_sha256": outcomes_digest,
        "train_outcomes_sha256_expected": EXPECTED_OUTCOMES_SHA256,
    }
    return TrainBundle(
        episodes=episodes,
        texts=texts,
        families=families,
        group_keys=group_keys,
        scores=np.asarray(arrays.scores, dtype=np.float64),
        costs=np.asarray(arrays.costs, dtype=np.float64),
        input_tokens=np.asarray(arrays.input_tokens, dtype=np.float64),
        output_tokens=np.asarray(arrays.output_tokens, dtype=np.float64),
        light_total=light_total,
        identity=identity,
        inputs=inputs,
        outcomes=outcomes,
        policy=policy,
    )


def feature_matrix(texts: Sequence[str], *, bins: int) -> np.ndarray:
    """Dense ``[intercept | structural | hashed]`` matrix.

    Intercept policy: column 0 is ``1.0`` and is the unpenalized
    intercept consumed by ``ridge_fit``. Structural columns follow
    ``STRUCTURAL_FEATURE_NAMES``. Hashed columns are the dense
    ``bins``-wide signed-FNV block (zeros for empty buckets). Built by
    calling ``research.lab.prompt_features.feature_row`` so training and
    runtime share one implementation.
    """

    width = int(bins)
    if width not in HASH_BINS:
        raise ValueError(f"hash bins must be one of {HASH_BINS}; got {bins!r}")
    rows = len(texts)
    n_struct = len(STRUCTURAL_FEATURE_NAMES)
    matrix = np.zeros((rows, 1 + n_struct + width), dtype=np.float64)
    matrix[:, 0] = 1.0
    struct_start = 1
    hash_start = 1 + n_struct
    for row, text in enumerate(texts):
        structural, hashed = feature_row(text, bins=width)
        matrix[row, struct_start:hash_start] = structural
        for bucket in sorted(hashed):
            matrix[row, hash_start + int(bucket)] = float(hashed[bucket])
    return matrix


def group_folds(
    episodes: Sequence[Episode], *, folds: int = FOLDS, seed: int
) -> Tuple[int, ...]:
    """Thin wrapper over ``assign_group_folds``."""

    return assign_group_folds(episodes, folds=folds, seed=seed)


def family_folds(
    families: Sequence[str],
) -> Tuple[Tuple[str, np.ndarray], ...]:
    """Leave-one-family-out index sets, sorted by family name."""

    names = tuple(sorted(dict.fromkeys(families)))
    array = np.asarray(list(families))
    held_out = []
    for name in names:
        held_out.append((name, np.flatnonzero(array == name)))
    return tuple(held_out)


def ridge_fit(X: np.ndarray, y: np.ndarray, *, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalized intercept (column 0).

    Solves ``(X'X + α·diag(0, 1, …, 1)) β = X'y`` in float64. The Gram
    matrix is explicitly symmetrized as ``0.5 * (G + G.T)`` before the
    penalty is added so floating-point asymmetry cannot change the
    factorization. Conditioning safeguard: if ``numpy.linalg.solve``
    raises ``LinAlgError``, a ``1e-12`` jitter is added to every
    diagonal entry and solve is retried. The jitter is a numerical
    guard only; it is not part of the statistical estimator when G is
    already SPD. No sklearn at runtime.
    """

    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError("ridge_fit requires 2-d X aligned with y")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("ridge_fit requires a non-empty design")
    gram = features.T @ features
    gram = 0.5 * (gram + gram.T)
    penalty = np.full(features.shape[1], float(alpha), dtype=np.float64)
    penalty[0] = 0.0
    gram = gram + np.diag(penalty)
    rhs = features.T @ target
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        gram = gram.copy()
        gram.flat[:: gram.shape[0] + 1] += 1e-12
        return np.linalg.solve(gram, rhs)


def ridge_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)
    weights = np.asarray(coef, dtype=np.float64).reshape(-1)
    return features @ weights


def oof_predict(
    X: np.ndarray, y: np.ndarray, folds: Sequence[int], *, alpha: float
) -> np.ndarray:
    """Out-of-fold predictions using ``ridge_fit`` on every complement."""

    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    fold_ids = np.asarray(list(folds))
    if features.shape[0] != target.shape[0] or target.shape[0] != fold_ids.shape[0]:
        raise ValueError("oof_predict requires aligned X, y, and folds")
    predicted = np.empty(target.shape[0], dtype=np.float64)
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        test = fold_ids == fold
        coef = ridge_fit(features[train], target[train], alpha=alpha)
        predicted[test] = ridge_predict(coef, features[test])
    return predicted


def pav_nonincreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators enforcing a non-increasing sequence.

    Contract matches ``tools/the rank recalibration study.py``: merge adjacent blocks
    while ``left + 1e-12 < right``, using weighted averages, then expand
    each block back to its original length. Does not import the rank recalibration study.
    """

    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if raw.shape != mass.shape or raw.size == 0:
        raise ProtocolError("recal-bin-undefined")
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ProtocolError("recal-bin-undefined")
    block_value = raw.tolist()
    block_weight = mass.tolist()
    block_size = [1] * len(raw)
    index = 0
    while index < len(block_value) - 1:
        if block_value[index] + FINITE_COMPARE >= block_value[index + 1]:
            index += 1
            continue
        merged = (
            block_value[index] * block_weight[index]
            + block_value[index + 1] * block_weight[index + 1]
        ) / (block_weight[index] + block_weight[index + 1])
        block_value[index] = merged
        block_weight[index] += block_weight[index + 1]
        block_size[index] += block_size[index + 1]
        del block_value[index + 1]
        del block_weight[index + 1]
        del block_size[index + 1]
        if index > 0:
            index -= 1
    expanded: list[float] = []
    for value, size in zip(block_value, block_size):
        expanded.extend([float(value)] * int(size))
    result = np.asarray(expanded, dtype=np.float64)
    if result.size != raw.size:
        raise ProtocolError("recal-bin-undefined")
    return result


def rank_recalibration(
    pred_inc: np.ndarray,
    actual_inc: np.ndarray,
    *,
    n_bins: int = N_BINS,
    clip: Tuple[float, float] = FACTOR_CLIP,
) -> RankRecal:
    """the feasibility ladder/the rank recalibration study rank isotonic recalibration (deployment.md steps 3–6).

    1. Stable argsort of ``pred_inc``.
    2. ``numpy.array_split`` into ``n_bins`` equal-count groups.
    3. Per-bin factor ``sum(actual_inc) / sum(pred_inc)``.
    4. PAV non-increasing, weights = predicted incremental sums.
    5. Clip to ``clip`` (charter / the feasibility ladder: ``[0.5, 6.0]``).
    6. Edges are midpoints of adjacent bin max/min (the rank recalibration study rule).
    """

    predicted = np.asarray(pred_inc, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_inc, dtype=np.float64).reshape(-1)
    bins = int(n_bins)
    if predicted.shape != actual.shape or predicted.size < bins:
        raise ProtocolError("recal-bins-insufficient")
    if np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(actual)):
        raise ProtocolError("recal-leak-detected")
    order = np.argsort(predicted, kind="stable")
    groups = np.array_split(order, bins)
    if any(group.size == 0 for group in groups):
        raise ProtocolError("recal-bins-insufficient")
    raw_factors: list[float] = []
    weights: list[float] = []
    bin_pred_max: list[float] = []
    bin_pred_min: list[float] = []
    for group in groups:
        pred_sum = float(predicted[group].sum())
        act_sum = float(actual[group].sum())
        if (not math.isfinite(pred_sum)) or pred_sum <= 0.0:
            raise ProtocolError("recal-bin-undefined")
        if not math.isfinite(act_sum):
            raise ProtocolError("recal-bin-undefined")
        raw_factors.append(act_sum / pred_sum)
        weights.append(pred_sum)
        bin_pred_max.append(float(predicted[group].max()))
        bin_pred_min.append(float(predicted[group].min()))
    raw = np.asarray(raw_factors, dtype=np.float64)
    pav = pav_nonincreasing(raw, np.asarray(weights, dtype=np.float64))
    clipped = np.clip(pav, float(clip[0]), float(clip[1]))
    edges: list[float] = []
    for index in range(bins - 1):
        left = bin_pred_max[index]
        right = bin_pred_min[index + 1]
        if left < right:
            edges.append(0.5 * (left + right))
        else:
            edges.append(left)
    return RankRecal(
        edges=np.asarray(edges, dtype=np.float64),
        raw_factors=raw,
        pav_factors=np.asarray(pav, dtype=np.float64),
        clipped_factors=np.asarray(clipped, dtype=np.float64),
    )


def paired_group_bootstrap(
    gain_per_episode: np.ndarray,
    group_keys: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Resample GROUPS (not episodes) with a pinned Generator."""

    gains = np.asarray(gain_per_episode, dtype=np.float64).reshape(-1)
    if gains.shape[0] != len(group_keys) or int(draws) < 1:
        raise ValueError("bootstrap inputs must align and draws must be >= 1")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        grouped[key].append(index)
    group_values = []
    for key in sorted(grouped):
        members = np.asarray(grouped[key], dtype=np.int64)
        group_values.append((float(gains[members].sum()), float(members.size)))
    matrix = np.asarray(group_values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    n_groups = len(matrix)
    for iteration in range(int(draws)):
        selected = rng.integers(0, n_groups, size=n_groups)
        totals = matrix[selected].sum(axis=0)
        samples[iteration] = totals[0] / totals[1]
    return {
        "mean": _json_float(samples.mean()),
        "q2_5": _json_float(np.quantile(samples, 0.025)),
        "q50": _json_float(np.quantile(samples, 0.50)),
        "q97_5": _json_float(np.quantile(samples, 0.975)),
    }


def budget_ratio(
    selection_indices_or_columns: Sequence[Any],
    costs: np.ndarray,
    light_total: float,
) -> float:
    """Realized cost / all-light total for a column-index (or model-id) pick."""

    matrix = np.asarray(costs, dtype=np.float64)
    if light_total <= 0.0:
        raise ValueError("light_total must be positive")
    selected = list(selection_indices_or_columns)
    if len(selected) != matrix.shape[0]:
        raise ValueError("selection length must match the cost rows")
    if selected and isinstance(selected[0], str):
        columns = [MODEL_IDS.index(model_id) for model_id in selected]
    else:
        columns = [int(item) for item in selected]
    rows = np.arange(matrix.shape[0], dtype=np.int64)
    spent = float(matrix[rows, np.asarray(columns, dtype=np.int64)].sum())
    return spent / float(light_total)


def weighted_final(fast_q: float, balanced_q: float, premium_q: float) -> float:
    """Official 0.4 / 0.3 / 0.3 weighted combination."""

    return (
        TIER_WEIGHTS["fast"] * float(fast_q)
        + TIER_WEIGHTS["balanced"] * float(balanced_q)
        + TIER_WEIGHTS["premium"] * float(premium_q)
    )


def submission_from_models(
    inputs: InputBatch,
    policy: RoutingPolicy,
    tier: str,
    model_ids: Sequence[str],
) -> Submission:
    """In-memory submission that still passes the official parser."""

    if len(model_ids) != len(inputs.episodes):
        raise ValueError("model_ids must align with input episodes")
    raw = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, model_ids)
        ),
    )
    return parse_submission(submission_to_dict(raw))


def official_score(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    submissions_by_tier: Mapping[str, Union[Submission, Sequence[str]]],
) -> Mapping[str, Any]:
    """Thin call into the repo Decimal scorer (``ossp_router.scoring``).

    ``submissions_by_tier`` may hold real ``Submission`` objects or a
    per-episode model-id sequence. Sequences are wrapped with an
    in-memory shim that constructs the same parser objects the CLI
    self-check path uses. No on-disk submission files are written.
    """

    submissions = []
    for tier in TIERS:
        if tier not in submissions_by_tier:
            raise ValueError(f"official_score missing tier {tier!r}")
        item = submissions_by_tier[tier]
        if isinstance(item, Submission):
            submissions.append(parse_submission(submission_to_dict(item)))
        else:
            submissions.append(submission_from_models(inputs, policy, tier, item))
    return score_submissions(inputs, outcomes, submissions, policy)


def constant_model_submissions(
    inputs: InputBatch, policy: RoutingPolicy, model_id: str
) -> dict[str, Submission]:
    chosen = tuple(model_id for _ in inputs.episodes)
    return {
        tier: submission_from_models(inputs, policy, tier, chosen) for tier in TIERS
    }


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "feature_signature": {
                str(bins): feature_signature(bins) for bins in HASH_BINS
            },
            "feature_version": FEATURE_VERSION,
            "fold_seed": FOLD_SEED,
            "folds": FOLDS,
            "hash_bins": list(HASH_BINS),
            "intercept_policy": INTERCEPT_POLICY,
            "official_caps": dict(OFFICIAL_CAPS),
            "official_weights": dict(TIER_WEIGHTS),
            "operating_targets": dict(OPERATING_TARGETS),
            "stress_backstop": STRESS_BACKSTOP,
        }
    )


def assemble_report(
    *,
    identity: Mapping[str, Any],
    locked: Mapping[str, Any],
    observed: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    report = {
        "decision": DECISION,
        "dev_opened": False,
        "diagnostic": diagnostic,
        "experiment": EXPERIMENT,
        "identity": identity,
        "locked": locked,
        "observed": observed,
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
    }
    if report["dev_opened"] is not False:
        raise RuntimeError("the modeling foundation report must assert dev_opened is false")
    if report["decision"] != DECISION:
        raise RuntimeError("the modeling foundation report decision is locked")
    return sort_mapping(report)


# Re-export for later stages that only import this module.
__all__ = (
    "BOOTSTRAP_SEED",
    "DECISION",
    "EXPERIMENT",
    "FEATURE_VERSION",
    "FOLD_SEED",
    "FOLDS",
    "SINGLE_MODEL",
    "HASH_BINS",
    "INTERCEPT_POLICY",
    "OFFICIAL_CAPS",
    "OPERATING_TARGETS",
    "RankRecal",
    "STRUCTURAL_FEATURE_NAMES",
    "TIER_WEIGHTS",
    "TrainBundle",
    "assemble_report",
    "budget_ratio",
    "constant_model_submissions",
    "episode_text_of",
    "family_folds",
    "feature_matrix",
    "feature_row",
    "feature_signature",
    "group_folds",
    "load_train",
    "locked_record",
    "official_score",
    "oof_predict",
    "paired_group_bootstrap",
    "pav_nonincreasing",
    "quantile_higher",
    "rank_recalibration",
    "reject_dev_reference",
    "ridge_fit",
    "ridge_predict",
    "sha256_file",
    "sort_mapping",
    "submission_from_models",
    "weighted_final",
)
