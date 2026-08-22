# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Data-scale diagnostic — does more public data help the brake head?

Compares two fold-local fits of the identical frozen ExtraTrees recipe
inside the same runtime-in-the-loop protocol as E5:

- ``fit-train-split-only``: outer-train restricted to Train-split episodes
  (mirrors the shipped 1,760-fitted head),
- ``fit-outer-full``: the full outer-train including Dev-split episodes
  (mirrors the full-public final fit).

Both arms share the parent allocation from the bundled artifact and rank
promotions by predicted uplift. The paired difference isolates the one
variable the final-public-refit swap changes: how many public episodes
the ranking head saw.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.e5_brake_conditioned import (
    BASELINE_NAME,
    MODEL_INDEX,
    ProtocolError,
    XTREES_PARAMS,
)
from research.lab.grouped_crossfit import assign_balanced_group_folds, fold_leakage_count
from research.lab.modeling import sort_mapping
from research.lab.public_pool import load_public_pool


EXPERIMENT = "data-scale-diagnostic-v1"
REPORT_TYPE = "scrooge-data-scale-diagnostic-v1"
SCHEMA_VERSION = 1
AUDIT_RELATIVE = "build/run-data-scale-diagnostic/episode-audit.json"
REPORT_RELATIVE = "build/run-data-scale-diagnostic/report.json"
FOLDS = 5
TVBALL_EPSILON = 0.014204545454545449
VIEW_MIN_N = 20

BASELINE_ARM = "fit-train-split-only"
TREATMENT_ARM = "fit-outer-full"
EVAL_ARMS: Tuple[str, ...] = (BASELINE_ARM, TREATMENT_ARM)


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(dict(protocol)))


def derive_fresh_seeds(
    prefix: str, core_sha256: str, count: int, forbidden: Sequence[int]
) -> Tuple[int, ...]:
    forbidden_set = {int(value) for value in forbidden}
    seen: set[int] = set()
    seeds: list[int] = []
    for index in range(count):
        digest = hashlib.sha256(
            prefix.encode("utf-8")
            + b"\x00"
            + bytes.fromhex(core_sha256)
            + index.to_bytes(4, "big")
        ).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed in seen or seed in forbidden_set:
            raise ProtocolError(f"fresh seed collision at index {index}: {seed}")
        seen.add(seed)
        seeds.append(seed)
    return tuple(seeds)


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
    derivation = protocol["seed_derivation"]
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    expected_seeds = derive_fresh_seeds(
        str(derivation["prefix"]),
        str(derivation["core_sha256"]),
        int(derivation["n"]),
        [int(value) for value in derivation["forbidden_previous_seeds"]],
    )
    if fresh != expected_seeds or len(set(fresh)) != len(fresh):
        raise ProtocolError("sealed fresh seeds drifted from the derivation")
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
    thresholds = protocol["thresholds"]
    if float(thresholds["mean_delta_min"]) <= 0.0:
        raise ProtocolError("mean_delta_min must be positive")
    return digest


def _fit_forest(harness, indexes: Sequence[int]):
    """Fit ONLY the frozen ExtraTrees recipe on the given rows."""

    from sklearn.ensemble import ExtraTreesRegressor

    outer = np.asarray(indexes, dtype=np.int64)
    features = harness.features[outer]
    target = harness.pool.scores[outer, 2] - harness.pool.scores[outer, 1]
    weights = np.minimum(
        harness.generations[outer, 1].astype(np.float64),
        harness.generations[outer, 2].astype(np.float64),
    )
    model = ExtraTreesRegressor(**XTREES_PARAMS)
    model.fit(features, target, sample_weight=weights)
    return model


def evaluate_seed(harness, seed: int) -> Mapping[str, Any]:
    pool = harness.pool
    n_episodes = len(pool.episodes)
    folds = assign_balanced_group_folds(
        pool.group_keys, pool.families, folds=FOLDS, seed=seed
    )
    leaked = fold_leakage_count(pool.group_keys, folds)
    if leaked:
        raise ProtocolError(f"grouped fold leakage: {leaked}")

    chosen_scores: dict[str, list[float]] = {arm: [] for arm in EVAL_ARMS}
    family_order: list[str] = []
    audit_rows: list[dict[str, Any]] = []

    for fold in range(FOLDS):
        held = tuple(i for i, value in enumerate(folds) if value == fold)
        outer = [i for i, value in enumerate(folds) if value != fold]
        train_only = [i for i in outer if pool.split_labels[i] == "train"]
        if not train_only or len(train_only) == len(outer):
            raise ProtocolError("fold composition leaves no split contrast")

        forest_a = _fit_forest(harness, train_only)
        forest_b = _fit_forest(harness, outer)
        features_held = harness.features[np.asarray(held, dtype=np.int64)]
        quality_a = forest_a.predict(features_held)
        quality_b = forest_b.predict(features_held)

        parent = harness.parent_allocation(held)
        selections = {
            BASELINE_ARM: harness.promote(held, parent, quality_a, BASELINE_NAME),
            TREATMENT_ARM: harness.promote(held, parent, quality_b, BASELINE_NAME),
        }
        for arm in EVAL_ARMS:
            for index, model in zip(held, selections[arm]):
                column = MODEL_INDEX[model]
                chosen_scores[arm].append(float(pool.scores[index, column]))
        for position, index in enumerate(held):
            family_order.append(harness.families[index])
            audit_rows.append(
                {
                    "episode_id": pool.episodes[index].episode_id,
                    "family": harness.families[index],
                    "fold": int(fold),
                    BASELINE_ARM: selections[BASELINE_ARM][position],
                    TREATMENT_ARM: selections[TREATMENT_ARM][position],
                }
            )

    pooled = {
        arm: math.fsum(values) / float(n_episodes)
        for arm, values in chosen_scores.items()
    }
    delta = float(pooled[TREATMENT_ARM] - pooled[BASELINE_ARM])

    buckets: dict[str, list[float]] = {}
    for family_value, treatment_score, base_score in zip(
        family_order, chosen_scores[TREATMENT_ARM], chosen_scores[BASELINE_ARM]
    ):
        buckets.setdefault(family_value, []).append(treatment_score - base_score)
    family_deltas = {
        name: {"delta": math.fsum(v) / len(v), "n": len(v)}
        for name, v in sorted(buckets.items())
    }
    eligible = [
        float(row["delta"])
        for row in family_deltas.values()
        if int(row["n"]) >= VIEW_MIN_N
    ]
    spread = min(eligible) - max(eligible)
    tvball_worst = delta + TVBALL_EPSILON * spread

    changed = sum(
        row[BASELINE_ARM] != row[TREATMENT_ARM] for row in audit_rows
    )
    return {
        "seed": int(seed),
        "pooled_quality": {arm: pooled[arm] for arm in EVAL_ARMS},
        "delta": delta,
        "tvball_worst": tvball_worst,
        "family_deltas": family_deltas,
        "n_changed_decisions": int(changed),
        "audit_rows": audit_rows,
    }


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("diagnostic output exists; refuse overwrite")

    pool = load_public_pool()
    from research.lab.e5_brake_conditioned import Harness

    harness = Harness.build(pool)
    thresholds = dict(protocol["thresholds"])
    seeds = [int(seed) for seed in protocol["fresh_seeds"]]

    results: dict[str, Any] = {}
    audits: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        result = evaluate_seed(harness, seed)
        audits[str(seed)] = result.pop("audit_rows")
        results[str(seed)] = result

    deltas = [float(results[str(seed)]["delta"]) for seed in seeds]
    mean_delta = math.fsum(deltas) / len(deltas)
    worst_delta = min(deltas)
    passed = bool(
        mean_delta >= float(thresholds["mean_delta_min"])
        and worst_delta >= float(thresholds["worst_seed_delta_min_inclusive"])
    )
    decision = str(protocol["decisions"]["pass" if passed else "fail"])
    reason = str(protocol["decision_reasons"]["pass" if passed else "fail"])

    audit_document = {
        "arms": list(EVAL_ARMS),
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "rows": {seed: audits[seed] for seed in sorted(audits)},
    }

    gate_block = {
        "mean_delta": mean_delta,
        "worst_delta": worst_delta,
        "deltas_by_seed": {str(s): d for s, d in zip(seeds, deltas)},
        "changed_decisions_by_seed": {
            str(seed): int(results[str(seed)]["n_changed_decisions"])
            for seed in seeds
        },
        "passed": passed,
    }
    report = {
        "audit": {
            "n_rows": sum(len(rows) for rows in audits.values()),
            "relative_path": AUDIT_RELATIVE,
            "sha256": sha256_text(canonical_json_text(audit_document)),
        },
        "candidate_treatment": TREATMENT_ARM,
        "decision": decision,
        "decision_reason": reason,
        "experiment": EXPERIMENT,
        "fold_seeds": seeds,
        "gate": gate_block,
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": {
            seed: {k: v for k, v in results[seed].items()} for seed in results
        },
        "thresholds": thresholds,
    }
    core = sort_mapping(
        {
            key: report[key]
            for key in (
                "audit",
                "candidate_treatment",
                "decision",
                "decision_reason",
                "experiment",
                "fold_seeds",
                "gate",
                "protocol_sha256",
                "report_type",
                "schema_version",
                "seed_results",
                "thresholds",
            )
        }
    )
    encoded = json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    report["decision_core_sha256"] = sha256_text(encoded)
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
