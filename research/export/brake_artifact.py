# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Fit the frozen ExtraTrees quality head and write the budget-brake artifact.

Starts from a copy of family-guard-router.v1.json, changes artifact_type, and
adds the budget_brake forest. Train only. Does not write the parent files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from ossp_router.cost_calibrated_router import structural_features
from research.lab.modeling import load_train
from research.lab.validation import public_arrays


ROOT = Path(__file__).resolve().parents[2]
PARENT_PATH = ROOT / "src" / "ossp_router" / "resources" / "family-guard-router.v1.json"
DEFAULT_OUTPUT = ROOT / "src" / "ossp_router" / "resources" / "budget-brake-router.v1.json"
FORBIDDEN_OUTPUT_NAMES = {
    "cost-calibrated-router.v1.json",
    "family-guard-router.v1.json",
    "feasibility-ladder.v1.json",
}

ARTIFACT_TYPE = "scrooge-budget-brake-router-v1"
FEATURE_SIGNATURE = "ossp_router.cost_calibrated_router.structural_features/14"
BRAKE_RATIO = 3.8
COUNT_CAP = 48
DENYLIST_FAMILIES = (
    "korean_reasoning",
    "python_program",
    "rule_reasoning",
)
RUNAWAY_LIGHT_FRACTION = 0.02
TRAIN_FULL_PRED_LIGHT = 8.576375372816607
RUNAWAY_ABSOLUTE = 0.17152750745633214
CLIP = (-1.0, 1.0)
EXPECTED_N_TRAIN = 1760
EXPECTED_N_TREES = 200
N_ESTIMATORS = 200
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 20
MAX_FEATURES = 1.0
BOOTSTRAP = False
N_JOBS = 1
RANDOM_STATE = 20260816
N_STRUCTURAL = 14


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(f"refusing to write protected artifact path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _tree_payload(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    values = tree.value
    if values.ndim == 3:
        value = [float(item) for item in values[:, 0, 0]]
    elif values.ndim == 2:
        value = [float(item) for item in values[:, 0]]
    else:
        value = [float(item) for item in values]
    return {
        "left": [int(item) for item in tree.children_left],
        "right": [int(item) for item in tree.children_right],
        "feature": [int(item) for item in tree.feature],
        "threshold": [float(item) for item in tree.threshold],
        "value": value,
    }


def _structural_matrix(episodes: Sequence[Any]) -> np.ndarray:
    matrix = np.asarray([structural_features(episode) for episode in episodes], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != N_STRUCTURAL:
        raise ValueError(
            f"structural_features must be {N_STRUCTURAL}-d; got {matrix.shape}"
        )
    return matrix


def export(*, parent_path: Path, output_path: Path) -> Mapping[str, Any]:
    if output_path.resolve() == parent_path.resolve():
        raise ValueError("refusing to overwrite the family-guard artifact")
    if output_path.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(f"refusing to write protected artifact path: {output_path}")
    if abs(RUNAWAY_LIGHT_FRACTION * TRAIN_FULL_PRED_LIGHT - RUNAWAY_ABSOLUTE) > 1e-18:
        raise ValueError("pinned runaway_absolute drifted from 0.02 * train light")

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if not isinstance(parent, dict):
        raise ValueError("family-guard artifact must be an object")
    if parent.get("artifact_type") != "scrooge-family-guard-router-v1":
        raise ValueError("parent artifact type is not the frozen family-guard router")

    bundle = load_train(None)
    n_train = len(bundle.inputs.episodes)
    if n_train != EXPECTED_N_TRAIN:
        raise ValueError(f"expected {EXPECTED_N_TRAIN} Train episodes; got {n_train}")
    arrays = public_arrays(bundle.inputs, bundle.outcomes, bundle.policy)
    target = np.asarray(arrays.scores[:, 2] - arrays.scores[:, 1], dtype=np.float64)
    weights = np.minimum(
        np.asarray(arrays.generations[:, 1], dtype=np.float64),
        np.asarray(arrays.generations[:, 2], dtype=np.float64),
    )
    features = _structural_matrix(bundle.inputs.episodes)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        criterion="squared_error",
        bootstrap=BOOTSTRAP,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    model.fit(features, target, sample_weight=weights)
    trees = [_tree_payload(estimator) for estimator in model.estimators_]
    if len(trees) != EXPECTED_N_TREES:
        raise ValueError(f"expected {EXPECTED_N_TREES} trees; got {len(trees)}")
    n_nodes = sum(len(tree["left"]) for tree in trees)

    artifact = dict(parent)
    artifact["artifact_type"] = ARTIFACT_TYPE
    artifact["budget_brake"] = {
        "enabled": True,
        "brake_ratio": BRAKE_RATIO,
        "count_cap": COUNT_CAP,
        "denylist_families": list(DENYLIST_FAMILIES),
        "runaway_light_fraction": RUNAWAY_LIGHT_FRACTION,
        "train_full_pred_light": TRAIN_FULL_PRED_LIGHT,
        "runaway_absolute": RUNAWAY_ABSOLUTE,
        "feature_signature": FEATURE_SIGNATURE,
        "clip": [CLIP[0], CLIP[1]],
        "forest": {"n_trees": len(trees), "trees": trees},
    }
    _write_json_atomic(output_path, artifact)
    digest = _sha256(output_path)
    print(
        f"OK: wrote {output_path} digest={digest} n_trees={len(trees)} "
        f"n_nodes={n_nodes} bytes={output_path.stat().st_size}"
    )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        export(parent_path=args.parent, output_path=args.output)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
