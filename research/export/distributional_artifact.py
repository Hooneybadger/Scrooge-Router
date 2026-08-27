# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile fitted distributional tree ensembles into a standard-library artifact.

The input component bundle is the output of the sealed fit and
certification run.  Compilation removes the sklearn dependency and
preserves every numeric tree threshold and leaf value used by serving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np

from ossp_router import distributional_router
from ossp_router.protocol import MODEL_IDS, load_bundled_policy
from research.lab.distributional_knapsack import (
    BALANCED_K1_MIN_GROUPS,
    BATCH_RISK_FEATURE_NAMES,
    DEFAULT_TIER_CONFIG,
    FAMILY_NAMES,
    FEATURE_VERSION,
    MIN_CONTENT_GROUPS,
    PREMIUM_K1_MAX_TV,
    PREMIUM_K1_MIN_GROUPS,
    STRUCTURAL_FEATURE_NAMES,
    FamilyCalibration,
    FitBundle,
    fit_family_calibration,
    predict_distributional,
)
from research.lab.public_pool import (
    DEV_INPUTS,
    DEV_OUTCOMES,
    TRAIN_INPUTS,
    TRAIN_OUTCOMES,
    load_public_pool,
    sha256_path,
)


EXPERIMENT_ID = "distributional-knapsack-v1"
BASE_COMMIT = "356b73e737efc4220490fa47e323009b76be87fd"


def _nodes_from_classic(model: Any) -> list[list[list[Any]]]:
    learning_rate = float(model.learning_rate)
    trees: list[list[list[Any]]] = []
    for estimator in model.estimators_.reshape(-1):
        tree = estimator.tree_
        nodes: list[list[Any]] = []
        for index in range(tree.node_count):
            left = int(tree.children_left[index])
            right = int(tree.children_right[index])
            leaf = left == -1 and right == -1
            nodes.append(
                [
                    -1 if leaf else int(tree.feature[index]),
                    0.0 if leaf else float(tree.threshold[index]),
                    left,
                    right,
                    learning_rate * float(tree.value[index][0][0]) if leaf else 0.0,
                    False,
                ]
            )
        trees.append(nodes)
    return trees


def _classic_head(model: Any, transform: str) -> Mapping[str, Any]:
    constant = np.asarray(model.init_.constant_, dtype=np.float64).reshape(-1)
    if constant.size != 1:
        raise ValueError("distributional classic tree head has a non-scalar initializer")
    return {
        "base": float(constant[0]),
        "transform": transform,
        "trees": _nodes_from_classic(model),
    }


def _hist_head(model: Any) -> Mapping[str, Any]:
    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).reshape(-1)
    if baseline.size != 1:
        raise ValueError("distributional histogram head has a non-scalar initializer")
    trees: list[list[list[Any]]] = []
    for stage in model._predictors:
        if len(stage) != 1:
            raise ValueError("distributional histogram head is unexpectedly multi-output")
        predictor = stage[0]
        nodes = []
        for node in predictor.nodes:
            if int(node["is_categorical"]):
                raise ValueError("distributional runtime does not support categorical tree nodes")
            leaf = bool(node["is_leaf"])
            nodes.append(
                [
                    -1 if leaf else int(node["feature_idx"]),
                    0.0 if leaf else float(node["num_threshold"]),
                    -1 if leaf else int(node["left"]),
                    -1 if leaf else int(node["right"]),
                    float(node["value"]) if leaf else 0.0,
                    bool(node["missing_go_to_left"]),
                ]
            )
        trees.append(nodes)
    return {"base": float(baseline[0]), "transform": "identity", "trees": trees}


def _calibration_mapping(calibration: FamilyCalibration) -> Mapping[str, Any]:
    return {
        "family_names": list(calibration.family_names),
        "reference_proportions": list(calibration.reference_proportions),
        "mean_scales": calibration.mean_scales.tolist(),
        "q90_scales": calibration.q90_scales.tolist(),
    }


def build_artifact(components: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build the frozen artifact from a fitted, certified component bundle."""

    policy = load_bundled_policy()
    pool = load_public_pool(policy=policy)
    fit = FitBundle(
        vocabulary=tuple(components["vocabulary"]),
        quality_models=tuple(components["quality_models"]),
        cost_mean_models=tuple(components["cost_mean_models"]),
        cost_q50_models=tuple(components["cost_q50_models"]),
        cost_q90_models=tuple(components["cost_q90_models"]),
    )
    raw = predict_distributional(fit, pool.episodes)
    calibration = fit_family_calibration(raw, pool.costs, pool.families)
    if tuple(calibration.family_names) != FAMILY_NAMES:
        raise RuntimeError("distributional family contract drifted")
    risk_models = components["risk_models"]
    if set(risk_models) != {"fast", "balanced"}:
        raise ValueError("distributional component bundle has the wrong risk heads")

    quality_heads = {
        model_id: _classic_head(model, "clip01")
        for model_id, model in zip(MODEL_IDS, fit.quality_models)
    }
    mean_heads = {
        model_id: _classic_head(model, "positive")
        for model_id, model in zip(MODEL_IDS, fit.cost_mean_models)
    }
    q50_heads = {
        model_id: _classic_head(model, "expm1_nonnegative")
        for model_id, model in zip(MODEL_IDS, fit.cost_q50_models)
    }
    q90_heads = {
        model_id: _classic_head(model, "expm1_nonnegative")
        for model_id, model in zip(MODEL_IDS, fit.cost_q90_models)
    }
    tier_config = {
        tier: {
            "base_fraction": config.base_fraction,
            "composition_penalty": config.composition_penalty,
            "risk_reserve": config.risk_reserve,
            "ax31_tail_weight": config.ax31_tail_weight,
            "k1_tail_weight": config.k1_tail_weight,
        }
        for tier, config in DEFAULT_TIER_CONFIG.items()
    }
    dataset_pins = {
        "train_inputs_sha256": sha256_path(TRAIN_INPUTS),
        "train_outcomes_sha256": sha256_path(TRAIN_OUTCOMES),
        "dev_inputs_sha256": sha256_path(DEV_INPUTS),
        "dev_outcomes_sha256": sha256_path(DEV_OUTCOMES),
    }
    return {
        "artifact_type": distributional_router.ARTIFACT_TYPE,
        "schema_version": distributional_router.SCHEMA_VERSION,
        "feature_contract": distributional_router.FEATURE_CONTRACT,
        "feature_version": FEATURE_VERSION,
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": pool.identity["policy_sha256"],
        "structural_feature_names": list(STRUCTURAL_FEATURE_NAMES),
        "batch_risk_feature_names": list(BATCH_RISK_FEATURE_NAMES),
        "vocabulary": list(fit.vocabulary),
        "quality_heads": quality_heads,
        "cost_mean_heads": mean_heads,
        "cost_q50_heads": q50_heads,
        "cost_q90_heads": q90_heads,
        "family_calibration": _calibration_mapping(calibration),
        "risk_heads": {
            tier: _hist_head(risk_models[tier]) for tier in ("fast", "balanced")
        },
        "tier_config": tier_config,
        "finite_sample_gates": {
            "min_content_groups": MIN_CONTENT_GROUPS,
            "balanced_k1_min_groups": BALANCED_K1_MIN_GROUPS,
            "premium_k1_min_groups": PREMIUM_K1_MIN_GROUPS,
            "premium_k1_max_tv": PREMIUM_K1_MAX_TV,
        },
        "training_summary": {
            "fit_scope": "pinned-public-train-dev",
            "num_episodes": len(pool.episodes),
            "num_train": 1760,
            "num_dev": 880,
            "dataset_pins": dataset_pins,
            "risk_catalog_seeds": [2026082204, 2026083104],
            "risk_quantile": 0.005,
            "cost_inflation": 1.054,
            "required_margin_fraction": 0.01,
        },
        "certification_summary": {
            "official_dev_weighted_quality": 0.7157670454545454,
            "compact_binding_ev": 0.7034118265993267,
            "compact_red_team_ev": 0.5962533076720336,
            "primary_views": 11680,
            "primary_margin_violations": 0,
            "wide_catalogs": 6,
            "wide_views": 20610,
            "wide_margin_violations": 0,
            "max_inflated_budget_ratio": {
                "fast": 1.2279638846683032,
                "balanced": 1.9714489261211643,
                "premium": 3.6945688135994135,
            },
        },
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "base_commit": BASE_COMMIT,
            "fit_implementation": (
                "research.lab.distributional_knapsack.fit_distributional_models"
            ),
            "exporter": "research/export/distributional_artifact.py",
            "selection_basis": (
                "additional public-data distributional and composition-shift "
                "experiments from the pinned base commit"
            ),
            "design": (
                "explicit lexicon + distributional tree heads + finite-sample "
                "batch risk + canonical concave-prefix allocation"
            ),
        },
    }


def export(components_path: Path, output_path: Path) -> None:
    components = joblib.load(components_path)
    artifact = build_artifact(components)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.components, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
