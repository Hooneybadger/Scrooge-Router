# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify family_guard_router reproduces the the final lock simulated policy exactly."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, load_train
from research.lab.hidden_set_gates import INFLATION, SplitContext
from research.lab.validation import prompt_family as validation_family, public_arrays
from ossp_router import family_guard_router
from ossp_router.protocol import MODEL_IDS, TIERS, load_input, load_outcomes
from research.experiments.try_family_costing import evaluate


ROOT = Path(__file__).resolve().parents[2]
GUARD_ARTIFACT_PATH = ROOT / "src" / "ossp_router" / "resources" / "family-guard-router.v1.json"
LADDER_PATH = ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json"


def main() -> int:
    bundle = load_train(None)
    policy = bundle.policy
    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)

    art_dict = json.loads(GUARD_ARTIFACT_PATH.read_text(encoding="utf-8"))
    artifact = family_guard_router.load_artifact_mapping(copy.deepcopy(art_dict))
    print("loaded selected artifact; multipliers:", dict(artifact.multipliers))

    base = json.loads(LADDER_PATH.read_text(encoding="utf-8"))
    sim_art = copy.deepcopy(base)
    sim_art["predicted_caps"] = dict(art_dict["predicted_caps"])
    sim_art["runaway_fraction"] = float(art_dict["runaway_fraction"])
    sim_art["max_upgrade_fraction"] = float(art_dict["max_upgrade_fraction"])
    multipliers = dict(artifact.multipliers)

    splits = {
        "train": (bundle.inputs, bundle.scores, bundle.costs, list(bundle.families)),
        "dev": (
            dev_inputs,
            np.asarray(arrays.scores),
            np.asarray(arrays.costs),
            [validation_family(ep) for ep in dev_inputs.episodes],
        ),
    }

    ok = True
    for label, (inputs, scores, costs, families) in splits.items():
        ctx = SplitContext.build(
            label=label,
            inputs=inputs,
            policy=policy,
            scores=scores,
            costs=costs,
            families=families,
        )
        cache = ctx.prediction_cache(sim_art)
        sim = evaluate(ctx, cache, art=sim_art, multipliers=multipliers)

        # family classifier parity between runtime module and validation tool
        runtime_families = [family_guard_router.prompt_family(ep) for ep in inputs.episodes]
        mismatched = sum(1 for a, b in zip(runtime_families, families) if a != b)
        print(f"{label}: family mismatch {mismatched}")
        ok = ok and mismatched == 0

        col = {m: i for i, m in enumerate(MODEL_IDS)}
        for tier in TIERS:
            plan = family_guard_router.make_submission(inputs, policy, artifact, tier)
            selection = tuple(d.model_id for d in plan.submission.decisions)
            rows = np.arange(len(selection))
            cols = np.asarray([col[m] for m in selection])
            realized = float(costs[rows, cols].sum() / costs[:, 0].sum())
            quality = float(scores[rows, cols].mean())
            sim_tier = sim["tiers"][tier]
            same_q = abs(quality - float(sim_tier["quality"])) < 1e-12
            same_r = abs(realized - float(sim_tier["realized"])) < 1e-9
            ok = ok and same_q and same_r
            cap = float(OFFICIAL_CAPS[tier])
            print(
                f"  {tier:9} q={quality:.6f} r={realized:.4f} "
                f"inflated={realized * INFLATION:.4f}/{cap} "
                f"match_q={same_q} match_r={same_r}"
            )
    print("ALL_MATCH", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
