# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Check the cached fast path reproduces the real router's full-batch choice."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from research.lab.modeling import load_train
from research.lab.hidden_set_gates import SplitContext, select_tier
from ossp_router.protocol import TIERS, load_input, load_outcomes
from ossp_router.feasibility_ladder import load_artifact_mapping, make_submission
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]


def real_selection(art_dict, inputs, policy, tier):
    art = load_artifact_mapping(copy.deepcopy(dict(art_dict)))
    return tuple(
        d.model_id for d in make_submission(inputs, policy, art, tier).submission.decisions
    )


def main() -> int:
    bundle = load_train(None)
    policy = bundle.policy
    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)

    ladder = json.loads(
        (ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json").read_text()
    )

    contexts = {
        "train": SplitContext.build(
            label="train",
            inputs=bundle.inputs,
            policy=policy,
            scores=bundle.scores,
            costs=bundle.costs,
            families=list(bundle.families),
        ),
        "dev": SplitContext.build(
            label="dev",
            inputs=dev_inputs,
            policy=policy,
            scores=np.asarray(arrays.scores),
            costs=np.asarray(arrays.costs),
            families=[prompt_family(ep) for ep in dev_inputs.episodes],
        ),
    }

    ok = True
    for name, art in (("the feasibility ladder", ladder),):
        for split, ctx in contexts.items():
            cache = ctx.prediction_cache(art)
            for tier in TIERS:
                fast = select_tier(cache, art, tier)
                real = real_selection(art, ctx.inputs, policy, tier)
                same = tuple(fast) == tuple(real)
                ok = ok and same
                diff = sum(1 for a, b in zip(fast, real) if a != b)
                print(f"{name:5} {split:5} {tier:9} match={same} diff={diff}")
    print("ALL_MATCH", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
