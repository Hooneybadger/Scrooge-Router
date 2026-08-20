# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""How much quality is left at the selected's own spend level?

Compares the router's tier quality against an oracle that spends the same
realized budget with perfect knowledge of scores and costs. The gap says
whether the remaining lever is the quality head (better picks) or the
budget (more spend).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, load_train
from research.lab.hidden_set_gates import INFLATION, NEAR_FRAC, SplitContext, select_tier
from ossp_router.protocol import MODEL_IDS, TIERS, load_input, load_outcomes
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]


def oracle_at_budget(
    scores: np.ndarray, costs: np.ndarray, budget_ratio: float
) -> tuple[float, int]:
    """Greedy score-per-credit knapsack over ax31/k1 upgrades from light."""

    light_cost = costs[:, 0]
    light_score = scores[:, 0]
    budget = float(budget_ratio) * float(light_cost.sum())
    spent = float(light_cost.sum())
    total = float(light_score.sum())
    options = []
    for model in (1, 2):
        gain = scores[:, model] - light_score
        extra = costs[:, model] - light_cost
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(extra > 0, gain / np.maximum(extra, 1e-15), np.inf)
        for i in range(scores.shape[0]):
            if gain[i] > 0:
                options.append((float(density[i]), i, model, float(gain[i]), float(extra[i])))
    options.sort(key=lambda row: row[0], reverse=True)
    used: dict[int, float] = {}
    upgrades = 0
    for _density, i, _model, gain, extra in options:
        if i in used:
            continue
        if extra <= 0 or spent + extra <= budget:
            spent += max(extra, 0.0)
            total += gain
            used[i] = extra
            upgrades += 1
    return total / scores.shape[0], upgrades


def main() -> int:
    bundle = load_train(None)
    policy = bundle.policy
    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)
    ladder = json.loads(
        (ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json").read_text()
    )

    ctx = SplitContext.build(
        label="dev",
        inputs=dev_inputs,
        policy=policy,
        scores=np.asarray(arrays.scores),
        costs=np.asarray(arrays.costs),
        families=[prompt_family(ep) for ep in dev_inputs.episodes],
    )
    cache = ctx.prediction_cache(ladder)
    col = {m: i for i, m in enumerate(MODEL_IDS)}

    print("tier      router_q  realized   oracle@same  oracle@limit   gap@same")
    out = {}
    for tier in TIERS:
        selection = select_tier(cache, ladder, tier)
        rows = np.arange(len(selection))
        cols = np.asarray([col[m] for m in selection])
        realized = float(ctx.costs[rows, cols].sum() / ctx.costs[:, 0].sum())
        q = float(ctx.scores[rows, cols].mean())
        same, n_same = oracle_at_budget(ctx.scores, ctx.costs, realized)
        limit = min(float(OFFICIAL_CAPS[tier]) / INFLATION, NEAR_FRAC * float(OFFICIAL_CAPS[tier]))
        at_limit, n_limit = oracle_at_budget(ctx.scores, ctx.costs, limit)
        out[tier] = {
            "router_quality": q,
            "realized": realized,
            "oracle_same_budget": same,
            "oracle_operating_limit": at_limit,
            "operating_limit": limit,
            "oracle_upgrades_same": n_same,
            "oracle_upgrades_limit": n_limit,
        }
        print(
            f"{tier:9} {q:.6f}  {realized:.4f}    {same:.6f}     {at_limit:.6f}   "
            f"{same - q:+.6f}"
        )
    print("\nweighted router", sum(
        w * out[t]["router_quality"] for t, w in (("fast", 0.4), ("balanced", 0.3), ("premium", 0.3))
    ))
    print("weighted oracle@same", sum(
        w * out[t]["oracle_same_budget"] for t, w in (("fast", 0.4), ("balanced", 0.3), ("premium", 0.3))
    ))
    print("weighted oracle@limit", sum(
        w * out[t]["oracle_operating_limit"] for t, w in (("fast", 0.4), ("balanced", 0.3), ("premium", 0.3))
    ))
    (ROOT / "build" / "hidden-diag" / "oracle-gap.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
