# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Where does the Fast drift tail come from: prediction error or mixture?

Compares predicted vs actual ax31-light incremental cost per episode, on
Train (in-sample for the frozen heads) and Dev (out-of-sample), sliced by
predicted-incremental decile and by prompt family.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.lab.modeling import load_train
from research.lab.hidden_set_gates import SplitContext
from ossp_router.protocol import load_input, load_outcomes
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "hidden-diag"


def slice_report(label: str, ctx: SplitContext, cache) -> dict:
    pred_inc = np.maximum(cache.pred_ax31 - cache.pred_light, 0.0)
    actual_inc = np.maximum(ctx.costs[:, 1] - ctx.costs[:, 0], 0.0)
    order = np.argsort(pred_inc, kind="stable")
    deciles = np.array_split(order, 10)
    print(f"\n[{label}] predicted vs actual ax31 incremental, by predicted decile")
    print("  decile   n   sum_pred   sum_actual   ratio")
    rows = []
    for i, idx in enumerate(deciles):
        sp = float(pred_inc[idx].sum())
        sa = float(actual_inc[idx].sum())
        ratio = sa / max(sp, 1e-15)
        rows.append({"decile": i, "n": int(idx.size), "ratio": ratio})
        print(f"  {i:6d} {idx.size:4d} {sp:10.4f} {sa:12.4f} {ratio:7.3f}")

    print(f"\n[{label}] by family")
    fam = np.asarray(ctx.families)
    fam_rows = []
    for name in sorted(set(fam.tolist())):
        idx = np.flatnonzero(fam == name)
        sp = float(pred_inc[idx].sum())
        sa = float(actual_inc[idx].sum())
        light = float(ctx.costs[idx, 0].sum())
        ratio = sa / max(sp, 1e-15)
        fam_rows.append({"family": name, "n": int(idx.size), "ratio": ratio})
        print(
            f"  {name:22} n={idx.size:5d} pred={sp:9.4f} act={sa:10.4f} "
            f"ratio={ratio:6.3f} all_ax31_r={1 + sa / max(light, 1e-15):6.3f}"
        )
    return {"deciles": rows, "families": fam_rows}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_train(None)
    policy = bundle.policy
    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)
    ladder = json.loads(
        (ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json").read_text()
    )

    out = {}
    for label, inputs, scores, costs, families in (
        ("train", bundle.inputs, bundle.scores, bundle.costs, list(bundle.families)),
        (
            "dev",
            dev_inputs,
            np.asarray(arrays.scores),
            np.asarray(arrays.costs),
            [prompt_family(ep) for ep in dev_inputs.episodes],
        ),
    ):
        ctx = SplitContext.build(
            label=label,
            inputs=inputs,
            policy=policy,
            scores=scores,
            costs=costs,
            families=families,
        )
        out[label] = slice_report(label, ctx, ctx.prediction_cache(ladder))

    (OUT / "cost-error.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
