# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: per-view-kind realized ratios for the incumbent routers.

Decides which the cap certification layer view kinds can serve as hard hidden-set gates (plausible
composition drift, per docs/DATA_CARD.md) and which stay red-team only
(famdom 75% single family, LOFO). Also reports the fixed-selection
(non-adaptive) bound used to catch the an earlier policy that kept the public selection Premium failure mode.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, STRESS_BACKSTOP, load_train
from research.lab.prefix_certificates import _realized_ratio
from research.lab.cap_certification import (
    LADDER_MAX_UPGRADE,
    LADDER_RUNAWAY,
    build_stress_views,
    cache_predictions,
    derived_runaway_fraction,
    sweep_tier_views,
)
from ossp_router.protocol import MODEL_IDS, TIERS, load_input, load_outcomes
from ossp_router.feasibility_ladder import load_artifact_mapping, make_submission
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "hidden-diag"
CANDIDATES = {
    "the feasibility ladder": ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json",
}


def fixed_selection_by_kind(
    *,
    selection: Sequence[str],
    costs: np.ndarray,
    views: Sequence[Any],
    cap: float,
) -> dict[str, dict[str, Any]]:
    index_of = {model_id: i for i, model_id in enumerate(MODEL_IDS)}
    chosen = np.asarray(
        [costs[i, index_of[selection[i]]] for i in range(len(selection))],
        dtype=np.float64,
    )
    light = np.asarray(costs[:, 0], dtype=np.float64)
    out: dict[str, dict[str, Any]] = {}
    for view in views:
        idx = np.asarray(view.index, dtype=np.int64)
        if idx.size == 0:
            continue
        realized = float(chosen[idx].sum() / max(float(light[idx].sum()), 1e-15))
        bucket = out.setdefault(view.kind, {"n": 0, "n_ruin": 0, "max_realized": 0.0})
        bucket["n"] += 1
        bucket["n_ruin"] += int(realized > cap + 1e-15)
        bucket["max_realized"] = max(bucket["max_realized"], realized)
    return out


def run_split(
    *,
    label: str,
    art_dict: Mapping[str, Any],
    inputs: Any,
    policy: Any,
    scores: np.ndarray,
    costs: np.ndarray,
    families: Sequence[str],
) -> dict[str, Any]:
    art = load_artifact_mapping(copy.deepcopy(dict(art_dict)))
    views, catalogue = build_stress_views(families)
    cache = cache_predictions(inputs.episodes, policy, art)
    value = art.value if hasattr(art, "value") else art

    selections = {
        tier: tuple(
            d.model_id
            for d in make_submission(inputs, policy, art, tier).submission.decisions
        )
        for tier in TIERS
    }

    result: dict[str, Any] = {"catalogue_n": catalogue["n_views"], "tiers": {}}
    for tier in TIERS:
        cap = float(OFFICIAL_CAPS[tier])
        full = float(_realized_ratio(costs, selections[tier]))
        runaway = (
            derived_runaway_fraction(float(value["predicted_caps"]["fast"]))
            if tier == "fast"
            else float(LADDER_RUNAWAY)
        )
        swept = sweep_tier_views(
            views,
            cache,
            costs,
            tier=tier,
            cap=float(value["predicted_caps"][tier]),
            runaway_fraction=float(runaway),
            max_upgrade_fraction=float(value.get("max_upgrade_fraction", LADDER_MAX_UPGRADE)),
        )
        fixed = fixed_selection_by_kind(
            selection=selections[tier], costs=costs, views=views, cap=cap
        )
        print(f"\n[{label}] {tier} cap={cap} full_r={full:.4f} infl={full * STRESS_BACKSTOP:.4f}")
        print("  kind            adaptive_max  ruin   fixed_max   ruin")
        for kind in sorted(set(swept["per_kind"]) | set(fixed)):
            ad = swept["per_kind"].get(kind, {})
            fx = fixed.get(kind, {})
            print(
                f"  {kind:15} {float(ad.get('max_realized', 0)):11.3f} "
                f"{int(ad.get('n_ruin', 0)):6d} "
                f"{float(fx.get('max_realized', 0)):10.3f} {int(fx.get('n_ruin', 0)):6d}"
            )
        result["tiers"][tier] = {
            "full_realized": full,
            "full_inflated": full * float(STRESS_BACKSTOP),
            "adaptive_per_kind": swept["per_kind"],
            "fixed_per_kind": fixed,
        }
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_train(None)
    policy = bundle.policy

    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)

    splits = {
        "train": (
            bundle.inputs,
            bundle.scores,
            bundle.costs,
            list(bundle.families),
        ),
        "dev": (
            dev_inputs,
            np.asarray(arrays.scores),
            np.asarray(arrays.costs),
            [prompt_family(ep) for ep in dev_inputs.episodes],
        ),
    }

    report: dict[str, Any] = {"stress_backstop": float(STRESS_BACKSTOP), "results": {}}
    for name, path in CANDIDATES.items():
        art_dict = json.loads(path.read_text(encoding="utf-8"))
        report["results"][name] = {}
        for split, (inputs, scores, costs, families) in splits.items():
            report["results"][name][split] = run_split(
                label=f"{name}/{split}",
                art_dict=art_dict,
                inputs=inputs,
                policy=policy,
                scores=scores,
                costs=costs,
                families=families,
            )

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nDONE diag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
