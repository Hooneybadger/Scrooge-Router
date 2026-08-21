# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the final lock — lock the family-guard selected with Train scoring and a Dev veto.

the guard selection proved Train drift views cannot certify a Fast cap: the cost head is
in-sample on Train, so Train says "safe" where Dev shows 132 ruined views.
The organizers' own baseline uses the split the same way we now do
(``baselines/README.md``): "Train으로 회귀계수를 학습하고 Dev로 등급별
안전계수 ... 정하는 데만 사용" — coefficients from Train, safety factors
verified on Dev.

So the rule here is:

* score with **Train** weighted quality (Dev never picks the winner),
* **Dev is a veto only**: hard gates plus zero drift ruin plus the extra
  margin gate A3 (inflated ratio under 95% of the cap, because the
  documented hash-regex failure sat at 99.6% of it),
* the guard multiplier stays inside the charter's registered clip.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, TIER_WEIGHTS, load_train
from research.lab.cap_certification import derived_runaway_fraction, sha256_file
from research.lab.hidden_set_gates import INFLATION, NEAR_FRAC, SplitContext
from research.lab.validation import prompt_family, public_arrays
from ossp_router.protocol import TIERS, load_input, load_outcomes
from research.experiments.try_family_costing import FAMILY_MULT_CLIP, RESIDUAL_FAMILY, evaluate, train_family_multipliers


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "lock-final-policy"
PREVIOUS_GUARD_PATH = ROOT / "build" / "lock-static-caps" / "static-cap-router.v1.json"
GUARD_ARTIFACT_PATH = OUT / "family-guard-router.v1.json"
MULT_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
FAST_GRID = (1.03, 1.05, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12)
BAL_GRID = (1.30, 1.38, 1.45, 1.50)
MARGIN_FRAC = 0.95
GUARD_EXPORT_NOTE = (
    "Runtime artifact for the submitted family-guard router. Every head, the "
    "recalibration and the Premium path are byte-copied from the feasibility-ladder "
    "artifact. The only additions are the raised Fast and Balanced caps and the "
    "per-family accounting multipliers, both selected on Train and confirmed once "
    "on Dev. No Dev path was opened, hashed, parsed or scored while fitting."
)


def gates(row: Mapping[str, Any]) -> dict[str, bool]:
    tiers = row["tiers"]
    return {
        "A1_inflated_under_cap": all(tiers[t]["cost_ok"] for t in TIERS),
        "A2_near_budget": all(tiers[t]["near_ok"] for t in TIERS),
        "A3_inflated_margin": all(
            float(tiers[t]["inflated"]) < MARGIN_FRAC * float(OFFICIAL_CAPS[t]) - 1e-15
            for t in TIERS
        ),
        "B_zero_drift_ruin": all(
            int(tiers[t]["drift_ruin_inflated"]) == 0 for t in TIERS
        ),
    }


def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading", flush=True)
    bundle = load_train(None)
    policy = bundle.policy
    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)

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
    selected = json.loads(PREVIOUS_GUARD_PATH.read_text(encoding="utf-8"))
    caches = {label: ctx.prediction_cache(selected) for label, ctx in contexts.items()}
    train_mult = train_family_multipliers(contexts["train"], caches["train"])

    print("\n=== the final lock grid: Train scores, Dev vetoes ===", flush=True)
    rows: list[dict[str, Any]] = []
    for mult_value in MULT_GRID:
        multipliers = {} if mult_value == 1.0 else {RESIDUAL_FAMILY: float(mult_value)}
        for fast in FAST_GRID:
            for bal in BAL_GRID:
                art = copy.deepcopy(selected)
                art["predicted_caps"]["fast"] = float(fast)
                art["predicted_caps"]["balanced"] = float(bal)
                art["runaway_fraction"] = float(derived_runaway_fraction(fast))
                splits = {
                    label: evaluate(
                        contexts[label], caches[label], art=art, multipliers=multipliers
                    )
                    for label in contexts
                }
                per_split = {label: gates(splits[label]) for label in splits}
                passed = all(all(g.values()) for g in per_split.values())
                entry = {
                    "name": f"m={mult_value:g}|f={fast:.2f}|b={bal:.2f}",
                    "multiplier": float(mult_value),
                    "fast_cap": float(fast),
                    "bal_cap": float(bal),
                    "splits": splits,
                    "gates": per_split,
                    "eligible": bool(passed),
                    "train_weighted": float(splits["train"]["weighted"]),
                    "dev_weighted": float(splits["dev"]["weighted"]),
                }
                rows.append(entry)
                if passed:
                    print(
                        f"  ok  {entry['name']:22} train={entry['train_weighted']:.6f} "
                        f"dev={entry['dev_weighted']:.6f} "
                        f"devFi={splits['dev']['tiers']['fast']['inflated']:.4f}",
                        flush=True,
                    )

    eligible = [r for r in rows if r["eligible"]]
    eligible.sort(key=lambda r: (r["train_weighted"], r["dev_weighted"]), reverse=True)
    print(f"\neligible {len(eligible)} / {len(rows)}", flush=True)
    if not eligible:
        print("no eligible configuration", flush=True)
        return 1
    winner = eligible[0]
    baseline = next(
        r for r in rows if r["multiplier"] == 1.0 and r["fast_cap"] == 1.03 and r["bal_cap"] == 1.38
    )
    print(
        f"WINNER {winner['name']} train={winner['train_weighted']:.6f} "
        f"dev={winner['dev_weighted']:.6f}",
        flush=True,
    )
    print(
        f"the static cap lock selected  train={baseline['train_weighted']:.6f} "
        f"dev={baseline['dev_weighted']:.6f} eligible={baseline['eligible']}",
        flush=True,
    )

    def expected(row: Mapping[str, Any]) -> float:
        total = 0.0
        for tier in TIERS:
            risk = max(
                int(row["splits"][s]["tiers"][tier]["drift_ruin_inflated"])
                / max(int(row["splits"][s]["tiers"][tier]["drift_n"]), 1)
                for s in row["splits"]
            )
            total += (
                float(TIER_WEIGHTS[tier])
                * float(row["splits"]["dev"]["tiers"][tier]["quality"])
                * (1.0 - risk)
            )
        return float(total)

    art = copy.deepcopy(selected)
    art["predicted_caps"]["fast"] = float(winner["fast_cap"])
    art["predicted_caps"]["balanced"] = float(winner["bal_cap"])
    art["runaway_fraction"] = float(derived_runaway_fraction(winner["fast_cap"]))
    art["artifact_type"] = "scrooge-family-guard-router-v1"
    art["selected_policy"] = "family-guard-router-v1"
    art["provenance"] = dict(art["provenance"])
    art["provenance"]["export_note"] = GUARD_EXPORT_NOTE
    if winner["multiplier"] != 1.0:
        art["family_guard"] = {
            "activation": "content-only prompt family bucket",
            "multiplier_clip": list(FAMILY_MULT_CLIP),
            "multipliers": {RESIDUAL_FAMILY: float(winner["multiplier"])},
            "scope": "fast-and-balanced accounting cost only",
            "selection": "train weighted score, dev veto (zero drift ruin + 95% margin)",
            "train_family_ratios": train_mult,
        }
    GUARD_ARTIFACT_PATH.write_text(
        json.dumps(art, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "experiment": "the final lock",
        "decision": "record-final_lock-selected-" + winner["name"].replace("=", "p").replace("|", "_").replace(".", "p"),
        "protocol": {
            "score_split": "train",
            "veto_split": "dev",
            "gates": {
                "A1": f"realized * {INFLATION} <= cap",
                "A2": f"realized < {NEAR_FRAC} * cap",
                "A3": f"realized * {INFLATION} < {MARGIN_FRAC} * cap",
                "B": "zero inflated drift ruin on both splits",
            },
            "multiplier_grid": list(MULT_GRID),
            "multiplier_clip": list(FAMILY_MULT_CLIP),
        },
        "winner": {k: v for k, v in winner.items()},
        "static_caps_baseline": {k: v for k, v in baseline.items()},
        "expected": {"winner": expected(winner), "static_caps": expected(baseline)},
        "artifact": {
            "path": str(GUARD_ARTIFACT_PATH),
            "sha256": sha256_file(GUARD_ARTIFACT_PATH),
            "note": "simulated policy; the runtime router must implement family_guard",
        },
        "eligible": [{k: v for k, v in r.items() if k != "splits"} for r in eligible[:20]],
        "grid": [{k: v for k, v in r.items() if k != "splits"} for r in rows],
        "timing_s": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"E[score] winner={report['expected']['winner']:.6f} "
        f"static_caps={report['expected']['static_caps']:.6f}",
        flush=True,
    )
    print("DONE", report["decision"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
