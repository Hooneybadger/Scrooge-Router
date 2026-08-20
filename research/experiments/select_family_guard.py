# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the guard selection — select the residual-bucket guard on Train, confirm on Dev once.

the family costing attempt showed that inflating the accounting cost of the content-only
residual family buys back Fast budget. This leaf fixes the protocol so the
choice cannot be a Dev artefact:

* Stage A selects the multiplier and the caps using **Train only** — the
  Train full batch must clear the inflated cap and the near-budget line,
  and every Train drift view must survive; among those, take the highest
  Train weighted quality.
* Stage B scores the Stage A winner on Dev exactly once, and reports the
  gates there.

The multiplier is bounded by ``the project charter.md`` 4.2's registered
family-multiplier clip of 3.0, so its range is pre-registered rather than
read off the Dev residual error.
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
from research.lab.hidden_set_gates import INFLATION, SplitContext
from research.lab.validation import prompt_family, public_arrays
from ossp_router.protocol import TIERS, load_input, load_outcomes
from research.experiments.try_family_costing import (
    FAMILY_MULT_CLIP,
    RESIDUAL_FAMILY,
    evaluate,
    train_family_multipliers,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "select-family-guard"
PREVIOUS_GUARD_PATH = ROOT / "src" / "ossp_router" / "resources" / "selected-router.v1.json"
MULT_GRID = (1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
FAST_GRID = (1.05, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.15)
BAL_GRID = (1.38, 1.50)


def train_ok(row: Mapping[str, Any]) -> bool:
    tiers = row["tiers"]
    return all(
        tiers[t]["cost_ok"] and tiers[t]["near_ok"] and tiers[t]["drift_ruin_inflated"] == 0
        for t in TIERS
    )


def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading", flush=True)
    bundle = load_train(None)
    policy = bundle.policy
    ctx_train = SplitContext.build(
        label="train",
        inputs=bundle.inputs,
        policy=policy,
        scores=bundle.scores,
        costs=bundle.costs,
        families=list(bundle.families),
    )
    selected = json.loads(PREVIOUS_GUARD_PATH.read_text(encoding="utf-8"))
    cache_train = ctx_train.prediction_cache(selected)
    train_mult = train_family_multipliers(ctx_train, cache_train)

    print("\n=== Stage A: Train-only selection ===", flush=True)
    rows: list[dict[str, Any]] = []
    for mult_value in MULT_GRID:
        multipliers = {RESIDUAL_FAMILY: float(mult_value)}
        for fast in FAST_GRID:
            for bal in BAL_GRID:
                art = copy.deepcopy(selected)
                art["predicted_caps"]["fast"] = float(fast)
                art["predicted_caps"]["balanced"] = float(bal)
                art["runaway_fraction"] = float(derived_runaway_fraction(fast))
                row = evaluate(ctx_train, cache_train, art=art, multipliers=multipliers)
                entry = {
                    "name": f"other={mult_value:g}|f={fast:.2f}|b={bal:.2f}",
                    "multiplier": float(mult_value),
                    "fast_cap": float(fast),
                    "bal_cap": float(bal),
                    "train": row,
                    "train_ok": train_ok(row),
                    "train_weighted": float(row["weighted"]),
                }
                rows.append(entry)
                print(
                    f"{entry['name']:26} train={entry['train_weighted']:.6f} "
                    f"ok={entry['train_ok']} "
                    f"drift={{f:{row['tiers']['fast']['drift_ruin_inflated']}, "
                    f"b:{row['tiers']['balanced']['drift_ruin_inflated']}}} "
                    f"Fr={row['tiers']['fast']['realized']:.4f}",
                    flush=True,
                )

    survivors = [r for r in rows if r["train_ok"]]
    survivors.sort(key=lambda r: r["train_weighted"], reverse=True)
    if not survivors:
        print("no Train survivor; keeping selected", flush=True)
        return 1
    winner = survivors[0]
    print(f"\nStage A winner: {winner['name']} train={winner['train_weighted']:.6f}", flush=True)

    print("\n=== Stage B: Dev confirmation (once) ===", flush=True)
    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)
    ctx_dev = SplitContext.build(
        label="dev",
        inputs=dev_inputs,
        policy=policy,
        scores=np.asarray(arrays.scores),
        costs=np.asarray(arrays.costs),
        families=[prompt_family(ep) for ep in dev_inputs.episodes],
    )
    cache_dev = ctx_dev.prediction_cache(selected)

    art = copy.deepcopy(selected)
    art["predicted_caps"]["fast"] = float(winner["fast_cap"])
    art["predicted_caps"]["balanced"] = float(winner["bal_cap"])
    art["runaway_fraction"] = float(derived_runaway_fraction(winner["fast_cap"]))
    art["family_guard"] = {
        "activation": "content-only prompt family bucket",
        "clip": list(FAMILY_MULT_CLIP),
        "multipliers": {RESIDUAL_FAMILY: float(winner["multiplier"])},
        "scope": "fast-and-balanced-accounting-cost",
        "selected_on": "train-drift-views-only",
        "train_family_ratios": train_mult,
    }
    multipliers = {RESIDUAL_FAMILY: float(winner["multiplier"])}

    dev_row = evaluate(ctx_dev, cache_dev, art=art, multipliers=multipliers)
    base_dev = evaluate(ctx_dev, cache_dev, art=selected, multipliers={})
    base_train = evaluate(ctx_train, cache_train, art=selected, multipliers={})

    def summary(label: str, row: Mapping[str, Any]) -> None:
        t = row["tiers"]
        print(
            f"{label:14} w={row['weighted']:.6f} "
            f"F(q={t['fast']['quality']:.6f} r={t['fast']['realized']:.4f} "
            f"i={t['fast']['inflated']:.4f} ruin={t['fast']['drift_ruin_inflated']}) "
            f"B(q={t['balanced']['quality']:.6f} r={t['balanced']['realized']:.4f} "
            f"ruin={t['balanced']['drift_ruin_inflated']}) "
            f"P(q={t['premium']['quality']:.6f} ruin={t['premium']['drift_ruin_inflated']})",
            flush=True,
        )

    summary("selected/train", base_train)
    summary("guard/train", winner["train"])
    summary("selected/dev", base_dev)
    summary("guard/dev", dev_row)

    def expected(dev: Mapping[str, Any], train: Mapping[str, Any]) -> float:
        total = 0.0
        for tier in TIERS:
            risk = max(
                int(dev["tiers"][tier]["drift_ruin_inflated"])
                / max(int(dev["tiers"][tier]["drift_n"]), 1),
                int(train["tiers"][tier]["drift_ruin_inflated"])
                / max(int(train["tiers"][tier]["drift_n"]), 1),
            )
            total += (
                float(TIER_WEIGHTS[tier])
                * float(dev["tiers"][tier]["quality"])
                * (1.0 - risk)
            )
        return float(total)

    guard_e = expected(dev_row, winner["train"])
    base_e = expected(base_dev, base_train)
    accept = (
        all(
            dev_row["tiers"][t]["cost_ok"] and dev_row["tiers"][t]["near_ok"] for t in TIERS
        )
        and all(
            dev_row["tiers"][t]["drift_ruin_inflated"]
            <= base_dev["tiers"][t]["drift_ruin_inflated"]
            for t in TIERS
        )
        and guard_e > base_e + 1e-12
    )
    print(f"\nE[score] guard={guard_e:.6f} selected={base_e:.6f} accept={accept}", flush=True)

    art_path = OUT / "family-guard-router.v1.json"
    art_path.write_text(
        json.dumps(art, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    art_path.chmod(0o644)

    report = {
        "experiment": "the guard selection",
        "decision": (
            f"record-guard_selection-promote-{winner['name'].replace('=', 'p').replace('|', '_').replace('.', 'p')}"
            if accept
            else "record-guard_selection-retain-static_caps-selected"
        ),
        "accepted": bool(accept),
        "protocol": {
            "stage_a": "Train full batch + all Train drift views must pass; max Train weighted",
            "stage_b": "Dev scored once, must not increase drift ruin and must raise E[score]",
            "multiplier_range": list(MULT_GRID),
            "charter_clip": list(FAMILY_MULT_CLIP),
        },
        "winner": {k: v for k, v in winner.items()},
        "dev": dev_row,
        "baseline": {"train": base_train, "dev": base_dev},
        "expected": {"guard": guard_e, "selected": base_e},
        "artifact": {"path": str(art_path), "sha256": sha256_file(art_path)},
        "inflation": INFLATION,
        "official_caps": {t: float(OFFICIAL_CAPS[t]) for t in TIERS},
        "train_rows": [{k: v for k, v in r.items() if k != "train"} for r in rows],
        "timing_s": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("DONE", report["decision"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
