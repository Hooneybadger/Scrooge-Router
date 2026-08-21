# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the family costing attempt — family-pessimistic cost accounting to buy back Fast/Balanced budget.

The reason the selected cannot spend more is not the quality head, it is
cost under-prediction on content the head has not really learned: on Dev
the residual ``other`` family costs 2.465x its predicted ax31 increment
(Train: 1.229). A static cap has to hold margin for that everywhere.

This leaf prices the risk where it lives. The router already may read the
prompt (``docs/CHALLENGE_RULES.md``), so it can bucket each episode with
the content-only family rule and inflate the *accounting* cost of
unfamiliar buckets before the allocator spends. Actual scoring still uses
real costs, so any mispricing shows up in the same hidden-set gates.

The multiplier is never fitted on Dev: it is either a pre-registered
constant for the residual bucket or the Train family ratio clipped to
``[1, 3]`` (the project charter.md 4.2 registered cap).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, TIER_WEIGHTS, load_train, weighted_final
from research.lab.prefix_certificates import _realized_ratio, json_float
from research.lab.cap_certification import (
    LADDER_MAX_UPGRADE,
    LADDER_RUNAWAY,
    allocate_numpy,
    ax31_count,
    derived_runaway_fraction,
    score_mean,
    select_premium_cached,
)
from research.lab.hidden_set_gates import INFLATION, NEAR_FRAC, SplitContext
from ossp_router.protocol import TIERS, load_input, load_outcomes
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "try-family-costing"
LADDER_PATH = ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json"
PREVIOUS_GUARD_PATH = ROOT / "build" / "lock-static-caps" / "static-cap-router.v1.json"
RESIDUAL_FAMILY = "other"
FAMILY_MULT_CLIP = (1.0, 3.0)


def train_family_multipliers(ctx: SplitContext, cache: Any) -> dict[str, float]:
    """actual/predicted ax31 increment per family, clipped to pessimism only."""

    pred_inc = np.maximum(cache.pred_ax31 - cache.pred_light, 0.0)
    actual_inc = np.maximum(ctx.costs[:, 1] - ctx.costs[:, 0], 0.0)
    fam = np.asarray(ctx.families)
    out: dict[str, float] = {}
    lo, hi = FAMILY_MULT_CLIP
    for name in sorted(set(fam.tolist())):
        idx = np.flatnonzero(fam == name)
        ratio = float(actual_inc[idx].sum()) / max(float(pred_inc[idx].sum()), 1e-15)
        out[name] = float(min(hi, max(lo, ratio)))
    return out


def guarded_costs(
    cache: Any, families: Sequence[str], multipliers: Mapping[str, float]
) -> np.ndarray:
    inc = np.maximum(cache.pred_ax31 - cache.pred_light, 0.0)
    mult = np.asarray([float(multipliers.get(f, 1.0)) for f in families], dtype=np.float64)
    return cache.pred_light + inc * mult


def allocate_guarded(
    cache: Any,
    pred_ax31: np.ndarray,
    idx: np.ndarray,
    *,
    tier: str,
    cap: float,
    max_up: float,
) -> tuple[str, ...]:
    runaway = (
        derived_runaway_fraction(float(cap)) if tier == "fast" else float(LADDER_RUNAWAY)
    )
    models, _pred, _bound = allocate_numpy(
        cache.uplift[idx],
        cache.pred_light[idx],
        pred_ax31[idx],
        cap=float(cap),
        runaway_fraction=float(runaway),
        max_upgrade_fraction=float(max_up),
    )
    return tuple(models.tolist())


def evaluate(
    ctx: SplitContext,
    cache: Any,
    *,
    art: Mapping[str, Any],
    multipliers: Mapping[str, float],
) -> dict[str, Any]:
    pred_ax31 = guarded_costs(cache, ctx.families, multipliers)
    max_up = float(art.get("max_upgrade_fraction", LADDER_MAX_UPGRADE))
    full = np.arange(len(cache.digests), dtype=np.int64)
    tiers: dict[str, Any] = {}
    for tier in TIERS:
        cap_official = float(OFFICIAL_CAPS[tier])
        if tier == "premium":
            selection = select_premium_cached(
                cache.digests,
                cache.premium_uplift,
                cache.premium_costs,
                float(art["predicted_caps"]["premium"]),
            )[0]
            drift = np.asarray(
                [
                    _realized_ratio(
                        ctx.costs[idx],
                        select_premium_cached(
                            tuple(cache.digests[int(i)] for i in idx),
                            cache.premium_uplift[idx],
                            cache.premium_costs[idx],
                            float(art["predicted_caps"]["premium"]),
                        )[0],
                    )
                    for idx in ctx.drift.indexes
                ],
                dtype=np.float64,
            )
        else:
            selection = allocate_guarded(
                cache,
                pred_ax31,
                full,
                tier=tier,
                cap=float(art["predicted_caps"][tier]),
                max_up=max_up,
            )
            drift = np.asarray(
                [
                    _realized_ratio(
                        ctx.costs[idx],
                        allocate_guarded(
                            cache,
                            pred_ax31,
                            idx,
                            tier=tier,
                            cap=float(art["predicted_caps"][tier]),
                            max_up=max_up,
                        ),
                    )
                    for idx in ctx.drift.indexes
                ],
                dtype=np.float64,
            )
        realized = float(_realized_ratio(ctx.costs, selection))
        tiers[tier] = {
            "quality": json_float(score_mean(ctx.scores, selection)),
            "realized": json_float(realized),
            "inflated": json_float(realized * INFLATION),
            "ax31": ax31_count(selection),
            "cost_ok": bool(realized * INFLATION <= cap_official + 1e-12),
            "near_ok": bool(realized < NEAR_FRAC * cap_official - 1e-15),
            "drift_max": json_float(float(drift.max())),
            "drift_ruin_inflated": int(
                np.count_nonzero(drift * INFLATION > cap_official + 1e-15)
            ),
            "drift_n": int(drift.size),
        }
    weighted = weighted_final(
        float(tiers["fast"]["quality"]),
        float(tiers["balanced"]["quality"]),
        float(tiers["premium"]["quality"]),
    )
    return {"weighted": json_float(float(weighted)), "tiers": tiers}


def expected_score(splits: Mapping[str, Mapping[str, Any]]) -> float:
    dev = splits["dev"]["tiers"]
    total = 0.0
    for tier in TIERS:
        risk = max(
            int(splits[s]["tiers"][tier]["drift_ruin_inflated"])
            / max(int(splits[s]["tiers"][tier]["drift_n"]), 1)
            for s in splits
        )
        total += float(TIER_WEIGHTS[tier]) * float(dev[tier]["quality"]) * (1.0 - risk)
    return float(total)


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
    print("train family multipliers (clipped):", {k: round(v, 3) for k, v in train_mult.items()}, flush=True)

    schemes: dict[str, dict[str, float]] = {
        "none": {},
        "other=2": {RESIDUAL_FAMILY: 2.0},
        "other=3": {RESIDUAL_FAMILY: 3.0},
        "trainfam": dict(train_mult),
        "trainfam|other=3": {**train_mult, RESIDUAL_FAMILY: 3.0},
    }

    print("\n=== the family costing attempt sweep: guard scheme x Fast cap x Balanced cap ===", flush=True)
    rows: list[dict[str, Any]] = []
    for scheme, mult in schemes.items():
        for fast in (1.03, 1.05, 1.07, 1.09, 1.12):
            for bal in (1.38, 1.60):
                art = copy.deepcopy(selected)
                art["predicted_caps"]["fast"] = float(fast)
                art["predicted_caps"]["balanced"] = float(bal)
                art["runaway_fraction"] = float(derived_runaway_fraction(fast))
                splits = {
                    label: evaluate(
                        contexts[label], caches[label], art=art, multipliers=mult
                    )
                    for label in contexts
                }
                name = f"{scheme}|f={fast:.2f}|b={bal:.2f}"
                hard_ok = all(
                    splits[s]["tiers"][t]["cost_ok"] and splits[s]["tiers"][t]["near_ok"]
                    for s in splits
                    for t in TIERS
                )
                row = {
                    "name": name,
                    "scheme": scheme,
                    "fast_cap": fast,
                    "bal_cap": bal,
                    "multipliers": dict(mult),
                    "splits": splits,
                    "hard_ok": bool(hard_ok),
                    "dev_weighted": float(splits["dev"]["weighted"]),
                    "expected": expected_score(splits),
                    "drift": {
                        t: max(
                            int(splits[s]["tiers"][t]["drift_ruin_inflated"])
                            for s in splits
                        )
                        for t in TIERS
                    },
                }
                rows.append(row)
                dev = splits["dev"]["tiers"]
                print(
                    f"{name:28} dev={row['dev_weighted']:.6f} E={row['expected']:.6f} "
                    f"hard={hard_ok} drift={row['drift']} "
                    f"Fq={dev['fast']['quality']:.6f} Fr={dev['fast']['realized']:.4f} "
                    f"Bq={dev['balanced']['quality']:.6f}",
                    flush=True,
                )

    base = next(
        r for r in rows if r["scheme"] == "none" and r["fast_cap"] == 1.03 and r["bal_cap"] == 1.38
    )
    ok = [r for r in rows if r["hard_ok"] and r["drift"]["fast"] <= base["drift"]["fast"]
          and r["drift"]["balanced"] <= base["drift"]["balanced"]
          and r["drift"]["premium"] <= base["drift"]["premium"]]
    ok.sort(key=lambda r: (r["expected"], r["dev_weighted"]), reverse=True)
    print("\nbaseline (current selected)", base["name"], base["dev_weighted"], base["expected"], base["drift"])
    print("best under parent-level risk:", flush=True)
    for row in ok[:8]:
        print(
            f"  {row['name']:28} dev={row['dev_weighted']:.6f} E={row['expected']:.6f} "
            f"drift={row['drift']}",
            flush=True,
        )

    payload = {
        "experiment": "the family costing attempt",
        "residual_family": RESIDUAL_FAMILY,
        "family_mult_clip": list(FAMILY_MULT_CLIP),
        "train_family_multipliers": train_mult,
        "baseline": {k: v for k, v in base.items() if k != "splits"},
        "rows": rows,
        "best_under_parent_risk": [
            {k: v for k, v in row.items() if k != "splits"} for row in ok[:10]
        ],
        "timing_s": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nDONE the family costing attempt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
