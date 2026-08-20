# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the adaptive guard attempt — simulate a composition-adaptive budget guard.

the feasibility ladder leaves most of the Fast budget unspent (Dev realized 1.043 against a
1.186 documented operating limit) because a single static predicted cap has
to be safe for every possible hidden composition. But ``docs/RUNTIME.md``
hands the router the whole tier batch, so the router can measure the batch
it was given and price its own prediction error.

Mechanism:

1. On Train drift views only, run the allocator at a reference cap and
   record the inflation ``iota = realized_ratio / predicted_ratio``.
2. Regress ``log iota`` on permutation-invariant statistics of the batch's
   predicted costs (means, quantile shape, Herfindahl concentration).
3. At run time the guard estimates ``iota`` for the batch it sees, adds a
   residual margin, and solves for the predicted cap that keeps the
   *realized* ratio under the operating limit.

Nothing here reads Dev during fitting; Dev is only scored afterwards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, load_train
from research.lab.prefix_certificates import _realized_ratio, json_float
from research.lab.cap_certification import (
    LADDER_MAX_UPGRADE,
    LADDER_RUNAWAY,
    allocate_numpy,
    derived_runaway_fraction,
    score_mean,
)
from research.lab.hidden_set_gates import INFLATION, NEAR_FRAC, SplitContext
from ossp_router.protocol import TIERS, load_input, load_outcomes
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "try-adaptive-guard"
LADDER_PATH = ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json"
GUARD_TIERS = ("fast", "balanced")
REFERENCE_CAP = {"fast": 1.12, "balanced": 1.80}
RIDGE_ALPHA = 1.0
FEATURE_NAMES = (
    "log_mean_light",
    "log_med_light",
    "log_neff_frac",
    "inc_over_light",
    "q90_over_med_light",
    "q90_over_med_inc",
    "mean_uplift",
    "log_n",
)


def operating_limit(tier: str) -> float:
    cap = float(OFFICIAL_CAPS[tier])
    return min(cap / INFLATION, NEAR_FRAC * cap)


def batch_features(
    pred_light: np.ndarray, pred_ax31: np.ndarray, uplift: np.ndarray
) -> np.ndarray:
    light = np.asarray(pred_light, dtype=np.float64)
    inc = np.maximum(np.asarray(pred_ax31, dtype=np.float64) - light, 0.0)
    total = max(float(light.sum()), 1e-15)
    share = light / total
    neff = 1.0 / max(float(np.square(share).sum()), 1e-15)
    med_light = max(float(np.median(light)), 1e-15)
    med_inc = max(float(np.median(inc)), 1e-15)
    return np.asarray(
        [
            np.log(max(float(light.mean()), 1e-15)),
            np.log(med_light),
            np.log(max(neff / max(light.size, 1), 1e-15)),
            float(inc.sum()) / total,
            float(np.quantile(light, 0.9)) / med_light,
            float(np.quantile(inc, 0.9)) / med_inc,
            float(np.mean(uplift)),
            np.log(max(float(light.size), 1.0)),
        ],
        dtype=np.float64,
    )


def allocate(cache: Any, idx: np.ndarray, *, tier: str, cap: float, max_up: float):
    runaway = (
        derived_runaway_fraction(float(cap)) if tier == "fast" else float(LADDER_RUNAWAY)
    )
    models, pred, _bound = allocate_numpy(
        cache.uplift[idx],
        cache.pred_light[idx],
        cache.pred_ax31[idx],
        cap=float(cap),
        runaway_fraction=float(runaway),
        max_upgrade_fraction=float(max_up),
    )
    return tuple(models.tolist()), float(pred)


class InflationModel:
    """Ridge on log(realized / predicted) with a frozen residual margin."""

    def __init__(self, coef: np.ndarray, intercept: float, sigma: float, z: float):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.sigma = float(sigma)
        self.z = float(z)

    @classmethod
    def fit(cls, features: np.ndarray, iota: np.ndarray, *, z: float) -> "InflationModel":
        target = np.log(np.maximum(iota, 1e-9))
        design = np.concatenate([np.ones((features.shape[0], 1)), features], axis=1)
        penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        resid = target - design @ beta
        sigma = float(np.sqrt(max(float(np.mean(np.square(resid))), 0.0)))
        return cls(beta[1:], float(beta[0]), sigma, z)

    def predict(self, features: np.ndarray) -> float:
        raw = self.intercept + float(features @ self.coef)
        return float(np.exp(raw + self.z * self.sigma))

    def as_dict(self, names: Sequence[str]) -> dict[str, Any]:
        return {
            "coefficients": {n: float(c) for n, c in zip(names, self.coef)},
            "intercept": float(self.intercept),
            "residual_sigma": float(self.sigma),
            "z": float(self.z),
        }


def guard_cap(
    model: InflationModel,
    features: np.ndarray,
    *,
    tier: str,
    base_cap: float,
    target: float,
    floor: float,
) -> tuple[float, float]:
    iota = max(model.predict(features), 1e-9)
    allowed = float(target) / iota
    return float(min(base_cap, max(floor, allowed))), iota


def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading", flush=True)
    bundle = load_train(None)
    policy = bundle.policy
    dev_inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    dev_outcomes = load_outcomes(ROOT / "data" / "dev" / "outcomes.json")
    arrays = public_arrays(dev_inputs, dev_outcomes, policy)

    ctx_train = SplitContext.build(
        label="train",
        inputs=bundle.inputs,
        policy=policy,
        scores=bundle.scores,
        costs=bundle.costs,
        families=list(bundle.families),
    )
    ctx_dev = SplitContext.build(
        label="dev",
        inputs=dev_inputs,
        policy=policy,
        scores=np.asarray(arrays.scores),
        costs=np.asarray(arrays.costs),
        families=[prompt_family(ep) for ep in dev_inputs.episodes],
    )
    ladder = json.loads(LADDER_PATH.read_text(encoding="utf-8"))
    max_up = float(ladder.get("max_upgrade_fraction", LADDER_MAX_UPGRADE))
    cache_train = ctx_train.prediction_cache(ladder)
    cache_dev = ctx_dev.prediction_cache(ladder)

    print("\n=== fit the inflation model on Train drift views ===", flush=True)
    models: dict[str, InflationModel] = {}
    fit_stats: dict[str, Any] = {}
    train_views = list(ctx_train.drift.indexes) + [
        np.arange(len(cache_train.digests), dtype=np.int64)
    ]
    for tier in GUARD_TIERS:
        feats = []
        iotas = []
        for idx in train_views:
            selection, pred = allocate(
                cache_train, idx, tier=tier, cap=REFERENCE_CAP[tier], max_up=max_up
            )
            realized = float(_realized_ratio(ctx_train.costs[idx], selection))
            if pred <= 1e-9:
                continue
            feats.append(
                batch_features(
                    cache_train.pred_light[idx],
                    cache_train.pred_ax31[idx],
                    cache_train.uplift[idx],
                )
            )
            iotas.append(realized / pred)
        features = np.asarray(feats, dtype=np.float64)
        iota = np.asarray(iotas, dtype=np.float64)
        model = InflationModel.fit(features, iota, z=2.0)
        models[tier] = model
        pred_iota = np.asarray(
            [model.predict(row) - 0.0 for row in features], dtype=np.float64
        )
        fit_stats[tier] = {
            "n_views": int(iota.size),
            "iota_observed": {
                "mean": json_float(float(iota.mean())),
                "p50": json_float(float(np.quantile(iota, 0.5))),
                "p99": json_float(float(np.quantile(iota, 0.99))),
                "max": json_float(float(iota.max())),
            },
            "coverage_upper": json_float(
                float(np.mean(pred_iota >= iota))
            ),
            "model": model.as_dict(FEATURE_NAMES),
        }
        print(
            f"{tier:9} n={iota.size} iota p50={np.quantile(iota, 0.5):.3f} "
            f"p99={np.quantile(iota, 0.99):.3f} max={iota.max():.3f} "
            f"upper-coverage={np.mean(pred_iota >= iota):.4f}",
            flush=True,
        )

    print("\n=== guard sweep (fit on Train, scored on both) ===", flush=True)
    limits = {tier: operating_limit(tier) for tier in TIERS}
    print("operating limits", {t: round(limits[t], 4) for t in TIERS}, flush=True)

    def score_guard(
        ctx: SplitContext,
        cache: Any,
        *,
        margin: float,
        base_cap: Mapping[str, float],
        floor: Mapping[str, float],
    ) -> dict[str, Any]:
        full_index = np.arange(len(cache.digests), dtype=np.int64)
        out: dict[str, Any] = {"tiers": {}}
        for tier in GUARD_TIERS:
            target = limits[tier] * float(margin)
            rows = []
            for idx in [full_index] + list(ctx.drift.indexes):
                feats = batch_features(
                    cache.pred_light[idx], cache.pred_ax31[idx], cache.uplift[idx]
                )
                cap, iota = guard_cap(
                    models[tier],
                    feats,
                    tier=tier,
                    base_cap=float(base_cap[tier]),
                    target=target,
                    floor=float(floor[tier]),
                )
                selection, _pred = allocate(
                    cache, idx, tier=tier, cap=cap, max_up=max_up
                )
                realized = float(_realized_ratio(ctx.costs[idx], selection))
                rows.append((idx, selection, realized, cap, iota))
            full_row = rows[0]
            drift = np.asarray([r[2] for r in rows[1:]], dtype=np.float64)
            cap_official = float(OFFICIAL_CAPS[tier])
            out["tiers"][tier] = {
                "quality": json_float(score_mean(ctx.scores, full_row[1])),
                "realized": json_float(full_row[2]),
                "inflated": json_float(full_row[2] * INFLATION),
                "cap_used": json_float(full_row[3]),
                "iota_full": json_float(full_row[4]),
                "cost_ok": bool(full_row[2] * INFLATION <= cap_official + 1e-12),
                "near_ok": bool(full_row[2] < NEAR_FRAC * cap_official - 1e-15),
                "drift_max": json_float(float(drift.max())),
                "drift_ruin": int(np.count_nonzero(drift > cap_official + 1e-15)),
                "drift_ruin_inflated": int(
                    np.count_nonzero(drift * INFLATION > cap_official + 1e-15)
                ),
                "drift_n": int(drift.size),
            }
        return out

    grid = []
    for margin in (0.90, 0.95, 1.00):
        for base_fast in (1.10, 1.15, 1.20):
            grid.append({"margin": margin, "base": {"fast": base_fast, "balanced": 1.80}})

    results = []
    for cfg in grid:
        base = cfg["base"]
        floor = {"fast": 1.0, "balanced": 1.0}
        train_row = score_guard(
            ctx_train, cache_train, margin=cfg["margin"], base_cap=base, floor=floor
        )
        dev_row = score_guard(
            ctx_dev, cache_dev, margin=cfg["margin"], base_cap=base, floor=floor
        )
        name = f"m={cfg['margin']:.2f}|bf={base['fast']:.2f}"
        results.append(
            {"name": name, "config": cfg, "train": train_row, "dev": dev_row}
        )
        dt = dev_row["tiers"]
        print(
            f"{name:20} devFast q={dt['fast']['quality']:.6f} r={dt['fast']['realized']:.4f} "
            f"cap={dt['fast']['cap_used']:.3f} driftmax={dt['fast']['drift_max']:.3f} "
            f"ruin_i={dt['fast']['drift_ruin_inflated']:4d} | "
            f"devBal q={dt['balanced']['quality']:.6f} r={dt['balanced']['realized']:.4f} "
            f"driftmax={dt['balanced']['drift_max']:.3f} "
            f"ruin_i={dt['balanced']['drift_ruin_inflated']:4d}",
            flush=True,
        )

    payload = {
        "experiment": "the adaptive guard attempt",
        "reference_cap": REFERENCE_CAP,
        "operating_limits": {t: json_float(limits[t]) for t in TIERS},
        "features": list(FEATURE_NAMES),
        "fit": fit_stats,
        "grid": results,
        "note": (
            "Premium is excluded: its ax31 upgrades are already saturated and "
            "the remaining lever there is K1, whose per-episode cost tail is "
            "not a composition effect."
        ),
        "timing_s": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nDONE the adaptive guard attempt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
