# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Hidden-set stress suite and gates, tagged to the organizer documents.

Every gate here traces to a sentence the organizers wrote:

* ``docs/SCORING.md`` — a tier over its cap scores ``0``; self-check flags
  ``near_budget`` at 95% of the cap.
* ``baselines/README.md`` — the official hash-regex baseline passed public
  Dev Premium at ``3.985`` and then measured ``~4.2`` on the graded set,
  so its Premium tier scored ``0``. That ratio inflation (``~1.054``) is
  the documented hidden-set failure mode.
* ``docs/DATA_CARD.md`` — public composition does not imply the private
  composition, so plausible family-mixture and batch-size drift must be
  survivable.
* ``docs/RUNTIME.md`` — the router is handed the whole tier batch in one
  run, so it re-allocates on the hidden batch: drift views replay the
  allocator rather than freezing the public selection.
* ``the earlier fixed-selection policy report`` — an earlier policy kept the public selection but blew Premium apart under
  composition shift, so Premium
  additionally has to survive the non-adaptive (fixed-selection) bound.

View kinds are split into gate-bearing ``drift`` views (plausible) and
``red-team`` views (famdom 75% single family, LOFO) that are reported but
never disqualify: those are doubly pessimistic by construction.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, STRESS_BACKSTOP, TIER_WEIGHTS, weighted_final
from research.lab.prefix_certificates import _realized_ratio, json_float
from research.lab.cap_certification import (
    LADDER_MAX_UPGRADE,
    LADDER_RUNAWAY,
    allocate_numpy,
    ax31_count,
    build_stress_views,
    cache_predictions,
    derived_runaway_fraction,
    k1_count,
    score_mean,
    select_premium_cached,
)
from ossp_router.protocol import MODEL_IDS, TIERS
from ossp_router.feasibility_ladder import load_artifact_mapping


INFLATION = float(STRESS_BACKSTOP)
NEAR_FRAC = 0.95
DRIFT_KINDS = ("dirichlet", "half", "small")
REDTEAM_KINDS = ("famdom", "lofo", "lofo-combined")
CAP_KEYS = ("predicted_caps", "max_upgrade_fraction", "runaway_fraction")

DOC_TAGS = {
    "full_batch_inflated": "baselines/README.md hash-regex 3.985 -> ~4.2",
    "near_budget": "docs/SCORING.md near_budget at 95% of cap",
    "drift_views": "docs/DATA_CARD.md private composition undisclosed",
    "adaptive_replay": "docs/RUNTIME.md whole tier batch in one run",
    "premium_fixed": "the earlier fixed-selection policy fixed-selection Premium blowup",
    "redteam_only": "the project charter.md famdom/LOFO doubly pessimistic",
}


def is_binding(view: Any) -> bool:
    if view.kind in ("lofo", "lofo-combined"):
        return False
    if view.kind == "small" and str(view.name).startswith("small-100-"):
        return False
    return True


@dataclass(frozen=True)
class ViewGroup:
    name: str
    indexes: tuple[np.ndarray, ...]


@dataclass
class SplitContext:
    """Everything that does not depend on the candidate artifact."""

    label: str
    inputs: Any
    policy: Any
    scores: np.ndarray
    costs: np.ndarray
    families: list[str]
    drift: ViewGroup
    redteam: ViewGroup
    catalogue: Mapping[str, Any]
    _cache: dict[str, Any]

    @classmethod
    def build(
        cls,
        *,
        label: str,
        inputs: Any,
        policy: Any,
        scores: np.ndarray,
        costs: np.ndarray,
        families: Sequence[str],
    ) -> "SplitContext":
        views, catalogue = build_stress_views(list(families))
        drift = tuple(
            np.asarray(view.index, dtype=np.int64)
            for view in views
            if is_binding(view) and view.kind in DRIFT_KINDS
        )
        redteam = tuple(
            np.asarray(view.index, dtype=np.int64)
            for view in views
            if view.kind in REDTEAM_KINDS
        )
        return cls(
            label=label,
            inputs=inputs,
            policy=policy,
            scores=np.asarray(scores, dtype=np.float64),
            costs=np.asarray(costs, dtype=np.float64),
            families=list(families),
            drift=ViewGroup("drift", drift),
            redteam=ViewGroup("red-team", redteam),
            catalogue=catalogue,
            _cache={},
        )

    def prediction_cache(self, art_dict: Mapping[str, Any]) -> Any:
        """Predictions only depend on the heads, not on the cap knobs."""

        head = {k: v for k, v in dict(art_dict).items() if k not in CAP_KEYS}
        key = hashlib.sha256(
            json.dumps(head, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        hit = self._cache.get(key)
        if hit is None:
            artifact = load_artifact_mapping(copy.deepcopy(dict(art_dict)))
            hit = cache_predictions(self.inputs.episodes, self.policy, artifact)
            self._cache[key] = hit
        return hit


def select_tier(cache: Any, art: Mapping[str, Any], tier: str, index: np.ndarray | None = None):
    """Replay the router allocator for one tier over ``index`` (default: all)."""

    caps = art["predicted_caps"]
    if index is None:
        index = np.arange(len(cache.digests), dtype=np.int64)
    if tier == "premium":
        digests = tuple(cache.digests[int(i)] for i in index)
        models, _pred = select_premium_cached(
            digests,
            cache.premium_uplift[index],
            cache.premium_costs[index],
            float(caps["premium"]),
        )
        return models
    runaway = (
        derived_runaway_fraction(float(caps["fast"]))
        if tier == "fast"
        else float(art.get("balanced_runaway_fraction", LADDER_RUNAWAY))
    )
    models, _pred, _bound = allocate_numpy(
        cache.uplift[index],
        cache.pred_light[index],
        cache.pred_ax31[index],
        cap=float(caps[tier]),
        runaway_fraction=float(runaway),
        max_upgrade_fraction=float(art.get("max_upgrade_fraction", LADDER_MAX_UPGRADE)),
    )
    return tuple(models.tolist())


def _ratio(costs: np.ndarray, selection: Sequence[str]) -> float:
    return float(_realized_ratio(costs, selection))


def _fixed_ratios(
    *, selection: Sequence[str], costs: np.ndarray, indexes: Sequence[np.ndarray]
) -> np.ndarray:
    col = {model_id: i for i, model_id in enumerate(MODEL_IDS)}
    chosen = np.asarray(
        [costs[i, col[selection[i]]] for i in range(len(selection))], dtype=np.float64
    )
    light = np.asarray(costs[:, 0], dtype=np.float64)
    out = np.empty(len(indexes), dtype=np.float64)
    for i, idx in enumerate(indexes):
        out[i] = float(chosen[idx].sum() / max(float(light[idx].sum()), 1e-15))
    return out


def _summarize(ratios: np.ndarray, cap: float) -> dict[str, Any]:
    if ratios.size == 0:
        return {"n": 0, "max_realized": 0.0, "n_ruin": 0, "n_ruin_inflated": 0, "p99": 0.0}
    inflated = ratios * INFLATION
    return {
        "n": int(ratios.size),
        "max_realized": json_float(float(ratios.max())),
        "max_inflated": json_float(float(inflated.max())),
        "n_ruin": int(np.count_nonzero(ratios > cap + 1e-15)),
        "n_ruin_inflated": int(np.count_nonzero(inflated > cap + 1e-15)),
        "p99": json_float(float(np.quantile(ratios, 0.99))),
    }


def evaluate_candidate(
    art_dict: Mapping[str, Any],
    ctx: SplitContext,
    *,
    with_redteam: bool = False,
) -> dict[str, Any]:
    """Full-batch score plus hidden-set stress for one artifact on one split."""

    cache = ctx.prediction_cache(art_dict)
    tiers: dict[str, Any] = {}
    selections: dict[str, tuple[str, ...]] = {}
    for tier in TIERS:
        cap = float(OFFICIAL_CAPS[tier])
        selection = select_tier(cache, art_dict, tier)
        selections[tier] = tuple(selection)
        realized = _ratio(ctx.costs, selection)
        drift_ratios = np.asarray(
            [
                _ratio(ctx.costs[idx], select_tier(cache, art_dict, tier, idx))
                for idx in ctx.drift.indexes
            ],
            dtype=np.float64,
        )
        block: dict[str, Any] = {
            "quality": json_float(score_mean(ctx.scores, selection)),
            "realized": json_float(realized),
            "inflated": json_float(realized * INFLATION),
            "ax31": ax31_count(selection),
            "k1": k1_count(selection),
            "cost_ok": bool(realized * INFLATION <= cap + 1e-12),
            "near_ok": bool(realized < NEAR_FRAC * cap - 1e-15),
            "headroom_inflated": json_float(cap - realized * INFLATION),
            "drift": _summarize(drift_ratios, cap),
        }
        if tier == "premium":
            block["fixed_composition"] = _summarize(
                _fixed_ratios(
                    selection=selection, costs=ctx.costs, indexes=ctx.drift.indexes
                ),
                cap,
            )
        if with_redteam and ctx.redteam.indexes:
            red = np.asarray(
                [
                    _ratio(ctx.costs[idx], select_tier(cache, art_dict, tier, idx))
                    for idx in ctx.redteam.indexes
                ],
                dtype=np.float64,
            )
            block["red_team"] = _summarize(red, cap)
            if tier == "premium":
                block["red_team_fixed"] = _summarize(
                    _fixed_ratios(
                        selection=selection,
                        costs=ctx.costs,
                        indexes=ctx.redteam.indexes,
                    ),
                    cap,
                )
        tiers[tier] = block

    weighted = weighted_final(
        float(tiers["fast"]["quality"]),
        float(tiers["balanced"]["quality"]),
        float(tiers["premium"]["quality"]),
    )
    return {
        "split": ctx.label,
        "weighted": json_float(float(weighted)),
        "tiers": tiers,
        "selections_sig": {
            tier: hashlib.sha256("|".join(selections[tier]).encode("utf-8")).hexdigest()[:16]
            for tier in TIERS
        },
    }


def evaluate_fixed_selection(
    selections: Mapping[str, Sequence[str]], ctx: SplitContext
) -> dict[str, Any]:
    """Same metrics for a non-adaptive baseline (no artifact / no allocator)."""

    tiers: dict[str, Any] = {}
    for tier in TIERS:
        cap = float(OFFICIAL_CAPS[tier])
        selection = list(selections[tier])
        realized = _ratio(ctx.costs, selection)
        drift = _fixed_ratios(
            selection=selection, costs=ctx.costs, indexes=ctx.drift.indexes
        )
        tiers[tier] = {
            "quality": json_float(score_mean(ctx.scores, selection)),
            "realized": json_float(realized),
            "inflated": json_float(realized * INFLATION),
            "ax31": ax31_count(selection),
            "k1": k1_count(selection),
            "cost_ok": bool(realized * INFLATION <= cap + 1e-12),
            "near_ok": bool(realized < NEAR_FRAC * cap - 1e-15),
            "headroom_inflated": json_float(cap - realized * INFLATION),
            "drift": _summarize(drift, cap),
            "method": "fixed-selection",
        }
        if tier == "premium":
            tiers[tier]["fixed_composition"] = tiers[tier]["drift"]
    weighted = weighted_final(
        float(tiers["fast"]["quality"]),
        float(tiers["balanced"]["quality"]),
        float(tiers["premium"]["quality"]),
    )
    return {"split": ctx.label, "weighted": json_float(float(weighted)), "tiers": tiers}


def gate_report(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    parent_drift: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Apply the hidden-set gates across every split that was measured."""

    checks: dict[str, Any] = {}
    cost_ok = all(rows[s]["tiers"][t]["cost_ok"] for s in rows for t in TIERS)
    near_ok = all(rows[s]["tiers"][t]["near_ok"] for s in rows for t in TIERS)
    checks["A_full_batch_inflated"] = bool(cost_ok)
    checks["A_near_budget"] = bool(near_ok)

    prem_fixed_ok = True
    for split in rows:
        block = rows[split]["tiers"]["premium"].get("fixed_composition")
        if block is None:
            continue
        if int(block["n_ruin"]) != 0 or float(block["max_realized"]) > float(
            OFFICIAL_CAPS["premium"]
        ) + 1e-12:
            prem_fixed_ok = False
    checks["C_premium_fixed_composition"] = bool(prem_fixed_ok)

    drift_counts = {
        tier: max(int(rows[s]["tiers"][tier]["drift"]["n_ruin_inflated"]) for s in rows)
        for tier in TIERS
    }
    if parent_drift is None:
        checks["B_drift_not_worse_than_parent"] = True
    else:
        checks["B_drift_not_worse_than_parent"] = all(
            drift_counts[tier] <= int(parent_drift[tier]) for tier in TIERS
        )
    eligible = all(bool(value) for value in checks.values())
    return {
        "checks": checks,
        "drift_ruin_inflated": drift_counts,
        "eligible": bool(eligible),
        "doc_tags": DOC_TAGS,
    }


def headroom_key(rows: Mapping[str, Mapping[str, Any]]) -> float:
    """Weighted worst-case headroom, used only to break score ties."""

    worst = 0.0
    for tier in TIERS:
        cap = float(OFFICIAL_CAPS[tier])
        margin = min(
            (cap - float(rows[s]["tiers"][tier]["drift"]["max_inflated"])) / cap
            for s in rows
        )
        worst += float(TIER_WEIGHTS[tier]) * margin
    return float(worst)


__all__ = [
    "DOC_TAGS",
    "DRIFT_KINDS",
    "INFLATION",
    "NEAR_FRAC",
    "SplitContext",
    "evaluate_candidate",
    "evaluate_fixed_selection",
    "gate_report",
    "headroom_key",
    "select_tier",
]
