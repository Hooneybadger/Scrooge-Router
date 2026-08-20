# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the policy search — selected search under the document-tagged hidden-set gates.

Walks the strategy tree from the public baselines upward. A candidate is
eligible only if, on **both** public splits:

A. every tier passes ``realized * 1.054 <= cap`` and stays below the
   ``near_budget`` line (0.95 * cap);
B. its inflated ruin count on plausible drift views is no worse than the
   certified rollback parent (the feasibility ladder) tier by tier;
C. Premium survives the non-adaptive fixed-selection composition bound
   with zero ruin.

Among the survivors the selected is the highest Dev weighted score, ties
broken by worst-case headroom.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from research.lab.baseline_bakeoff import build_live_submissions
from research.lab.modeling import OFFICIAL_CAPS, load_train
from research.lab.cap_certification import derived_runaway_fraction, sha256_file
from research.lab.hidden_set_gates import (
    DOC_TAGS,
    INFLATION,
    NEAR_FRAC,
    SplitContext,
    evaluate_candidate,
    evaluate_fixed_selection,
    gate_report,
    headroom_key,
)
from ossp_router.cost_calibrated_router import structural_features
from ossp_router.protocol import TIERS, load_input, load_outcomes
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "search-policy-space"
LADDER_PATH = ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json"
GUARD_RESOURCE = ROOT / "src" / "ossp_router" / "resources" / "selected-router.v1.json"
HR_PATH = ROOT / "baselines" / "hash-regex-public.v1.json"


def fit_structural_qk(episodes: Sequence[Any], scores: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray([structural_features(ep) for ep in episodes], dtype=np.float64)
    target = scores[:, 2] - scores[:, 0]
    mean = matrix.mean(0)
    scale = matrix.std(0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    alpha = 100.0
    design = np.concatenate([np.ones((len(target), 1)), standardized], axis=1)
    penalty = np.eye(matrix.shape[1] + 1)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ target)
    coef = beta[1:]
    intercept = float(beta[0]) - float((mean / scale) @ coef)
    return {
        "alpha": alpha,
        "artifact_type": "scrooge-policy_search-k1-uplift-ridge-v1",
        "coefficients": [float(v) for v in coef],
        "feature": {
            "char_ngrams": False,
            "dimension": 14,
            "hash_bins": 0,
            "name": "structural",
            "word_ngrams": False,
        },
        "intercept": [float(intercept)],
        "scale": [float(v) for v in scale],
        "schema_version": 1,
        "target": "k1-light.score",
    }


def variant(
    base: Mapping[str, Any],
    *,
    fast: Optional[float] = None,
    balanced: Optional[float] = None,
    premium: Optional[float] = None,
    max_up: Optional[float] = None,
    kappa: Optional[float] = None,
    k1: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    art = copy.deepcopy(dict(base))
    caps = art["predicted_caps"]
    if fast is not None:
        caps["fast"] = float(fast)
        art["runaway_fraction"] = float(derived_runaway_fraction(float(fast)))
    if balanced is not None:
        caps["balanced"] = float(balanced)
    if premium is not None:
        caps["premium"] = float(premium)
    if max_up is not None:
        art["max_upgrade_fraction"] = float(max_up)
    if kappa is not None:
        art.setdefault("premium_overlay", {})["kappa_q999"] = float(kappa)
    if k1 is None:
        art["k1_enabled"] = False
    else:
        art["k1_enabled"] = True
        art["k1"] = {
            "activation": "artifact-flag-plus-quality-head",
            "enabled": True,
            "quality": copy.deepcopy(dict(k1)),
            "scope": "premium-only",
        }
    return art


class Search:
    def __init__(self, contexts: Mapping[str, SplitContext], parent_drift: Mapping[str, int] | None):
        self.contexts = contexts
        self.parent_drift = parent_drift
        self.rows: list[dict[str, Any]] = []
        self.seen: set[str] = set()

    def key(self, art: Mapping[str, Any]) -> str:
        caps = art["predicted_caps"]
        return json.dumps(
            {
                "f": caps["fast"],
                "b": caps["balanced"],
                "p": caps["premium"],
                "m": art.get("max_upgrade_fraction"),
                "k": art.get("premium_overlay", {}).get("kappa_q999"),
                "k1": bool(art.get("k1_enabled")),
            },
            sort_keys=True,
        )

    def run(self, name: str, art: Mapping[str, Any], *, stage: str) -> Optional[dict[str, Any]]:
        key = self.key(art)
        if key in self.seen:
            return None
        self.seen.add(key)
        splits = {
            label: evaluate_candidate(art, ctx) for label, ctx in self.contexts.items()
        }
        gates = gate_report(splits, parent_drift=self.parent_drift)
        row = {
            "name": name,
            "stage": stage,
            "art": copy.deepcopy(dict(art)),
            "splits": splits,
            "gates": gates,
            "dev_weighted": float(splits["dev"]["weighted"]),
            "train_weighted": float(splits["train"]["weighted"]),
            "headroom": headroom_key(splits),
        }
        self.rows.append(row)
        dev = splits["dev"]["tiers"]
        print(
            f"{stage:7} {name:26} dev={row['dev_weighted']:.6f} "
            f"train={row['train_weighted']:.6f} elig={gates['eligible']} "
            f"drift={gates['drift_ruin_inflated']} "
            f"infl(F/B/P)={dev['fast']['inflated']:.3f}/"
            f"{dev['balanced']['inflated']:.3f}/{dev['premium']['inflated']:.3f}",
            flush=True,
        )
        return row

    def eligible(self) -> list[dict[str, Any]]:
        rows = [r for r in self.rows if r["gates"]["eligible"]]
        rows.sort(key=lambda r: (r["dev_weighted"], r["headroom"]), reverse=True)
        return rows


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "art"}


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

    print("\n=== public baselines (fixed selection) ===", flush=True)
    baselines: list[dict[str, Any]] = []
    inputs_by_split = {"train": bundle.inputs, "dev": dev_inputs}
    for name in ("always_light", "prompt_heuristic", "feature_budget"):
        splits = {}
        for label, ctx in contexts.items():
            subs = build_live_submissions(inputs_by_split[label], policy, name)
            sels = {
                tier: tuple(d.model_id for d in subs[tier].decisions) for tier in TIERS
            }
            splits[label] = evaluate_fixed_selection(sels, ctx)
        gates = gate_report(splits, parent_drift=None)
        baselines.append({"name": name, "splits": splits, "gates": gates})
        print(
            f"base    {name:26} dev={splits['dev']['weighted']:.6f} "
            f"elig(A/C only)={gates['eligible']} "
            f"infl(F/B/P)={splits['dev']['tiers']['fast']['inflated']:.3f}/"
            f"{splits['dev']['tiers']['balanced']['inflated']:.3f}/"
            f"{splits['dev']['tiers']['premium']['inflated']:.3f}",
            flush=True,
        )
    if HR_PATH.exists():
        sys.path.insert(0, str(ROOT / "baselines"))
        from hash_regex import load_artifact, make_hash_regex_submission

        hr = load_artifact(HR_PATH)
        splits = {}
        for label, ctx in contexts.items():
            sels = {
                tier: tuple(
                    d.model_id
                    for d in make_hash_regex_submission(
                        inputs_by_split[label], policy, hr, tier
                    ).submission.decisions
                )
                for tier in TIERS
            }
            splits[label] = evaluate_fixed_selection(sels, ctx)
        gates = gate_report(splits, parent_drift=None)
        baselines.append({"name": "hash_regex", "splits": splits, "gates": gates})
        print(
            f"base    {'hash_regex':26} dev={splits['dev']['weighted']:.6f} "
            f"elig(A/C only)={gates['eligible']} "
            f"infl(F/B/P)={splits['dev']['tiers']['fast']['inflated']:.3f}/"
            f"{splits['dev']['tiers']['balanced']['inflated']:.3f}/"
            f"{splits['dev']['tiers']['premium']['inflated']:.3f}",
            flush=True,
        )

    print("\n=== parent feasibility ladder (sets the risk budget) ===", flush=True)
    ladder = json.loads(LADDER_PATH.read_text(encoding="utf-8"))
    probe = Search(contexts, None)
    parent_row = probe.run("the feasibility ladder", ladder, stage="parent")
    assert parent_row is not None
    parent_drift = dict(parent_row["gates"]["drift_ruin_inflated"])
    print("parent drift budget", parent_drift, flush=True)

    search = Search(contexts, parent_drift)
    search.rows.append(parent_row)
    search.seen.add(search.key(ladder))
    parent_row["gates"] = gate_report(parent_row["splits"], parent_drift=parent_drift)

    print("\n=== the policy search single-axis sweeps (K1 off) ===", flush=True)
    for fast in (1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.12):
        search.run(f"fast={fast:.2f}", variant(ladder, fast=fast), stage="fast")
    for bal in (1.40, 1.45, 1.50, 1.60, 1.70, 1.80, 1.90):
        search.run(f"bal={bal:.2f}", variant(ladder, balanced=bal), stage="bal")
    for max_up in (0.55, 0.65, 0.75, 0.82, 0.85, 0.95):
        search.run(f"maxup={max_up:.2f}", variant(ladder, max_up=max_up), stage="maxup")
    for prem in (2.50, 3.00, 3.25, 3.50, 3.75):
        search.run(f"prem={prem:.2f}", variant(ladder, premium=prem), stage="prem")
    for kappa in (0.80, 1.00, 1.1025, 1.25, 1.50, 2.00):
        search.run(f"kappa={kappa:.4g}", variant(ladder, kappa=kappa), stage="kappa")

    print("\n=== joint of surviving axes ===", flush=True)
    elig = search.eligible()

    def best_axis(stage: str, field: str, default: float) -> float:
        rows = [r for r in elig if r["stage"] == stage]
        if not rows:
            return default
        art = rows[0]["art"]
        if field == "max_up":
            return float(art.get("max_upgrade_fraction", default))
        if field == "kappa":
            return float(art.get("premium_overlay", {}).get("kappa_q999", default))
        return float(art["predicted_caps"][field])

    best_fast = best_axis("fast", "fast", float(ladder["predicted_caps"]["fast"]))
    best_bal = best_axis("bal", "balanced", float(ladder["predicted_caps"]["balanced"]))
    best_up = best_axis("maxup", "max_up", float(ladder.get("max_upgrade_fraction", 0.75)))
    best_prem = best_axis("prem", "premium", float(ladder["predicted_caps"]["premium"]))
    best_kappa = best_axis(
        "kappa", "kappa", float(ladder.get("premium_overlay", {}).get("kappa_q999", 1.1025))
    )
    print(
        "axis winners",
        {
            "fast": best_fast,
            "balanced": best_bal,
            "max_up": best_up,
            "premium": best_prem,
            "kappa": best_kappa,
        },
        flush=True,
    )
    joint = variant(
        ladder,
        fast=best_fast,
        balanced=best_bal,
        premium=best_prem,
        max_up=best_up,
        kappa=best_kappa,
    )
    search.run("joint", joint, stage="joint")
    # Neighbourhood around the joint point.
    for fast in sorted({round(best_fast + d, 3) for d in (-0.02, -0.01, 0.01, 0.02)}):
        if fast <= 1.0:
            continue
        search.run(f"joint|fast={fast:.3f}", variant(joint, fast=fast), stage="joint")
    for max_up in sorted({round(best_up + d, 2) for d in (-0.1, -0.05, 0.05, 0.1)}):
        if not 0.0 < max_up <= 1.0:
            continue
        search.run(f"joint|maxup={max_up:.2f}", variant(joint, max_up=max_up), stage="joint")

    print("\n=== Premium K1 band on the joint parent ===", flush=True)
    quality = fit_structural_qk(bundle.inputs.episodes, bundle.scores)
    for kappa in (1.25, 1.50, 1.75, 2.00, 3.00):
        for prem in (2.25, 2.50, 2.75, 3.00, 3.25):
            search.run(
                f"k1|k={kappa}|c={prem}",
                variant(joint, premium=prem, kappa=kappa, k1=quality),
                stage="k1",
            )

    print("\n=== lock ===", flush=True)
    elig = search.eligible()
    parent_dev = float(parent_row["dev_weighted"])
    selected = parent_row
    for row in elig:
        if row["dev_weighted"] > parent_dev + 1e-12:
            selected = row
            break
        if (
            abs(row["dev_weighted"] - parent_dev) <= 1e-12
            and row["headroom"] > selected["headroom"] + 1e-12
        ):
            selected = row
            break
    promoted = selected is not parent_row
    print(
        "SELECTED",
        selected["name"],
        "dev",
        selected["dev_weighted"],
        "train",
        selected["train_weighted"],
        "headroom",
        round(selected["headroom"], 5),
        "promoted",
        promoted,
        flush=True,
    )

    print("re-measuring selected with the red-team layer", flush=True)
    selected_full = {
        label: evaluate_candidate(selected["art"], ctx, with_redteam=True)
        for label, ctx in contexts.items()
    }

    art_path = OUT / "policy_search-selected-router.v1.json"
    art_path.write_text(
        json.dumps(selected["art"], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    art_path.chmod(0o644)
    GUARD_RESOURCE.write_text(art_path.read_text(encoding="utf-8"), encoding="utf-8")
    GUARD_RESOURCE.chmod(0o644)

    safe = (
        selected["name"].replace("=", "p").replace("|", "_").replace(".", "p").lower()[:48]
    )
    report = {
        "experiment": "the policy search",
        "decision": (
            f"record-policy_search-promote-{safe}" if promoted else "record-policy_search-retain-ladder"
        ),
        "promoted": bool(promoted),
        "gates": {
            "A_full_batch_inflated": f"realized * {INFLATION} <= cap on train and dev",
            "A_near_budget": f"realized < {NEAR_FRAC} * cap on train and dev",
            "B_drift_not_worse_than_parent": "inflated ruin count on drift views <= the feasibility ladder",
            "C_premium_fixed_composition": "premium fixed-selection ruin == 0",
            "doc_tags": DOC_TAGS,
            "official_caps": {t: float(OFFICIAL_CAPS[t]) for t in TIERS},
            "parent_drift_budget": parent_drift,
        },
        "selected": {
            "name": selected["name"],
            "stage": selected["stage"],
            "dev_weighted": selected["dev_weighted"],
            "train_weighted": selected["train_weighted"],
            "headroom": selected["headroom"],
            "gates": selected["gates"],
            "splits": selected_full,
            "artifact_sha256": sha256_file(art_path),
            "path": str(art_path),
            "resource": str(GUARD_RESOURCE),
        },
        "parent": public_row(parent_row),
        "baselines": baselines,
        "eligible_top": [public_row(r) for r in elig[:15]],
        "all_rows": [public_row(r) for r in search.rows],
        "timing_s": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("DONE", report["decision"], selected["dev_weighted"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
