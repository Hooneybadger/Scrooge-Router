# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the static cap lock — lock the selected under the documented hidden-set policy.

Final candidate set only: the parent and the few knob moves that survived
the policy search's gates. Ranking is by expected score under drift,

    E[score] = sum_t w_t * q_t * (1 - p_t),

because the contest zeroes a busted tier rather than discounting it. The
lock also runs the ID/order audit from ``docs/ENFORCEMENT.md``.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.lab.modeling import OFFICIAL_CAPS, TIER_WEIGHTS, load_train
from research.lab.cap_certification import sha256_file
from research.lab.hidden_set_gates import (
    DOC_TAGS,
    INFLATION,
    NEAR_FRAC,
    SplitContext,
    evaluate_candidate,
    gate_report,
    headroom_key,
)
from ossp_router.protocol import TIERS, InputBatch, load_input, load_outcomes
from ossp_router.feasibility_ladder import load_artifact_mapping, make_submission
from research.lab.validation import prompt_family, public_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "lock-static-caps"
LADDER_PATH = ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json"
STATIC_CAP_ARTIFACT = "static-cap-router.v1.json"


def variant(base: Mapping[str, Any], **kwargs: float) -> dict[str, Any]:
    art = copy.deepcopy(dict(base))
    if "balanced" in kwargs:
        art["predicted_caps"]["balanced"] = float(kwargs["balanced"])
    if "fast" in kwargs:
        art["predicted_caps"]["fast"] = float(kwargs["fast"])
    if "premium" in kwargs:
        art["predicted_caps"]["premium"] = float(kwargs["premium"])
    if "max_up" in kwargs:
        art["max_upgrade_fraction"] = float(kwargs["max_up"])
    if "kappa" in kwargs:
        art.setdefault("premium_overlay", {})["kappa_q999"] = float(kwargs["kappa"])
    art["k1_enabled"] = False
    return art


def tier_risk(splits: Mapping[str, Mapping[str, Any]], tier: str) -> float:
    worst = 0.0
    for row in splits.values():
        drift = row["tiers"][tier]["drift"]
        worst = max(worst, int(drift["n_ruin_inflated"]) / max(int(drift["n"]), 1))
    return worst


def expected_score(splits: Mapping[str, Mapping[str, Any]], shrink: float = 1.0) -> float:
    dev = splits["dev"]["tiers"]
    return sum(
        float(TIER_WEIGHTS[t])
        * float(dev[t]["quality"])
        * (1.0 - shrink * tier_risk(splits, t))
        for t in TIERS
    )


def audit_id_order(art: Mapping[str, Any], inputs: Any, policy: Any) -> dict[str, Any]:
    """docs/ENFORCEMENT.md: same prompt content must route the same way."""

    mapped = load_artifact_mapping(copy.deepcopy(dict(art)))
    rng = np.random.default_rng(20260821)
    order = rng.permutation(len(inputs.episodes))
    shuffled = InputBatch(
        schema_version=inputs.schema_version,
        challenge_id="audit-shuffle",
        split=inputs.split,
        episodes=tuple(inputs.episodes[int(i)] for i in order),
    )
    result: dict[str, Any] = {"tiers": {}, "passed": True}
    for tier in TIERS:
        base = {
            d.episode_id: d.model_id
            for d in make_submission(inputs, policy, mapped, tier).submission.decisions
        }
        shuf = {
            d.episode_id: d.model_id
            for d in make_submission(shuffled, policy, mapped, tier).submission.decisions
        }
        same = base == shuf
        result["tiers"][tier] = {
            "identical": bool(same),
            "n_diff": int(sum(1 for k in base if base[k] != shuf.get(k))),
        }
        result["passed"] = bool(result["passed"] and same)
    return result


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

    ladder = json.loads(LADDER_PATH.read_text(encoding="utf-8"))
    candidates: list[tuple[str, dict[str, Any]]] = [("the feasibility ladder", copy.deepcopy(ladder))]
    for bal in (1.15, 1.20, 1.25, 1.28, 1.30, 1.32, 1.35, 1.38, 1.40):
        candidates.append((f"bal={bal:.2f}", variant(ladder, balanced=bal)))
    for bal in (1.30, 1.35):
        candidates.append(
            (f"bal={bal:.2f}|maxup=0.82", variant(ladder, balanced=bal, max_up=0.82))
        )
        candidates.append(
            (f"bal={bal:.2f}|fast=1.04", variant(ladder, balanced=bal, fast=1.04))
        )

    print("\n=== the static cap lock final candidates (drift + red-team) ===", flush=True)
    rows: list[dict[str, Any]] = []
    parent_drift: dict[str, int] | None = None
    for name, art in candidates:
        splits = {
            label: evaluate_candidate(art, ctx, with_redteam=True)
            for label, ctx in contexts.items()
        }
        if name == "the feasibility ladder":
            parent_drift = {
                t: max(
                    int(splits[s]["tiers"][t]["drift"]["n_ruin_inflated"]) for s in splits
                )
                for t in TIERS
            }
        rows.append({"name": name, "art": art, "splits": splits})

    assert parent_drift is not None
    for row in rows:
        row["gates"] = gate_report(row["splits"], parent_drift=parent_drift)
        row["expected"] = {
            f"{s:g}": expected_score(row["splits"], s) for s in (1.0, 0.5, 0.25)
        }
        row["dev_weighted"] = float(row["splits"]["dev"]["weighted"])
        row["train_weighted"] = float(row["splits"]["train"]["weighted"])
        row["headroom"] = headroom_key(row["splits"])
        dev = row["splits"]["dev"]["tiers"]
        print(
            f"{row['name']:20} dev={row['dev_weighted']:.6f} "
            f"E1={row['expected']['1']:.6f} elig={row['gates']['eligible']} "
            f"drift={row['gates']['drift_ruin_inflated']} "
            f"redteamP={dev['premium']['red_team']['max_realized']:.3f}",
            flush=True,
        )

    eligible = [r for r in rows if r["gates"]["eligible"]]
    eligible.sort(key=lambda r: (r["expected"]["1"], r["dev_weighted"]), reverse=True)
    selected = eligible[0]
    parent = next(r for r in rows if r["name"] == "the feasibility ladder")
    print(
        f"\nSELECTED {selected['name']} E1={selected['expected']['1']:.6f} "
        f"dev={selected['dev_weighted']:.6f} "
        f"(parent E1={parent['expected']['1']:.6f} dev={parent['dev_weighted']:.6f})",
        flush=True,
    )

    print("running ID/order audit", flush=True)
    audit = {
        "dev": audit_id_order(selected["art"], dev_inputs, policy),
        "train": audit_id_order(selected["art"], bundle.inputs, policy),
    }
    print("audit passed", audit["dev"]["passed"] and audit["train"]["passed"], flush=True)

    art_path = OUT / STATIC_CAP_ARTIFACT
    art_path.write_text(
        json.dumps(selected["art"], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    art_path.chmod(0o644)

    safe = selected["name"].replace("=", "p").replace("|", "_").replace(".", "p").lower()
    report = {
        "experiment": "the static cap lock",
        "decision": f"record-static_caps-selected-{safe}",
        "objective": "max E[score] = sum_t w_t q_t (1 - p_t) subject to hidden-set gates",
        "gates": {
            "A_full_batch_inflated": f"realized * {INFLATION} <= cap (train and dev)",
            "A_near_budget": f"realized < {NEAR_FRAC} * cap (train and dev)",
            "B_drift_not_worse_than_parent": parent_drift,
            "C_premium_fixed_composition": "premium fixed-selection ruin == 0",
            "doc_tags": DOC_TAGS,
            "official_caps": {t: float(OFFICIAL_CAPS[t]) for t in TIERS},
        },
        "selected": {
            "name": selected["name"],
            "dev_weighted": selected["dev_weighted"],
            "train_weighted": selected["train_weighted"],
            "expected": selected["expected"],
            "headroom": selected["headroom"],
            "gates": selected["gates"],
            "splits": selected["splits"],
            "predicted_caps": selected["art"]["predicted_caps"],
            "max_upgrade_fraction": selected["art"].get("max_upgrade_fraction"),
            "premium_kappa_q999": selected["art"].get("premium_overlay", {}).get("kappa_q999"),
            "k1_enabled": bool(selected["art"].get("k1_enabled")),
            "artifact_sha256": sha256_file(art_path),
            "path": str(art_path),
            "artifact": str(art_path),
        },
        "id_order_audit": audit,
        "candidates": [
            {k: v for k, v in row.items() if k != "art"} for row in rows
        ],
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
