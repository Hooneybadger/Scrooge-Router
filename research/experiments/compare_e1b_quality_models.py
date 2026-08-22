# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1B — hashed / logistic / residual quality heads on exact public costs.

Writes ``build/compare-e1b-quality-models/`` only. Does not touch E1/E2
reports or runtime artifacts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ossp_router.protocol import TIERS
from research.lab.e1_objectives import canonical_json_text, sha256_text, write_json_atomic
from research.lab.e1b_quality_models import (
    AUDIT_RELATIVE_PATH,
    CANDIDATE_ORDER,
    assemble,
)
from research.lab.public_pool import ROOT, load_public_pool


OUT = ROOT / "build" / "compare-e1b-quality-models"


def _print_tier(label: str, block: Mapping[str, Any]) -> None:
    counts = block["model_counts"]
    print(
        f"  {label:8s}  q={block['quality_score']:>12}  "
        f"tier={block['tier_score']:>12}  "
        f"ratio={block['budget_ratio']:>12}  "
        f"pass={block['budget_passed']!s:5s}  "
        f"L/A/K={counts['ax31-light']}/{counts['ax31']}/{counts['axk1-think']}",
        flush=True,
    )


def print_summary(report: Mapping[str, Any]) -> None:
    identity = report["identity"]
    gate = report["promotion_gate"]
    repro = report["baseline_reproduction"]
    print(
        f"E1B public pool n={identity['n_episodes']} "
        f"train={identity['n_train']} dev={identity['n_dev']}",
        flush=True,
    )
    print(
        f"baseline reproduction expected={repro['expected_quality_weighted']} "
        f"observed={repro['observed_quality_weighted']} matched={repro['matched']}",
        flush=True,
    )
    print(
        "allocator=exact public costs; Phase-2 predicted costs are not used",
        flush=True,
    )
    baseline_q = float(gate["baseline_quality_weighted"])
    for name in CANDIDATE_ORDER:
        pooled = report["results"][name]["pooled"]
        delta = float(pooled["quality_weighted"]) - baseline_q
        print(
            f"\n{name}  quality={pooled['quality_weighted']:.12f}  "
            f"official={pooled['official_final_score']}  "
            f"delta={delta:+.6f}  abs_ok={float(pooled['quality_weighted']) >= 0.690}",
            flush=True,
        )
        for tier in TIERS:
            _print_tier(tier, pooled["tiers"][tier])
        rank = report["results"][name]["ranking"]
        print(
            f"  rank qa mae={rank['qa_vs_delta_ax31_light']['mae']:.6f} "
            f"sign={rank['qa_vs_delta_ax31_light']['sign_accuracy_unequal']} "
            f"qk mae={rank['qk_vs_delta_k1_ax31']['mae']:.6f} "
            f"sign={rank['qk_vs_delta_k1_ax31']['sign_accuracy_unequal']}",
            flush=True,
        )
        clf = report["results"][name]["classifier"]
        if clf:
            print(
                f"  clf qa auc={clf['qa_p_delta_ax31_light']['roc_auc']} "
                f"brier={clf['qa_p_delta_ax31_light']['brier']:.6f} "
                f"qk auc={clf['qk_p_delta_k1_ax31']['roc_auc']} "
                f"brier={clf['qk_p_delta_k1_ax31']['brier']:.6f}",
                flush=True,
            )
        print(f"  split {report['split_deltas'][name]}", flush=True)
        print("  per-fold realloc:", flush=True)
        for row in report["results"][name]["per_fold"]:
            print(
                f"    fold {row['fold']}: quality={row['quality_weighted']:.12f}  "
                f"official={row['official_final_score']}",
                flush=True,
            )
    print("\npromotion gate", flush=True)
    print(
        f"  baseline={gate['baseline']} q={gate['baseline_quality_weighted']:.12f} "
        f"champion_abs={gate['champion_absolute']}",
        flush=True,
    )
    for row in gate["candidates"]:
        print(
            f"  {row['candidate']}: delta={row['delta_vs_baseline']:+.6f} "
            f"gain_ok={row['gain_ok']} abs_ok={row['absolute_ok']} "
            f"views_ok={row['views_ok']} caps={row['pooled_caps_ok']}/{row['fold_caps_ok']} "
            f"pass={row['pass']} view_fail={row['view_failures']}",
            flush=True,
        )
    print(
        f"  passed={gate['passed']} recommended={gate['recommended']}",
        flush=True,
    )
    print(
        f"decision={report['decision']} core_sha256={report['decision_core_sha256']}",
        flush=True,
    )
    print(f"decision_reason={report['decision_reason']}", flush=True)
    audit = report["audit"]
    print(
        f"audit path={audit['relative_path']} n_rows={audit['n_rows']} "
        f"sha256={audit['sha256']}",
        flush=True,
    )
    elapsed = report.get("runtime", {}).get("elapsed_s")
    if elapsed is not None:
        print(f"elapsed_s={elapsed:.3f}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "report.json")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / AUDIT_RELATIVE_PATH,
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    print("loading public Train+Dev pool", flush=True)
    pool = load_public_pool()
    print(
        f"loaded n={pool.identity['n_episodes']} groups={pool.grouping['n_groups']}",
        flush=True,
    )
    print("fitting E1B quality heads (exact public costs)", flush=True)
    report, audit_document = assemble(pool)
    elapsed = time.perf_counter() - started
    runtime = dict(report["runtime"])
    runtime["elapsed_s"] = float(elapsed)
    report = dict(report)
    report["runtime"] = runtime
    if sha256_text(canonical_json_text(audit_document)) != report["audit"]["sha256"]:
        raise RuntimeError("audit dump SHA drifted from the decision-core digest")
    write_json_atomic(args.audit_output, audit_document)
    write_json_atomic(args.output, report)
    print_summary(report)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
