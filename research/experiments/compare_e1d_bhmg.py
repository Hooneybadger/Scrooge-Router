# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1D — frozen bhmg-v1 on exact public costs.

Writes ``build/compare-e1d-bhmg/`` only. Does not touch E1–E4 reports,
runtime artifacts, or predicted-cost Phase 2.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ossp_router.protocol import TIERS
from research.lab.e1_objectives import canonical_json_text, sha256_text, write_json_atomic
from research.lab.e1d_bhmg import (
    AUDIT_RELATIVE_PATH,
    BASELINE_NAME,
    CANDIDATE_NAME,
    FOLD_SEEDS,
    assemble,
)
from research.lab.public_pool import ROOT, load_public_pool


OUT = ROOT / "build" / "compare-e1d-bhmg"


def _print_tier(label: str, block: Mapping[str, Any]) -> None:
    counts = block["model_counts"]
    print(
        f"    {label:8s}  q={block['quality_score']:>12}  "
        f"tier={block['tier_score']:>12}  "
        f"ratio={block['budget_ratio']:>12}  "
        f"pass={block['budget_passed']!s:5s}  "
        f"L/A/K={counts['ax31-light']}/{counts['ax31']}/{counts['axk1-think']}",
        flush=True,
    )


def print_summary(report: Mapping[str, Any]) -> None:
    identity = report["identity"]
    gate = report["promotion_gate"]
    print(
        f"E1D public pool n={identity['n_episodes']} "
        f"train={identity['n_train']} dev={identity['n_dev']}",
        flush=True,
    )
    print(f"seeds={report['fold_seeds']} candidate={report['candidate']}", flush=True)
    print(
        "allocator=exact public costs; Phase-2 predicted costs are not used",
        flush=True,
    )
    print(f"solver={report['solver']['name']} {report['solver']['damping']}", flush=True)
    print(f"constants={report['constants']}", flush=True)
    print(f"label_checks={report['label_checks']}", flush=True)
    for seed in report["fold_seeds"]:
        block = report["seed_results"][str(seed)]
        print(
            f"\nseed {seed}  baseline={block['baseline_quality']:.12f}  "
            f"candidate={block['candidate_quality']:.12f}  "
            f"delta={block['delta']:+.6f}  matched={block['matched_e1_baseline']}",
            flush=True,
        )
        worst = block["worst_view"]
        if worst:
            print(
                f"  worst_view {worst['kind']}:{worst['name']} "
                f"n={worst['n']} delta={worst['delta']:+.6f} "
                f"fail={worst['worse_than_gate']}",
                flush=True,
            )
        print(
            f"  singular={block['singular']} "
            f"fold_converged={[row['converged'] for row in block['fold_fits']]}",
            flush=True,
        )
        for name in (BASELINE_NAME, CANDIDATE_NAME):
            pooled = block["results"][name]["pooled"]
            print(
                f"  {name} official={pooled['official_final_score']} "
                f"caps_ok={all(pooled['tiers'][tier]['within_hard_cap'] for tier in TIERS)} "
                f"fold_caps={block['results'][name]['fold_caps_ok']} "
                f"k1_ok={block['results'][name]['k1_fast_balanced_zero']}",
                flush=True,
            )
            for tier in TIERS:
                _print_tier(tier, pooled["tiers"][tier])
        abstention = block["diagnostics"]["abstention"]
        print(f"  abstention={abstention}", flush=True)
        print(f"  ranking={block['diagnostics']['ranking']}", flush=True)
        print(f"  h1_h2={block['diagnostics']['h1_h2_views']}", flush=True)
    print("\npromotion gate", flush=True)
    print(
        f"  mean_delta={gate['mean_delta']:+.6f}  worst_delta={gate['worst_delta']:+.6f}  "
        f"mean_abs={gate['mean_absolute']:.12f}  worst_abs={gate['worst_absolute']:.12f}",
        flush=True,
    )
    print(
        f"  matched_20260821={gate['matched_e1_baseline_20260821']} "
        f"valid={gate['experiment_valid']} "
        f"cap_failures={gate['cap_failures']} view_failures={gate['view_failures']} "
        f"k1_failures={gate['k1_failures']} singular={gate['singular_failures']}",
        flush=True,
    )
    print(f"  phase1_passed={gate['phase1_passed']} phase2={gate['phase2_executed']}", flush=True)
    print(f"decision={report['decision']} core_sha256={report['decision_core_sha256']}", flush=True)
    print(f"decision_reason={report['decision_reason']}", flush=True)
    print(f"sequential_testing={report['sequential_testing']}", flush=True)
    print(f"export_preview_selection_use={report['export_preview']['selection_use']}", flush=True)
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
    print(f"loading public Train+Dev pool; seeds={list(FOLD_SEEDS)}", flush=True)
    pool = load_public_pool()
    print(
        f"loaded n={pool.identity['n_episodes']} groups={pool.grouping['n_groups']}",
        flush=True,
    )
    print("fitting bhmg-v1 (exact public costs; no Phase 2)", flush=True)
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
