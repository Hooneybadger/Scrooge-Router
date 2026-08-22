# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1 — compare quality objectives on grouped Train+Dev OOF.

Locks the current 14-d structural quality features and compares continuous
uplift Ridge against adjacent-step delta, a sign surrogate, and a hybrid.
Writes ``build/compare-e1-quality-objectives/report.json`` and the episode
audit dump. Does not touch runtime artifacts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ossp_router.protocol import TIERS
from research.lab.e1_objectives import (
    AUDIT_RELATIVE_PATH,
    CANDIDATE_ORDER,
    GATE_VIEW_KINDS,
    assemble,
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.public_pool import ROOT, load_public_pool


OUT = ROOT / "build" / "compare-e1-quality-objectives"


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
    grouping = report["grouping"]
    gate = report["promotion_gate"]
    kinds = report.get("stress_view_kinds", list(GATE_VIEW_KINDS))
    print(
        f"E1 public pool n={identity['n_episodes']} "
        f"train={identity['n_train']} dev={identity['n_dev']}",
        flush=True,
    )
    print(
        f"groups={grouping['n_groups']} exact={grouping['n_exact_groups']} "
        f"template={grouping['n_template_groups']} "
        f"near_unions={grouping['n_near_duplicate_unions']} "
        f"jaccard_cmp={grouping['n_jaccard_comparisons']}",
        flush=True,
    )
    print("fold balance:", flush=True)
    for row in report["fold_table"]:
        print(
            f"  fold {row['fold']}: n={row['n_episodes']} groups={row['n_groups']}",
            flush=True,
        )
    print(
        f"allocator={report['allocator']['name']} "
        f"caps={report['allocator']['caps']}",
        flush=True,
    )
    print(
        f"gated view kinds={kinds} "
        f"(fold=pooled allocation slice; per-fold realloc is observational)",
        flush=True,
    )
    cost = report["cost_diagnostic"]
    print(
        f"cost diagnostic: ax31<light={cost['ax31_lt_light_rows']} "
        f"k1<ax31={cost['k1_lt_ax31_rows']} clamped={cost['clamped']}",
        flush=True,
    )
    baseline_q = float(gate["baseline_quality_weighted"])
    for name in CANDIDATE_ORDER:
        pooled = report["results"][name]["pooled"]
        delta = float(pooled["quality_weighted"]) - baseline_q
        print(
            f"\n{name}  quality={pooled['quality_weighted']:.12f}  "
            f"official={pooled['official_final_score']}  "
            f"delta={delta:+.6f}",
            flush=True,
        )
        for tier in TIERS:
            _print_tier(tier, pooled["tiers"][tier])
        print("  per-fold realloc (observational, not gated):", flush=True)
        for row in report["results"][name]["per_fold"]:
            print(
                f"    fold {row['fold']}: quality={row['quality_weighted']:.12f}  "
                f"official={row['official_final_score']}",
                flush=True,
            )
        print("  gated views:", flush=True)
        for row in report["stress_views"][name]:
            if not row["gated"] and row["kind"] not in {"split", "fold"}:
                continue
            mark = " FAIL" if row["worse_than_gate"] else ""
            print(
                f"    {row['kind']}:{row['name']} n={row['n']} "
                f"d={row['delta']} gated={row['gated']}{mark}",
                flush=True,
            )
    print("\npromotion gate", flush=True)
    print(
        f"  baseline={gate['baseline']} quality={gate['baseline_quality_weighted']:.12f}",
        flush=True,
    )
    print(
        f"  view kinds={gate['thresholds']['gated_view_kinds']} "
        f"fold_view={gate['thresholds']['fold_view']}",
        flush=True,
    )
    for row in gate["candidates"]:
        print(
            f"  {row['candidate']}: delta={row['delta_vs_baseline']:+.6f} "
            f"quality_ok={row['quality_ok']} views_ok={row['views_ok']} "
            f"caps_ok={row['caps_ok']} pass={row['pass']} "
            f"view_fail={row['view_failures']}",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "report.json",
        help="JSON report path under build/ (ignored by git)",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / AUDIT_RELATIVE_PATH,
        help="Episode audit dump path under build/ (ignored by git)",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    print("loading public Train+Dev pool", flush=True)
    pool = load_public_pool()
    print(
        f"loaded n={pool.identity['n_episodes']} groups={pool.grouping['n_groups']}",
        flush=True,
    )
    print("fitting OOF quality objectives", flush=True)
    report, audit_document = assemble(pool)
    elapsed = time.perf_counter() - started
    runtime = dict(report["runtime"])
    runtime["elapsed_s"] = float(elapsed)
    report = dict(report)
    report["runtime"] = runtime
    audit_text = canonical_json_text(audit_document)
    if sha256_text(audit_text) != report["audit"]["sha256"]:
        raise RuntimeError("audit dump SHA drifted from the decision-core digest")
    write_json_atomic(args.audit_output, audit_document)
    write_json_atomic(args.output, report)
    print_summary(report)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
