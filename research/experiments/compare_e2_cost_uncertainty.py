# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E2/E3 — OOF point-cost, item sigma, and two-price settlement.

Locks the E1 baseline_continuous_uplift quality signal and compares cost
uncertainty heads on the Phase-1 grouped folds. Writes
``build/compare-e2-cost-uncertainty/`` only. Does not touch runtime
artifacts or the E1 report.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ossp_router.protocol import TIERS
from research.lab.e1_objectives import canonical_json_text, sha256_text, write_json_atomic
from research.lab.e2_cost_uncertainty import (
    AUDIT_RELATIVE_PATH,
    CANDIDATE_ORDER,
    QUALITY_SIGNAL,
    assemble,
)
from research.lab.public_pool import ROOT, load_public_pool


OUT = ROOT / "build" / "compare-e2-cost-uncertainty"


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
    print(
        f"E2 public pool n={identity['n_episodes']} "
        f"train={identity['n_train']} dev={identity['n_dev']} "
        f"quality={QUALITY_SIGNAL}",
        flush=True,
    )
    print(
        f"inner={report['inner_protocol']['residual_source']}",
        flush=True,
    )
    print(
        "allocator denom is OOF predicted light; actual light is scorer/stress only",
        flush=True,
    )
    acc = report["cost_accuracy"]
    print(
        f"light denom pred={acc['light_denominator']['predicted_total']} "
        f"actual={acc['light_denominator']['actual_total']} "
        f"bias={acc['light_denominator']['bias']}",
        flush=True,
    )
    for model_id, block in acc["models"].items():
        print(
            f"  {model_id} mae={block['point']['mae']:.6f} "
            f"q90={block['q90_coverage']['empirical']:.4f} "
            f"q99={block['q99_coverage']['empirical']:.4f}",
            flush=True,
        )
    overlap = report["family_guard_overlap"]
    print(
        f"sigma vs family-ratio corr={overlap['correlation_sigma_vs_train_family_ratio']}",
        flush=True,
    )
    baseline_q = float(gate["baseline_quality_weighted"])
    for name in CANDIDATE_ORDER:
        pooled = report["results"][name]["pooled"]
        delta = float(pooled["quality_weighted"]) - baseline_q
        print(
            f"\n{name}  quality={pooled['quality_weighted']:.12f}  "
            f"official={pooled['official_final_score']}  "
            f"delta={delta:+.6f}  pred_light={report['results'][name]['predicted_light_total']}",
            flush=True,
        )
        for tier in TIERS:
            _print_tier(tier, pooled["tiers"][tier])
        print("  per-fold realloc (observational):", flush=True)
        for row in report["results"][name]["per_fold"]:
            print(
                f"    fold {row['fold']}: quality={row['quality_weighted']:.12f}  "
                f"official={row['official_final_score']}",
                flush=True,
            )
        print("  bootstrap / mixture:", flush=True)
        for tier in TIERS:
            boot = report["stress"][name]["bootstrap"][tier]
            mix = report["stress"][name]["family_mixture"][tier]
            print(
                f"    {tier}: boot max={boot['max']} q99.9={boot['q99_9']} "
                f"hard_over={boot['hard_cap_overrun']} "
                f"mix max={mix['max']} q99.9={mix['q99_9']}",
                flush=True,
            )
    print("\npromotion gate", flush=True)
    print(
        f"  baseline={gate['baseline']} quality={gate['baseline_quality_weighted']:.12f} "
        f"reference_budget_valid={gate['reference_budget_valid']}",
        flush=True,
    )
    print(f"  quality_ok_is_not_sufficient={gate['quality_ok_is_not_sufficient']}", flush=True)
    print(f"  thresholds={gate['thresholds']}", flush=True)
    for row in gate["candidates"]:
        print(
            f"  {row['candidate']}: delta={row['delta_vs_point_baseline']:+.6f} "
            f"quality_ok={row['quality_ok']} views_ok={row['views_ok']} "
            f"caps={row['pooled_caps_ok']}/{row['fold_caps_ok']} "
            f"stress_ok={row['stress_ok']} coverage_ok={row['coverage_ok']} "
            f"pass={row['pass']} view_fail={row['view_failures']} "
            f"ratio_fail={row['ratio_view_failures']} "
            f"stress_fail={row['stress_failures']}",
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
    print("fitting OOF cost uncertainty + two-price settlement", flush=True)
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
