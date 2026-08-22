# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E4 — point buy + aggregate conformal rollback.

Writes ``build/compare-e4-aggregate-risk/`` only. Does not touch E1/E1B/E1C/E2
reports or runtime artifacts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ossp_router.protocol import TIERS
from research.lab.e1_objectives import canonical_json_text, sha256_text, write_json_atomic
from research.lab.e4_aggregate_risk import (
    AUDIT_RELATIVE_PATH,
    CANDIDATE_ORDER,
    FOLD_SEEDS,
    assemble,
)
from research.lab.public_pool import ROOT, load_public_pool


OUT = ROOT / "build" / "compare-e4-aggregate-risk"


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
        f"E4 public pool n={identity['n_episodes']} "
        f"train={identity['n_train']} dev={identity['n_dev']}",
        flush=True,
    )
    print(f"seeds={report['fold_seeds']} quality={report['quality_signal']}", flush=True)
    print(report["allocator"]["buy"], flush=True)
    print(report["feature"]["bound_rule"], flush=True)
    print(f"used_actual_light={report['feature']['used_actual_light_in_allocator']}", flush=True)
    print(f"sigma_in_price={report['feature']['sigma_in_price']}", flush=True)
    print(f"runtime_scope={report['runtime_scope']['provenance']}", flush=True)
    for seed in report["fold_seeds"]:
        block = report["seed_results"][str(seed)]
        print(
            f"\nseed {seed}  baseline_valid={block['baseline_valid']}  "
            f"pred_light={block['predicted_light_total']}",
            flush=True,
        )
        for name in CANDIDATE_ORDER:
            pooled = block["results"][name]["pooled"]
            rollback = block["results"][name]["rollback"]
            row = next(item for item in block["gate_rows"] if item["candidate"] == name)
            print(
                f"  {name}  quality={pooled['quality_weighted']:.12f}  "
                f"official={pooled['official_final_score']}  "
                f"delta={row['delta_vs_safe_baseline']}  "
                f"pass={row['pass']}  "
                f"caps={row['pooled_caps_ok']}/{row['fold_caps_ok']}",
                flush=True,
            )
            for tier in TIERS:
                _print_tier(tier, pooled["tiers"][tier])
                rb = rollback[tier]
                print(
                    f"      rollback groups={rb['n_groups_rolled']}  "
                    f"ax31 {rb['n_ax31_bought']}->{rb['n_ax31_final']}  "
                    f"k1 {rb['n_k1_bought']}->{rb['n_k1_final']}  "
                    f"bound={rb['bound_mean']}",
                    flush=True,
                )
            print(
                f"    stress_fail={row['stress_failures']}  "
                f"ratio_fail={row['ratio_view_failures']}  "
                f"view_fail={row['view_failures']}",
                flush=True,
            )
    print("\npromotion gate", flush=True)
    print(
        f"  baseline_valid_all={gate['baseline_valid_all_seeds']}  "
        f"passed={gate['passed']}  recommended={gate['recommended']}",
        flush=True,
    )
    for row in gate["candidates"]:
        print(
            f"  {row['candidate']}: all_seeds={row['pass_all_seeds']}  "
            f"worst_q={row['worst_quality']}  worst_delta={row['worst_delta']}",
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
    print(f"loading public Train+Dev pool; seeds={list(FOLD_SEEDS)}", flush=True)
    pool = load_public_pool()
    print(
        f"loaded n={pool.identity['n_episodes']} groups={pool.grouping['n_groups']}",
        flush=True,
    )
    print("fitting E4 aggregate-risk candidates", flush=True)
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
