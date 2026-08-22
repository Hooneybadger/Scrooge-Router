# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Data-scale diagnostic runner.

Requires the sealed ``data-scale-diagnostic.v1`` protocol and its
canonical SHA. Writes only ``build/run-data-scale-diagnostic/`` and
refuses to overwrite. This is recommendation evidence for the
final-public-refit swap; it promotes no runtime candidate by itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.data_scale_diagnostic import (
    AUDIT_RELATIVE,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


OUT = ROOT / "build" / "run-data-scale-diagnostic"


def print_summary(report) -> None:
    gate = report["gate"]
    print(
        f"DATA-SCALE seeds={report['fold_seeds']} "
        f"treatment={report['candidate_treatment']}",
        flush=True,
    )
    for seed in report["fold_seeds"]:
        block = report["seed_results"][str(seed)]
        print(
            f"seed {seed}  baseline={block['pooled_quality']['fit-train-split-only']:.12f}  "
            f"treatment={block['pooled_quality']['fit-outer-full']:.12f}  "
            f"delta={block['delta']:+.8f}  tvball={block['tvball_worst']:+.8f}  "
            f"changed={block['n_changed_decisions']}",
            flush=True,
        )
    print("\nrecommendation gate", flush=True)
    print(
        f"  mean_delta={gate['mean_delta']:+.8f}  "
        f"worst_delta={gate['worst_delta']:+.8f}  "
        f"thresholds=({report['thresholds']['mean_delta_min']}, "
        f"{report['thresholds']['worst_seed_delta_min_inclusive']})",
        flush=True,
    )
    print(f"decision={report['decision']} core={report['decision_core_sha256']}", flush=True)
    audit = report["audit"]
    print(f"audit n_rows={audit['n_rows']} sha256={audit['sha256']}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / REPORT_RELATIVE)
    parser.add_argument("--audit-output", type=Path, default=ROOT / AUDIT_RELATIVE)
    args = parser.parse_args(argv)
    if not args.protocol.is_file():
        raise RuntimeError(f"protocol file missing: {args.protocol}")
    if args.output.exists() or args.audit_output.exists():
        raise RuntimeError("diagnostic output already exists; refuse overwrite")
    started = time.perf_counter()
    try:
        report = run_from_protocol(
            args.protocol,
            args.expected_protocol_sha256,
            output=args.output,
            audit_output=args.audit_output,
        )
    except (ProtocolError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print_summary(report)
    print(f"elapsed_s={time.perf_counter() - started:.3f}", flush=True)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
