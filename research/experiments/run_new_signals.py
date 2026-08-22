# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""New-signals runner.

Requires the sealed ``new-signals.v1`` protocol and its canonical SHA.
Writes only ``build/run-new-signals/`` and refuses to overwrite. This
tests pure-stdlib extractor blocks (AST/numeric/choice) as new signal
sources through the actual serving path; it opens a promotion window
only and integrates nothing by itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.new_signals import (
    AUDIT_RELATIVE,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


OUT = ROOT / "build" / "run-new-signals"


def print_summary(report) -> None:
    gate = report["gate"]
    print(f"NEW-SIGNALS seeds={report['fold_seeds']}", flush=True)
    for seed in report["fold_seeds"]:
        block = report["seed_results"][str(seed)]
        parts = [
            f"{arm}={delta:+.8f}"
            for arm, delta in sorted(block["pooled_delta"].items())
        ]
        print(f"seed {seed}  " + "  ".join(parts), flush=True)
    print("\nper-arm gate", flush=True)
    for arm, row in gate["arms"].items():
        worst_fast = row["actual_ratio_worst"]["fast"]
        worst_balanced = row["actual_ratio_worst"]["balanced"]
        print(
            f"  {arm:18s} mean={row['mean_delta']:+.8f}  "
            f"worst={row['worst_delta']:+.8f}  "
            f"tvball={row['tvball_worst_min']:+.8f}  "
            f"ratio(F/B)={worst_fast:.4f}/{worst_balanced:.4f}  "
            f"passed={row['passed']} {row['failures']}",
            flush=True,
        )
    print(f"\nwindow_arm={gate.get('window_arm')}", flush=True)
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
        raise RuntimeError("new-signals output already exists; refuse overwrite")
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
