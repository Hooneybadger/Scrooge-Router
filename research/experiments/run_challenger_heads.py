# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Challenger-heads runner.

Requires the sealed ``challenger-heads.v1`` protocol and its canonical
SHA. Writes only ``build/run-challenger-heads/`` and refuses to
overwrite. This is the unified generalization diagnostic for the three
competitor head classes never tested here; it opens a promotion window
only and integrates nothing by itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.challenger_heads import (
    AUDIT_RELATIVE,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


OUT = ROOT / "build" / "run-challenger-heads"


def print_summary(report) -> None:
    gate = report["gate"]
    print(
        f"CHALLENGER-HEADS seeds={report['fold_seeds']}", flush=True,
    )
    for seed in report["fold_seeds"]:
        block = report["seed_results"][str(seed)]
        parts = []
        for arm, delta in block["pooled_delta"].items():
            parts.append(f"{arm}={delta:+.8f}")
        print(f"seed {seed}  " + "  ".join(parts), flush=True)
    print("\nper-arm gate", flush=True)
    for arm, row in gate["arms"].items():
        print(
            f"  {arm:24s} mean={row['mean_delta']:+.8f}  "
            f"worst={row['worst_delta']:+.8f}  "
            f"tvball={row['tvball_worst_min']:+.8f}  "
            f"safety_fail={len(row['safety_failures'])}  "
            f"passed={row['passed']} {row['failures']}",
            flush=True,
        )
    print(f"\nwindow_arm={gate.get('window_arm')}", flush=True)
    print(f"decision={report['decision']} core={report['decision_core_sha256']}", flush=True)
    print(f"decision_reason={report['decision_reason']}", flush=True)
    audit = report["audit"]
    print(
        f"audit n_rows={audit['n_rows']} sha256={audit['sha256']}",
        flush=True,
    )


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
        raise RuntimeError("challenger-heads output already exists; refuse overwrite")
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
