# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E12 Fast-cap tighten runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.e12_fast_cap_tighten import (
    AUDIT_RELATIVE,
    CANDIDATE_ARMS,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


def print_summary(report) -> None:
    gate = report["gate"]
    pin = gate["pin"]
    print(
        f"E12 pin matched={pin['matched']} fidelity={pin['replica_fidelity']} "
        f"dev={pin['final_score']:.12f}",
        flush=True,
    )
    for arm in CANDIDATE_ARMS:
        block = gate["arms"][arm]
        print(
            f"arm {arm} train_delta={block['train_delta']:+.8f} "
            f"dev_delta={block['dev_delta']:+.8f} "
            f"passed={block['passed']} failures={block['failures']}",
            flush=True,
        )
        if block["fast_view_failures"]:
            for row in block["fast_view_failures"]:
                print(
                    f"  fast fail {row['view']} "
                    f"{row['actual_ratio']:.6f}/{row['inflated_ratio']:.6f}",
                    flush=True,
                )
    print(f"decision={report['decision']} window={gate['window_arm']}", flush=True)
    print(f"decision_reason={report['decision_reason']}", flush=True)
    print(f"core={report['decision_core_sha256']}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / REPORT_RELATIVE)
    parser.add_argument("--audit-output", type=Path, default=ROOT / AUDIT_RELATIVE)
    args = parser.parse_args(argv)
    if args.output.exists() or args.audit_output.exists():
        raise RuntimeError("e12 output already exists; refuse overwrite")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
