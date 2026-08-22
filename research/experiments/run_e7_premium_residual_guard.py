# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E7 Premium residual-guard runner.

Requires the sealed e7-premium-residual-guard.v1 protocol. Writes only
``build/run-e7-premium-residual-guard/`` and refuses to overwrite. A pass
opens a promotion window; runtime export stays out of scope of this
runner.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.e7_premium_residual_guard import (
    AUDIT_RELATIVE,
    CANDIDATE_ARM,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


def print_summary(report) -> None:
    gate = report["gate"]
    pin = gate["pin"]
    print(
        f"E7 pin matched={pin['matched']} fidelity={pin['replica_fidelity']} "
        f"dev={pin['final_score']:.12f}",
        flush=True,
    )
    print(
        f"arm {CANDIDATE_ARM} train_delta={gate['train_delta']:+.8f} "
        f"dev_delta={gate['dev_delta']:+.8f} "
        f"fast_bal_identical={gate['fast_balanced_identical']} "
        f"residual_ok={gate['residual_ok']} "
        f"residual_improved={gate['residual_improved']} "
        f"passed={gate['passed']} failures={gate['failures']}",
        flush=True,
    )
    for row in gate["residual_rows"]:
        base = row["baseline"]
        cand = row["candidate"]
        print(
            f"residual {row['split']} "
            f"baseline={base['actual_ratio']:.6f}/{base['inflated_ratio']:.6f} "
            f"candidate={cand['actual_ratio']:.6f}/{cand['inflated_ratio']:.6f}",
            flush=True,
        )
    print(
        f"decision={report['decision']} window={gate['window_arm']}",
        flush=True,
    )
    print(f"decision_reason={report['decision_reason']}", flush=True)
    print(f"core={report['decision_core_sha256']}", flush=True)
    audit = report["audit"]
    print(
        f"audit path={audit['relative_path']} n_rows={audit['n_rows']} "
        f"sha256={audit['sha256']}",
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
        raise RuntimeError(
            "e7 premium-residual-guard output already exists; refuse overwrite"
        )
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
    elapsed = time.perf_counter() - started
    print_summary(report)
    print(f"elapsed_s={elapsed:.3f}", flush=True)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
