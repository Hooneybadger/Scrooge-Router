# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Champion robustness audit runner.

Requires the sealed champion-robustness-audit.v1 protocol. Writes only
``build/run-champion-robustness-audit/`` and refuses to overwrite.
This is a diagnostic; it never exports runtime artifacts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.champion_robustness_audit import (
    AUDIT_RELATIVE,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


def print_summary(report) -> None:
    verdict = report["verdict"]
    pin = verdict["axes"]["pin"]
    safety = verdict["axes"]["safety"]
    print(
        f"audit overall={verdict['overall']} decision={report['decision']}",
        flush=True,
    )
    print(
        f"pin matched={pin['matched']} dev_final={pin['dev_final']:.12f}",
        flush=True,
    )
    print(
        f"safety={safety['status']} "
        f"official_failures={len(safety['official_failures'])} "
        f"view_failures={len(safety['view_failures'])}",
        flush=True,
    )
    for name, axis in verdict["axes"].items():
        print(f"  axis {name}={axis['status']}", flush=True)
    for label in ("train", "dev"):
        block = report["splits"][label]
        official = block["official"]
        print(
            f"{label} final={block['final_score']:.12f} "
            f"fast={official['fast']['budget_ratio']:.6f} "
            f"balanced={official['balanced']['budget_ratio']:.6f} "
            f"premium={official['premium']['budget_ratio']:.6f} "
            f"residual={block['residual_fraction']:.4f}",
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
        raise RuntimeError("champion robustness audit output already exists; refuse overwrite")
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
