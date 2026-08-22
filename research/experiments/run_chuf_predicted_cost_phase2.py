# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""CHUF predicted-cost Phase 2 runner.

Requires a sealed protocol and its canonical SHA. Writes only
``build/phase2-chuf-predicted-cost/``. Refuses overwrite and will not
touch E1F or confirmation reports. Runtime export is out of scope.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.chuf_predicted_cost_phase2 import (
    AUDIT_RELATIVE,
    OUT_RELATIVE,
    load_protocol,
    refuse_foreign_output_path,
    run_phase2,
    verify_protocol,
)
from research.lab.e1_objectives import write_json_atomic
from research.lab.public_pool import ROOT


OUT = ROOT / OUT_RELATIVE


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUT / "report.json")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / AUDIT_RELATIVE,
    )
    args = parser.parse_args(argv)
    if not args.protocol.is_file():
        raise RuntimeError(f"protocol file missing: {args.protocol}")
    refuse_foreign_output_path(args.output)
    refuse_foreign_output_path(args.audit_output)
    if args.output.exists() or args.audit_output.exists():
        raise RuntimeError("phase2 output already exists; refuse overwrite")
    protocol = load_protocol(args.protocol)
    verify_protocol(protocol, args.expected_protocol_sha256)
    started = time.perf_counter()
    report = run_phase2(
        protocol, output=args.output, audit_output=args.audit_output
    )
    elapsed = time.perf_counter() - started
    runtime = dict(report.get("runtime", {}))
    runtime["elapsed_s"] = float(elapsed)
    report["runtime"] = runtime
    write_json_atomic(args.output, report)
    print(
        f"decision={report['decision']} "
        f"protocol={report['protocol_sha256']} "
        f"core={report['decision_core_sha256']}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
