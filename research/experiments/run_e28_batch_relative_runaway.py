# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E28 — batch-relative Premium runaway guard runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.e28_batch_relative_runaway import (
    AUDIT_RELATIVE,
    EXPECTED_PROTOCOL_SHA256,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


def print_summary(report) -> None:
    gate = report["gate"]
    pin = report["pin_dev_replay"]
    print(
        f"E28 seeds={report['fold_seeds']} pin={pin['matched']} "
        f"final={pin['final_score']:.12f}",
        flush=True,
    )
    for label, row in sorted(report["full_batch_identity_by_split"].items()):
        base = report["full_batch"]["baseline"][label]
        cand = report["full_batch"]["primary"][label]
        print(
            f"  {label:7} identical={row} k1 {base['k1']}->{cand['k1']} "
            f"actual {base['premium_ratio']:.6f}->{cand['premium_ratio']:.6f} "
            f"runaway {base['runaway_threshold']:.6f}->{cand['runaway_threshold']:.6f}",
            flush=True,
        )
    for label in sorted(report["stress"]["primary"]):
        base = report["stress"]["baseline"][label]
        cand = report["stress"]["primary"][label]
        print(
            f"  stress/{label:5} ruin {base['n_ruin']}->{cand['n_ruin']} "
            f"binding {base['n_ruin_binding']}->{cand['n_ruin_binding']} "
            f"freq {base['binding_ruin_frequency']:.5f}->"
            f"{cand['binding_ruin_frequency']:.5f} "
            f"worst {base['worst_realized']:.4f}->{cand['worst_realized']:.4f}",
            flush=True,
        )
    for share, row in sorted(report["falsifiability_probe"].items()):
        print(
            f"  probe share={share} identity={row['gate_identity_passed']} "
            f"ruin_ok={row['gate_ruin_passed']} "
            f"failed_a_gate={row['some_gate_failed']}",
            flush=True,
        )
    for name, ok in sorted(gate["rows"].items()):
        print(f"  gate {name:42} {ok}", flush=True)
    print(f"failures={gate['failures']} passed={gate['passed']}", flush=True)
    print(f"decision={report['decision']} core={report['decision_core_sha256']}", flush=True)
    print(f"decision_reason={report['decision_reason']}", flush=True)


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
        raise RuntimeError("e28 output already exists; refuse overwrite")
    if str(args.expected_protocol_sha256) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "expected protocol sha drifted from the sealed lab constant"
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
    print_summary(report)
    print(f"elapsed_s={time.perf_counter() - started:.3f}", flush=True)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
