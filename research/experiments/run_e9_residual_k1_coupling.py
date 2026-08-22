# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E9 residual K1-coupling runner.

Requires the sealed e9-residual-k1-coupling.v1 protocol. Writes only
``build/run-e9-residual-k1-coupling/`` and refuses to overwrite. A pass
opens a promotion window; runtime export stays out of scope of this
runner.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.e9_residual_k1_coupling import (
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
        f"E9 pin matched={pin['matched']} fidelity={pin['replica_fidelity']} "
        f"dev={pin['final_score']:.12f}",
        flush=True,
    )
    for arm in CANDIDATE_ARMS:
        block = gate["arms"][arm]
        knobs = block["knobs"]
        print(
            f"arm {arm} parent={knobs['guard_parent']} "
            f"denylist={knobs['denylist_other']} "
            f"k1_guard={knobs['guard_brake_k1']} "
            f"train_delta={block['train_delta']:+.8f} "
            f"dev_delta={block['dev_delta']:+.8f} "
            f"residual_ok={block['residual_ok']} "
            f"passed={block['passed']} failures={block['failures']}",
            flush=True,
        )
        for row in block["residual_rows"]:
            base = row["baseline"]
            cand = row["candidate"]
            print(
                f"  residual {row['split']} "
                f"baseline={base['actual_ratio']:.6f}/{base['inflated_ratio']:.6f} "
                f"candidate={cand['actual_ratio']:.6f}/{cand['inflated_ratio']:.6f} "
                f"cand_k1={cand['counts']['axk1-think']}",
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
            "e9 residual-k1-coupling output already exists; refuse overwrite"
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
