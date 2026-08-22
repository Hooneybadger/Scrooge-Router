# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Final-public-refit runner.

Requires the sealed ``final-public-refit.v1`` protocol and its canonical
SHA. Writes only ``build/run-final-public-refit/`` and refuses to
overwrite. The runtime resource file is never touched here; a passing
report recommends the swap and an explicit human approval step follows.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.final_public_refit import (
    AUDIT_RELATIVE,
    CANDIDATE_RELATIVE,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


OUT = ROOT / "build" / "run-final-public-refit"


def print_summary(report) -> None:
    gates = report["gates"]
    candidate = report["candidate"]
    print(
        f"FINAL-REFIT protocol={report['protocol_sha256'][:16]}… "
        f"trees={candidate['n_trees']} nodes={candidate['n_nodes']}",
        flush=True,
    )
    for split in ("train", "dev"):
        old = report["current"][f"{split}_replay"]
        new = report["candidate_replays"][split]
        print(
            f"{split:8s} current final={old['official_final_score']:.12f} "
            f"k1={old['premium_n_k1']} ratio={old['premium_actual_ratio']:.6f} | "
            f"candidate k1={new['premium_n_k1']} "
            f"ratio={new['premium_actual_ratio']:.6f}",
            flush=True,
        )
    dev_final = gates["g5_observed_dev_final"]
    print(
        f"candidate dev final={dev_final:.12f} (pin 0.669517045455) "
        f"identity={gates['fast_balanced_identity']} "
        f"determinism={gates['g2_determinism']}",
        flush=True,
    )
    worst = gates.get("g4_worst_family_ratio")
    print(
        f"gates load={gates['g1_contract_load']} brake={gates['g3_predicted_brake']} "
        f"budget={gates['g4_actual_budget']} floor={gates['g5_dev_floor']} "
        f"worst_family_ratio={worst:.6f}",
        flush=True,
    )
    print(f"failures={report['failures']}", flush=True)
    print(f"decision={report['decision']} core={report['decision_core_sha256']}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / REPORT_RELATIVE)
    parser.add_argument("--audit-output", type=Path, default=ROOT / AUDIT_RELATIVE)
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=ROOT / CANDIDATE_RELATIVE,
    )
    args = parser.parse_args(argv)
    if not args.protocol.is_file():
        raise RuntimeError(f"protocol file missing: {args.protocol}")
    if args.output.exists() or args.audit_output.exists() or args.candidate_output.exists():
        raise RuntimeError("final-refit output already exists; refuse overwrite")
    started = time.perf_counter()
    try:
        report = run_from_protocol(
            args.protocol,
            args.expected_protocol_sha256,
            output=args.output,
            audit_output=args.audit_output,
            candidate_output=args.candidate_output,
        )
    except (ProtocolError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print_summary(report)
    print(f"elapsed_s={time.perf_counter() - started:.3f}", flush=True)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.audit_output}", flush=True)
    print(f"wrote {args.candidate_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
