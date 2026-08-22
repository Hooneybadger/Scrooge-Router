# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E5 — brake-conditioned Premium ranking runner.

Requires the sealed ``e5-brake-conditioned.v1`` protocol and its canonical
SHA. Writes only ``build/run-e5-brake-conditioned/`` and refuses to
overwrite. Runtime export is out of scope; a pass opens the promotion
window for a separate sealed integration protocol only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.e5_brake_conditioned import (
    AUDIT_RELATIVE,
    PRIMARY_NAME,
    REPORT_RELATIVE,
    ProtocolError,
    run_from_protocol,
)
from research.lab.public_pool import ROOT


OUT = ROOT / "build" / "run-e5-brake-conditioned"


def print_summary(report) -> None:
    gate = report["gate"]
    pin = report["pin_dev_replay"]
    equivalence = report["equivalence"]
    print(
        f"E5 seeds={report['fold_seeds']} primary={report['candidate_primary']}",
        flush=True,
    )
    print(
        f"pin_dev_replay matched={pin['matched']} final={pin['final_score']:.12f} "
        f"premium_n_k1={pin['premium_n_k1']} "
        f"premium_ratio={pin['premium_budget_ratio']:.6f}",
        flush=True,
    )
    print(
        f"equivalence matched={equivalence['matched']} "
        f"train_n_k1={equivalence['runtime_n_k1']}/{equivalence['composed_n_k1']}",
        flush=True,
    )
    for seed in report["fold_seeds"]:
        block = report["seed_results"][str(seed)]
        delta = block["delta_vs_baseline"]
        print(
            f"seed {seed}  baseline={block['quality_premium']['uplift-xtrees-refit']:.12f}  "
            f"primary_delta={delta[PRIMARY_NAME]:+.8f}  "
            f"secondary_delta={delta['density-ridge-standardized-wls']:+.8f}  "
            f"tvball={block['tvball_worst'][PRIMARY_NAME]:+.8f}  "
            f"k1={block['k1_counts']['uplift-xtrees-refit']}/"
            f"{block['k1_counts'][PRIMARY_NAME]}/"
            f"{block['k1_counts']['density-ridge-standardized-wls']}",
            flush=True,
        )
    print("\npromotion gate", flush=True)
    print(
        f"  mean_delta={gate['mean_delta']:+.8f}  "
        f"worst_delta={gate['worst_delta']:+.8f}  "
        f"tvball_worst_min={gate['tvball_worst_min']:+.8f}",
        flush=True,
    )
    print(
        f"  safety_failures={gate['safety_failures']} "
        f"determinism={gate['determinism_passed']} "
        f"passed={gate['passed']}",
        flush=True,
    )
    print(f"decision={report['decision']} core={report['decision_core_sha256']}", flush=True)
    print(f"decision_reason={report['decision_reason']}", flush=True)
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
        raise RuntimeError("e5 output already exists; refuse overwrite")
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
