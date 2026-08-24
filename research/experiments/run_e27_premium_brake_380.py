# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E27 — 3.80-only Premium brake confirmation runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.e27_premium_brake_380 import (
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
    primary = report["primary"]
    print(
        f"E27 seeds={report['fold_seeds']} pin={pin['matched']} "
        f"final={pin['final_score']:.12f}",
        flush=True,
    )
    print(
        f"fast_balanced={gate['fast_balanced_identical']} "
        f"conformal_identical={gate['conformal_identical_all_seeds']} "
        f"dev_delta={gate['dev_delta']:+.12f} "
        f"train_q={gate['train_premium_quality_delta']:+.12f}",
        flush=True,
    )
    print(
        f"k1 train {report['shipped']['train']['premium_k1']}->"
        f"{primary['train']['premium_k1']} "
        f"dev {report['shipped']['dev']['premium_k1']}->"
        f"{primary['dev']['premium_k1']} "
        f"pred {primary['train']['predicted_ratio']:.6f}/"
        f"{primary['dev']['predicted_ratio']:.6f} "
        f"actual {primary['train']['premium_ratio']:.6f}/"
        f"{primary['dev']['premium_ratio']:.6f}",
        flush=True,
    )
    print(
        f"bootstrap_q2_5_min={gate['bootstrap_q2_5_min']:+.8f} "
        f"other_share={report['deltas']['dev_other_official_share_evidence_only']} "
        f"passed={gate['passed']}",
        flush=True,
    )
    print(f"safety_failures={gate['safety_failures']}", flush=True)
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
        raise RuntimeError("e27 output already exists; refuse overwrite")
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
