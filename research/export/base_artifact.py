# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the the Premium overlay Premium overlay on the audited the calibrated base fallback artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def export(
    *,
    base_artifact_path: Path,
    overlay_report_path: Path,
    overlay_artifact_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    base = json.loads(base_artifact_path.read_text(encoding="utf-8"))
    report = json.loads(overlay_report_path.read_text(encoding="utf-8"))
    source = json.loads(overlay_artifact_path.read_text(encoding="utf-8"))
    if (
        base.get("artifact_type") != "scrooge-base-final-router-v1"
        or base.get("k1_enabled") is not False
    ):
        raise ValueError("the calibrated base requires the audited the calibrated base fallback artifact")
    if (
        not report.get("promotion_gate", {}).get("passed")
        or report.get("decision") != "promote-premium-for-deployment-validation"
    ):
        raise ValueError("the Premium overlay Premium overlay did not pass promotion")
    premium = source.get("premium_e600", {})
    if premium.get("group_method") != "exact-content-sha256-v1":
        raise ValueError("the Premium overlay runtime grouping was not revalidated")
    comparison = report["premium_candidate"]["runtime_group_comparison"]
    if (
        comparison.get("train_selection_equal") is not True
        or comparison.get("dev_selection_equal") is not True
        or not comparison.get("loso_selection_digest")
    ):
        raise ValueError("the Premium overlay runtime grouping does not match validation")

    residual = premium["cost"]["conditional_residual_upper"]
    artifact = dict(base)
    artifact.update(
        {
            "schema_version": 2,
            "artifact_type": "scrooge-cost-calibrated-router-v1",
            "selected_policy": "overlay-premium-overlay-on-base-fallback",
            "premium_overlay": {
                "tier": "premium",
                "group_method": premium["group_method"],
                "kappa_q999": premium["tier_kappa_q999"],
                "safe_cap": source["safe_tier_caps"]["premium"],
                "minimum_model_cost_step_ratio": base["cost"][
                    "optimizer_minimum_model_cost_step_ratio"
                ],
                "quality": {
                    "positive_uplift_shrink_only": True,
                    "ood": premium["quality"]["ood"],
                },
                "cost": {
                    "token_upper_bounds": premium["cost"]["token_upper_bounds"],
                    "residual_upper": {
                        "ax31": residual["ax31"]["bounds"][0],
                        "axk1-think": residual["axk1-think"]["bounds"][0],
                    },
                },
            },
            "provenance": {
                **base["provenance"],
                "base_artifact_sha256": _sha256(base_artifact_path),
                "overlay_report_sha256": _sha256(overlay_report_path),
                "overlay_artifact_sha256": _sha256(overlay_artifact_path),
            },
        }
    )
    _write_json_atomic(output_path, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=ROOT / "build/base-artifact/base-router.v1.json",
    )
    parser.add_argument(
        "--overlay-report",
        type=Path,
        default=ROOT / "build/premium-overlay/report.json",
    )
    parser.add_argument(
        "--overlay-artifact",
        type=Path,
        default=ROOT / "build/premium-overlay/artifact.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = export(
            base_artifact_path=args.base_artifact,
            overlay_report_path=args.overlay_report,
            overlay_artifact_path=args.overlay_artifact,
            output_path=args.output,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"OK: exported {artifact['selected_policy']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
