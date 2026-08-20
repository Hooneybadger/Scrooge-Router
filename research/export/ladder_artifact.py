# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a Train-only the feasibility ladder runtime artifact. Never opens Dev. Never writes the calibrated base files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
BASE_ARTIFACT_PATH = ROOT / "src" / "ossp_router" / "resources" / "cost-calibrated-router.v1.json"
LADDER_REPORT_PATH = ROOT / "build" / "feasibility-ladder" / "report.json"
DEFAULT_OUTPUT = ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json"

BASE_ARTIFACT_SHA256 = (
    "2287c12c880a42b88909408be32ef43ea0962f1925bb6422bccdc89feb0ab7f5"
)
LADDER_REPORT_SHA256 = (
    "de6af64cadf5a16608a0b8c86a572a6c427176dd6e91b12f0f3a776c91fc4717"
)
TRAIN_INPUTS_SHA256 = (
    "029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4"
)
TRAIN_OUTCOMES_SHA256 = (
    "97a5a787086b3e1d9fa9c7945518543540e527ea248df4a4760de581b612a4ba"
)

ARTIFACT_TYPE = "scrooge-feasibility-ladder-v1"
SCHEMA_VERSION = 1
CANDIDATE_NAME = "ladder-recal-recal-fast-drift-cap-base-quality-v1"
PREDICTED_CAPS = {"fast": 1.03, "balanced": 1.50, "premium": 3.25}
RUNAWAY_FRACTION = 0.05
MAX_UPGRADE_FRACTION = 0.75
N_BINS = 10
FACTOR_CLIP = [0.5, 6.0]
FORBIDDEN_OUTPUT_NAMES = {
    "cost-calibrated-router.v1.json",
    "final-router.v1.json",
    "recalibrated-router.v1.json",
}
DEV_PATH_NEEDLES = (
    "data/materialized/dev/inputs.json",
    "data/dev/outcomes.json",
)
# Pinned to the the feasibility ladder Dev replay Train-only full-fit transform. The exporter
# never opens Dev paths; it only checks that Stage 1 full_fit_bins match.
PINNED_DEV_REPLAY_EDGES = [
    0.0007097631570864036,
    0.0009625135607888608,
    0.0011950115357821517,
    0.0015632830285995957,
    0.0021962900901607687,
    0.003147182667285221,
    0.004689852462570289,
    0.007135930207493405,
    0.011367695469560648,
]
PINNED_DEV_REPLAY_FACTORS = [
    2.447787889555669,
    1.7135489365004517,
    1.5130756804934848,
    1.4447803368708958,
    1.4447803368708958,
    1.0340609140515808,
    1.0340609140515808,
    1.0340609140515808,
    0.8595288085681846,
    0.8595288085681846,
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_dev_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_dev_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_dev_path(item) for item in value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return any(needle in normalized for needle in DEV_PATH_NEEDLES)
    return False


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(f"refusing to write protected artifact path: {path}")
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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _extract_full_fit_bins(report: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostic = _require_mapping(report.get("diagnostic"), "report.diagnostic")
    bins = _require_mapping(diagnostic.get("full_fit_bins"), "diagnostic.full_fit_bins")
    fast = _require_mapping(bins.get("fast"), "full_fit_bins.fast")
    balanced = _require_mapping(bins.get("balanced"), "full_fit_bins.balanced")
    for key in ("edges", "factors", "raw_factors", "pav_factors", "weights", "n"):
        if fast.get(key) != balanced.get(key):
            raise ValueError(f"full-fit {key} differs between Fast and Balanced")
    edges = list(fast["edges"])
    factors = list(fast["factors"])
    if len(edges) != N_BINS - 1 or len(factors) != N_BINS:
        raise ValueError("full-fit recalibration bin shape is not 10 equal-count bins")
    clip = list(fast.get("clip", FACTOR_CLIP))
    if clip != FACTOR_CLIP:
        raise ValueError("full-fit clip is not the locked [0.5, 6.0] range")
    if edges != PINNED_DEV_REPLAY_EDGES or factors != PINNED_DEV_REPLAY_FACTORS:
        raise ValueError(
            "Stage 1 full-fit bins do not match the frozen the feasibility ladder Dev replay "
            "Train-only transform"
        )
    return {
        "bin_max": [float(item) for item in fast["bin_max"]],
        "bin_min": [float(item) for item in fast["bin_min"]],
        "clip": FACTOR_CLIP,
        "dev_data_used": False,
        "edges": [float(item) for item in edges],
        "factors": [float(item) for item in factors],
        "fit": "full-train-fixed",
        "method": "rank-decile-incremental-cost",
        "n": int(fast["n"]),
        "n_bins": N_BINS,
        "pav": "non-increasing",
        "pav_factors": [float(item) for item in fast["pav_factors"]],
        "production": (
            "Copied verbatim from build/feasibility-ladder/report.json "
            "diagnostic.full_fit_bins.fast (identical to .balanced). "
            "Those values are the single full-fit Train transform that the "
            "frozen the feasibility ladder Dev replay also computed: research.lab.cap_certification.full_recalibrated_costs "
            "on Train only (assign_group_folds seed 2026082105); predicted "
            "incremental cost = the calibrated base OOF cost-head ax31-light incrementals "
            "from family_quality_study.shared_costs; actual incremental cost = Train outcomes "
            "ax31-light incrementals; 10 equal-count bins via numpy.array_split "
            "on a stable argsort; per-bin factor = sum(actual_inc)/sum(pred_inc); "
            "pool-adjacent-violators non-increasing (the rank recalibration study pav_nonincreasing, "
            "weights = pred_inc sums); clip [0.5, 6.0]. Inference uses "
            "numpy.digitize(pred_inc, edges, right=True) clipped to {0..9}. "
            "Runtime applies these frozen edges and factors and does not refit. "
            "No Dev input, outcome, hash, stat, or score was read to produce "
            "this block."
        ),
        "raw_factors": [float(item) for item in fast["raw_factors"]],
        "report_sha256": LADDER_REPORT_SHA256,
        "source": "build/feasibility-ladder/report.json:diagnostic.full_fit_bins",
        "train_inputs_sha256": TRAIN_INPUTS_SHA256,
        "train_outcomes_sha256": TRAIN_OUTCOMES_SHA256,
        "weights": [float(item) for item in fast["weights"]],
    }


def export(
    *,
    base_artifact_path: Path,
    ladder_report_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    if output_path.resolve() == base_artifact_path.resolve():
        raise ValueError("refusing to overwrite the the calibrated base artifact")
    if output_path.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(f"refusing to write protected artifact path: {output_path}")
    base_digest = _sha256(base_artifact_path)
    if base_digest != BASE_ARTIFACT_SHA256:
        raise ValueError(
            f"the calibrated base artifact digest mismatch: {base_digest} != {BASE_ARTIFACT_SHA256}"
        )
    report_digest = _sha256(ladder_report_path)
    if report_digest != LADDER_REPORT_SHA256:
        raise ValueError(
            f"the feasibility ladder report digest mismatch: {report_digest} != {LADDER_REPORT_SHA256}"
        )
    base = json.loads(base_artifact_path.read_text(encoding="utf-8"))
    report = json.loads(ladder_report_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(report, dict):
        raise ValueError("the calibrated base artifact and the feasibility ladder report must be objects")
    if base.get("artifact_type") != "scrooge-cost-calibrated-router-v1":
        raise ValueError("the calibrated base artifact type is not the frozen v2 router")
    if base.get("k1_enabled") is not False:
        raise ValueError("the calibrated base artifact must keep K1 disabled")
    if report.get("decision") != "record-ladder-ready-for-one-dev":
        raise ValueError("the feasibility ladder report is not the frozen ready-for-one-dev record")
    observed_caps = (
        report.get("diagnostic", {}).get("candidate_predicted_caps")
        or report.get("observed", {}).get("ladder", {}).get("selected")
    )
    if observed_caps != PREDICTED_CAPS:
        raise ValueError(
            f"frozen the feasibility ladder caps are {PREDICTED_CAPS}, report has {observed_caps}"
        )
    recalibration = _extract_full_fit_bins(report)
    provenance = dict(base.get("provenance") or {})
    provenance.update(
        {
            "dev_data_used": False,
            "base_artifact_sha256": base_digest,
            "export_note": (
                "the feasibility ladder runtime artifact. Quality, cost, and Premium overlay heads "
                "are byte-copied from the pinned the calibrated base artifact. Recalibration "
                "edges and factors are the frozen the feasibility ladder full-fit Train-only "
                "values, identical to the the feasibility ladder Dev replay transform. Caps are "
                "Fast 1.03 / Balanced 1.50 / Premium 3.25. K1 stays disabled "
                "and can later be turned on for Premium by setting k1_enabled "
                "and supplying k1.quality; that is an artifact configuration, "
                "not a router rewrite. No Dev path was opened, hashed, parsed, "
                "stated, or scored."
            ),
            "ladder_candidate_name": CANDIDATE_NAME,
            "ladder_report_sha256": report_digest,
        }
    )
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "cost": base["cost"],
        "k1": {
            "activation": "artifact-flag-plus-quality-head",
            "enabled": False,
            "quality": None,
            "scope": "premium-only",
        },
        "k1_enabled": False,
        "max_upgrade_fraction": MAX_UPGRADE_FRACTION,
        "policy_id": base["policy_id"],
        "policy_sha256": base["policy_sha256"],
        "predicted_caps": dict(PREDICTED_CAPS),
        "premium_overlay": base["premium_overlay"],
        "provenance": provenance,
        "quality": base["quality"],
        "recalibration": recalibration,
        "runaway_fraction": RUNAWAY_FRACTION,
        "safe_tier_caps": base["safe_tier_caps"],
        "schema_version": SCHEMA_VERSION,
        "selected_policy": CANDIDATE_NAME,
        "tier_kappa_q999": base["tier_kappa_q999"],
    }
    if _contains_dev_path(artifact):
        raise ValueError("Dev path leaked into the the feasibility ladder artifact")
    _write_json_atomic(output_path, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=BASE_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--ladder-report",
        type=Path,
        default=LADDER_REPORT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = export(
            base_artifact_path=args.base_artifact,
            ladder_report_path=args.ladder_report,
            output_path=args.output,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        f"OK: exported {artifact['artifact_type']} "
        f"caps={artifact['predicted_caps']} k1={artifact['k1_enabled']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
