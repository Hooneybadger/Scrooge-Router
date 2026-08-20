# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Policy-independence and determinism audit the official checker does not run.

Charter A12: repeat the same input, shuffle episode order, and replace
episode IDs. The official runtime checker does not cover those properties.
This tool is stdlib-only and does not import router internals beyond the
caller-selected ``--router-module`` entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
REPORT_TYPE = "g-series-submission-contract-audit"
LEGAL_MODEL_IDS = ("ax31-light", "ax31", "axk1-think")
LEGAL_TIERS = ("fast", "balanced", "premium")
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
REQUIRED_MODE = 0o644
ORDER_SHUFFLE_SEED = 2026082201
ID_PERMUTATION_SEED = 2026082202
SUBMISSION_NAME = "submission.json"
REQUIRED_SUBMISSION_FIELDS = (
    "schema_version",
    "challenge_id",
    "policy_id",
    "split",
    "tier",
    "decisions",
)

ROOT = Path(__file__).resolve().parents[2]


class AuditError(RuntimeError):
    """Raised when the auditor cannot execute a check."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(REQUIRED_MODE)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_content(episode: Mapping[str, Any]) -> str:
    prompt = episode.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    messages = episode.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for item in messages:
            if not isinstance(item, Mapping):
                raise AuditError("episode messages must be objects")
            content = item.get("content")
            if not isinstance(content, str):
                raise AuditError("message content must be a string")
            parts.append(content)
        return "\n".join(parts)
    raise AuditError("episode has neither prompt nor messages")


def _content_digest(episode: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_content(episode).encode("utf-8")).hexdigest()


def load_input_document(path: Path) -> dict[str, Any]:
    document = _load_json(path)
    if not isinstance(document, dict):
        raise AuditError("input JSON must be an object")
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise AuditError("input JSON must contain a non-empty episodes array")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise AuditError(f"episodes[{index}] must be an object")
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise AuditError(f"episodes[{index}].episode_id is missing")
    return document


def import_router(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AuditError(f"cannot import router module {module_name!r}: {exc}") from exc


def import_router_fresh(module_name: str) -> Any:
    dropped = [
        name
        for name in list(sys.modules)
        if name == module_name or name.startswith(module_name + ".")
    ]
    for name in dropped:
        del sys.modules[name]
    return import_router(module_name)


def run_router(
    module: Any,
    *,
    input_path: Path,
    output_path: Path,
    tier: str,
) -> dict[str, Any]:
    if not hasattr(module, "main"):
        raise AuditError("router module has no main()")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for leftover in output_path.parent.iterdir():
        leftover.unlink()
    argv = [
        "--input",
        str(input_path),
        "--tier",
        tier,
        "--output",
        str(output_path),
    ]
    code = module.main(argv)
    if code not in (0, None):
        raise AuditError(f"router main() exited {code} for tier {tier}")
    return {
        "exit_code": 0 if code is None else int(code),
        "output_path": str(output_path),
    }


def _decision_map(submission: Mapping[str, Any]) -> dict[str, str]:
    decisions = submission.get("decisions")
    if not isinstance(decisions, list):
        raise AuditError("submission.decisions must be an array")
    mapping: dict[str, str] = {}
    for item in decisions:
        if not isinstance(item, Mapping):
            raise AuditError("each decision must be an object")
        episode_id = item.get("episode_id")
        model_id = item.get("model_id")
        if not isinstance(episode_id, str) or not isinstance(model_id, str):
            raise AuditError("decision fields must be strings")
        mapping[episode_id] = model_id
    return mapping


def _check_result(
    passed: bool,
    **fields: Any,
) -> dict[str, Any]:
    record = {"passed": bool(passed)}
    record.update(fields)
    return record


def check_output_contract(
    output_directory: Path,
    input_document: Mapping[str, Any],
    *,
    expected_tier: str,
) -> dict[str, Any]:
    findings: list[str] = []
    try:
        entries = sorted(output_directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        return _check_result(False, findings=[f"cannot list output directory: {exc}"])
    names = [entry.name for entry in entries]
    if names != [SUBMISSION_NAME]:
        findings.append(f"output entries {names!r} are not exactly [{SUBMISSION_NAME!r}]")
        return _check_result(False, findings=findings, entries=names)

    path = entries[0]
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        size = path.stat().st_size
        payload = path.read_bytes()
    except OSError as exc:
        return _check_result(False, findings=[f"cannot read submission: {exc}"])

    if mode != REQUIRED_MODE:
        findings.append(f"mode {mode:04o} is not 0644")
    if size >= MAX_OUTPUT_BYTES:
        findings.append(f"size {size} bytes is not under 4 MiB")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        findings.append(f"submission is not UTF-8 JSON: {exc}")
        return _check_result(
            False,
            findings=findings,
            mode=f"{mode:04o}",
            size_bytes=size,
        )

    schema_findings = _validate_submission_schema(document, expected_tier=expected_tier)
    findings.extend(schema_findings)

    expected_ids = [episode["episode_id"] for episode in input_document["episodes"]]
    expected_set = set(expected_ids)
    if isinstance(document, dict) and isinstance(document.get("decisions"), list):
        actual_ids = [
            item.get("episode_id")
            for item in document["decisions"]
            if isinstance(item, Mapping)
        ]
        actual_set = {item for item in actual_ids if isinstance(item, str)}
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing:
            findings.append(f"omitted episode_id count={len(missing)}")
        if extra:
            findings.append(f"unexpected episode_id count={len(extra)}")
        if len(actual_ids) != len(actual_set):
            findings.append("duplicate episode_id in decisions")
        if len(actual_ids) != len(expected_ids):
            findings.append(
                f"decision count {len(actual_ids)} != input count {len(expected_ids)}"
            )
    return _check_result(
        not findings,
        findings=findings,
        mode=f"{mode:04o}",
        size_bytes=size,
        file_count=len(entries),
    )


def _validate_submission_schema(document: Any, *, expected_tier: str) -> list[str]:
    findings: list[str] = []
    if not isinstance(document, dict):
        return ["submission is not a JSON object"]
    extra = sorted(set(document) - set(REQUIRED_SUBMISSION_FIELDS))
    missing = [name for name in REQUIRED_SUBMISSION_FIELDS if name not in document]
    if extra:
        findings.append(f"additionalProperties {extra}")
    if missing:
        findings.append(f"missing fields {missing}")
    if document.get("schema_version") != 1:
        findings.append("schema_version must be 1")
    for field in ("challenge_id", "policy_id", "split"):
        value = document.get(field)
        if not isinstance(value, str) or not value:
            findings.append(f"{field} must be a non-empty string")
    if document.get("tier") != expected_tier:
        findings.append(
            f"tier {document.get('tier')!r} does not match requested {expected_tier!r}"
        )
    elif document.get("tier") not in LEGAL_TIERS:
        findings.append(f"tier {document.get('tier')!r} is not a legal tier")
    decisions = document.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        findings.append("decisions must be a non-empty array")
        return findings
    seen: set[str] = set()
    for index, item in enumerate(decisions):
        if not isinstance(item, dict):
            findings.append(f"decisions[{index}] is not an object")
            continue
        extra_item = sorted(set(item) - {"episode_id", "model_id"})
        if extra_item:
            findings.append(f"decisions[{index}] extra fields {extra_item}")
        episode_id = item.get("episode_id")
        model_id = item.get("model_id")
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 128:
            findings.append(f"decisions[{index}].episode_id is invalid")
        elif episode_id in seen:
            findings.append(f"duplicate episode_id {episode_id!r}")
        else:
            seen.add(episode_id)
        if model_id not in LEGAL_MODEL_IDS:
            findings.append(f"decisions[{index}].model_id {model_id!r} is not legal")
    return findings


def check_repeat_determinism(first: bytes, second: bytes) -> dict[str, Any]:
    return _check_result(
        first == second,
        first_sha256=hashlib.sha256(first).hexdigest(),
        second_sha256=hashlib.sha256(second).hexdigest(),
        findings=[] if first == second else ["byte-identical submission.json failed"],
    )


def check_input_order_invariance(
    baseline: Mapping[str, Any],
    shuffled: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_map = _decision_map(baseline)
    shuffled_map = _decision_map(shuffled)
    baseline_sorted = sorted(baseline_map.items())
    shuffled_sorted = sorted(shuffled_map.items())
    passed = baseline_sorted == shuffled_sorted
    mismatches = [
        {"episode_id": episode_id, "baseline": baseline_map[episode_id], "shuffled": model}
        for episode_id, model in shuffled_sorted
        if baseline_map.get(episode_id) != model
    ]
    return _check_result(
        passed,
        compared_after_sort_by_episode_id=True,
        mismatch_count=len(mismatches),
        mismatches=mismatches[:20],
        findings=[]
        if passed
        else [f"order shuffle changed {len(mismatches)} selections"],
    )


def check_episode_id_invariance(
    baseline_document: Mapping[str, Any],
    baseline_submission: Mapping[str, Any],
    remapped_document: Mapping[str, Any],
    remapped_submission: Mapping[str, Any],
    original_to_token: Mapping[str, str],
) -> dict[str, Any]:
    baseline_by_id = _decision_map(baseline_submission)
    remapped_by_id = _decision_map(remapped_submission)
    mismatches = []
    for episode, remapped_episode in zip(
        baseline_document["episodes"], remapped_document["episodes"]
    ):
        original_id = episode["episode_id"]
        token = remapped_episode["episode_id"]
        expected_token = original_to_token[original_id]
        if token != expected_token:
            mismatches.append(
                {
                    "original_id": original_id,
                    "expected_token": expected_token,
                    "actual_token": token,
                    "reason": "permutation mapping drifted",
                }
            )
            continue
        left = baseline_by_id.get(original_id)
        right = remapped_by_id.get(token)
        if left != right:
            mismatches.append(
                {
                    "original_id": original_id,
                    "token": token,
                    "content_sha256": _content_digest(episode),
                    "baseline": left,
                    "remapped": right,
                }
            )
    passed = not mismatches
    return _check_result(
        passed,
        mismatch_count=len(mismatches),
        mismatches=mismatches[:20],
        findings=[]
        if passed
        else [f"episode_id permutation changed {len(mismatches)} selections"],
    )


def check_duplicate_content_consistency(
    input_document: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    decisions = _decision_map(submission)
    groups: dict[str, list[dict[str, str]]] = {}
    for episode in input_document["episodes"]:
        digest = _content_digest(episode)
        groups.setdefault(digest, []).append(
            {
                "episode_id": episode["episode_id"],
                "model_id": decisions.get(episode["episode_id"], ""),
            }
        )
    duplicate_groups = {
        digest: members for digest, members in groups.items() if len(members) > 1
    }
    inconsistent = []
    for digest, members in sorted(duplicate_groups.items()):
        models = {member["model_id"] for member in members}
        if len(models) != 1:
            inconsistent.append(
                {
                    "content_sha256": digest,
                    "size": len(members),
                    "models": sorted(models),
                    "members": members,
                }
            )
    passed = not inconsistent
    findings = []
    if inconsistent:
        findings.append(
            f"{len(inconsistent)} duplicate-content group(s) selected different models"
        )
        findings.append(
            "legitimate only for a global-knapsack allocator that breaks ties "
            "by something other than content (index, id, or unstable sort)"
        )
    return _check_result(
        passed,
        duplicate_group_count=len(duplicate_groups),
        unique_content_count=len(groups),
        inconsistent_group_count=len(inconsistent),
        inconsistent_groups=inconsistent[:20],
        findings=findings,
    )


def check_tier_isolation(
    fast_after_balanced: bytes,
    fast_first: bytes,
) -> dict[str, Any]:
    passed = fast_after_balanced == fast_first
    return _check_result(
        passed,
        fast_after_balanced_sha256=hashlib.sha256(fast_after_balanced).hexdigest(),
        fast_first_sha256=hashlib.sha256(fast_first).hexdigest(),
        findings=[]
        if passed
        else ["fast selections changed after balanced ran in the same process"],
    )


def _write_input(path: Path, document: Mapping[str, Any]) -> None:
    _write_json(path, document)


def _opaque_tokens(count: int, *, seed: int) -> list[str]:
    rng = random.Random(seed)
    order = list(range(count))
    rng.shuffle(order)
    return [f"opaque-{value:08x}" for value in order]


def _shuffled_document(document: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    episodes = list(document["episodes"])
    rng = random.Random(seed)
    rng.shuffle(episodes)
    clone = dict(document)
    clone["episodes"] = episodes
    return clone


def _id_permuted_document(
    document: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    tokens = _opaque_tokens(len(document["episodes"]), seed=seed)
    original_to_token = {}
    remapped = []
    for episode, token in zip(document["episodes"], tokens):
        clone = dict(episode)
        original_to_token[episode["episode_id"]] = token
        clone["episode_id"] = token
        remapped.append(clone)
    out = dict(document)
    out["episodes"] = remapped
    return out, original_to_token


def _read_submission_bytes(path: Path) -> bytes:
    return path.read_bytes()


def audit_router(
    *,
    router_module: str,
    input_path: Path,
    workdir: Path,
    tiers: Sequence[str] = LEGAL_TIERS,
    order_seed: int = ORDER_SHUFFLE_SEED,
    id_seed: int = ID_PERMUTATION_SEED,
) -> dict[str, Any]:
    input_document = load_input_document(input_path)
    workdir.mkdir(parents=True, exist_ok=True)
    baseline_input = workdir / "baseline-input.json"
    _write_input(baseline_input, input_document)

    tier_records: dict[str, Any] = {}
    all_passed = True
    for tier in tiers:
        module = import_router_fresh(router_module)
        first_dir = workdir / f"{tier}-repeat-1"
        second_dir = workdir / f"{tier}-repeat-2"
        first_out = first_dir / SUBMISSION_NAME
        second_out = second_dir / SUBMISSION_NAME
        run_router(module, input_path=baseline_input, output_path=first_out, tier=tier)
        run_router(module, input_path=baseline_input, output_path=second_out, tier=tier)
        first_bytes = _read_submission_bytes(first_out)
        second_bytes = _read_submission_bytes(second_out)
        first_submission = json.loads(first_bytes.decode("utf-8"))

        shuffled = _shuffled_document(input_document, seed=order_seed)
        shuffled_input = workdir / f"{tier}-shuffled-input.json"
        shuffled_out = workdir / f"{tier}-shuffled" / SUBMISSION_NAME
        _write_input(shuffled_input, shuffled)
        run_router(module, input_path=shuffled_input, output_path=shuffled_out, tier=tier)
        shuffled_submission = json.loads(shuffled_out.read_text(encoding="utf-8"))

        remapped, original_to_token = _id_permuted_document(input_document, seed=id_seed)
        remapped_input = workdir / f"{tier}-idperm-input.json"
        remapped_out = workdir / f"{tier}-idperm" / SUBMISSION_NAME
        _write_input(remapped_input, remapped)
        run_router(module, input_path=remapped_input, output_path=remapped_out, tier=tier)
        remapped_submission = json.loads(remapped_out.read_text(encoding="utf-8"))

        records = {
            "repeat_determinism": check_repeat_determinism(first_bytes, second_bytes),
            "input_order_invariance": check_input_order_invariance(
                first_submission, shuffled_submission
            ),
            "episode_id_invariance": check_episode_id_invariance(
                input_document,
                first_submission,
                remapped,
                remapped_submission,
                original_to_token,
            ),
            "duplicate_content_consistency": check_duplicate_content_consistency(
                input_document, first_submission
            ),
            "output_contract": check_output_contract(
                first_dir, input_document, expected_tier=tier
            ),
        }
        tier_passed = all(item["passed"] for item in records.values())
        records["passed"] = tier_passed
        tier_records[tier] = records
        all_passed = all_passed and tier_passed

    isolation = {"passed": True, "findings": ["tier isolation skipped; fast not requested"]}
    if "fast" in tiers and "balanced" in tiers:
        after_balanced_dir = workdir / "isolation-balanced-then-fast"
        first_fast_dir = workdir / "isolation-fast-first"
        after_balanced_out = after_balanced_dir / SUBMISSION_NAME
        first_fast_out = first_fast_dir / SUBMISSION_NAME
        balanced_scratch = workdir / "isolation-balanced" / SUBMISSION_NAME

        module_a = import_router_fresh(router_module)
        run_router(
            module_a,
            input_path=baseline_input,
            output_path=balanced_scratch,
            tier="balanced",
        )
        run_router(
            module_a,
            input_path=baseline_input,
            output_path=after_balanced_out,
            tier="fast",
        )

        module_b = import_router_fresh(router_module)
        run_router(
            module_b,
            input_path=baseline_input,
            output_path=first_fast_out,
            tier="fast",
        )
        isolation = check_tier_isolation(
            _read_submission_bytes(after_balanced_out),
            _read_submission_bytes(first_fast_out),
        )
        all_passed = all_passed and isolation["passed"]

    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "router_module": router_module,
        "input": str(input_path),
        "seeds": {
            "order_shuffle": order_seed,
            "id_permutation": id_seed,
        },
        "tiers": tier_records,
        "tier_isolation": isolation,
        "passed": all_passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-module",
        required=True,
        help="Import path of a router with main(), e.g. ossp_router.cost_calibrated_router",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Scratch directory; a temporary directory is used when omitted.",
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=LEGAL_TIERS,
        dest="tiers",
        help="Audit one tier; omit to audit all three.",
    )
    parser.add_argument("--order-seed", type=int, default=ORDER_SHUFFLE_SEED)
    parser.add_argument("--id-seed", type=int, default=ID_PERMUTATION_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    workdir_context = None
    workdir = args.workdir
    if workdir is None:
        workdir_context = tempfile.TemporaryDirectory(prefix="ossp-g-audit-")
        workdir = Path(workdir_context.name)
    try:
        report = audit_router(
            router_module=args.router_module,
            input_path=args.input,
            workdir=workdir,
            tiers=tuple(args.tiers or LEGAL_TIERS),
            order_seed=args.order_seed,
            id_seed=args.id_seed,
        )
    except (OSError, AuditError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if workdir_context is not None:
            workdir_context.cleanup()
        return 2
    if args.report is not None:
        try:
            _write_json(args.report, report)
        except OSError as exc:
            print(f"error: cannot write report: {exc}", file=sys.stderr)
            if workdir_context is not None:
                workdir_context.cleanup()
            return 2
    print(
        f"{'PASS' if report['passed'] else 'FAIL'}: "
        f"{report['router_module']} on {report['input']}"
    )
    for tier, record in report["tiers"].items():
        status = "PASS" if record["passed"] else "FAIL"
        failed = [
            name
            for name, item in record.items()
            if name != "passed" and isinstance(item, Mapping) and not item.get("passed", True)
        ]
        extra = f" ({', '.join(failed)})" if failed else ""
        print(f"  {tier}: {status}{extra}")
    isolation = report["tier_isolation"]
    print(f"  tier_isolation: {'PASS' if isolation['passed'] else 'FAIL'}")
    if workdir_context is not None:
        workdir_context.cleanup()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
