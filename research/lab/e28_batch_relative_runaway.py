# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E28 — make the Premium runaway guard hold as a share of the incoming batch.

``promote_premium_brake`` compares every candidate K1 increment against
``runaway_absolute``, which is 0.02 * ``train_full_pred_light`` and therefore
2% of the *Train* batch, frozen. The artifact already carries
``runaway_light_fraction``; serve time never reads it. On a short or cheap
Premium batch the frozen absolute lets one item take a large slice of that
batch's own light, and the realized ratio leaves the official cap behind.

The candidate adds ``runaway_share`` and takes
``min(runaway_absolute, runaway_share * batch_predicted_light)``. It is never
looser than the shipped rule, so batches at or above the pinned split sizes
keep their selection bit for bit.

Unlike E27 this protocol separates gates from invariants. A gate is a claim
that could come back false; an invariant is true by construction and is
recorded as evidence, never counted as a pass. The falsifiability probe runs
pre-registered neighbour shares that must fail, so a silently vacuous gate is
itself a failure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router import budget_brake_router
from ossp_router.feasibility_ladder import _select_premium_configured
from ossp_router.protocol import TIERS, InputBatch, load_input, load_outcomes, policy_sha256
from research.lab.cap_certification import build_stress_views
from research.lab.density_ordering import view_layer
from research.lab.e1_objectives import (
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.e5_brake_conditioned import ProtocolError, derive_fresh_seeds
from research.lab.modeling import STRESS_BACKSTOP, sort_mapping
from research.lab.public_pool import (
    DEV_INPUTS,
    DEV_OUTCOMES,
    EXPECTED_N_DEV,
    EXPECTED_N_TRAIN,
    TRAIN_INPUTS,
    TRAIN_OUTCOMES,
    sha256_path,
    subset_inputs,
)
from research.lab.serving_replica import (
    K1,
    MODEL_INDEX,
    OFFICIAL_CAPS,
    PINNED_DEV_FINAL_SCORE,
    PIN_TOLERANCE,
    RESIDUAL_FAMILY,
    ServingReplica,
    SplitReplica,
    json_float,
    official_tier_block,
    score_models,
)


EXPERIMENT = "e28-batch-relative-runaway-v1"
REPORT_TYPE = "scrooge-e28-batch-relative-runaway-v1"
SCHEMA_VERSION = 1
BASELINE_NAME = "shipped-frozen-absolute"
PRIMARY_NAME = "runaway-share-0.06"
RUNAWAY_SHARE = 0.06
BRAKE_RATIO = 3.8
COUNT_CAP = 48
RUIN_FREQ_MAX = 0.0025
PUBLIC_LABEL = "public"
AUDIT_RELATIVE = "build/run-e28-batch-relative-runaway/view-audit.json"
REPORT_RELATIVE = "build/run-e28-batch-relative-runaway/report.json"
E27_DECISION_CORE = (
    "7deca711b20d3992bb1dd551a2600542f549601e1f220c4633f3fe303054fc6b"
)
EXPECTED_PROTOCOL_SHA256 = (
    "b7564f96b6ae8378ba2634ea0a50c0db22fcfd6215c3e4c1f825a667d4f81625"
)


def brake_block(
    replica: ServingReplica, *, share: Optional[float]
) -> dict[str, Any]:
    block = dict(replica.brake.budget_brake)
    block.pop("runaway_share", None)
    if share is not None:
        block["runaway_share"] = float(share)
    return block


def premium_models(
    replica: ServingReplica,
    split: SplitReplica,
    indexes: Sequence[int],
    *,
    share: Optional[float],
) -> Tuple[str, ...]:
    """Shipped Premium serving path with the E28 lever wired in.

    Parent ladder, E10 residual-majority hedge, and the brake loop are the
    real ``budget_brake_router`` entry points; only the block changes.
    """

    families = tuple(split.families[index] for index in indexes)
    active = budget_brake_router.premium_residual_composition_guard(families)
    raw_rows = tuple(split.premium_rows[index] for index in indexes)
    if active:
        parent_rows = tuple(
            (
                row[0],
                budget_brake_router.guard_premium_parent_costs(
                    split.inputs.episodes[index], row[1], replica.brake
                ),
            )
            for index, row in zip(indexes, raw_rows)
        )
    else:
        parent_rows = raw_rows
    parent, _ratio = _select_premium_configured(
        subset_inputs(split.inputs, indexes),
        parent_rows,
        float(replica.shipped_caps["premium"]),
        replica.brake.family_guard.base,
    )
    block = brake_block(replica, share=share)
    if active:
        block["denylist_families"] = list(block["denylist_families"]) + [
            RESIDUAL_FAMILY
        ]
    return budget_brake_router.promote_premium_brake(
        parent,
        tuple(split.premium_quality[index] for index in indexes),
        families,
        tuple(row[1] for row in raw_rows),
        tuple(split.digests[index] for index in indexes),
        block,
    )


def predicted_light(split: SplitReplica, indexes: Sequence[int]) -> float:
    return math.fsum(float(split.premium_rows[index][1][0]) for index in indexes)


def predicted_ratio(
    split: SplitReplica, indexes: Sequence[int], models: Sequence[str]
) -> float:
    light = predicted_light(split, indexes)
    if light <= 0.0:
        raise ProtocolError("predicted light sum is not positive")
    spend = math.fsum(
        float(split.premium_rows[index][1][MODEL_INDEX[model]])
        for index, model in zip(indexes, models)
    )
    return float(spend / light)


def combine_splits(train: SplitReplica, dev: SplitReplica) -> SplitReplica:
    """Train+Dev as one batch. The runtime check submits 2,640 rows per tier."""

    inputs = InputBatch(
        schema_version=train.inputs.schema_version,
        challenge_id=train.inputs.challenge_id,
        split=PUBLIC_LABEL,
        episodes=train.inputs.episodes + dev.inputs.episodes,
    )
    return SplitReplica(
        label=PUBLIC_LABEL,
        inputs=inputs,
        outcomes=train.outcomes,
        scores=np.concatenate([train.scores, dev.scores]),
        costs=np.concatenate([train.costs, dev.costs]),
        families=train.families + dev.families,
        digests=train.digests + dev.digests,
        fb_predictions=train.fb_predictions + dev.fb_predictions,
        premium_rows=train.premium_rows + dev.premium_rows,
        premium_quality=train.premium_quality + dev.premium_quality,
    )


def full_batch_block(
    replica: ServingReplica, split: SplitReplica, *, share: Optional[float]
) -> dict[str, Any]:
    indexes = list(range(split.n))
    models = premium_models(replica, split, indexes, share=share)
    scored = score_models(split.scores, split.costs, indexes, models)
    light = predicted_light(split, indexes)
    return {
        "batch_predicted_light": json_float(light),
        "k1": int(sum(model == K1 for model in models)),
        "models": models,
        "predicted_ratio": json_float(predicted_ratio(split, indexes, models)),
        "premium_quality": json_float(scored["quality"]),
        "premium_ratio": json_float(scored["actual_ratio"]),
        "runaway_threshold": json_float(
            budget_brake_router.batch_runaway_threshold(
                brake_block(replica, share=share), light
            )
        ),
    }


def sweep_views(
    replica: ServingReplica,
    split: SplitReplica,
    views: Sequence[Any],
    *,
    share: Optional[float],
) -> dict[str, Any]:
    """Realized Premium ratio of the arm over the frozen stress catalogue."""

    official = float(OFFICIAL_CAPS["premium"])
    n_binding = 0
    n_ruin = 0
    n_ruin_binding = 0
    worst = 0.0
    worst_name = ""
    ruined_names: list[str] = []
    for view in views:
        indexes = [int(item) for item in np.asarray(view.index, dtype=np.int64)]
        models = premium_models(replica, split, indexes, share=share)
        realized = float(
            score_models(split.scores, split.costs, indexes, models)["actual_ratio"]
        )
        ruined = realized > official + 1e-15
        if realized > worst:
            worst = realized
            worst_name = str(view.name)
        if ruined:
            n_ruin += 1
            ruined_names.append(str(view.name))
        if view_layer(view) == "binding":
            n_binding += 1
            n_ruin_binding += int(ruined)
    if n_binding <= 0:
        raise ProtocolError("stress catalogue has no binding layer")
    return {
        "binding_ruin_frequency": json_float(n_ruin_binding / float(n_binding)),
        "n_binding": int(n_binding),
        "n_ruin": int(n_ruin),
        "n_ruin_binding": int(n_ruin_binding),
        "n_views": int(len(views)),
        "ruined_names": ruined_names,
        "worst_inflated": json_float(worst * float(STRESS_BACKSTOP)),
        "worst_realized": json_float(worst),
        "worst_view": worst_name,
    }


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(dict(protocol)))


def verify_protocol(
    protocol: Mapping[str, Any],
    expected_sha256: str,
    *,
    pool_identity: Optional[Mapping[str, Any]] = None,
) -> str:
    digest = protocol_sha256(protocol)
    if digest != expected_sha256:
        raise ProtocolError(
            f"protocol sha mismatch: got {digest}, expected {expected_sha256}"
        )
    if (
        protocol.get("experiment") != EXPERIMENT
        or protocol.get("protocol_id") != EXPERIMENT
    ):
        raise ProtocolError("protocol experiment id drifted")
    if protocol["arms"]["baseline"]["name"] != BASELINE_NAME:
        raise ProtocolError("baseline arm drifted")
    if protocol["arms"]["primary"]["name"] != PRIMARY_NAME:
        raise ProtocolError("primary arm drifted")
    thresholds = protocol["thresholds"]
    if abs(float(thresholds["runaway_share"]) - RUNAWAY_SHARE) > 1e-15:
        raise ProtocolError("runaway_share drifted")
    if abs(float(thresholds["binding_ruin_frequency_max"]) - RUIN_FREQ_MAX) > 1e-15:
        raise ProtocolError("binding_ruin_frequency_max drifted")
    if abs(float(thresholds["brake_ratio"]) - BRAKE_RATIO) > 1e-15:
        raise ProtocolError("brake_ratio must stay at the E27 value")
    if int(thresholds["count_cap"]) != COUNT_CAP:
        raise ProtocolError("count_cap must stay at the shipped value")
    probes = tuple(float(value) for value in protocol["falsifiability_probe"]["shares"])
    if not probes or RUNAWAY_SHARE in probes:
        raise ProtocolError("falsifiability probe must use neighbour shares")
    if not protocol["gates"] or not protocol["invariants"]:
        raise ProtocolError("protocol must declare both gates and invariants")
    overlap = set(protocol["gates"]) & set(protocol["invariants"])
    if overlap:
        raise ProtocolError(f"a claim cannot be both gate and invariant: {sorted(overlap)}")
    derivation = protocol["seed_derivation"]
    if str(derivation["core_sha256"]) != E27_DECISION_CORE:
        raise ProtocolError("e28 core must be the e27 decision core")
    fresh = tuple(int(seed) for seed in protocol["fresh_seeds"])
    expected_seeds = derive_fresh_seeds(
        str(derivation["prefix"]),
        str(derivation["core_sha256"]),
        int(derivation["n"]),
        [int(value) for value in derivation["forbidden_previous_seeds"]],
    )
    if fresh != expected_seeds or len(set(fresh)) != len(fresh):
        raise ProtocolError("sealed fresh seeds drifted from the derivation")
    if pool_identity is not None:
        pins = protocol["pins"]
        for key in (
            "train_inputs_sha256",
            "train_outcomes_sha256",
            "dev_inputs_sha256",
            "dev_outcomes_sha256",
            "policy_sha256",
        ):
            if str(pins[key]) != str(pool_identity[key]):
                raise ProtocolError(f"pinned {key} drifted")
    return digest


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "audit": report["audit"],
            "candidate_primary": report["candidate_primary"],
            "constants": report["constants"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "falsifiability_probe": report["falsifiability_probe"],
            "fold_seeds": report["fold_seeds"],
            "gate": report["gate"],
            "invariants": report["invariants"],
            "pin_dev_replay": report["pin_dev_replay"],
            "protocol_sha256": report["protocol_sha256"],
            "report_type": report["report_type"],
            "schema_version": report["schema_version"],
            "thresholds": report["thresholds"],
        }
    )


def decision_core_sha256(report: Mapping[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            decision_core_payload(report),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def assemble(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    *,
    output: Path,
    audit_output: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output.exists() or audit_output.exists():
        raise ProtocolError("e28 output exists; refuse overwrite")
    if "src/" in str(output) or "src/" in str(audit_output):
        raise ProtocolError("e28 must not write under src/")

    replica = ServingReplica.load()
    if "runaway_share" in replica.brake.budget_brake:
        raise ProtocolError(
            "shipped artifact already carries runaway_share; baseline is not the shipped arm"
        )
    identity_pins = {
        "dev_inputs_sha256": sha256_path(DEV_INPUTS),
        "dev_outcomes_sha256": sha256_path(DEV_OUTCOMES),
        "policy_sha256": policy_sha256(replica.policy),
        "train_inputs_sha256": sha256_path(TRAIN_INPUTS),
        "train_outcomes_sha256": sha256_path(TRAIN_OUTCOMES),
    }
    verify_protocol(protocol, protocol_digest, pool_identity=identity_pins)

    train = replica.build_split(
        "train", load_input(TRAIN_INPUTS), load_outcomes(TRAIN_OUTCOMES)
    )
    dev = replica.build_split(
        "dev", load_input(DEV_INPUTS), load_outcomes(DEV_OUTCOMES)
    )
    if train.n != EXPECTED_N_TRAIN or dev.n != EXPECTED_N_DEV:
        raise ProtocolError("split n drifted")
    public = combine_splits(train, dev)
    batches = {"train": train, "dev": dev, PUBLIC_LABEL: public}

    shipped_dev = {tier: replica.runtime_models(dev, tier) for tier in TIERS}
    pin_official = replica.official(dev, shipped_dev)
    pin = {
        "final_score": float(pin_official["final_score"]),
        "matched": bool(
            abs(float(pin_official["final_score"]) - PINNED_DEV_FINAL_SCORE)
            <= PIN_TOLERANCE
        ),
        "pinned_final_score": PINNED_DEV_FINAL_SCORE,
    }
    if not pin["matched"]:
        raise ProtocolError("shipped Dev pin failed")

    thresholds = dict(protocol["thresholds"])
    share = float(thresholds["runaway_share"])

    full: dict[str, Any] = {"baseline": {}, "primary": {}}
    identity_rows: dict[str, bool] = {}
    for label, split in batches.items():
        base = full_batch_block(replica, split, share=None)
        cand = full_batch_block(replica, split, share=share)
        identity_rows[label] = bool(base["models"] == cand["models"])
        for arm, row in (("baseline", base), ("primary", cand)):
            full[arm][label] = {
                key: value for key, value in row.items() if key != "models"
            }

    dev_primary = premium_models(replica, dev, list(range(dev.n)), share=share)
    dev_official = replica.official(
        dev,
        {
            "fast": shipped_dev["fast"],
            "balanced": shipped_dev["balanced"],
            "premium": dev_primary,
        },
    )
    dev_final = float(dev_official["final_score"])

    stress: dict[str, Any] = {"baseline": {}, "primary": {}}
    catalogue_rows: dict[str, Any] = {}
    new_ruins: dict[str, list[str]] = {}
    for label in ("train", "dev"):
        split = batches[label]
        views, catalogue = build_stress_views(split.families)
        catalogue_rows[label] = {
            "n_views": int(catalogue["n_views"]),
            "view_kind_counts": dict(catalogue["view_kind_counts"]),
        }
        base = sweep_views(replica, split, views, share=None)
        cand = sweep_views(replica, split, views, share=share)
        stress["baseline"][label] = base
        stress["primary"][label] = cand
        new_ruins[label] = sorted(
            set(cand["ruined_names"]) - set(base["ruined_names"])
        )

    probe_shares = [float(value) for value in protocol["falsifiability_probe"]["shares"]]
    probe: dict[str, Any] = {}
    for value in probe_shares:
        rows = {
            label: bool(
                full_batch_block(replica, split, share=None)["models"]
                == premium_models(replica, split, list(range(split.n)), share=value)
            )
            for label, split in batches.items()
        }
        views, _catalogue = build_stress_views(dev.families)
        swept = sweep_views(replica, dev, views, share=value)
        gate_identity = all(rows.values())
        gate_ruin = float(swept["binding_ruin_frequency"]) <= RUIN_FREQ_MAX + 1e-15
        probe[f"{value:.4f}"] = {
            "dev_binding_ruin_frequency": swept["binding_ruin_frequency"],
            "full_batch_identity": gate_identity,
            "gate_identity_passed": bool(gate_identity),
            "gate_ruin_passed": bool(gate_ruin),
            "some_gate_failed": bool((not gate_identity) or (not gate_ruin)),
            "worst_realized": swept["worst_realized"],
        }

    official_cap = float(OFFICIAL_CAPS["premium"])
    gate_rows = {
        "full_batch_identity": bool(all(identity_rows.values())),
        "dev_official_unchanged": bool(
            abs(dev_final - float(pin_official["final_score"])) <= PIN_TOLERANCE
        ),
        "binding_ruin_frequency_within_budget": bool(
            all(
                float(stress["primary"][label]["binding_ruin_frequency"])
                <= RUIN_FREQ_MAX + 1e-15
                for label in ("train", "dev")
            )
        ),
        "worst_realized_under_official_cap": bool(
            all(
                float(stress["primary"][label]["worst_realized"]) <= official_cap
                for label in ("train", "dev")
            )
        ),
        "dev_ruin_strictly_improved": bool(
            int(stress["primary"]["dev"]["n_ruin"])
            < int(stress["baseline"]["dev"]["n_ruin"])
        ),
        "no_new_ruined_view": bool(
            not any(new_ruins[label] for label in ("train", "dev"))
        ),
        "probe_shares_fail_a_gate": bool(
            probe and all(row["some_gate_failed"] for row in probe.values())
        ),
    }
    failures = sorted(name for name, ok in gate_rows.items() if not ok)
    declared = set(protocol["gates"])
    if declared != set(gate_rows):
        raise ProtocolError(
            f"gate set drifted from the protocol: {sorted(declared ^ set(gate_rows))}"
        )
    gates_ok = (not failures) and bool(pin["matched"])
    decision = str(protocol["decisions"]["pass" if gates_ok else "fail"])
    reason = str(protocol["decision_reasons"]["pass" if gates_ok else "fail"])

    audit_document = {
        "baseline_ruined_views": {
            label: sorted(stress["baseline"][label]["ruined_names"])
            for label in ("train", "dev")
        },
        "experiment": EXPERIMENT,
        "new_ruined_views": new_ruins,
        "primary_ruined_views": {
            label: sorted(stress["primary"][label]["ruined_names"])
            for label in ("train", "dev")
        },
        "prompt_text_included": False,
    }
    for arm in ("baseline", "primary"):
        for label in ("train", "dev"):
            stress[arm][label] = {
                key: value
                for key, value in stress[arm][label].items()
                if key != "ruined_names"
            }

    report = {
        "audit": {
            "relative_path": AUDIT_RELATIVE,
            "sha256": sha256_text(canonical_json_text(audit_document)),
        },
        "candidate_primary": PRIMARY_NAME,
        "constants": {
            "brake_ratio": BRAKE_RATIO,
            "count_cap": COUNT_CAP,
            "official_premium_cap": official_cap,
            "runaway_absolute": json_float(
                replica.brake.budget_brake["runaway_absolute"]
            ),
            "runaway_share": RUNAWAY_SHARE,
            "stress_backstop": json_float(STRESS_BACKSTOP),
        },
        "decision": decision,
        "decision_reason": reason,
        "dev_official_primary": {
            tier: official_tier_block(dev_official, tier) for tier in TIERS
        },
        "experiment": EXPERIMENT,
        "falsifiability_probe": probe,
        "fold_seeds": [int(seed) for seed in protocol["fresh_seeds"]],
        "full_batch": full,
        "full_batch_identity_by_split": identity_rows,
        "gate": {
            "failures": failures,
            "passed": bool(gates_ok),
            "pin_matched": bool(pin["matched"]),
            "rows": gate_rows,
        },
        "invariants": {
            "brake_loop_bounds_predicted_ratio": (
                "promote_premium_brake skips any increment that would push "
                "predicted spend past brake_ratio * predicted light, so a "
                "predicted-ratio ceiling can never come back false"
            ),
            "fast_balanced_untouched_by_lever": (
                "runaway_share is read only inside promote_premium_brake; the "
                "Fast and Balanced vectors are the shipped runtime_models"
            ),
            "note": (
                "Recorded as evidence, not as gates. Each is true by "
                "construction and cannot come back false, so counting it as a "
                "pass would overstate what the run tested."
            ),
            "shipped_block_without_runaway_share_is_bit_identical": (
                "batch_runaway_threshold returns runaway_absolute unchanged "
                "when the field is absent; covered by the unit tests"
            ),
        },
        "pin_dev_replay": pin,
        "protocol_id": EXPERIMENT,
        "protocol_sha256": protocol_digest,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "stress": stress,
        "stress_catalogue": catalogue_rows,
        "thresholds": thresholds,
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    write_json_atomic(audit_output, audit_document)
    write_json_atomic(output, report)
    return report, audit_document


def run_from_protocol(
    protocol_path: Path,
    expected_protocol_sha256: str,
    *,
    output: Path,
    audit_output: Path,
) -> Mapping[str, Any]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("protocol is not a JSON object")
    digest = verify_protocol(payload, expected_protocol_sha256)
    report, _audit = assemble(
        payload, digest, output=output, audit_output=audit_output
    )
    return report
