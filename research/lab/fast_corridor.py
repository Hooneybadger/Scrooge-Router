# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the Fast corridor sweep — Fast-safe corridor. Freeze the the feasibility ladder head; sweep Fast caps only.

Balanced and Premium stay at the feasibility ladder values. K1 stays off. Train gates use the
the yardstick run the feasibility ladder Fast-ruin yardstick on the same the cap certification layer design. Dev opens once, and
only for one raised Fast cap that already passed Train.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.protocol import MODEL_IDS, TIERS, InputBatch, OutcomeBatch, RoutingPolicy
from ossp_router.feasibility_ladder import (
    load_artifact_mapping,
    load_bundled_artifact,
    make_submission,
)
from research.lab.modeling import (
    OFFICIAL_CAPS,
    STRESS_BACKSTOP,
    TrainBundle,
    official_score,
    sort_mapping,
    weighted_final,
)
from research.lab.prefix_certificates import json_float
from research.lab.cap_certification import (
    PREMIUM_CAP,
    RUIN_FREQ_MAX,
    LADDER_ARTIFACT_PATH,
    LADDER_BALANCED_CAP,
    LADDER_DEV_WEIGHTED,
    LADDER_FAST_CAP,
    LADDER_MAX_UPGRADE,
    LADDER_RUNAWAY,
    CapConfig,
    ReproductionError,
    Stage2Refused,
    allocate_frozen,
    artifact_for_config,
    ax31_count,
    binding_constraint_frozen,
    build_stress_views,
    cache_predictions,
    derived_runaway_fraction,
    k1_count,
    official_tier_block,
    official_weighted_text,
    score_mean,
    select_premium_cached,
    sweep_tier_views,
    write_selected_artifact,
    _permute_ids,
    _selection_by_digest,
    _selection_by_id,
    _shuffle_inputs,
    _subset_rows,
)
from research.lab.prefix_certificates import _realized_ratio
from research.lab.validation import public_arrays


EXPERIMENT = "the Fast corridor sweep"
REPORT_TYPE = "scrooge-fast_corridor-fast-corridor-v1"
SCHEMA_VERSION = 1
DECISION_PROMOTE = "record-fast_corridor-promote-fast-corridor"
DECISION_NO_ELIGIBLE = "record-fast_corridor-close-no-eligible-config"
DECISION_DEV_REJECT = "record-fast_corridor-close-dev-reject"
DECISIONS = (DECISION_PROMOTE, DECISION_NO_ELIGIBLE, DECISION_DEV_REJECT)
GRID_FAST: Tuple[float, ...] = (1.03, 1.04, 1.05, 1.06, 1.07, 1.08)
LADDER_TRAIN_WEIGHTED = 0.6476136363636363
YARDSTICK_DECISION = "record-yardstick-ruin-yardstick"
LADDER_DEV_FAST_RUIN = 0.0013986013986013986
LADDER_TRAIN_FAST_RUIN = 0.0
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YARDSTICK_REPORT = ROOT / "build" / "yardstick" / "report.json"
_K1 = MODEL_IDS[2]
SELECTION_RULE = (
    "Among raised Fast caps (fast > 1.03) that pass Train gates (a)(b)(c), "
    "maximize Train weighted quality; tie-break toward the smaller Fast cap. "
    "The 1.03 cell is the the feasibility ladder reference and is never promoted."
)
GATE_TEXTS: Tuple[str, ...] = (
    "(a) Train Fast binding ruin <= max(the yardstick run the feasibility ladder Fast ruin on this design, 0.25%)",
    "(b) Train weighted >= the feasibility ladder Train weighted",
    "(c) Balanced and Premium full-batch realized * 1.054 <= official caps; "
    "Fast uses the same backstop and is fail-closed",
)


class YardstickError(RuntimeError):
    """the yardstick run report is missing or does not match the locked yardstick."""


@dataclass(frozen=True)
class FastCorridorConfig:
    predicted_caps_fast: float

    @property
    def predicted_caps_balanced(self) -> float:
        return float(LADDER_BALANCED_CAP)

    @property
    def max_upgrade_fraction(self) -> float:
        return float(LADDER_MAX_UPGRADE)

    @property
    def key(self) -> str:
        return (
            f"fast={self.predicted_caps_fast:.2f}"
            f"|balanced={self.predicted_caps_balanced:.2f}"
            f"|max_upgrade={self.max_upgrade_fraction:.2f}"
        )

    def label(self) -> str:
        if abs(self.predicted_caps_fast - float(LADDER_FAST_CAP)) <= 1e-15:
            return "the feasibility ladder"
        return "corridor"

    def is_incumbent(self) -> bool:
        return self.label() == "the feasibility ladder"

    def cap(self, tier: str) -> float:
        if tier == "fast":
            return float(self.predicted_caps_fast)
        if tier == "balanced":
            return float(LADDER_BALANCED_CAP)
        return float(PREMIUM_CAP)

    def runaway(self, tier: str) -> float:
        if tier == "fast":
            return derived_runaway_fraction(self.predicted_caps_fast)
        if tier == "balanced":
            return float(LADDER_RUNAWAY)
        return derived_runaway_fraction(PREMIUM_CAP)

    def as_cap_cert_config(self) -> CapConfig:
        return CapConfig(
            predicted_caps_fast=float(self.predicted_caps_fast),
            predicted_caps_balanced=float(LADDER_BALANCED_CAP),
            max_upgrade_fraction=float(LADDER_MAX_UPGRADE),
        )


def pre_registered_grid() -> Tuple[FastCorridorConfig, ...]:
    grid = tuple(FastCorridorConfig(predicted_caps_fast=float(cap)) for cap in GRID_FAST)
    if len(grid) != 6:
        raise RuntimeError(f"the Fast corridor sweep grid must be 6 Fast caps; got {len(grid)}")
    if grid[0].label() != "the feasibility ladder":
        raise RuntimeError("the Fast corridor sweep grid must start with the the feasibility ladder Fast cap")
    return grid


def fast_ruin_threshold(ladder_fast_ruin: float, *, floor: float = RUIN_FREQ_MAX) -> float:
    """Yardstick is the feasibility ladder Fast ruin; the §13.1 floor binds when the feasibility ladder is safer."""

    return max(float(ladder_fast_ruin), float(floor))


def load_yardstick_fast_yardstick(path: Path) -> dict[str, Any]:
    """Read the yardstick run as a locked fact. Does not re-run the yardstick."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "the yardstick run":
        raise YardstickError("the yardstick run report experiment field drifted")
    if payload.get("decision") != YARDSTICK_DECISION:
        raise YardstickError(
            f"the yardstick run decision {payload.get('decision')!r} != {YARDSTICK_DECISION!r}"
        )
    train_fast = float(
        payload["splits"]["train"]["policies"]["ladder"]["fast"]["binding"]["ruin_frequency"]
    )
    dev_fast = float(payload["gate"]["per_policy"]["ladder"]["tiers"]["fast"]["ruin_frequency"])
    if abs(dev_fast - LADDER_DEV_FAST_RUIN) > 1e-15:
        raise YardstickError(
            f"the yardstick run Dev the feasibility ladder Fast ruin {dev_fast} != locked {LADDER_DEV_FAST_RUIN}"
        )
    if abs(train_fast - LADDER_TRAIN_FAST_RUIN) > 1e-15:
        raise YardstickError(
            f"the yardstick run Train the feasibility ladder Fast ruin {train_fast} != locked {LADDER_TRAIN_FAST_RUIN}"
        )
    return {
        "decision": YARDSTICK_DECISION,
        "dev_fast_ruin": json_float(dev_fast),
        "dev_threshold": json_float(fast_ruin_threshold(dev_fast)),
        "discriminates": bool(payload["gate"]["discriminates"]),
        "path": str(path),
        "safest": payload["gate"]["safest"],
        "train_fast_ruin": json_float(train_fast),
        "train_threshold": json_float(fast_ruin_threshold(train_fast)),
    }


def evaluate_train_gates(
    *,
    fast_ruin: float,
    fast_ruin_limit: float,
    train_weighted: float,
    ladder_train_weighted: float,
    realized: Mapping[str, float],
    is_incumbent: bool,
) -> dict[str, Any]:
    """Evaluate the three pre-registered Train gates independently."""

    gate_a = float(fast_ruin) <= float(fast_ruin_limit) + 1e-15
    gate_b = float(train_weighted) + 1e-15 >= float(ladder_train_weighted)
    backstops: dict[str, bool] = {}
    gate_c = True
    for tier in TIERS:
        official = float(OFFICIAL_CAPS[tier])
        ok = float(realized[tier]) * float(STRESS_BACKSTOP) <= official + 1e-15
        backstops[tier] = bool(ok)
        if tier in ("balanced", "premium") and not ok:
            gate_c = False
        if tier == "fast" and not ok:
            gate_c = False
    failed = []
    if not gate_a:
        failed.append("a")
    if not gate_b:
        failed.append("b")
    if not gate_c:
        failed.append("c")
    passed = bool(gate_a and gate_b and gate_c)
    return {
        "backstops": backstops,
        "eligible_raised": bool(passed and (not is_incumbent)),
        "failed": failed,
        "gate_a_fast_ruin": bool(gate_a),
        "gate_b_train_weighted": bool(gate_b),
        "gate_c_cost_backstop": bool(gate_c),
        "is_incumbent": bool(is_incumbent),
        "passed": bool(passed),
    }


def _config_sort_key(row: Mapping[str, Any]) -> Tuple[float, float]:
    return (-float(row["train_weighted_float"]), float(row["predicted_caps_fast"]))


def select_certified(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    certified = [row for row in rows if row.get("eligible_raised") is True]
    if not certified:
        return None
    return sorted(certified, key=_config_sort_key)[0]


def locked_record(yardstick: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "balanced_cap": json_float(LADDER_BALANCED_CAP),
            "balanced_runaway": json_float(LADDER_RUNAWAY),
            "derived_guard_formula": "max(0.05, 1.5 * (predicted_caps.fast - 1))",
            "fast_ruin_floor": json_float(RUIN_FREQ_MAX),
            "yardstick_yardstick": dict(yardstick),
            "gates": list(GATE_TEXTS),
            "grid_fast": [json_float(item) for item in GRID_FAST],
            "k1": "off",
            "max_upgrade_fraction": json_float(LADDER_MAX_UPGRADE),
            "n_configs": int(len(GRID_FAST)),
            "official_caps": {tier: json_float(OFFICIAL_CAPS[tier]) for tier in TIERS},
            "premium_cap": json_float(PREMIUM_CAP),
            "premium_untouched": True,
            "selection_rule": SELECTION_RULE,
            "stress_backstop": json_float(STRESS_BACKSTOP),
            "ladder_dev_weighted": json_float(LADDER_DEV_WEIGHTED),
            "ladder_fast_cap": json_float(LADDER_FAST_CAP),
            "ladder_train_weighted_lock": json_float(LADDER_TRAIN_WEIGHTED),
        }
    )


def _tier_row(
    *,
    tier: str,
    config: FastCorridorConfig,
    selection: Sequence[str],
    scores: np.ndarray,
    costs: np.ndarray,
    bound: str,
) -> dict[str, Any]:
    realized = float(_realized_ratio(costs, selection))
    return {
        "ax31": ax31_count(selection),
        "binding_constraint": bound,
        "k1": k1_count(selection),
        "predicted_cap": json_float(config.cap(tier)),
        "quality_float": score_mean(scores, selection),
        "runaway_fraction": json_float(config.runaway(tier)),
        "realized": json_float(realized),
        "realized_times_1054": json_float(realized * float(STRESS_BACKSTOP)),
    }


def _public_config_row(row: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        "eligible_raised": bool(row.get("eligible_raised", False)),
        "gates": row["gates"],
        "is_incumbent": bool(row["is_incumbent"]),
        "key": row["key"],
        "label": row["label"],
        "max_upgrade_fraction": json_float(row["max_upgrade_fraction"]),
        "predicted_caps_balanced": json_float(row["predicted_caps_balanced"]),
        "predicted_caps_fast": json_float(row["predicted_caps_fast"]),
        "runaway_fast": json_float(row["runaway_fast"]),
        "tiers": row["tiers"],
        "train_weighted_float": row["train_weighted_float"],
    }
    if row.get("fast_views") is not None:
        public["fast_views"] = {
            "binding": row["fast_views"]["binding"],
            "red_team": {
                "max_realized": row["fast_views"]["red_team"]["max_realized"],
                "n": row["fast_views"]["red_team"]["n"],
                "n_ruin": row["fast_views"]["red_team"]["n_ruin"],
                "ruin_frequency": row["fast_views"]["red_team"]["ruin_frequency"],
            },
        }
    if "official_train" in row:
        public["official_train"] = row["official_train"]
    return public


def run_stage1(
    bundle: TrainBundle,
    *,
    yardstick_report_path: Path,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> dict[str, Any]:
    yardstick = load_yardstick_fast_yardstick(yardstick_report_path)
    ladder_dict = json.loads(LADDER_ARTIFACT_PATH.read_text(encoding="utf-8"))
    ladder_artifact = load_artifact_mapping(copy.deepcopy(ladder_dict))
    cache = cache_predictions(bundle.inputs.episodes, bundle.policy, ladder_artifact)
    bundled = load_bundled_artifact()
    ladder_sel: dict[str, Tuple[str, ...]] = {}
    for tier in TIERS:
        plan = make_submission(bundle.inputs, bundle.policy, bundled, tier)
        ladder_sel[tier] = tuple(d.model_id for d in plan.submission.decisions)
    cached_fast, _ = allocate_frozen(
        cache.rows,
        cap=LADDER_FAST_CAP,
        runaway_fraction=LADDER_RUNAWAY,
        max_upgrade_fraction=LADDER_MAX_UPGRADE,
    )
    if cached_fast != ladder_sel["fast"]:
        raise ReproductionError("cached the feasibility ladder Fast allocation disagrees with make_submission")
    premium_sel, _ = select_premium_cached(
        cache.digests, cache.premium_uplift, cache.premium_costs, PREMIUM_CAP
    )
    if premium_sel != ladder_sel["premium"]:
        raise ReproductionError("cached Premium allocator disagrees with frozen the feasibility ladder")
    balanced_sel, _ = allocate_frozen(
        cache.rows,
        cap=LADDER_BALANCED_CAP,
        runaway_fraction=LADDER_RUNAWAY,
        max_upgrade_fraction=LADDER_MAX_UPGRADE,
    )
    if balanced_sel != ladder_sel["balanced"]:
        raise ReproductionError("the feasibility ladder-valued Balanced allocation disagrees with make_submission")
    if any(k1_count(ladder_sel[tier]) != 0 for tier in TIERS):
        raise ReproductionError("the feasibility ladder selected K1; charter §13.2 is violated")

    incumbent = FastCorridorConfig(predicted_caps_fast=LADDER_FAST_CAP)
    ladder_train = {
        tier: _tier_row(
            tier=tier,
            config=incumbent,
            selection=ladder_sel[tier],
            scores=bundle.scores,
            costs=bundle.costs,
            bound=binding_constraint_frozen(
                cache.rows,
                ladder_sel[tier],
                cap=incumbent.cap(tier),
                runaway_fraction=incumbent.runaway(tier),
                max_upgrade_fraction=LADDER_MAX_UPGRADE,
            )
            if tier != "premium"
            else "none",
        )
        for tier in TIERS
    }
    ladder_weighted = float(
        weighted_final(
            float(ladder_train["fast"]["quality_float"]),
            float(ladder_train["balanced"]["quality_float"]),
            float(ladder_train["premium"]["quality_float"]),
        )
    )
    if abs(ladder_weighted - float(LADDER_TRAIN_WEIGHTED)) > 5.5e-7:
        raise ReproductionError(
            f"recomputed the feasibility ladder Train weighted {ladder_weighted} != locked {LADDER_TRAIN_WEIGHTED}"
        )

    views, catalogue = build_stress_views(bundle.families)
    train_threshold = float(yardstick["train_threshold"])
    rows: list[dict[str, Any]] = []
    for config in pre_registered_grid():
        if config.is_incumbent():
            fast_sel = ladder_sel["fast"]
            fast_bound = ladder_train["fast"]["binding_constraint"]
        else:
            fast_sel, _ = allocate_frozen(
                cache.rows,
                cap=config.cap("fast"),
                runaway_fraction=config.runaway("fast"),
                max_upgrade_fraction=config.max_upgrade_fraction,
            )
            fast_bound = binding_constraint_frozen(
                cache.rows,
                fast_sel,
                cap=config.cap("fast"),
                runaway_fraction=config.runaway("fast"),
                max_upgrade_fraction=config.max_upgrade_fraction,
            )
        selection = {
            "fast": fast_sel,
            "balanced": ladder_sel["balanced"],
            "premium": ladder_sel["premium"],
        }
        if k1_count(fast_sel) != 0:
            raise ReproductionError(f"{config.key} selected K1 on Fast")
        train_q = {tier: score_mean(bundle.scores, selection[tier]) for tier in TIERS}
        realized = {tier: float(_realized_ratio(bundle.costs, selection[tier])) for tier in TIERS}
        swept = sweep_tier_views(
            views,
            cache,
            bundle.costs,
            tier="fast",
            cap=config.cap("fast"),
            runaway_fraction=config.runaway("fast"),
            max_upgrade_fraction=config.max_upgrade_fraction,
        )
        train_weighted = float(
            weighted_final(train_q["fast"], train_q["balanced"], train_q["premium"])
        )
        gates = evaluate_train_gates(
            fast_ruin=float(swept["binding"]["ruin_frequency"]),
            fast_ruin_limit=train_threshold,
            train_weighted=train_weighted,
            ladder_train_weighted=ladder_weighted,
            realized=realized,
            is_incumbent=config.is_incumbent(),
        )
        row = {
            "eligible_raised": bool(gates["eligible_raised"]),
            "fast_views": swept,
            "gates": gates,
            "is_incumbent": bool(config.is_incumbent()),
            "key": config.key,
            "label": config.label(),
            "max_upgrade_fraction": json_float(config.max_upgrade_fraction),
            "predicted_caps_balanced": json_float(config.predicted_caps_balanced),
            "predicted_caps_fast": json_float(config.predicted_caps_fast),
            "runaway_fast": json_float(config.runaway("fast")),
            "selection": selection,
            "tiers": {
                "fast": {
                    "ax31": ax31_count(fast_sel),
                    "binding_constraint": fast_bound,
                    "quality_float": train_q["fast"],
                    "realized": json_float(realized["fast"]),
                    "realized_times_1054": json_float(
                        realized["fast"] * float(STRESS_BACKSTOP)
                    ),
                },
                "balanced": {
                    "ax31": ax31_count(selection["balanced"]),
                    "binding_constraint": ladder_train["balanced"]["binding_constraint"],
                    "quality_float": train_q["balanced"],
                    "realized": json_float(realized["balanced"]),
                    "realized_times_1054": json_float(
                        realized["balanced"] * float(STRESS_BACKSTOP)
                    ),
                },
                "premium": {
                    "ax31": ax31_count(selection["premium"]),
                    "binding_constraint": "none",
                    "quality_float": train_q["premium"],
                    "realized": json_float(realized["premium"]),
                    "realized_times_1054": json_float(
                        realized["premium"] * float(STRESS_BACKSTOP)
                    ),
                },
            },
            "train_weighted_float": json_float(train_weighted),
        }
        rows.append(row)
        if progress is not None:
            progress(
                {
                    "key": config.key,
                    "eligible_raised": row["eligible_raised"],
                    "fast_ruin": swept["binding"]["ruin_frequency"],
                    "phase": "stage1-cap",
                }
            )

    selected = select_certified(rows)
    official_train = None
    if selected is not None:
        official_train = official_score(
            bundle.inputs,
            bundle.outcomes,
            bundle.policy,
            {tier: selected["selection"][tier] for tier in TIERS},
        )
        selected = dict(selected)
        selected["official_train"] = {
            "final_score": official_weighted_text(official_train),
            "tiers": {tier: official_tier_block(official_train, tier) for tier in TIERS},
            "weighted": json_float(float(official_train["final_score"])),
        }
        selected["train_weighted_official"] = json_float(float(official_train["final_score"]))
    ladder_official = official_score(bundle.inputs, bundle.outcomes, bundle.policy, ladder_sel)
    incumbent_live_ruin = float(rows[0]["fast_views"]["binding"]["ruin_frequency"])
    stage1 = {
        "catalogue": catalogue,
        "decision_if_stop": DECISION_NO_ELIGIBLE if selected is None else None,
        "yardstick_yardstick": yardstick,
        "grid": [_public_config_row(row) for row in rows],
        "n_eligible_raised": int(sum(1 for row in rows if row["eligible_raised"])),
        "selected": None if selected is None else _public_config_row(selected),
        "selected_key": None if selected is None else selected["key"],
        "ladder_fast_ruin_live": json_float(incumbent_live_ruin),
        "ladder_fast_ruin_matches_yardstick": bool(
            abs(incumbent_live_ruin - float(yardstick["train_fast_ruin"])) <= 1e-15
        ),
        "ladder_train": {
            "official": {
                "final_score": official_weighted_text(ladder_official),
                "tiers": {tier: official_tier_block(ladder_official, tier) for tier in TIERS},
            },
            "tiers": ladder_train,
            "weighted_float": json_float(ladder_weighted),
        },
    }
    private = {
        "selected": selected,
        "ladder_dict": ladder_dict,
        "ladder_selection": ladder_sel,
        "yardstick": yardstick,
    }
    return {"private": private, "stage1": sort_mapping(stage1)}


def run_stage2(
    *,
    selected: Optional[Mapping[str, Any]],
    **_kwargs: Any,
) -> Mapping[str, Any]:
    if selected is None:
        raise Stage2Refused("Stage 2 refuses to run when Stage 1 selected nothing")
    raise RuntimeError("run_stage2_on_bundle is the Dev entry; tests use the refuse path")


def run_stage2_on_bundle(
    *,
    selected: Optional[Mapping[str, Any]],
    selected_full: Optional[Mapping[str, Any]],
    ladder_dict: Mapping[str, Any],
    yardstick: Mapping[str, Any],
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    families: Sequence[str],
) -> dict[str, Any]:
    if selected is None or selected_full is None:
        raise Stage2Refused("Stage 2 refuses to run when Stage 1 selected nothing")
    ladder_artifact = load_artifact_mapping(copy.deepcopy(dict(ladder_dict)))
    cache = cache_predictions(inputs.episodes, policy, ladder_artifact)
    arrays = public_arrays(inputs, outcomes, policy)
    costs = np.asarray(arrays.costs, dtype=np.float64)
    config = FastCorridorConfig(predicted_caps_fast=float(selected["predicted_caps_fast"]))
    if config.is_incumbent():
        raise RuntimeError("Stage 2 must not open Dev for the the feasibility ladder incumbent cell")
    _candidate_artifact = load_artifact_mapping(artifact_for_config(ladder_dict, config.as_cap_cert_config()))

    selections: dict[str, dict[str, Tuple[str, ...]]] = {"candidate": {}, "ladder": {}}
    for tier in TIERS:
        selections["ladder"][tier] = tuple(
            d.model_id
            for d in make_submission(inputs, policy, ladder_artifact, tier).submission.decisions
        )
    fast_sel, _ = allocate_frozen(
        cache.rows,
        cap=config.cap("fast"),
        runaway_fraction=config.runaway("fast"),
        max_upgrade_fraction=config.max_upgrade_fraction,
    )
    selections["candidate"]["fast"] = fast_sel
    selections["candidate"]["balanced"] = selections["ladder"]["balanced"]
    selections["candidate"]["premium"] = selections["ladder"]["premium"]
    if any(k1_count(selections["candidate"][tier]) != 0 for tier in TIERS):
        raise ReproductionError("the Fast corridor sweep candidate selected K1")

    official = {
        name: official_score(inputs, outcomes, policy, selections[name])
        for name in ("candidate", "ladder")
    }
    ladder_weighted = float(official["ladder"]["final_score"])
    if abs(ladder_weighted - float(LADDER_DEV_WEIGHTED)) > 5.5e-7:
        raise ReproductionError(
            f"recomputed the feasibility ladder Dev weighted {ladder_weighted} != locked {LADDER_DEV_WEIGHTED}"
        )
    cand_weighted = float(official["candidate"]["final_score"])
    quality_ok = cand_weighted + 1e-15 >= float(LADDER_DEV_WEIGHTED)
    cost_ok = True
    k1_ok = True
    tier_blocks: dict[str, Any] = {}
    for name in ("candidate", "ladder"):
        tier_blocks[name] = {
            "final_score": official_weighted_text(official[name]),
            "tiers": {tier: official_tier_block(official[name], tier) for tier in TIERS},
            "weighted": json_float(float(official[name]["final_score"])),
        }
    cand_tiers = official["candidate"]["tiers"]
    for tier in TIERS:
        realized = float(cand_tiers[tier]["budget_ratio"])
        if realized * float(STRESS_BACKSTOP) > float(OFFICIAL_CAPS[tier]) + 1e-15:
            cost_ok = False
        if int(cand_tiers[tier]["model_counts"].get(_K1, 0)) != 0:
            k1_ok = False

    repeat_ok = True
    shuffle_ok = True
    permute_ok = True
    rng = np.random.default_rng(2026082210)
    shuffled = rng.permutation(len(inputs.episodes))
    id_perm = rng.permutation(len(inputs.episodes))
    again, _ = allocate_frozen(
        cache.rows,
        cap=config.cap("fast"),
        runaway_fraction=config.runaway("fast"),
        max_upgrade_fraction=config.max_upgrade_fraction,
    )
    if again != selections["candidate"]["fast"]:
        repeat_ok = False
    shuf_sel, _ = allocate_frozen(
        _subset_rows(cache.rows, shuffled),
        cap=config.cap("fast"),
        runaway_fraction=config.runaway("fast"),
        max_upgrade_fraction=config.max_upgrade_fraction,
    )
    orig_by_id = _selection_by_id(inputs, selections["candidate"]["fast"])
    shuf_inputs = _shuffle_inputs(inputs, shuffled)
    if orig_by_id != _selection_by_id(shuf_inputs, shuf_sel):
        shuffle_ok = False
    perm_inputs = _permute_ids(inputs, id_perm)
    perm_sel, _ = allocate_frozen(
        cache.rows,
        cap=config.cap("fast"),
        runaway_fraction=config.runaway("fast"),
        max_upgrade_fraction=config.max_upgrade_fraction,
    )
    if _selection_by_digest(inputs, selections["candidate"]["fast"]) != _selection_by_digest(
        perm_inputs, perm_sel
    ):
        permute_ok = False

    views, catalogue = build_stress_views(families)
    fast_stress = sweep_tier_views(
        views,
        cache,
        costs,
        tier="fast",
        cap=config.cap("fast"),
        runaway_fraction=config.runaway("fast"),
        max_upgrade_fraction=config.max_upgrade_fraction,
    )
    ladder_fast_stress = sweep_tier_views(
        views,
        cache,
        costs,
        tier="fast",
        cap=LADDER_FAST_CAP,
        runaway_fraction=LADDER_RUNAWAY,
        max_upgrade_fraction=LADDER_MAX_UPGRADE,
    )
    dev_threshold = float(yardstick["dev_threshold"])
    fast_ruin = float(fast_stress["binding"]["ruin_frequency"])
    fast_ruin_ok = fast_ruin <= dev_threshold + 1e-15
    predicates = {
        "cost_backstop": bool(cost_ok),
        "determinism_invariance": bool(repeat_ok and shuffle_ok and permute_ok),
        "fast_ruin": bool(fast_ruin_ok),
        "k1_off": bool(k1_ok),
        "quality_vs_ladder": bool(quality_ok),
    }
    passed = all(predicates.values())
    return sort_mapping(
        {
            "catalogue": catalogue,
            "determinism": {
                "permute_ids": bool(permute_ok),
                "repeat": bool(repeat_ok),
                "shuffle": bool(shuffle_ok),
            },
            "fast_ruin_limit": json_float(dev_threshold),
            "passed": bool(passed),
            "predicates": predicates,
            "quality_ok": bool(quality_ok),
            "refs": {"ladder": tier_blocks["ladder"]},
            "selected": tier_blocks["candidate"],
            "stress": {
                "candidate_fast": {
                    "binding": fast_stress["binding"],
                    "red_team": {
                        "max_realized": fast_stress["red_team"]["max_realized"],
                        "n": fast_stress["red_team"]["n"],
                        "n_ruin": fast_stress["red_team"]["n_ruin"],
                        "ruin_frequency": fast_stress["red_team"]["ruin_frequency"],
                    },
                },
                "ladder_fast": {
                    "binding": ladder_fast_stress["binding"],
                    "red_team": {
                        "max_realized": ladder_fast_stress["red_team"]["max_realized"],
                        "n": ladder_fast_stress["red_team"]["n"],
                        "n_ruin": ladder_fast_stress["red_team"]["n_ruin"],
                        "ruin_frequency": ladder_fast_stress["red_team"]["ruin_frequency"],
                    },
                },
            },
            "weighted": {
                "candidate": json_float(cand_weighted),
                "ladder": json_float(ladder_weighted),
            },
        }
    )


def assemble_report(
    *,
    identity: Mapping[str, Any],
    locked: Mapping[str, Any],
    stage1: Mapping[str, Any],
    stage2: Optional[Mapping[str, Any]],
    decision: str,
    diagnostic: Mapping[str, Any],
    artifact: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if decision not in DECISIONS:
        raise ValueError(f"unknown the Fast corridor sweep decision {decision!r}")
    dev_opened = stage2 is not None
    if decision == DECISION_NO_ELIGIBLE and dev_opened:
        raise RuntimeError("no-eligible decision must not open Dev")
    return sort_mapping(
        {
            "artifact": artifact,
            "decision": decision,
            "dev_opened": bool(dev_opened),
            "diagnostic": diagnostic,
            "experiment": EXPERIMENT,
            "identity": identity,
            "locked": locked,
            "report_type": REPORT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "stage1": stage1,
            "stage2": stage2,
        }
    )


__all__ = (
    "DECISION_DEV_REJECT",
    "DECISION_NO_ELIGIBLE",
    "DECISION_PROMOTE",
    "EXPERIMENT",
    "FastCorridorConfig",
    "YardstickError",
    "Stage2Refused",
    "assemble_report",
    "evaluate_train_gates",
    "fast_ruin_threshold",
    "load_yardstick_fast_yardstick",
    "locked_record",
    "pre_registered_grid",
    "run_stage1",
    "run_stage2",
    "run_stage2_on_bundle",
    "select_certified",
    "write_selected_artifact",
)
