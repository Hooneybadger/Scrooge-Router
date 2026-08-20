# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the cap certification layer — certify raised the feasibility ladder/the recalibrated router numeric policy caps. No new head, no refit.

Reuses the prefix certificate layer view constructors and the density ordering layer binding/red-team split. Drives the
frozen ``ossp_router.feasibility_ladder`` selection path against a re-parameterized
artifact. Dev is never opened from this module; the runner injects an
already-loaded Dev batch only after Stage 1 is on disk.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.heuristic import episode_text
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Episode,
    InputBatch,
    RoutingPolicy,
)
from ossp_router.feasibility_ladder import (
    FINITE_COMPARE,
    LadderArtifact,
    apply_runaway_guard,
    apply_upgrade_count_cap,
    load_artifact_mapping,
    load_bundled_artifact,
    make_submission,
    predict_fast_balanced_row,
    select_fast_balanced,
    spend_ratio,
)
from ossp_router.cost_calibrated_router import _premium_prediction, _select_ax31
from research.lab.modeling import (
    OFFICIAL_CAPS,
    STRESS_BACKSTOP,
    TrainBundle,
    official_score,
    sort_mapping,
    weighted_final,
)
from research.lab.prefix_certificates import (
    DIRICHLET_ALPHA,
    DIRICHLET_N,
    FAMDOM_DUP_CAP,
    FAMDOM_SHARE,
    FAMDOM_TARGET_N,
    HALF_N,
    MIN_VIEW_N,
    View,
    _dirichlet_counts,
    _famdom_sizes,
    _realized_ratio,
    _sample_capped,
    family_folds,
    json_float,
)
from research.lab.density_ordering import view_layer


EXPERIMENT = "the cap certification layer"
REPORT_TYPE = "scrooge-cap_cert-raised-caps-v1"
SCHEMA_VERSION = 1
DECISION_PROMOTE = "record-cap_cert-promote-raised-caps"
DECISION_NO_ELIGIBLE = "record-cap_cert-close-no-eligible-config"
DECISION_DEV_REJECT = "record-cap_cert-close-dev-reject"
LADDER_DEV_WEIGHTED = 0.665341
RECALIBRATED_DEV_WEIGHTED = 0.668324
FAST_DRIFT_COEF = 2.2235
NEAR_BUDGET_FRAC = 0.95
RUIN_FREQ_MAX = 0.0025
FAMDOM_SEED = 2026082204
DIRICHLET_SEED = 2026082205
HALF_SEED = 2026082206
SMALL_SEED = 2026082207
FAMDOM_DRAWS = 500
DIRICHLET_DRAWS = 500
HALF_DRAWS = 20
SMALL_BINDING_SIZES: Tuple[int, ...] = (300, 880)
SMALL_REDTEAM_SIZE = 100
SMALL_DRAWS = 100
GRID_FAST: Tuple[float, ...] = (1.03, 1.05, 1.07, 1.08, 1.09)
GRID_BALANCED: Tuple[float, ...] = (1.50, 1.65, 1.80, 1.90)
GRID_MAX_UPGRADE: Tuple[float, ...] = (0.75, 0.85, 0.90, 0.95)
PREMIUM_CAP = 3.25
LADDER_FAST_CAP = 1.03
LADDER_BALANCED_CAP = 1.50
LADDER_MAX_UPGRADE = 0.75
LADDER_RUNAWAY = 0.05
RECALIBRATED_FAST_CAP = 1.07
_LIGHT = MODEL_IDS[0]
_AX31 = MODEL_IDS[1]
_K1 = MODEL_IDS[2]
ROOT = Path(__file__).resolve().parents[2]
LADDER_ARTIFACT_PATH = ROOT / "src" / "ossp_router" / "resources" / "feasibility-ladder.v1.json"
RECALIBRATED_ARTIFACT_PATH = ROOT / "research" / "artifacts" / "recalibrated-router.v1.json"
OVERRIDE_PATHS = (
    ("predicted_caps", "fast"),
    ("predicted_caps", "balanced"),
    ("max_upgrade_fraction",),
    ("runaway_fraction",),
)
PREDICATE_TEXTS: Tuple[str, ...] = (
    "Train_realized * 1.054 <= official limit",
    "1 + 2.2235 * (Train_realized - 1) <= official_limit / 1.054",
    "projected Dev realized < 0.95 * official limit",
    "non-vacuity: the config must change the selection versus the feasibility ladder",
    "binding-layer ruin frequency <= 0.25% per tier, "
    "where ruin means the view's realized ratio exceeds that tier's official limit",
)
SELECTION_RULE = (
    "Among configs passing all five predicates, maximize Train weighted "
    "quality with the official Decimal scorer; tie-break toward smaller "
    "predicted_caps.fast, then smaller predicted_caps.balanced, then "
    "smaller max_upgrade_fraction. Quality is essentially monotone in "
    "upgrade count, so this reduces to loosest certified parameters. "
    "the feasibility ladder's head is full-fit on Train so Train quality is in-sample, "
    "used only to order already-certified configs."
)
DERIVED_GUARD_FORMULA = "max(0.05, 1.5 * (predicted_caps.<tier> - 1))"
VIEW_ROLE = (
    "These views are cost-safety designs for a fixed policy under "
    "composition shift. The the feasibility ladder head is full-fit on Train and is not "
    "refit, so the views are not generalization folds."
)
BINDING_LAYER_REASON = (
    "lofo-{family} single-family batches, lofo-combined, and small-100 "
    "are red-team only, never binding (charter §14.4). Binding layer is "
    "famdom-{family} at 75%, dirichlet, half, and small-{300,880}."
)


class Stage2Refused(RuntimeError):
    """Stage 2 must not run when Stage 1 selected nothing."""


class ReproductionError(RuntimeError):
    """the feasibility ladder/the recalibrated router injection reproduction failed; fail closed."""


@dataclass(frozen=True)
class CapConfig:
    predicted_caps_fast: float
    predicted_caps_balanced: float
    max_upgrade_fraction: float

    @property
    def key(self) -> str:
        return (
            f"fast={self.predicted_caps_fast:.2f}"
            f"|balanced={self.predicted_caps_balanced:.2f}"
            f"|max_upgrade={self.max_upgrade_fraction:.2f}"
        )

    def label(self) -> str:
        if (
            self.predicted_caps_fast == LADDER_FAST_CAP
            and self.predicted_caps_balanced == LADDER_BALANCED_CAP
            and self.max_upgrade_fraction == LADDER_MAX_UPGRADE
        ):
            return "the feasibility ladder"
        if (
            self.predicted_caps_fast == RECALIBRATED_FAST_CAP
            and self.predicted_caps_balanced == LADDER_BALANCED_CAP
            and self.max_upgrade_fraction == LADDER_MAX_UPGRADE
        ):
            return "the recalibrated router"
        return "grid"

    def cap(self, tier: str) -> float:
        if tier == "fast":
            return float(self.predicted_caps_fast)
        if tier == "balanced":
            return float(self.predicted_caps_balanced)
        return float(PREMIUM_CAP)

    def runaway(self, tier: str) -> float:
        if tier == "premium":
            return derived_runaway_fraction(PREMIUM_CAP)
        return derived_runaway_fraction(self.cap(tier))


@dataclass(frozen=True)
class PredictionCache:
    uplift: np.ndarray
    pred_light: np.ndarray
    pred_ax31: np.ndarray
    premium_uplift: np.ndarray
    premium_costs: np.ndarray
    digests: Tuple[str, ...]
    rows: Tuple[Tuple[float, Tuple[float, float]], ...]
    premium_rows: Tuple[Tuple[float, Tuple[float, float, float]], ...]


def json_floats(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [json_float(item) for item in array]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_frozen_artifact_dict(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def derived_runaway_fraction(predicted_cap: float) -> float:
    """Prediction-independent tail guard; monotone in the cap and >= 0.05."""

    return max(0.05, 1.5 * (float(predicted_cap) - 1.0))


def _get_path(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def _set_path(value: dict[str, Any], path: Sequence[str], new_value: Any) -> None:
    current: Any = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = new_value


def _strip_overrides(value: Mapping[str, Any]) -> Any:
    stripped = copy.deepcopy(value)
    del stripped["max_upgrade_fraction"]
    del stripped["runaway_fraction"]
    del stripped["predicted_caps"]["fast"]
    del stripped["predicted_caps"]["balanced"]
    return stripped


def reparameterize_artifact(
    base: Mapping[str, Any],
    *,
    predicted_caps_fast: float,
    predicted_caps_balanced: float,
    max_upgrade_fraction: float,
    runaway_fraction: float,
) -> dict[str, Any]:
    """Override only the four numeric policy fields. Fail closed otherwise."""

    if not isinstance(base, Mapping):
        raise ValueError("frozen artifact must be an object")
    updated = copy.deepcopy(dict(base))
    _set_path(updated, ("predicted_caps", "fast"), float(predicted_caps_fast))
    _set_path(updated, ("predicted_caps", "balanced"), float(predicted_caps_balanced))
    _set_path(updated, ("max_upgrade_fraction",), float(max_upgrade_fraction))
    _set_path(updated, ("runaway_fraction",), float(runaway_fraction))
    left = json.dumps(_strip_overrides(base), sort_keys=True, ensure_ascii=False)
    right = json.dumps(_strip_overrides(updated), sort_keys=True, ensure_ascii=False)
    if left != right:
        raise ValueError("reparameterize_artifact changed a frozen key")
    touched = {
        "predicted_caps.fast",
        "predicted_caps.balanced",
        "max_upgrade_fraction",
        "runaway_fraction",
    }
    if touched != {
        "predicted_caps.fast",
        "predicted_caps.balanced",
        "max_upgrade_fraction",
        "runaway_fraction",
    }:
        raise ValueError("reparameterize_artifact override set drifted")
    load_artifact_mapping(updated)
    return updated


def field_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            keys = sorted(set(left) | set(right))
            for key in keys:
                child = f"{path}.{key}" if path else str(key)
                if key not in left:
                    rows.append({"path": child, "before": None, "after": right[key]})
                elif key not in right:
                    rows.append({"path": child, "before": left[key], "after": None})
                else:
                    walk(left[key], right[key], child)
            return
        if left != right:
            rows.append({"path": path, "before": left, "after": right})

    walk(before, after, "")
    return rows


def pre_registered_grid() -> Tuple[CapConfig, ...]:
    configs = []
    for fast in GRID_FAST:
        for balanced in GRID_BALANCED:
            for max_upgrade in GRID_MAX_UPGRADE:
                configs.append(
                    CapConfig(
                        predicted_caps_fast=float(fast),
                        predicted_caps_balanced=float(balanced),
                        max_upgrade_fraction=float(max_upgrade),
                    )
                )
    if len(configs) != 80:
        raise RuntimeError(f"the cap certification layer grid must be 80 configs; got {len(configs)}")
    return tuple(configs)


def artifact_for_config(base: Mapping[str, Any], config: CapConfig) -> dict[str, Any]:
    return reparameterize_artifact(
        base,
        predicted_caps_fast=config.predicted_caps_fast,
        predicted_caps_balanced=config.predicted_caps_balanced,
        max_upgrade_fraction=config.max_upgrade_fraction,
        runaway_fraction=derived_runaway_fraction(config.predicted_caps_fast),
    )


def cache_predictions(
    episodes: Sequence[Episode],
    policy: RoutingPolicy,
    artifact: LadderArtifact,
) -> PredictionCache:
    rows = []
    premium_rows = []
    digests = []
    for episode in episodes:
        rows.append(predict_fast_balanced_row(episode, policy, artifact))
        premium_rows.append(_premium_prediction(episode, policy, artifact))
        digests.append(hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest())
    uplift = np.asarray([row[0] for row in rows], dtype=np.float64)
    pred_light = np.asarray([row[1][0] for row in rows], dtype=np.float64)
    pred_ax31 = np.asarray([row[1][1] for row in rows], dtype=np.float64)
    premium_uplift = np.asarray([row[0] for row in premium_rows], dtype=np.float64)
    premium_costs = np.asarray([row[1] for row in premium_rows], dtype=np.float64)
    return PredictionCache(
        uplift=uplift,
        pred_light=pred_light,
        pred_ax31=pred_ax31,
        premium_uplift=premium_uplift,
        premium_costs=premium_costs,
        digests=tuple(digests),
        rows=tuple(rows),
        premium_rows=tuple(premium_rows),
    )


def allocate_frozen(
    rows: Sequence[Tuple[float, Tuple[float, float]]],
    *,
    cap: float,
    runaway_fraction: float,
    max_upgrade_fraction: float,
) -> Tuple[Tuple[str, ...], float]:
    """Drive the real feasibility_ladder Fast/Balanced selection path."""

    return select_fast_balanced(
        rows,
        cap=float(cap),
        runaway_fraction=float(runaway_fraction),
        max_upgrade_fraction=float(max_upgrade_fraction),
    )


def allocate_numpy(
    uplift: np.ndarray,
    pred_light: np.ndarray,
    pred_ax31: np.ndarray,
    *,
    cap: float,
    runaway_fraction: float,
    max_upgrade_fraction: float,
) -> Tuple[np.ndarray, float, str]:
    """Vectorized Fast/Balanced allocator matching ``select_fast_balanced``.

    Used for view sweeps. Stage 1a and reproduction use the frozen path.
    """

    n_rows = int(uplift.size)
    light = np.asarray(pred_light, dtype=np.float64).reshape(-1)
    ax31 = np.asarray(pred_ax31, dtype=np.float64).reshape(-1)
    quality = np.asarray(uplift, dtype=np.float64).reshape(-1)
    if light.size != n_rows or ax31.size != n_rows:
        raise ValueError("allocate_numpy requires aligned arrays")
    light_total = float(math.fsum(light.tolist()))
    if (not math.isfinite(light_total)) or light_total <= 0.0:
        raise ValueError("light denominator is not positive")
    threshold = float(runaway_fraction) * light_total
    raw_inc = ax31 - light
    runaway_hit = raw_inc > threshold
    guarded_inc = np.where(runaway_hit, 0.0, raw_inc)
    n_eligible = int(np.sum((quality > 0.0) & (raw_inc > 0.0)))
    n_after_runaway = int(np.sum((quality > 0.0) & (guarded_inc > 0.0)))
    density = np.full(n_rows, -np.inf, dtype=np.float64)
    eligible = (quality > 0.0) & (guarded_inc > 0.0)
    density[eligible] = quality[eligible] * light_total / guarded_inc[eligible]
    order = np.argsort(-density, kind="stable")
    budget = float(cap) * light_total
    used = 0.0
    selected = np.zeros(n_rows, dtype=bool)
    cursor = 0
    while cursor < n_rows:
        rank = density[order[cursor]]
        if not math.isfinite(float(rank)):
            break
        end = cursor + 1
        while end < n_rows and density[order[end]] == rank:
            end += 1
        group = order[cursor:end]
        group_cost = float(math.fsum(guarded_inc[group].tolist()))
        if math.fsum((light_total, used, group_cost)) > budget:
            break
        selected[group] = True
        used = math.fsum((used, group_cost))
        cursor = end
    n_after_cap = int(selected.sum())
    maximum = int(math.floor(float(max_upgrade_fraction) * float(n_rows)))
    extra = n_after_cap - maximum
    if extra > 0:
        ax31_idx = np.flatnonzero(selected)
        increments = guarded_inc[ax31_idx]
        demote = np.lexsort((np.arange(ax31_idx.size), -increments))
        selected[ax31_idx[demote[:extra]]] = False
    n_after_count = int(selected.sum())
    spend = light_total + float(math.fsum(guarded_inc[selected].tolist()))
    pred_ratio = spend / light_total
    fallback = False
    if pred_ratio > float(cap) + FINITE_COMPARE:
        selected[:] = False
        pred_ratio = 1.0
        fallback = True
    if fallback:
        bound = "cost cap"
    elif n_after_count < n_after_cap:
        bound = "upgrade-count cap"
    elif n_after_cap < n_after_runaway:
        bound = "cost cap"
    elif n_after_runaway < n_eligible:
        bound = "runaway guard"
    else:
        bound = "none"
    models = np.where(selected, _AX31, _LIGHT)
    return models, float(pred_ratio), bound


def binding_constraint_frozen(
    rows: Sequence[Tuple[float, Tuple[float, float]]],
    selected: Sequence[str],
    *,
    cap: float,
    runaway_fraction: float,
    max_upgrade_fraction: float,
) -> str:
    costs = tuple(pair for _uplift, pair in rows)
    guarded = apply_runaway_guard(costs, runaway_fraction)
    ranked = tuple((rows[index][0], guarded[index]) for index in range(len(rows)))
    after_cap, _ratio = _select_ax31(ranked, cap)
    after_count = apply_upgrade_count_cap(after_cap, guarded, max_upgrade_fraction)
    pred_ratio = spend_ratio(guarded, after_count)
    fallback = pred_ratio > float(cap) + FINITE_COMPARE
    n_eligible = sum(
        1
        for uplift, (light, ax31) in rows
        if uplift > 0.0 and (ax31 - light) > 0.0
    )
    n_after_runaway = sum(
        1
        for (uplift, _raw), (light, ax31) in zip(rows, guarded)
        if uplift > 0.0 and (ax31 - light) > 0.0
    )
    n_after_cap = sum(1 for model_id in after_cap if model_id == _AX31)
    n_after_count = sum(1 for model_id in after_count if model_id == _AX31)
    n_final = sum(1 for model_id in selected if model_id == _AX31)
    if fallback or (n_final == 0 and n_after_count > 0):
        return "cost cap"
    if n_after_count < n_after_cap:
        return "upgrade-count cap"
    if n_after_cap < n_after_runaway:
        return "cost cap"
    if n_after_runaway < n_eligible:
        return "runaway guard"
    return "none"


def projected_dev_realized(train_realized: float) -> float:
    return 1.0 + float(FAST_DRIFT_COEF) * (float(train_realized) - 1.0)


def evaluate_predicates(
    *,
    train_realized: Mapping[str, float],
    selection: Mapping[str, Sequence[str]],
    ladder_selection: Mapping[str, Sequence[str]],
    ruin_frequency: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Evaluate the five pre-registered predicates independently."""

    per_tier: dict[str, dict[str, Any]] = {}
    p1 = True
    p2 = True
    p3 = True
    for tier in TIERS:
        realized = float(train_realized[tier])
        official = float(OFFICIAL_CAPS[tier])
        projected = projected_dev_realized(realized)
        backstop_ok = realized * float(STRESS_BACKSTOP) <= official + 1e-15
        drift_ok = projected <= (official / float(STRESS_BACKSTOP)) + 1e-15
        near_ok = projected < float(NEAR_BUDGET_FRAC) * official
        per_tier[tier] = {
            "p1_backstop": bool(backstop_ok),
            "p2_drift": bool(drift_ok),
            "p3_projected_near": bool(near_ok),
            "projected_dev_realized": json_float(projected),
            "realized": json_float(realized),
        }
        p1 = p1 and backstop_ok
        p2 = p2 and drift_ok
        p3 = p3 and near_ok
    p4 = any(tuple(selection[tier]) != tuple(ladder_selection[tier]) for tier in TIERS)
    p5: Optional[bool]
    ruin_detail: dict[str, Any] = {}
    if ruin_frequency is None:
        p5 = None
    else:
        p5 = True
        for tier in TIERS:
            freq = float(ruin_frequency[tier])
            ok = freq <= float(RUIN_FREQ_MAX) + 1e-15
            ruin_detail[tier] = {"frequency": json_float(freq), "ok": bool(ok)}
            p5 = bool(p5 and ok)
    failed = []
    if not p1:
        failed.append("p1")
    if not p2:
        failed.append("p2")
    if not p3:
        failed.append("p3")
    if not p4:
        failed.append("p4")
    if p5 is False:
        failed.append("p5")
    eligible = bool(p1 and p2 and p3 and p4 and (p5 is True))
    stage1a_eligible = bool(p1 and p2 and p3 and p4)
    return {
        "eligible": eligible,
        "failed": failed,
        "p1": bool(p1),
        "p2": bool(p2),
        "p3": bool(p3),
        "p4": bool(p4),
        "p5": p5,
        "per_tier": per_tier,
        "ruin": ruin_detail,
        "stage1a_eligible": stage1a_eligible,
        "vacuous": (not p4),
    }


def ruin_frequency(n_ruin: int, n_views: int) -> float:
    if int(n_views) <= 0:
        return 0.0
    return float(n_ruin) / float(n_views)


def ruin_ok(frequency: float) -> bool:
    return float(frequency) <= float(RUIN_FREQ_MAX) + 1e-15


def score_mean(scores: np.ndarray, model_ids: Sequence[str]) -> float:
    columns = np.asarray([MODEL_IDS.index(model_id) for model_id in model_ids], dtype=np.int64)
    rows = np.arange(int(scores.shape[0]), dtype=np.int64)
    return json_float(float(scores[rows, columns].mean()))


def k1_count(model_ids: Sequence[str]) -> int:
    return int(sum(1 for model_id in model_ids if model_id == _K1))


def ax31_count(model_ids: Sequence[str]) -> int:
    return int(sum(1 for model_id in model_ids if model_id == _AX31))


def build_stress_views(families: Sequence[str]) -> Tuple[Tuple[View, ...], dict[str, Any]]:
    """the cap certification layer catalogue. Imports the prefix certificate layer resampling helpers; does not call build_views."""

    fam = np.asarray(list(families))
    names = tuple(sorted(dict.fromkeys(fam.tolist())))
    n_train = int(fam.size)
    views: list[View] = []
    skipped: list[dict[str, Any]] = []

    def _accept(view: View) -> None:
        if int(view.index.size) < MIN_VIEW_N:
            skipped.append(
                {
                    "kind": view.kind,
                    "n": int(view.index.size),
                    "name": view.name,
                    "reason": "n<20",
                }
            )
            return
        views.append(view)

    for name, index in family_folds(families):
        held = np.asarray(index, dtype=np.int64)
        _accept(View("lofo", f"lofo-{name}", held, "oof"))
        _accept(View("lofo-combined", f"lofo-combined-{name}", held.copy(), "oof"))

    fam_rng = np.random.default_rng(int(FAMDOM_SEED))
    famdom_fallback: dict[str, str] = {}
    for name in names:
        focus = np.flatnonzero(fam == name)
        other = np.flatnonzero(fam != name)
        n_focus, n_other, _n_batch, fallback = _famdom_sizes(int(focus.size), int(other.size))
        famdom_fallback[name] = fallback
        for draw in range(FAMDOM_DRAWS):
            chosen_focus, tag_f = _sample_capped(fam_rng, focus, n_focus)
            chosen_other, tag_o = _sample_capped(fam_rng, other, n_other)
            tag = fallback
            if tag_f is not None or tag_o is not None:
                tag = "+".join(item for item in (fallback, tag_f, tag_o) if item and item != "none")
            chosen = np.concatenate([chosen_focus, chosen_other])
            _accept(View("famdom", f"famdom-{name}-{draw:03d}", chosen, "oof", tag))

    dir_rng = np.random.default_rng(int(DIRICHLET_SEED))
    pools = {name: np.flatnonzero(fam == name) for name in names}
    for draw in range(DIRICHLET_DRAWS):
        counts = _dirichlet_counts(dir_rng, len(names), DIRICHLET_N, DIRICHLET_ALPHA)
        parts: list[np.ndarray] = []
        leftover = 0
        capacity = []
        for name, count in zip(names, counts):
            take, _tag = _sample_capped(dir_rng, pools[name], int(count))
            parts.append(take)
            short = int(count) - int(take.size)
            leftover += max(0, short)
            cap_left = FAMDOM_DUP_CAP * int(pools[name].size) - int(take.size)
            capacity.append(max(0, cap_left))
        if leftover > 0:
            for name, cap_left in zip(names, capacity):
                if leftover <= 0 or cap_left <= 0:
                    continue
                extra, _tag = _sample_capped(dir_rng, pools[name], min(leftover, cap_left))
                parts.append(extra)
                leftover -= int(extra.size)
        chosen = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        tag = "dirichlet-redistribute" if leftover > 0 or int(chosen.size) != DIRICHLET_N else None
        _accept(View("dirichlet", f"dirichlet-{draw:04d}", chosen, "oof", tag))

    half_rng = np.random.default_rng(int(HALF_SEED))
    universe = np.arange(n_train, dtype=np.int64)
    half_n = min(int(HALF_N), n_train)
    for draw in range(HALF_DRAWS):
        chosen = half_rng.choice(universe, size=int(half_n), replace=False)
        _accept(View("half", f"half-{draw:02d}", chosen, "oof"))

    small_rng = np.random.default_rng(int(SMALL_SEED))
    for size in (SMALL_REDTEAM_SIZE, *SMALL_BINDING_SIZES):
        take = min(int(size), n_train)
        for draw in range(SMALL_DRAWS):
            chosen = small_rng.choice(universe, size=int(take), replace=False)
            _accept(View("small", f"small-{size}-{draw:03d}", chosen, "oof"))

    kind_counts = {
        kind: int(sum(1 for view in views if view.kind == kind))
        for kind in ("lofo", "lofo-combined", "famdom", "dirichlet", "half", "small")
    }
    layer_counts = {"binding": 0, "red-team": 0}
    for view in views:
        layer_counts[view_layer(view)] += 1
    catalogue = {
        "binding_layer_reason": BINDING_LAYER_REASON,
        "dirichlet_alpha": json_float(DIRICHLET_ALPHA),
        "dirichlet_draws": int(DIRICHLET_DRAWS),
        "dirichlet_n": int(DIRICHLET_N),
        "famdom_draws": int(FAMDOM_DRAWS),
        "famdom_fallback_by_family": famdom_fallback,
        "famdom_share": json_float(FAMDOM_SHARE),
        "famdom_target_n": int(FAMDOM_TARGET_N),
        "half_draws": int(HALF_DRAWS),
        "layer_counts": layer_counts,
        "n_skipped": int(len(skipped)),
        "n_views": int(len(views)),
        "role": VIEW_ROLE,
        "small_draws": int(SMALL_DRAWS),
        "view_kind_counts": kind_counts,
        "view_role": VIEW_ROLE,
    }
    return tuple(views), catalogue


def select_premium_cached(
    digests: Sequence[str],
    uplift: np.ndarray,
    costs: np.ndarray,
    cap_ratio: float,
) -> Tuple[Tuple[str, ...], float]:
    """Same heap allocator as frozen ``_select_premium``, hashes precomputed."""

    grouped: dict[str, list[int]] = {}
    for index, digest in enumerate(digests):
        grouped.setdefault(str(digest), []).append(index)
    names = sorted(grouped)
    rows = [grouped[name] for name in names]
    quality = [
        (
            0.0,
            math.fsum(float(uplift[index]) for index in indexes),
            -1_000_000.0 * len(indexes),
        )
        for indexes in rows
    ]
    group_costs = [
        tuple(math.fsum(float(costs[index, model]) for index in indexes) for model in range(3))
        for indexes in rows
    ]
    states = [0] * len(names)
    versions = [0] * len(names)
    light_total = math.fsum(item[0] for item in group_costs)
    budget = float(cap_ratio) * light_total
    total_cost = light_total
    queue: list[Tuple[float, float, str, int, int, int, float, int]] = []

    def push_upgrades(group_index: int) -> None:
        source = states[group_index]
        for target in range(1, 3):
            if target == source:
                continue
            incremental = group_costs[group_index][target] - group_costs[group_index][source]
            gain = quality[group_index][target] - quality[group_index][source]
            if incremental <= 0.0 or gain <= 0.0:
                continue
            heapq.heappush(
                queue,
                (
                    -(gain / incremental),
                    -gain,
                    names[group_index],
                    target,
                    versions[group_index],
                    group_index,
                    incremental,
                    source,
                ),
            )

    for group_index in range(len(names)):
        push_upgrades(group_index)
    while queue:
        _density, _gain, _name, target, version, group_index, incremental, source = heapq.heappop(
            queue
        )
        if version != versions[group_index] or states[group_index] != source:
            continue
        if total_cost + incremental > budget + 1e-12:
            continue
        states[group_index] = target
        versions[group_index] += 1
        total_cost += incremental
        push_upgrades(group_index)
    selected = [_LIGHT] * int(uplift.size)
    for group_index, indexes in enumerate(rows):
        for index in indexes:
            selected[index] = MODEL_IDS[states[group_index]]
    return tuple(selected), total_cost / light_total if light_total else 1.0


def _subset_rows(
    rows: Sequence[Tuple[float, Tuple[float, float]]], index: np.ndarray
) -> Tuple[Tuple[float, Tuple[float, float]], ...]:
    return tuple(rows[int(item)] for item in np.asarray(index, dtype=np.int64))


def sweep_tier_views(
    views: Sequence[View],
    cache: PredictionCache,
    actual_costs: np.ndarray,
    *,
    tier: str,
    cap: float,
    runaway_fraction: float,
    max_upgrade_fraction: float,
) -> dict[str, Any]:
    official = float(OFFICIAL_CAPS[tier])
    binding_n = 0
    binding_ruin = 0
    red_n = 0
    red_ruin = 0
    per_kind: dict[str, dict[str, Any]] = {}
    lofo_rows: list[dict[str, Any]] = []
    worst_binding = 0.0
    worst_red = 0.0
    for view in views:
        layer = view_layer(view)
        index = np.asarray(view.index, dtype=np.int64)
        if tier == "premium":
            digests = tuple(cache.digests[int(item)] for item in index)
            models, _pred = select_premium_cached(
                digests,
                cache.premium_uplift[index],
                cache.premium_costs[index],
                float(cap),
            )
            model_tuple = models
        else:
            models_arr, _pred, _bound = allocate_numpy(
                cache.uplift[index],
                cache.pred_light[index],
                cache.pred_ax31[index],
                cap=float(cap),
                runaway_fraction=float(runaway_fraction),
                max_upgrade_fraction=float(max_upgrade_fraction),
            )
            model_tuple = tuple(models_arr.tolist())
        realized = float(_realized_ratio(actual_costs[index], model_tuple))
        ruined = realized > official + 1e-15
        bucket = per_kind.setdefault(
            view.kind, {"n": 0, "n_ruin": 0, "max_realized": 0.0}
        )
        bucket["n"] += 1
        bucket["n_ruin"] += int(ruined)
        bucket["max_realized"] = max(float(bucket["max_realized"]), realized)
        if layer == "binding":
            binding_n += 1
            binding_ruin += int(ruined)
            worst_binding = max(worst_binding, realized)
        else:
            red_n += 1
            red_ruin += int(ruined)
            worst_red = max(worst_red, realized)
            if view.kind in ("lofo", "lofo-combined") or view.name.startswith("small-100-"):
                lofo_rows.append(
                    {
                        "name": view.name,
                        "realized": json_float(realized),
                        "ruin": bool(ruined),
                    }
                )
    return {
        "binding": {
            "max_realized": json_float(worst_binding),
            "n": int(binding_n),
            "n_ruin": int(binding_ruin),
            "ruin_frequency": json_float(ruin_frequency(binding_ruin, binding_n)),
        },
        "per_kind": {
            kind: {
                "max_realized": json_float(row["max_realized"]),
                "n": int(row["n"]),
                "n_ruin": int(row["n_ruin"]),
                "ruin_frequency": json_float(ruin_frequency(int(row["n_ruin"]), int(row["n"]))),
            }
            for kind, row in sorted(per_kind.items())
        },
        "red_team": {
            "max_realized": json_float(worst_red),
            "n": int(red_n),
            "n_ruin": int(red_ruin),
            "rows": lofo_rows,
            "ruin_frequency": json_float(ruin_frequency(red_ruin, red_n)),
        },
    }


def official_weighted_text(report: Mapping[str, Any]) -> str:
    return str(report["final_score"])


def official_tier_block(report: Mapping[str, Any], tier: str) -> dict[str, Any]:
    row = report["tiers"][tier]
    return {
        "ax31": int(row["model_counts"].get(_AX31, 0)),
        "axk1_think": int(row["model_counts"].get(_K1, 0)),
        "budget_passed": bool(row["budget_passed"]),
        "budget_ratio": row["budget_ratio"],
        "light": int(row["model_counts"].get(_LIGHT, 0)),
        "near_budget": bool(row["near_budget"]),
        "quality_score": row["quality_score"],
        "realized": row["budget_ratio"],
        "realized_times_1054": json_float(float(row["budget_ratio"]) * float(STRESS_BACKSTOP)),
        "tier_score": row["tier_score"],
    }


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "binding_layer_reason": BINDING_LAYER_REASON,
            "derived_guard_formula": DERIVED_GUARD_FORMULA,
            "dirichlet_draws": int(DIRICHLET_DRAWS),
            "dirichlet_seed": int(DIRICHLET_SEED),
            "famdom_draws": int(FAMDOM_DRAWS),
            "famdom_seed": int(FAMDOM_SEED),
            "fast_drift_coef": json_float(FAST_DRIFT_COEF),
            "grid_balanced": [json_float(item) for item in GRID_BALANCED],
            "grid_fast": [json_float(item) for item in GRID_FAST],
            "grid_max_upgrade_fraction": [json_float(item) for item in GRID_MAX_UPGRADE],
            "half_draws": int(HALF_DRAWS),
            "half_seed": int(HALF_SEED),
            "k1": "off",
            "n_configs": 80,
            "near_budget_frac": json_float(NEAR_BUDGET_FRAC),
            "official_caps": {tier: json_float(OFFICIAL_CAPS[tier]) for tier in TIERS},
            "predicates": list(PREDICATE_TEXTS),
            "premium_cap": json_float(PREMIUM_CAP),
            "premium_untouched": True,
            "ruin_frequency_max": json_float(RUIN_FREQ_MAX),
            "selection_rule": SELECTION_RULE,
            "small_binding_sizes": list(SMALL_BINDING_SIZES),
            "small_draws": int(SMALL_DRAWS),
            "small_redteam_size": int(SMALL_REDTEAM_SIZE),
            "small_seed": int(SMALL_SEED),
            "stress_backstop": json_float(STRESS_BACKSTOP),
            "recal_grid": {
                "max_upgrade_fraction": json_float(LADDER_MAX_UPGRADE),
                "predicted_caps_balanced": json_float(LADDER_BALANCED_CAP),
                "predicted_caps_fast": json_float(RECALIBRATED_FAST_CAP),
            },
            "ladder_grid": {
                "max_upgrade_fraction": json_float(LADDER_MAX_UPGRADE),
                "predicted_caps_balanced": json_float(LADDER_BALANCED_CAP),
                "predicted_caps_fast": json_float(LADDER_FAST_CAP),
            },
            "view_role": VIEW_ROLE,
        }
    )


def _config_sort_key(row: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (
        -float(row["train_weighted_float"]),
        float(row["predicted_caps_fast"]),
        float(row["predicted_caps_balanced"]),
        float(row["max_upgrade_fraction"]),
    )


def select_certified(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    certified = [row for row in rows if row.get("eligible") is True]
    if not certified:
        return None
    ordered = sorted(certified, key=_config_sort_key)
    return ordered[0]


def run_stage2(
    *,
    selected: Optional[Mapping[str, Any]],
    **_kwargs: Any,
) -> Mapping[str, Any]:
    """Refuse unless Stage 1 recorded a selected config."""

    if selected is None:
        raise Stage2Refused("Stage 2 refuses to run when Stage 1 selected nothing")
    raise RuntimeError("run_stage2_on_bundle is the Dev entry; tests use the refuse path")


def _reproduce_references(
    bundle: TrainBundle,
    cache: PredictionCache,
    ladder_artifact: LadderArtifact,
    ladder_dict: Mapping[str, Any],
    recal_dict: Mapping[str, Any],
) -> dict[str, Any]:
    from ossp_router.recalibrated_router import (
        load_bundled_artifact as load_recal_bundled,
        make_submission as recal_make_submission,
    )

    ladder_bundled = load_bundled_artifact()
    ladder_injected = load_artifact_mapping(copy.deepcopy(dict(ladder_dict)))
    recal_like = load_artifact_mapping(
        reparameterize_artifact(
            ladder_dict,
            predicted_caps_fast=RECALIBRATED_FAST_CAP,
            predicted_caps_balanced=float(ladder_dict["predicted_caps"]["balanced"]),
            max_upgrade_fraction=float(ladder_dict["max_upgrade_fraction"]),
            runaway_fraction=float(ladder_dict["runaway_fraction"]),
        )
    )
    ladder_sel: dict[str, Tuple[str, ...]] = {}
    ladder_inj: dict[str, Tuple[str, ...]] = {}
    recal_from_ladder: dict[str, Tuple[str, ...]] = {}
    recal_native: dict[str, Tuple[str, ...]] = {}
    recal_artifact = load_recal_bundled()
    for tier in TIERS:
        bundled = make_submission(bundle.inputs, bundle.policy, ladder_bundled, tier)
        injected = make_submission(bundle.inputs, bundle.policy, ladder_injected, tier)
        ladder_sel[tier] = tuple(d.model_id for d in bundled.submission.decisions)
        ladder_inj[tier] = tuple(d.model_id for d in injected.submission.decisions)
        if ladder_sel[tier] != ladder_inj[tier]:
            raise ReproductionError(
                f"unmodified the feasibility ladder injection disagrees on {tier}: "
                f"bundled vs injected selection mismatch"
            )
        recal_plan = recal_make_submission(bundle.inputs, bundle.policy, recal_artifact, tier)
        recal_native[tier] = tuple(d.model_id for d in recal_plan.submission.decisions)
        injected_recal = make_submission(bundle.inputs, bundle.policy, recal_like, tier)
        recal_from_ladder[tier] = tuple(d.model_id for d in injected_recal.submission.decisions)
    cached_ladder_fast, _ = allocate_frozen(
        cache.rows,
        cap=LADDER_FAST_CAP,
        runaway_fraction=LADDER_RUNAWAY,
        max_upgrade_fraction=LADDER_MAX_UPGRADE,
    )
    if cached_ladder_fast != ladder_sel["fast"]:
        raise ReproductionError("cached the feasibility ladder Fast allocation disagrees with make_submission")
    numpy_models, _pred, _bound = allocate_numpy(
        cache.uplift,
        cache.pred_light,
        cache.pred_ax31,
        cap=LADDER_FAST_CAP,
        runaway_fraction=LADDER_RUNAWAY,
        max_upgrade_fraction=LADDER_MAX_UPGRADE,
    )
    if tuple(numpy_models.tolist()) != ladder_sel["fast"]:
        raise ReproductionError("numpy Fast allocator disagrees with frozen the feasibility ladder selection")
    recal_mismatch = {
        tier: ax31_count(recal_native[tier]) != ax31_count(recal_from_ladder[tier])
        or recal_native[tier] != recal_from_ladder[tier]
        for tier in TIERS
    }
    return {
        "recal_ax31": {tier: ax31_count(recal_native[tier]) for tier in TIERS},
        "recal_ax31_from_ladder_injection": {
            tier: ax31_count(recal_from_ladder[tier]) for tier in TIERS
        },
        "recal_match": (not any(recal_mismatch.values())),
        "recal_mismatch_tiers": [tier for tier, flag in recal_mismatch.items() if flag],
        "recal_selection": recal_native,
        "recal_selection_from_ladder_injection": recal_from_ladder,
        "ladder_ax31": {tier: ax31_count(ladder_sel[tier]) for tier in TIERS},
        "ladder_match": True,
        "ladder_selection": ladder_sel,
        "recal_artifact_predicted_caps_fast": json_float(recal_dict["predicted_caps"]["fast"]),
        "recal_frozen_runaway": json_float(recal_dict["runaway_fraction"]),
        "recal_grid_derived_runaway": json_float(derived_runaway_fraction(RECALIBRATED_FAST_CAP)),
        "ladder_vs_recal_artifact_diff": field_diff(ladder_dict, recal_dict),
    }


def _tier_train_row(
    *,
    tier: str,
    config: CapConfig,
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


def run_stage1(
    bundle: TrainBundle,
    *,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> dict[str, Any]:
    ladder_dict = load_frozen_artifact_dict(LADDER_ARTIFACT_PATH)
    recal_dict = load_frozen_artifact_dict(RECALIBRATED_ARTIFACT_PATH)
    ladder_artifact = load_artifact_mapping(copy.deepcopy(ladder_dict))
    cache = cache_predictions(bundle.inputs.episodes, bundle.policy, ladder_artifact)
    reproduction = _reproduce_references(bundle, cache, ladder_artifact, ladder_dict, recal_dict)
    ladder_selection = reproduction["ladder_selection"]
    recal_selection = reproduction["recal_selection"]
    premium_sel, _premium_ratio = select_premium_cached(
        cache.digests, cache.premium_uplift, cache.premium_costs, PREMIUM_CAP
    )
    if premium_sel != ladder_selection["premium"]:
        raise ReproductionError("cached Premium allocator disagrees with frozen the feasibility ladder")

    ladder_train = {
        tier: _tier_train_row(
            tier=tier,
            config=CapConfig(LADDER_FAST_CAP, LADDER_BALANCED_CAP, LADDER_MAX_UPGRADE),
            selection=ladder_selection[tier],
            scores=bundle.scores,
            costs=bundle.costs,
            bound=binding_constraint_frozen(
                cache.rows,
                ladder_selection[tier],
                cap=LADDER_FAST_CAP if tier == "fast" else LADDER_BALANCED_CAP,
                runaway_fraction=LADDER_RUNAWAY,
                max_upgrade_fraction=LADDER_MAX_UPGRADE,
            )
            if tier != "premium"
            else "none",
        )
        for tier in TIERS
    }
    recal_train = {
        tier: _tier_train_row(
            tier=tier,
            config=CapConfig(RECALIBRATED_FAST_CAP, LADDER_BALANCED_CAP, LADDER_MAX_UPGRADE),
            selection=recal_selection[tier],
            scores=bundle.scores,
            costs=bundle.costs,
            bound=binding_constraint_frozen(
                cache.rows,
                recal_selection[tier],
                cap=RECALIBRATED_FAST_CAP if tier == "fast" else LADDER_BALANCED_CAP,
                runaway_fraction=LADDER_RUNAWAY,
                max_upgrade_fraction=LADDER_MAX_UPGRADE,
            )
            if tier != "premium"
            else "none",
        )
        for tier in TIERS
    }
    binding_claims = {
        "balanced_upgrade_count_capped": {
            "claim": (
                "Balanced is upgrade-count-capped, not cost-capped. "
                "Raising predicted_caps.balanced alone will do nothing."
            ),
            "recal_ax31": int(recal_train["balanced"]["ax31"]),
            "recal_bound": recal_train["balanced"]["binding_constraint"],
            "train_n": int(len(bundle.inputs.episodes)),
            "upgrade_cap_075": int(math.floor(0.75 * len(bundle.inputs.episodes))),
            "ladder_ax31": int(ladder_train["balanced"]["ax31"]),
            "ladder_bound": ladder_train["balanced"]["binding_constraint"],
        },
        "fast_headroom": {
            "claim": (
                "Fast has very little headroom left. the recalibrated router predicted_caps.fast "
                "1.07 is already near the ceiling implied by the 1.054 "
                "backstop and the near_budget line."
            ),
            "recal_realized": recal_train["fast"]["realized"],
            "ladder_realized": ladder_train["fast"]["realized"],
        },
    }

    views, catalogue = build_stress_views(bundle.families)
    grid = pre_registered_grid()
    rows: list[dict[str, Any]] = []
    for config in grid:
        selection: dict[str, Tuple[str, ...]] = {"premium": premium_sel}
        bounds: dict[str, str] = {"premium": "none"}
        for tier in ("fast", "balanced"):
            selected, _pred = allocate_frozen(
                cache.rows,
                cap=config.cap(tier),
                runaway_fraction=config.runaway(tier),
                max_upgrade_fraction=config.max_upgrade_fraction,
            )
            selection[tier] = selected
            bounds[tier] = binding_constraint_frozen(
                cache.rows,
                selected,
                cap=config.cap(tier),
                runaway_fraction=config.runaway(tier),
                max_upgrade_fraction=config.max_upgrade_fraction,
            )
        realized = {
            tier: float(_realized_ratio(bundle.costs, selection[tier])) for tier in TIERS
        }
        predicates = evaluate_predicates(
            train_realized=realized,
            selection=selection,
            ladder_selection=ladder_selection,
        )
        train_q = {
            tier: score_mean(bundle.scores, selection[tier]) for tier in TIERS
        }
        row = {
            "binding_constraint": bounds,
            "key": config.key,
            "label": config.label(),
            "max_upgrade_fraction": json_float(config.max_upgrade_fraction),
            "p4_vacuous": bool(predicates["vacuous"]),
            "predicted_caps_balanced": json_float(config.predicted_caps_balanced),
            "predicted_caps_fast": json_float(config.predicted_caps_fast),
            "predicates": predicates,
            "runaway_fraction": {
                tier: json_float(config.runaway(tier)) for tier in ("fast", "balanced")
            },
            "selection": selection,
            "stage1a_eligible": bool(predicates["stage1a_eligible"]),
            "tiers": {
                tier: {
                    "ax31": ax31_count(selection[tier]),
                    "binding_constraint": bounds[tier],
                    "quality_float": train_q[tier],
                    "realized": json_float(realized[tier]),
                    "realized_times_1054": json_float(
                        realized[tier] * float(STRESS_BACKSTOP)
                    ),
                }
                for tier in TIERS
            },
            "train_weighted_float": json_float(
                weighted_final(train_q["fast"], train_q["balanced"], train_q["premium"])
            ),
        }
        rows.append(row)

    survivors = [row for row in rows if row["stage1a_eligible"]]
    failed_summary = _summarize_stage1a_failures(rows)
    if progress is not None:
        progress(
            {
                "phase": "stage1a",
                "n_survivors": len(survivors),
                "failed_summary": failed_summary,
            }
        )

    premium_views = None
    if survivors:
        premium_views = sweep_tier_views(
            views,
            cache,
            bundle.costs,
            tier="premium",
            cap=PREMIUM_CAP,
            runaway_fraction=derived_runaway_fraction(PREMIUM_CAP),
            max_upgrade_fraction=1.0,
        )

    for row in survivors:
        config = CapConfig(
            predicted_caps_fast=float(row["predicted_caps_fast"]),
            predicted_caps_balanced=float(row["predicted_caps_balanced"]),
            max_upgrade_fraction=float(row["max_upgrade_fraction"]),
        )
        ruin_freq = {}
        view_block = {"premium": premium_views}
        ruin_freq["premium"] = float(premium_views["binding"]["ruin_frequency"])
        for tier in ("fast", "balanced"):
            swept = sweep_tier_views(
                views,
                cache,
                bundle.costs,
                tier=tier,
                cap=config.cap(tier),
                runaway_fraction=config.runaway(tier),
                max_upgrade_fraction=config.max_upgrade_fraction,
            )
            view_block[tier] = swept
            ruin_freq[tier] = float(swept["binding"]["ruin_frequency"])
        predicates = evaluate_predicates(
            train_realized={tier: float(row["tiers"][tier]["realized"]) for tier in TIERS},
            selection=row["selection"],
            ladder_selection=ladder_selection,
            ruin_frequency=ruin_freq,
        )
        row["predicates"] = predicates
        row["eligible"] = bool(predicates["eligible"])
        row["views"] = view_block
        if progress is not None:
            progress({"phase": "stage1b", "key": row["key"], "eligible": row["eligible"]})

    for row in rows:
        if "eligible" not in row:
            row["eligible"] = False
        if "views" not in row:
            row["views"] = None

    selected = select_certified(rows)
    runner_ups: list[dict[str, Any]] = []
    official_train = None
    if selected is not None:
        certified = sorted(
            [row for row in rows if row.get("eligible") is True],
            key=_config_sort_key,
        )
        runner_ups = [_public_config_row(item) for item in certified[1:6]]
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

    ladder_official = official_score(
        bundle.inputs, bundle.outcomes, bundle.policy, ladder_selection
    )
    recal_official = official_score(
        bundle.inputs, bundle.outcomes, bundle.policy, recal_selection
    )
    decision = DECISION_NO_ELIGIBLE if selected is None else None
    public_rows = [_public_config_row(row) for row in rows]
    stage1 = {
        "binding_claims": binding_claims,
        "catalogue": catalogue,
        "decision_if_stop": decision,
        "failed_summary": failed_summary,
        "n_eligible": int(sum(1 for row in rows if row.get("eligible") is True)),
        "n_stage1a_survivors": int(len(survivors)),
        "reproduction": {
            "recal_ax31": reproduction["recal_ax31"],
            "recal_ax31_from_ladder_injection": reproduction["recal_ax31_from_ladder_injection"],
            "recal_frozen_runaway": reproduction["recal_frozen_runaway"],
            "recal_grid_derived_runaway": reproduction["recal_grid_derived_runaway"],
            "recal_match": bool(reproduction["recal_match"]),
            "recal_mismatch_tiers": reproduction["recal_mismatch_tiers"],
            "recal_note": (
                "the feasibility ladder injection with predicted_caps.fast=1.07 does NOT "
                "reproduce frozen recalibrated_router. The artifacts also differ in "
                "recalibration edges/factors, provenance, artifact_type, and "
                "selected_policy. The labeled the recalibrated router grid cell is the the feasibility ladder head "
                "with the recalibrated router's Fast cap and the derived runaway; it is not "
                "frozen the recalibrated router. Frozen the recalibrated router keeps runaway_fraction=0.05."
            ),
            "ladder_ax31": reproduction["ladder_ax31"],
            "ladder_match": True,
            "ladder_vs_recal_artifact_diff_paths": [
                row["path"] for row in reproduction["ladder_vs_recal_artifact_diff"]
            ],
        },
        "runner_ups": runner_ups,
        "selected": None if selected is None else _public_config_row(selected),
        "selected_key": None if selected is None else selected["key"],
        "recal_train": {
            "official": {
                "final_score": official_weighted_text(recal_official),
                "tiers": {tier: official_tier_block(recal_official, tier) for tier in TIERS},
            },
            "tiers": recal_train,
            "weighted_float": json_float(
                weighted_final(
                    float(recal_train["fast"]["quality_float"]),
                    float(recal_train["balanced"]["quality_float"]),
                    float(recal_train["premium"]["quality_float"]),
                )
            ),
        },
        "ladder_train": {
            "official": {
                "final_score": official_weighted_text(ladder_official),
                "tiers": {tier: official_tier_block(ladder_official, tier) for tier in TIERS},
            },
            "tiers": ladder_train,
            "weighted_float": json_float(
                weighted_final(
                    float(ladder_train["fast"]["quality_float"]),
                    float(ladder_train["balanced"]["quality_float"]),
                    float(ladder_train["premium"]["quality_float"]),
                )
            ),
        },
        "grid": public_rows,
    }
    if selected is not None and official_train is not None:
        stage1["selected"]["official_train"] = selected["official_train"]
    private = {
        "cache": cache,
        "rows": rows,
        "selected": selected,
        "recal_dict": recal_dict,
        "recal_selection": recal_selection,
        "ladder_artifact": ladder_artifact,
        "ladder_dict": ladder_dict,
        "ladder_selection": ladder_selection,
        "views": views,
    }
    return {"private": private, "stage1": sort_mapping(stage1)}


def _public_config_row(row: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        "binding_constraint": row["binding_constraint"],
        "eligible": bool(row.get("eligible", False)),
        "key": row["key"],
        "label": row["label"],
        "max_upgrade_fraction": row["max_upgrade_fraction"],
        "p4_vacuous": row["p4_vacuous"],
        "predicted_caps_balanced": row["predicted_caps_balanced"],
        "predicted_caps_fast": row["predicted_caps_fast"],
        "predicates": {
            key: row["predicates"][key]
            for key in ("eligible", "failed", "p1", "p2", "p3", "p4", "p5", "vacuous")
            if key in row["predicates"]
        },
        "runaway_fraction": row["runaway_fraction"],
        "stage1a_eligible": row["stage1a_eligible"],
        "tiers": row["tiers"],
        "train_weighted_float": row["train_weighted_float"],
    }
    if row.get("views") is not None:
        public["views"] = {
            tier: {
                "binding": row["views"][tier]["binding"],
                "red_team": {
                    "max_realized": row["views"][tier]["red_team"]["max_realized"],
                    "n": row["views"][tier]["red_team"]["n"],
                    "n_ruin": row["views"][tier]["red_team"]["n_ruin"],
                    "ruin_frequency": row["views"][tier]["red_team"]["ruin_frequency"],
                },
            }
            for tier in TIERS
        }
    if "official_train" in row:
        public["official_train"] = row["official_train"]
    return public


def _summarize_stage1a_failures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {"p1": 0, "p2": 0, "p3": 0, "p4": 0, "passed": 0}
    by_label = {"the feasibility ladder": 0, "the recalibrated router": 0, "grid": 0}
    for row in rows:
        predicates = row["predicates"]
        if predicates["stage1a_eligible"]:
            counts["passed"] += 1
            by_label[row["label"]] += 1
            continue
        for key in ("p1", "p2", "p3", "p4"):
            if not predicates[key]:
                counts[key] += 1
    return {"by_label_survivors": by_label, "failed_predicate_counts": counts}


def _permute_ids(inputs: InputBatch, perm: Sequence[int]) -> InputBatch:
    episodes = []
    source = inputs.episodes
    for index, episode in enumerate(source):
        donor = source[int(perm[index])]
        episodes.append(
            Episode(episode_id=donor.episode_id, prompt=episode.prompt, messages=episode.messages)
        )
    return InputBatch(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        split=inputs.split,
        episodes=tuple(episodes),
    )


def _shuffle_inputs(inputs: InputBatch, order: np.ndarray) -> InputBatch:
    episodes = tuple(inputs.episodes[int(index)] for index in order)
    return InputBatch(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        split=inputs.split,
        episodes=episodes,
    )


def _selection_by_id(inputs: InputBatch, model_ids: Sequence[str]) -> dict[str, str]:
    return {
        episode.episode_id: model_id
        for episode, model_id in zip(inputs.episodes, model_ids)
    }


def _selection_by_digest(inputs: InputBatch, model_ids: Sequence[str]) -> dict[str, str]:
    return {
        hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest(): model_id
        for episode, model_id in zip(inputs.episodes, model_ids)
    }


def write_selected_artifact(
    ladder_dict: Mapping[str, Any],
    selected: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    config = CapConfig(
        predicted_caps_fast=float(selected["predicted_caps_fast"]),
        predicted_caps_balanced=float(selected["predicted_caps_balanced"]),
        max_upgrade_fraction=float(selected["max_upgrade_fraction"]),
    )
    payload = artifact_for_config(ladder_dict, config)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o644)
    return {
        "diff": field_diff(ladder_dict, payload),
        "path": str(path),
        "sha256": sha256_bytes(text.encode("utf-8")),
    }


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
    if decision not in (DECISION_PROMOTE, DECISION_NO_ELIGIBLE, DECISION_DEV_REJECT):
        raise ValueError(f"unknown the cap certification layer decision {decision!r}")
    dev_opened = stage2 is not None
    if decision == DECISION_NO_ELIGIBLE and dev_opened:
        raise RuntimeError("no-eligible decision must not open Dev")
    report = {
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
    return sort_mapping(report)


__all__ = (
    "CapConfig",
    "DECISION_DEV_REJECT",
    "DECISION_NO_ELIGIBLE",
    "DECISION_PROMOTE",
    "EXPERIMENT",
    "Stage2Refused",
    "allocate_frozen",
    "allocate_numpy",
    "artifact_for_config",
    "assemble_report",
    "build_stress_views",
    "cache_predictions",
    "derived_runaway_fraction",
    "evaluate_predicates",
    "field_diff",
    "locked_record",
    "pre_registered_grid",
    "reparameterize_artifact",
    "ruin_frequency",
    "ruin_ok",
    "run_stage1",
    "run_stage2",
    "select_certified",
    "sha256_file",
    "write_selected_artifact",
)
