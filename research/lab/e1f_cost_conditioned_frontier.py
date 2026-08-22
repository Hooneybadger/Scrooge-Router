# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1F — frozen chuf-v1 cost-conditioned hierarchical uplift frontier.

AX31 quality stays the E1 ``baseline_continuous_uplift`` OOF head. K1 uses
a family × predicted-cost-bin hierarchical posterior with a pre-registered
weighted isotonic (nondecreasing) frontier. No new quality feature model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.e1_objectives import (
    ALLOCATOR,
    BASELINE_NAME as E1_BASELINE,
    GATE_VIEW_DROP,
    GATE_VIEW_KINDS,
    VIEW_MIN_N,
    allocate_all_tiers,
    canonical_json_text,
    current_quality_matrix,
    exact_cost_diagnostic,
    oof_candidate_predictions,
    score_decisions,
    sha256_text,
    stress_views,
    write_json_atomic,
)
from research.lab.e1b_quality_models import CHAMPION_ABS
from research.lab.e1c_regime_residual import relabel_folds
from research.lab.e2_cost_uncertainty import oof_cost_surfaces
from research.lab.modeling import OFFICIAL_CAPS, sort_mapping
from research.lab.public_pool import PublicPool, load_public_pool
from research.lab.quality_heads import (
    allocate_two_action,
    content_tie_keys,
    models_two_action,
)


EXPERIMENT = "e1f-cost-conditioned-hierarchical-uplift-frontier"
REPORT_TYPE = "scrooge-e1f-chuf-v1"
SCHEMA_VERSION = 1
CANDIDATE_NAME = "chuf-v1"
BASELINE_NAME = "baseline_continuous_uplift"
FOLD_SEEDS: Tuple[int, ...] = (20260821, 20260822, 20260823, 20260824, 20260825)
N_COST_BINS = 4
MIN_CELL_GROUPS = 20
BETA_PRIOR_A = 0.5
BETA_PRIOR_B = 0.5
COST_EPS = 1e-15
N_MODELS = 3
ALLOWED_GENERATIONS = (2, 4)
GATE_MEAN_DELTA = 0.002
GATE_WORST_DELTA = 0.001
EXPECTED_BASELINE_20260821 = 0.6877178030302
# Pre-audit-fix public OOF selection/gate. Extras repair must not change these.
PINNED_PUBLIC_DECISION = "record-e1f-no-promote"
PINNED_PUBLIC_GATE = {
    "ax31_identity_failures": [],
    "cap_failures": [],
    "experiment_valid": True,
    "k1_failures": [],
    "matched_e1_baseline_20260821": True,
    "mean_absolute": 0.6910246212122401,
    "mean_delta": 0.002022727272719971,
    "passed": False,
    "phase1_passed": False,
    "worst_absolute": 0.6901893939393999,
    "worst_delta": 0.0013920454545000016,
}
PINNED_PUBLIC_VIEW_FAILURES = [
    {
        "failures": ["family:english_multiple_choice", "family:symbolic_math"],
        "seed": 20260821,
    },
    {
        "failures": ["family:english_multiple_choice", "family:symbolic_math"],
        "seed": 20260822,
    },
    {
        "failures": ["family:english_multiple_choice", "family:symbolic_math"],
        "seed": 20260823,
    },
    {
        "failures": ["family:english_multiple_choice", "family:symbolic_math"],
        "seed": 20260824,
    },
    {
        "failures": ["family:english_multiple_choice", "family:symbolic_math"],
        "seed": 20260825,
    },
]
PINNED_PUBLIC_SEEDS = {
    20260821: {
        "baseline_quality": 0.6877178030302,
        "candidate_quality": 0.6901893939393999,
        "delta": 0.0024715909091999055,
        "matched_e1_baseline": True,
        "fast": {"ax31-light": 753, "ax31": 1887, "axk1-think": 0},
        "balanced": {"ax31-light": 67, "ax31": 2573, "axk1-think": 0},
        "premium": {"ax31-light": 58, "ax31": 1667, "axk1-think": 915},
    },
    20260822: {
        "baseline_quality": 0.6892424242427,
        "candidate_quality": 0.6913731060607999,
        "delta": 0.0021306818180999443,
        "matched_e1_baseline": None,
        "fast": {"ax31-light": 750, "ax31": 1890, "axk1-think": 0},
        "balanced": {"ax31-light": 70, "ax31": 2570, "axk1-think": 0},
        "premium": {"ax31-light": 58, "ax31": 1647, "axk1-think": 935},
    },
    20260823: {
        "baseline_quality": 0.6897159090909001,
        "candidate_quality": 0.6911079545454001,
        "delta": 0.0013920454545000016,
        "matched_e1_baseline": None,
        "fast": {"ax31-light": 745, "ax31": 1895, "axk1-think": 0},
        "balanced": {"ax31-light": 59, "ax31": 2581, "axk1-think": 0},
        "premium": {"ax31-light": 52, "ax31": 1642, "axk1-think": 946},
    },
    20260824: {
        "baseline_quality": 0.6885890151517,
        "candidate_quality": 0.6908901515154999,
        "delta": 0.002301136363799916,
        "matched_e1_baseline": None,
        "fast": {"ax31-light": 754, "ax31": 1886, "axk1-think": 0},
        "balanced": {"ax31-light": 64, "ax31": 2576, "axk1-think": 0},
        "premium": {"ax31-light": 59, "ax31": 1641, "axk1-think": 940},
    },
    20260825: {
        "baseline_quality": 0.6897443181820999,
        "candidate_quality": 0.6915625000001,
        "delta": 0.0018181818180000864,
        "matched_e1_baseline": None,
        "fast": {"ax31-light": 749, "ax31": 1891, "axk1-think": 0},
        "balanced": {"ax31-light": 54, "ax31": 2586, "axk1-think": 0},
        "premium": {"ax31-light": 47, "ax31": 1675, "axk1-think": 918},
    },
}
AUDIT_RELATIVE_PATH = "build/compare-e1f-cost-conditioned-frontier/episode-audit.json"
FAMILY_DEFINITION = "ossp_router.cost_calibrated_router.prompt_family"
COST_FEATURE_DEFINITION = (
    "r=log1p(max(cK_point-cA_point,0)/max(cL_point,1e-15)); "
    "outer-train uses E2 oof_cost_surfaces inner_train point costs; "
    "held-out uses the outer OOF point surface. Item-sigma / q90 / "
    "two-price / E2 allocator are not used."
)
AX31_POLICY = "e1_baseline_continuous_uplift_oof"
K1_POLICY = "cost_conditioned_hierarchical_isotonic_frontier"
EXPORT_PREVIEW_KEYS: Tuple[str, ...] = (
    "global_posterior",
    "family_posterior",
    "bin_edges",
    "family_bin_uplift",
    "family_bin_group_counts",
    "c_family",
    "c_cell",
    "min_cell_groups",
    "family_definition",
    "cost_feature_definition",
)
_LIGHT = 0
_AX31 = 1
_K1 = 2
SEQUENTIAL_TESTING = (
    "This phase is a single sequential follow-up after E1/E2/E1B/E1C/E4/"
    "E1D/E1E. Type-I error is not family-wise controlled. A Phase-1 pass "
    "is not a runtime export and does not authorize Phase 2 here."
)
MONOTONE_RATIONALE = {
    "constraint": (
        "weighted PAV nondecreasing in predicted incremental K1 cost "
        "bin 0..3, within family. Direction is pre-registered and is "
        "not reversed after seeing E1F scores."
    ),
    "ebsq_selected_vs_family_uk_seed21": {
        "english_multiple_choice": {"family": 0.0763, "selected": 0.0287},
        "korean_multiple_choice": {"family": 0.0507, "selected": 0.0273},
        "latex_math": {"family": 0.1683, "selected": 0.0086},
        "other": {"family": 0.3612, "selected": 0.2646},
        "python_program": {"family": 0.3806, "selected": 0.3395},
        "symbolic_math": {"family": 0.2793, "selected": 0.1409},
    },
    "note": (
        "EBSQ family-constant qk/cost selected the cheap tail whose "
        "realized uplift was below the family mean. These numbers are "
        "the prior shape rationale only. They do not retune bins, C, "
        "or the isotonic direction."
    ),
}
POSTERIOR_FORMULA = {
    "ax31": "pred_qa is byte-identical E1 baseline_continuous_uplift OOF",
    "bin_edges": (
        "global outer-train quantiles of r at 0.25/0.50/0.75; "
        "np.digitize(..., right=True). Duplicate edges stay deterministic. "
        "No per-family quantiles."
    ),
    "c_cell": (
        "median(total_trials_cell) over nonempty family×bin cells; "
        "total_trials_cell is n_A if n_A>0 else n_K. Empty median -> 1"
    ),
    "c_family": (
        "median(total_trials_f) over families with n_A>0 or n_K>0; "
        "total_trials_f is n_A if n_A>0 else n_K. Empty median -> 1"
    ),
    "cell": "p_m,f,b=(k_m,f,b + C_cell*p_m,f)/(n_m,f,b + C_cell)",
    "cell_fallback": (
        "unique groups < MIN_CELL_GROUPS or n_A=n_K=0 -> family p_m,f; "
        "PAV weight 0 on fallback cells"
    ),
    "family": "p_m,f=(k_m,f + C_family*p_m,global)/(n_m,f + C_family)",
    "global": "p_m=(sum k_m + 0.5)/(sum n_m + 1) for AX31 and K1",
    "isotonic": (
        "weighted PAV nondecreasing on u_f,b=pK-pA; weights=unique "
        "groups; all-zero weights -> family constant"
    ),
    "pred_qk": "max(isotonic_u_family_bin, 0); unseen family -> max(uk_global,0)",
    "r": "log1p(max(cK_point-cA_point,0)/max(cL_point,1e-15))",
    "unseen_family": "global p / uk_global",
}


def _json_float(value: Any) -> float:
    return float(np.float64(value))


def _json_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(np.float64(value))
    if not np.isfinite(number):
        return None
    return number


def binomial_counts(
    pool: PublicPool,
) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``n=num_generations``, ``k=score*n`` from outcome rows only."""

    n_rows = len(pool.episodes)
    n = np.zeros((n_rows, N_MODELS), dtype=np.int64)
    k = np.zeros((n_rows, N_MODELS), dtype=np.int64)
    index_of = {episode.episode_id: row for row, episode in enumerate(pool.episodes)}
    model_of = {model_id: column for column, model_id in enumerate(MODEL_IDS)}
    seen = np.zeros((n_rows, N_MODELS), dtype=np.bool_)
    non_integer = 0
    illegal_n = 0
    for outcome in pool.outcomes.outcomes:
        row = index_of.get(outcome.episode_id)
        column = model_of.get(outcome.model_id)
        if row is None or column is None:
            continue
        generations = int(outcome.num_generations)
        if generations not in ALLOWED_GENERATIONS:
            illegal_n += 1
        product = Decimal(outcome.score) * generations
        if product != product.to_integral_value():
            non_integer += 1
        n[row, column] = generations
        k[row, column] = int(product)
        seen[row, column] = True
    if not bool(np.all(seen)):
        raise RuntimeError("binomial counts missing an episode/model outcome row")
    if illegal_n:
        raise RuntimeError(f"num_generations outside {{2,4}}: {illegal_n}")
    if non_integer:
        raise RuntimeError(f"score*num_generations is not integer on {non_integer} rows")
    n_mismatch = int(np.count_nonzero(n.max(axis=1) != n.min(axis=1)))
    if n_mismatch:
        raise RuntimeError(f"per-model num_generations mismatch: {n_mismatch}")
    return n, k, {
        "illegal_n": illegal_n,
        "k_non_integer": non_integer,
        "n_mismatch": n_mismatch,
        "n_outcome_rows": int(n.size),
        "n_values": {
            str(value): int(np.count_nonzero(n == value)) for value in ALLOWED_GENERATIONS
        },
    }


def global_success_posterior(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    trials = np.asarray(n, dtype=np.float64).reshape(-1, N_MODELS)
    successes = np.asarray(k, dtype=np.float64).reshape(-1, N_MODELS)
    numer = successes.sum(axis=0) + BETA_PRIOR_A
    denom = trials.sum(axis=0) + BETA_PRIOR_A + BETA_PRIOR_B
    return numer / denom


def cost_scalar(point: np.ndarray) -> np.ndarray:
    """Predicted incremental K1 cost intensity. Never uses actual costs."""

    costs = np.asarray(point, dtype=np.float64).reshape(-1, N_MODELS)
    increment = np.maximum(costs[:, _K1] - costs[:, _AX31], 0.0)
    denom = np.maximum(costs[:, _LIGHT], COST_EPS)
    return np.log1p(increment / denom)


def cost_bin_edges(r_train: np.ndarray) -> np.ndarray:
    values = np.asarray(r_train, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(N_COST_BINS - 1, dtype=np.float64)
    return np.quantile(values, (0.25, 0.50, 0.75))


def assign_cost_bins(r: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Right-closed digitize. Duplicate quantile edges stay deterministic."""

    return np.digitize(
        np.asarray(r, dtype=np.float64).reshape(-1),
        np.asarray(edges, dtype=np.float64).reshape(-1),
        right=True,
    ).astype(np.int64)


def fold_predicted_points(
    surfaces: Mapping[str, Any], fold_ids: np.ndarray, fold: int
) -> np.ndarray:
    """Train = inner OOF point; held-out = outer OOF point. No actual costs."""

    ids = np.asarray(fold_ids, dtype=np.int64)
    n_rows = ids.shape[0]
    out = np.empty((n_rows, N_MODELS), dtype=np.float64)
    train = ids != int(fold)
    test = ids == int(fold)
    block = next(
        row for row in surfaces["inner_train"] if int(row["fold"]) == int(fold)
    )
    train_index = np.asarray(block["train_index"], dtype=np.int64)
    if not np.array_equal(train_index, np.flatnonzero(train)):
        raise RuntimeError("inner_train index drifted from the outer-train mask")
    out[train] = np.asarray(block["point"], dtype=np.float64)
    out[test] = np.asarray(surfaces["point"], dtype=np.float64)[test]
    return out


def _trial_strength(n_a: float, n_k: float) -> float:
    if n_a > 0.0:
        return float(n_a)
    return float(n_k)


def _shrink(successes: float, trials: float, prior_mean: float, strength: float) -> float:
    return (float(successes) + float(strength) * float(prior_mean)) / (
        float(trials) + float(strength)
    )


def weighted_isotonic_increasing(
    values: Sequence[float], weights: Sequence[float]
) -> np.ndarray:
    """Weighted PAV for a nondecreasing sequence. Zero-weight bins merge."""

    raw = [float(value) for value in values]
    mass = [float(weight) for weight in weights]
    if len(raw) != len(mass):
        raise ValueError("PAV values and weights must align")
    if not any(weight > 0.0 for weight in mass):
        return np.asarray(raw, dtype=np.float64)
    blocks: list[list[float]] = []
    for value, weight in zip(raw, mass):
        blocks.append([weight, weight * value if weight > 0.0 else 0.0, 1.0, value])
        while len(blocks) >= 2:
            w1, wy1, c1, v1 = blocks[-2]
            w2, wy2, c2, v2 = blocks[-1]
            mean1 = wy1 / w1 if w1 > 0.0 else v1
            mean2 = wy2 / w2 if w2 > 0.0 else v2
            if mean1 <= mean2:
                break
            total = w1 + w2
            if total > 0.0:
                mean = (wy1 + wy2) / total
            else:
                mean = (v1 + v2) / 2.0
            blocks[-2] = [total, total * mean if total > 0.0 else 0.0, c1 + c2, mean]
            blocks.pop()
    out: list[float] = []
    for weight, weighted, count, value in blocks:
        mean = weighted / weight if weight > 0.0 else value
        out.extend([mean] * int(count))
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class FoldFrontier:
    p_global: np.ndarray
    uk_global: float
    c_family: float
    c_cell: float
    bin_edges: np.ndarray
    family_p: dict[str, np.ndarray]
    family_uk: dict[str, float]
    family_isotonic: dict[str, np.ndarray]
    family_bin_raw: dict[str, np.ndarray]
    family_bin_groups: dict[str, np.ndarray]
    family_bin_fallback: dict[str, np.ndarray]
    cells: list[dict[str, Any]]


def fit_fold_frontier(
    families: Sequence[str],
    group_keys: Sequence[str],
    bins: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    bin_edges: np.ndarray,
) -> FoldFrontier:
    """Outer-train hierarchical posteriors + isotonic frontier. No held-out rows."""

    p_global = global_success_posterior(n, k)
    uk_global = float(p_global[_K1] - p_global[_AX31])
    trials = np.asarray(n, dtype=np.float64)
    successes = np.asarray(k, dtype=np.float64)
    bin_ids = np.asarray(bins, dtype=np.int64)
    family_n: dict[str, np.ndarray] = {}
    family_k: dict[str, np.ndarray] = {}
    family_groups: dict[str, set[str]] = {}
    cell_n: dict[tuple[str, int], np.ndarray] = {}
    cell_k: dict[tuple[str, int], np.ndarray] = {}
    cell_groups: dict[tuple[str, int], set[str]] = {}
    for family, group, bin_id, row_n, row_k in zip(
        families, group_keys, bin_ids, trials, successes
    ):
        name = str(family)
        family_n[name] = family_n.get(name, np.zeros(N_MODELS)) + row_n
        family_k[name] = family_k.get(name, np.zeros(N_MODELS)) + row_k
        family_groups.setdefault(name, set()).add(str(group))
        key = (name, int(bin_id))
        cell_n[key] = cell_n.get(key, np.zeros(N_MODELS)) + row_n
        cell_k[key] = cell_k.get(key, np.zeros(N_MODELS)) + row_k
        cell_groups.setdefault(key, set()).add(str(group))

    family_trial_values = [
        _trial_strength(float(totals[_AX31]), float(totals[_K1]))
        for totals in family_n.values()
        if float(totals[_AX31]) > 0.0 or float(totals[_K1]) > 0.0
    ]
    c_family = (
        float(np.median(np.asarray(family_trial_values, dtype=np.float64)))
        if family_trial_values
        else 1.0
    )
    cell_trial_values = [
        _trial_strength(float(totals[_AX31]), float(totals[_K1]))
        for totals in cell_n.values()
        if float(totals[_AX31]) > 0.0 or float(totals[_K1]) > 0.0
    ]
    c_cell = (
        float(np.median(np.asarray(cell_trial_values, dtype=np.float64)))
        if cell_trial_values
        else 1.0
    )

    family_p: dict[str, np.ndarray] = {}
    family_uk: dict[str, float] = {}
    for name, totals_n in family_n.items():
        totals_k = family_k[name]
        posterior = np.array(
            [
                _shrink(
                    float(totals_k[model]),
                    float(totals_n[model]),
                    float(p_global[model]),
                    c_family,
                )
                for model in range(N_MODELS)
            ],
            dtype=np.float64,
        )
        family_p[name] = posterior
        family_uk[name] = float(posterior[_K1] - posterior[_AX31])

    family_isotonic: dict[str, np.ndarray] = {}
    family_bin_raw: dict[str, np.ndarray] = {}
    family_bin_groups: dict[str, np.ndarray] = {}
    family_bin_fallback: dict[str, np.ndarray] = {}
    cells: list[dict[str, Any]] = []
    for name in sorted(family_n):
        raw = np.full(N_COST_BINS, family_uk[name], dtype=np.float64)
        weights = np.zeros(N_COST_BINS, dtype=np.float64)
        groups = np.zeros(N_COST_BINS, dtype=np.int64)
        fallback = np.ones(N_COST_BINS, dtype=np.bool_)
        for bin_id in range(N_COST_BINS):
            key = (name, bin_id)
            totals_n = cell_n.get(key, np.zeros(N_MODELS))
            totals_k = cell_k.get(key, np.zeros(N_MODELS))
            n_groups = int(len(cell_groups.get(key, set())))
            n_a = float(totals_n[_AX31])
            n_k = float(totals_n[_K1])
            empty = n_a <= 0.0 and n_k <= 0.0
            use_family = empty or n_groups < MIN_CELL_GROUPS
            groups[bin_id] = n_groups
            fallback[bin_id] = bool(use_family)
            if not use_family:
                p_cell = np.array(
                    [
                        _shrink(
                            float(totals_k[model]),
                            float(totals_n[model]),
                            float(family_p[name][model]),
                            c_cell,
                        )
                        for model in range(N_MODELS)
                    ],
                    dtype=np.float64,
                )
                raw[bin_id] = float(p_cell[_K1] - p_cell[_AX31])
                weights[bin_id] = float(n_groups)
                p_used = p_cell
                source = "cell"
            else:
                p_used = family_p[name]
                source = "family_fallback" if not empty else "empty_cell"
            cells.append(
                {
                    "bin": bin_id,
                    "family": name,
                    "fallback": bool(use_family),
                    "n_unique_groups": n_groups,
                    "p": {
                        MODEL_IDS[_AX31]: _json_float(p_used[_AX31]),
                        MODEL_IDS[_K1]: _json_float(p_used[_K1]),
                    },
                    "raw_uk": _json_float(raw[bin_id]),
                    "source": source,
                    "trials": {
                        "ax31": _json_float(n_a),
                        "axk1-think": _json_float(n_k),
                    },
                    "weight": _json_float(weights[bin_id]),
                }
            )
        if float(weights.sum()) <= 0.0:
            isotonic = np.full(N_COST_BINS, family_uk[name], dtype=np.float64)
        else:
            isotonic = weighted_isotonic_increasing(raw, weights)
        family_isotonic[name] = isotonic
        family_bin_raw[name] = raw
        family_bin_groups[name] = groups
        family_bin_fallback[name] = fallback
        for row in cells:
            if row["family"] == name:
                row["isotonic_uk"] = _json_float(isotonic[int(row["bin"])])

    return FoldFrontier(
        p_global=p_global,
        uk_global=uk_global,
        c_family=c_family,
        c_cell=c_cell,
        bin_edges=np.asarray(bin_edges, dtype=np.float64),
        family_p=family_p,
        family_uk=family_uk,
        family_isotonic=family_isotonic,
        family_bin_raw=family_bin_raw,
        family_bin_groups=family_bin_groups,
        family_bin_fallback=family_bin_fallback,
        cells=cells,
    )


def predict_qk(families: Sequence[str], bins: np.ndarray, frontier: FoldFrontier) -> np.ndarray:
    out = np.empty(len(families), dtype=np.float64)
    for index, (family, bin_id) in enumerate(zip(families, np.asarray(bins, dtype=np.int64))):
        curve = frontier.family_isotonic.get(str(family))
        if curve is None:
            uplift = frontier.uk_global
        else:
            uplift = float(curve[int(bin_id)])
        out[index] = max(uplift, 0.0)
    return out


def export_preview_coefficients(frontier: FoldFrontier) -> dict[str, Any]:
    return {
        "bin_edges": [_json_float(value) for value in frontier.bin_edges],
        "c_cell": _json_float(frontier.c_cell),
        "c_family": _json_float(frontier.c_family),
        "cost_feature_definition": COST_FEATURE_DEFINITION,
        "family_bin_group_counts": {
            name: [int(value) for value in frontier.family_bin_groups[name]]
            for name in sorted(frontier.family_bin_groups)
        },
        "family_bin_uplift": {
            name: [_json_float(value) for value in frontier.family_isotonic[name]]
            for name in sorted(frontier.family_isotonic)
        },
        "family_definition": FAMILY_DEFINITION,
        "family_posterior": {
            name: {
                MODEL_IDS[_AX31]: _json_float(frontier.family_p[name][_AX31]),
                MODEL_IDS[_K1]: _json_float(frontier.family_p[name][_K1]),
            }
            for name in sorted(frontier.family_p)
        },
        "global_posterior": {
            MODEL_IDS[_AX31]: _json_float(frontier.p_global[_AX31]),
            MODEL_IDS[_K1]: _json_float(frontier.p_global[_K1]),
        },
        "min_cell_groups": MIN_CELL_GROUPS,
    }


def _frontier_record(
    fold: int, train: np.ndarray, test: np.ndarray, frontier: FoldFrontier
) -> dict[str, Any]:
    return {
        "bin_edges": [_json_float(value) for value in frontier.bin_edges],
        "c_cell": _json_float(frontier.c_cell),
        "c_family": _json_float(frontier.c_family),
        "cells": list(frontier.cells),
        "export_preview": {
            "coefficients": export_preview_coefficients(frontier),
            "selection_use": False,
        },
        "fold": int(fold),
        "n_test": int(test.sum()),
        "n_train": int(train.sum()),
        "p_global": {
            MODEL_IDS[_AX31]: _json_float(frontier.p_global[_AX31]),
            MODEL_IDS[_K1]: _json_float(frontier.p_global[_K1]),
        },
        "uk_global": _json_float(frontier.uk_global),
    }


@dataclass(frozen=True)
class HeadPred:
    pred_qa: np.ndarray
    pred_qk: np.ndarray
    extras: dict[str, np.ndarray]


def premium_parent_models(
    pred_qa: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    tie_keys: Sequence[str],
) -> Tuple[str, ...]:
    upgrade = allocate_two_action(
        pred_qa, costs, light_total, float(OFFICIAL_CAPS["premium"]), tie_keys
    )
    return models_two_action(upgrade)


def ax31_selections_match(
    baseline_qa: np.ndarray,
    candidate_qa: np.ndarray,
    costs: np.ndarray,
    light_total: float,
    tie_keys: Sequence[str],
) -> dict[str, Any]:
    if not np.array_equal(
        np.asarray(baseline_qa, dtype=np.float64),
        np.asarray(candidate_qa, dtype=np.float64),
    ):
        return {"fast": False, "balanced": False, "premium_parent": False, "pred_qa": False}
    base = allocate_all_tiers(baseline_qa, np.zeros_like(baseline_qa), costs, light_total, tie_keys)
    cand = allocate_all_tiers(candidate_qa, np.zeros_like(candidate_qa), costs, light_total, tie_keys)
    parent_base = premium_parent_models(baseline_qa, costs, light_total, tie_keys)
    parent_cand = premium_parent_models(candidate_qa, costs, light_total, tie_keys)
    return {
        "balanced": tuple(base["balanced"]) == tuple(cand["balanced"]),
        "fast": tuple(base["fast"]) == tuple(cand["fast"]),
        "pred_qa": True,
        "premium_parent": parent_base == parent_cand,
    }


def oof_chuf_heads(
    pool: PublicPool,
    *,
    scores: Optional[np.ndarray] = None,
    n: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
    surfaces: Optional[Mapping[str, Any]] = None,
) -> Tuple[HeadPred, HeadPred, list[dict[str, Any]]]:
    y = pool.scores if scores is None else np.asarray(scores, dtype=np.float64)
    if n is None or k is None:
        trials, successes, _diag = binomial_counts(pool)
    else:
        trials = np.asarray(n, dtype=np.int64)
        successes = np.asarray(k, dtype=np.int64)
    structural = current_quality_matrix(pool.episodes)
    baseline_qa, baseline_qk = oof_candidate_predictions(structural, y, pool.folds)[
        E1_BASELINE
    ]
    cost_bundle = surfaces if surfaces is not None else oof_cost_surfaces(pool)
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    pred_qk = np.zeros(y.shape[0], dtype=np.float64)
    extras = {
        "cost_bin": np.full(y.shape[0], -1, dtype=np.int64),
        "r": np.full(y.shape[0], np.nan, dtype=np.float64),
    }
    assigned = np.zeros(y.shape[0], dtype=np.int64)
    fold_rows = []
    families = tuple(pool.families)
    groups = tuple(pool.group_keys)
    for fold in range(int(fold_ids.max()) + 1):
        train = fold_ids != int(fold)
        test = fold_ids == int(fold)
        points = fold_predicted_points(cost_bundle, fold_ids, int(fold))
        scalars = cost_scalar(points)
        edges = cost_bin_edges(scalars[train])
        bins = assign_cost_bins(scalars, edges)
        train_idx = np.flatnonzero(train)
        frontier = fit_fold_frontier(
            tuple(families[index] for index in train_idx),
            tuple(groups[index] for index in train_idx),
            bins[train],
            trials[train],
            successes[train],
            edges,
        )
        test_idx = np.flatnonzero(test)
        pred_qk[test] = predict_qk(
            tuple(families[index] for index in test_idx),
            bins[test],
            frontier,
        )
        extras["cost_bin"][test] = bins[test]
        extras["r"][test] = scalars[test]
        assigned[test] += 1
        fold_rows.append(_frontier_record(fold, train, test, frontier))
    hits = {
        int(value): int(np.count_nonzero(assigned == value))
        for value in np.unique(assigned)
    }
    if hits != {1: int(y.shape[0])}:
        raise RuntimeError(
            f"audit extras r/cost_bin must be assigned exactly once "
            f"on each outer held-out row; hits={hits}"
        )
    if np.any(extras["cost_bin"] < 0) or np.any(~np.isfinite(extras["r"])):
        raise RuntimeError("audit extras left unassigned or non-finite")
    return (
        HeadPred(baseline_qa, baseline_qk, extras={}),
        HeadPred(np.asarray(baseline_qa, dtype=np.float64).copy(), pred_qk, extras=extras),
        fold_rows,
    )


def _caps_ok(scored: Mapping[str, Any]) -> bool:
    return all(bool(scored["tiers"][tier]["within_hard_cap"]) for tier in TIERS)


def _k1_fast_balanced(scored: Mapping[str, Any]) -> Tuple[int, int]:
    fast = int(scored["tiers"]["fast"]["model_counts"]["axk1-think"])
    balanced = int(scored["tiers"]["balanced"]["model_counts"]["axk1-think"])
    return fast, balanced


def _evaluate_head(pool: PublicPool, name: str, head: HeadPred, tie_keys: Sequence[str]) -> dict[str, Any]:
    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    pooled_models = allocate_all_tiers(
        head.pred_qa, head.pred_qk, pool.costs, pool.light_total, tie_keys
    )
    pooled = score_decisions(pool, pooled_models)
    per_fold = []
    for fold in range(int(max(pool.folds)) + 1):
        indexes = [index for index, value in enumerate(pool.folds) if value == fold]
        mask = fold_ids == fold
        local_models = allocate_all_tiers(
            head.pred_qa[mask],
            head.pred_qk[mask],
            pool.costs[mask],
            float(pool.costs[mask, _LIGHT].sum()),
            tuple(tie_keys[index] for index in indexes),
        )
        local = score_decisions(pool, local_models, indexes=indexes)
        per_fold.append(
            {
                "fold": fold,
                "n": int(mask.sum()),
                "official_final_score": local["official_final_score"],
                "quality_weighted": local["quality_weighted"],
                "tiers": local["tiers"],
            }
        )
    fast_k1, balanced_k1 = _k1_fast_balanced(pooled)
    return {
        "fold_caps_ok": all(_caps_ok(row) for row in per_fold),
        "k1_balanced": balanced_k1,
        "k1_fast": fast_k1,
        "k1_fast_balanced_zero": bool(fast_k1 == 0 and balanced_k1 == 0),
        "name": name,
        "per_fold": per_fold,
        "pooled": pooled,
    }


def _worst_view(views: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    gated = [
        row
        for row in views
        if row["kind"] in GATE_VIEW_KINDS and row["gated"] and row["delta"] is not None
    ]
    if not gated:
        return None
    row = min(gated, key=lambda item: (item["delta"], item["kind"], item["name"]))
    return {
        "delta": row["delta"],
        "kind": row["kind"],
        "n": row["n"],
        "name": row["name"],
        "worse_than_gate": row["worse_than_gate"],
    }


def _assert_pinned_public_selection(
    seed_reports: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    decision: str,
    public_n: int,
    seeds: Sequence[int],
) -> None:
    """Freeze pre-audit-fix public OOF scores, counts, and gate."""

    if int(public_n) != 2640 or tuple(int(seed) for seed in seeds) != FOLD_SEEDS:
        return
    if decision != PINNED_PUBLIC_DECISION:
        raise RuntimeError(f"public OOF decision drifted: {decision}")
    for key, expected in PINNED_PUBLIC_GATE.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"public OOF gate[{key}] drifted: {gate.get(key)}")
    if list(gate["view_failures"]) != PINNED_PUBLIC_VIEW_FAILURES:
        raise RuntimeError(f"public OOF view failures drifted: {gate['view_failures']}")
    for row in seed_reports:
        seed = int(row["fold_seed"])
        pinned = PINNED_PUBLIC_SEEDS[seed]
        cand = row["candidate"]["pooled"]
        if _json_float(row["baseline"]["pooled"]["quality_weighted"]) != pinned[
            "baseline_quality"
        ]:
            raise RuntimeError(f"public OOF baseline quality drifted at {seed}")
        if _json_float(cand["quality_weighted"]) != pinned["candidate_quality"]:
            raise RuntimeError(f"public OOF candidate quality drifted at {seed}")
        if _json_float(row["delta"]) != pinned["delta"]:
            raise RuntimeError(f"public OOF delta drifted at {seed}")
        if row["matched_e1_baseline"] != pinned["matched_e1_baseline"]:
            raise RuntimeError(f"public OOF matched flag drifted at {seed}")
        for tier in ("fast", "balanced", "premium"):
            counts = cand["tiers"][tier]["model_counts"]
            if {key: int(counts[key]) for key in pinned[tier]} != pinned[tier]:
                raise RuntimeError(f"public OOF {tier} counts drifted at {seed}")


def _family_bin_k1(
    pool: PublicPool,
    baseline_models: Mapping[str, Sequence[str]],
    candidate_models: Mapping[str, Sequence[str]],
    bins: np.ndarray,
    scalars: np.ndarray,
) -> list[dict[str, Any]]:
    """Family×true-OOF-bin K1 mix. ``bins``/``scalars`` are held-out extras."""
    actual_uk = pool.scores[:, _K1] - pool.scores[:, _AX31]
    base_p = np.asarray(list(baseline_models["premium"]))
    cand_p = np.asarray(list(candidate_models["premium"]))
    rows = []
    for family in sorted(set(pool.families)):
        for bin_id in range(N_COST_BINS):
            mask = np.asarray(
                [
                    item == family and int(bin_value) == bin_id
                    for item, bin_value in zip(pool.families, bins)
                ]
            )
            if not np.any(mask):
                continue
            base_k1 = mask & (base_p == "axk1-think")
            cand_k1 = mask & (cand_p == "axk1-think")
            cand_not = mask & (cand_p != "axk1-think")
            rows.append(
                {
                    "actual_uk_candidate_k1": _json_optional_float(
                        float(actual_uk[cand_k1].mean()) if np.any(cand_k1) else None
                    ),
                    "actual_uk_candidate_not_k1": _json_optional_float(
                        float(actual_uk[cand_not].mean()) if np.any(cand_not) else None
                    ),
                    "bin": bin_id,
                    "family": family,
                    "mean_r": _json_float(float(scalars[mask].mean())),
                    "n": int(mask.sum()),
                    "n_baseline_k1": int(np.count_nonzero(base_k1)),
                    "n_candidate_k1": int(np.count_nonzero(cand_k1)),
                }
            )
    return rows


def _split_shift(
    pool: PublicPool, models: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    premium = np.asarray(list(models["premium"]))
    out = {}
    for name in ("train", "dev"):
        mask = np.asarray([label == name for label in pool.split_labels])
        if not np.any(mask):
            out[name] = None
            continue
        chosen = premium[mask]
        scores = pool.scores[mask]
        columns = np.asarray([MODEL_IDS.index(model_id) for model_id in chosen])
        quality = scores[np.arange(scores.shape[0]), columns]
        out[name] = {
            "mean_selected_quality": _json_float(float(quality.mean())),
            "n": int(mask.sum()),
            "n_k1": int(np.count_nonzero(chosen == "axk1-think")),
        }
    return out


def _diagnostics(
    pool: PublicPool,
    baseline_head: HeadPred,
    candidate: HeadPred,
    views: Sequence[Mapping[str, Any]],
    baseline_models: Mapping[str, Sequence[str]],
    candidate_models: Mapping[str, Sequence[str]],
    tie_keys: Sequence[str],
) -> dict[str, Any]:
    identity = ax31_selections_match(
        baseline_head.pred_qa,
        candidate.pred_qa,
        pool.costs,
        pool.light_total,
        tie_keys,
    )
    identity["fast_models"] = tuple(baseline_models["fast"]) == tuple(
        candidate_models["fast"]
    )
    identity["balanced_models"] = tuple(baseline_models["balanced"]) == tuple(
        candidate_models["balanced"]
    )
    parent_base = premium_parent_models(
        baseline_head.pred_qa, pool.costs, pool.light_total, tie_keys
    )
    parent_cand = premium_parent_models(
        candidate.pred_qa, pool.costs, pool.light_total, tie_keys
    )
    identity["premium_parent_models"] = parent_base == parent_cand
    identity["all"] = all(identity.values())
    return {
        "ax31_identical_to_baseline": identity,
        "ax31_policy": AX31_POLICY,
        "family_bin_k1": _family_bin_k1(
            pool,
            baseline_models,
            candidate_models,
            candidate.extras["cost_bin"],
            candidate.extras["r"],
        ),
        "k1_policy": K1_POLICY,
        "quality_feature_dimension": "e1_baseline_oof_only",
        "split_shift": _split_shift(pool, candidate_models),
        "views": [
            {
                "delta": row["delta"],
                "gated": row["gated"],
                "kind": row["kind"],
                "n": row["n"],
                "name": row["name"],
                "worse_than_gate": row["worse_than_gate"],
            }
            for row in views
            if row["kind"] in GATE_VIEW_KINDS
        ],
    }


def promotion_gate(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta"]) for row in seed_reports]
    qualities = [
        float(row["candidate"]["pooled"]["quality_weighted"]) for row in seed_reports
    ]
    baseline_qualities = [
        float(row["baseline"]["pooled"]["quality_weighted"]) for row in seed_reports
    ]
    view_fail = []
    cap_fail = []
    k1_fail = []
    identity_fail = []
    matched = None
    for row in seed_reports:
        seed = row["fold_seed"]
        if seed == 20260821:
            matched = bool(row.get("matched_e1_baseline"))
        pooled_ok = _caps_ok(row["candidate"]["pooled"]) and _caps_ok(
            row["baseline"]["pooled"]
        )
        fold_ok = bool(row["candidate"]["fold_caps_ok"] and row["baseline"]["fold_caps_ok"])
        if not (pooled_ok and fold_ok):
            cap_fail.append(seed)
        if not (
            row["candidate"].get("k1_fast_balanced_zero", False)
            and row["baseline"].get("k1_fast_balanced_zero", False)
        ):
            k1_fail.append(seed)
        identity = row.get("ax31_identical_to_baseline")
        if identity is False or (
            isinstance(identity, Mapping) and not identity.get("all", False)
        ):
            identity_fail.append(seed)
        fails = [
            f"{item['kind']}:{item['name']}"
            for item in row["views"]
            if item["kind"] in GATE_VIEW_KINDS and item["worse_than_gate"]
        ]
        if fails:
            view_fail.append({"failures": fails, "seed": seed})
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    worst_delta = float(np.min(deltas)) if deltas else float("nan")
    mean_quality = float(np.mean(qualities)) if qualities else float("nan")
    worst_quality = float(np.min(qualities)) if qualities else float("nan")
    baseline_matched = True if matched is None else bool(matched)
    experiment_valid = bool(baseline_matched and not identity_fail)
    phase1 = bool(
        experiment_valid
        and not cap_fail
        and not view_fail
        and not k1_fail
        and mean_delta >= GATE_MEAN_DELTA
        and worst_delta >= GATE_WORST_DELTA
        and mean_quality >= CHAMPION_ABS
    )
    return {
        "ax31_identity_failures": identity_fail,
        "baseline_mean_quality": _json_float(float(np.mean(baseline_qualities)))
        if baseline_qualities
        else None,
        "cap_failures": cap_fail,
        "experiment_valid": experiment_valid,
        "k1_failures": k1_fail,
        "matched_e1_baseline_20260821": matched,
        "mean_absolute": _json_float(mean_quality),
        "mean_delta": _json_float(mean_delta),
        "passed": phase1,
        "phase1_passed": phase1,
        "phase2_executed": False,
        "thresholds": {
            "mean_absolute": CHAMPION_ABS,
            "mean_delta": GATE_MEAN_DELTA,
            "stress_95_not_gated": True,
            "view_drop": GATE_VIEW_DROP,
            "view_min_n": VIEW_MIN_N,
            "worst_delta": GATE_WORST_DELTA,
        },
        "view_failures": view_fail,
        "worst_absolute": _json_float(worst_quality),
        "worst_delta": _json_float(worst_delta),
    }


def episode_audit_document(
    seed_pools: Mapping[int, PublicPool],
    heads: Mapping[int, Tuple[HeadPred, HeadPred]],
) -> dict[str, Any]:
    seed_blocks = {}
    for seed, pool in seed_pools.items():
        base_head, cand_head = heads[seed]
        ties = content_tie_keys(pool.texts)
        base_models = allocate_all_tiers(
            base_head.pred_qa, base_head.pred_qk, pool.costs, pool.light_total, ties
        )
        cand_models = allocate_all_tiers(
            cand_head.pred_qa, cand_head.pred_qk, pool.costs, pool.light_total, ties
        )
        rows = []
        for index, episode in enumerate(pool.episodes):
            rows.append(
                {
                    "cost_bin": int(cand_head.extras["cost_bin"][index]),
                    "episode_id": episode.episode_id,
                    "family": pool.families[index],
                    "fold": int(pool.folds[index]),
                    "group_key": pool.group_keys[index],
                    "pred_qa": _json_float(cand_head.pred_qa[index]),
                    "pred_qk": _json_float(cand_head.pred_qk[index]),
                    "r": _json_float(cand_head.extras["r"][index]),
                    "seed": int(seed),
                    "selected": {
                        BASELINE_NAME: {
                            tier: str(base_models[tier][index]) for tier in TIERS
                        },
                        CANDIDATE_NAME: {
                            tier: str(cand_models[tier][index]) for tier in TIERS
                        },
                    },
                    "split": pool.split_labels[index],
                }
            )
        seed_blocks[str(seed)] = {"n_rows": len(rows), "rows": rows}
    return {
        "experiment": EXPERIMENT,
        "prompt_text_included": False,
        "seeds": seed_blocks,
    }


def decision_core_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return sort_mapping(
        {
            "allocator": report["allocator"],
            "audit": report["audit"],
            "candidate": report["candidate"],
            "constants": report["constants"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "experiment": report["experiment"],
            "export_preview": report["export_preview"],
            "feature": report["feature"],
            "fold_seeds": report["fold_seeds"],
            "identity": report["identity"],
            "label_checks": report["label_checks"],
            "limitations": report["limitations"],
            "monotone_rationale": report["monotone_rationale"],
            "posterior_formula": report["posterior_formula"],
            "promotion_gate": report["promotion_gate"],
            "report_type": report["report_type"],
            "schema_version": report["schema_version"],
            "seed_results": report["seed_results"],
            "sequential_testing": report["sequential_testing"],
        }
    )


def decision_core_sha256(report: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        decision_core_payload(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(encoded)


def _example_preview(seed_reports: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    for seed in FOLD_SEEDS:
        report = seed_reports.get(int(seed))
        if report is None:
            continue
        for row in report["fold_frontiers"]:
            return {
                "coefficients": dict(row["export_preview"]["coefficients"]),
                "example_fold": row["fold"],
                "example_seed": int(seed),
                "note": (
                    "example outer-train frontier from one fold; not a "
                    "full-public refit and not for selection"
                ),
                "selection_use": False,
            }
    empty = FoldFrontier(
        p_global=np.full(N_MODELS, 0.5),
        uk_global=0.0,
        c_family=1.0,
        c_cell=1.0,
        bin_edges=np.zeros(3),
        family_p={},
        family_uk={},
        family_isotonic={},
        family_bin_raw={},
        family_bin_groups={},
        family_bin_fallback={},
        cells=[],
    )
    return {
        "coefficients": export_preview_coefficients(empty),
        "example_fold": None,
        "example_seed": None,
        "note": "no fold frontier; preview is empty and not for selection",
        "selection_use": False,
    }


def assemble(
    pool: PublicPool | None = None,
    *,
    seeds: Sequence[int] = FOLD_SEEDS,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    base_pool = pool or load_public_pool()
    trials, successes, label_checks = binomial_counts(base_pool)
    seed_pools: dict[int, PublicPool] = {}
    seed_reports: dict[int, dict[str, Any]] = {}
    heads: dict[int, Tuple[HeadPred, HeadPred]] = {}
    public_n = int(base_pool.identity.get("n_episodes", 0))
    for seed in seeds:
        current = relabel_folds(base_pool, int(seed))
        seed_pools[int(seed)] = current
        baseline_head, candidate_head, fold_rows = oof_chuf_heads(
            current, n=trials, k=successes
        )
        heads[int(seed)] = (baseline_head, candidate_head)
        tie_keys = content_tie_keys(current.texts)
        baseline = _evaluate_head(current, BASELINE_NAME, baseline_head, tie_keys)
        candidate = _evaluate_head(current, CANDIDATE_NAME, candidate_head, tie_keys)
        models_base = allocate_all_tiers(
            baseline_head.pred_qa,
            baseline_head.pred_qk,
            current.costs,
            current.light_total,
            tie_keys,
        )
        models_cand = allocate_all_tiers(
            candidate_head.pred_qa,
            candidate_head.pred_qk,
            current.costs,
            current.light_total,
            tie_keys,
        )
        views = stress_views(current, models_base, models_cand)
        identity = ax31_selections_match(
            baseline_head.pred_qa,
            candidate_head.pred_qa,
            current.costs,
            current.light_total,
            tie_keys,
        )
        identity["fast_models"] = tuple(models_base["fast"]) == tuple(models_cand["fast"])
        identity["balanced_models"] = tuple(models_base["balanced"]) == tuple(
            models_cand["balanced"]
        )
        identity["premium_parent_models"] = premium_parent_models(
            baseline_head.pred_qa, current.costs, current.light_total, tie_keys
        ) == premium_parent_models(
            candidate_head.pred_qa, current.costs, current.light_total, tie_keys
        )
        identity["all"] = all(identity.values())
        diagnostics = _diagnostics(
            current,
            baseline_head,
            candidate_head,
            views,
            models_base,
            models_cand,
            tie_keys,
        )
        baseline_quality = float(baseline["pooled"]["quality_weighted"])
        matched = None
        if int(seed) == 20260821 and public_n == 2640:
            matched = bool(_json_float(baseline_quality) == EXPECTED_BASELINE_20260821)
        seed_reports[int(seed)] = {
            "ax31_identical_to_baseline": identity,
            "baseline": baseline,
            "candidate": candidate,
            "delta": _json_float(
                float(candidate["pooled"]["quality_weighted"]) - baseline_quality
            ),
            "diagnostics": diagnostics,
            "fold_frontiers": fold_rows,
            "fold_seed": int(seed),
            "matched_e1_baseline": matched,
            "views": views,
            "worst_view": _worst_view(views),
        }
    ordered = [seed_reports[int(seed)] for seed in seeds]
    gate = promotion_gate(ordered)
    if gate["phase1_passed"]:
        decision = "record-e1f-quality-pass-await-phase2"
        decision_reason = (
            "Phase-1 exact-cost gates passed. This invocation does not run "
            "predicted-cost Phase 2 and does not export runtime artifacts. "
            "Hand off to independent review only."
        )
    else:
        decision = "record-e1f-no-promote"
        decision_reason = (
            "Phase-1 exact-cost gates failed. STOP. Do not open Phase 2, "
            "do not retune bins/C/isotonic direction, and do not add a "
            "second candidate or another public-OOF patch. Keep the "
            "current runtime."
        )
    _assert_pinned_public_selection(ordered, gate, decision, public_n, seeds)
    audit_document = episode_audit_document(seed_pools, heads)
    audit_sha = sha256_text(canonical_json_text(audit_document))
    export_block = _example_preview(seed_reports)
    seed_payload = {}
    for seed, report in seed_reports.items():
        seed_payload[str(seed)] = {
            "ax31_identical_to_baseline": report["ax31_identical_to_baseline"],
            "baseline_quality": report["baseline"]["pooled"]["quality_weighted"],
            "candidate_quality": report["candidate"]["pooled"]["quality_weighted"],
            "delta": report["delta"],
            "diagnostics": report["diagnostics"],
            "fold_frontiers": report["fold_frontiers"],
            "fold_table": list(seed_pools[seed].fold_table),
            "matched_e1_baseline": report["matched_e1_baseline"],
            "results": {
                BASELINE_NAME: report["baseline"],
                CANDIDATE_NAME: report["candidate"],
            },
            "views": report["views"],
            "worst_view": report["worst_view"],
        }
    report = {
        "allocator": dict(ALLOCATOR),
        "audit": {
            "n_rows": sum(block["n_rows"] for block in audit_document["seeds"].values()),
            "relative_path": AUDIT_RELATIVE_PATH,
            "sha256": audit_sha,
        },
        "candidate": CANDIDATE_NAME,
        "constants": {
            "beta_prior_a": BETA_PRIOR_A,
            "beta_prior_b": BETA_PRIOR_B,
            "cost_eps": COST_EPS,
            "min_cell_groups": MIN_CELL_GROUPS,
            "n_cost_bins": N_COST_BINS,
        },
        "cost_diagnostic": exact_cost_diagnostic(base_pool.costs),
        "decision": decision,
        "decision_reason": decision_reason,
        "experiment": EXPERIMENT,
        "export_preview": export_block,
        "feature": {
            "ax31_policy": AX31_POLICY,
            "cost_feature_definition": COST_FEATURE_DEFINITION,
            "family_definition": FAMILY_DEFINITION,
            "item_score_model": False,
            "k1_policy": K1_POLICY,
            "new_structural_quality_model": False,
            "runtime_artifact_changed": False,
        },
        "fold_seeds": [int(seed) for seed in seeds],
        "identity": dict(base_pool.identity),
        "label_checks": label_checks,
        "limitations": [
            SEQUENTIAL_TESTING,
            "Outer held-out score / n / k / actual cost never enter "
            "posteriors, bin edges, C strengths, or isotonic fits.",
            "Outer-train r uses inner OOF predicted point costs only.",
            "95% stress caps are observational.",
            "Phase 2 predicted-cost evaluation is not executed here.",
            "A pass is not a runtime export.",
            "exact-cost OOF and the current public Dev runtime replay "
            "are different protocols and must not be subtracted.",
            "Audit extras r / cost_bin store the outer held-out test-mask "
            "value only; each row is assigned exactly once. Selection uses "
            "pred_qa / pred_qk, not extras.",
        ],
        "monotone_rationale": dict(MONOTONE_RATIONALE),
        "phase2": {
            "executed": False,
            "reason": "this invocation never opens predicted-cost Phase 2",
        },
        "posterior_formula": dict(POSTERIOR_FORMULA),
        "promotion_gate": gate,
        "report_type": REPORT_TYPE,
        "runtime": {"excluded_from_core": ["elapsed_s"]},
        "schema_version": SCHEMA_VERSION,
        "seed_results": seed_payload,
        "sequential_testing": SEQUENTIAL_TESTING,
        "solver": {"name": None, "note": "closed-form posterior + weighted PAV"},
    }
    report["decision_core_sha256"] = decision_core_sha256(report)
    return sort_mapping(report), audit_document


__all__ = (
    "AUDIT_RELATIVE_PATH",
    "BASELINE_NAME",
    "BETA_PRIOR_A",
    "BETA_PRIOR_B",
    "CANDIDATE_NAME",
    "COST_EPS",
    "EXPECTED_BASELINE_20260821",
    "EXPORT_PREVIEW_KEYS",
    "FAMILY_DEFINITION",
    "FOLD_SEEDS",
    "MIN_CELL_GROUPS",
    "N_COST_BINS",
    "PINNED_PUBLIC_DECISION",
    "PINNED_PUBLIC_GATE",
    "PINNED_PUBLIC_SEEDS",
    "PINNED_PUBLIC_VIEW_FAILURES",
    "assemble",
    "assign_cost_bins",
    "ax31_selections_match",
    "binomial_counts",
    "cost_bin_edges",
    "cost_scalar",
    "export_preview_coefficients",
    "fit_fold_frontier",
    "fold_predicted_points",
    "global_success_posterior",
    "oof_chuf_heads",
    "predict_qk",
    "premium_parent_models",
    "promotion_gate",
    "weighted_isotonic_increasing",
    "write_json_atomic",
)
