# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the prefix certificate layer monotone prefix certificate: AX31 count + certified K1, no predicted denominators."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.lab.prompt_features import (
    FEATURE_VERSION,
    STRUCTURAL_FEATURE_NAMES,
    feature_row,
    feature_signature,
)
from ossp_router.protocol import MODEL_IDS, TIERS
from research.lab.modeling import (
    FACTOR_CLIP,
    FOLD_SEED,
    FOLDS,
    HASH_BINS,
    INTERCEPT_POLICY,
    N_BINS,
    OFFICIAL_CAPS,
    OPERATING_TARGETS,
    STRESS_BACKSTOP,
    TIER_WEIGHTS,
    TrainBundle,
    family_folds,
    feature_matrix,
    group_folds,
    load_train,
    official_score,
    oof_predict,
    paired_group_bootstrap,
    quantile_higher,
    reject_dev_reference,
    ridge_fit,
    ridge_predict,
    sort_mapping,
    weighted_final,
)
from research.lab.cost_certificates import (
    FittedHeads,
    actual_increments,
    apply_recal,
    fit_heads,
    fit_recal,
    floor_inc,
    oof_incremental_costs,
    predict_heads,
)
from research.lab.validation import prompt_family as _prompt_family


EXPERIMENT = "the prefix certificate layer"
REPORT_TYPE = "scrooge-prefix_cert-certified-allocator-v1"
SCHEMA_VERSION = 1
DECISION_PASS = "record-prefix_cert-certified-allocator"
DECISION_K1_OFF = "record-prefix_cert-certified-allocator-k1-off"
FOLD_SEED_PREFIX_CERT = FOLD_SEED  # 2026082202
BOOTSTRAP_SEED = 2026082203
BOOTSTRAP_DRAWS = 1000
FAMDOM_SEED = 2026082204
DIRICHLET_SEED = 2026082205
HALF_SEED = 2026082206
SMALL_SEED = 2026082207
LOCKED_VARIANT = "direct_log1p_inc"
LOCKED_ALPHA = 300.0
LOCKED_BINS = 512
RECAL_BINS = N_BINS
RECAL_CLIP = FACTOR_CLIP
F_GRID = tuple(float(index) / 100.0 for index in range(101))
F_GRID_ARRAY = np.asarray(F_GRID, dtype=np.float64)
SORT_RULES: Tuple[str, ...] = ("sortA_pred_inc", "sortA_pred_inc_per_light")
K_ORDERS: Tuple[str, ...] = ("orderK_qk", "orderK_density")
Q_ELIG: Tuple[float, ...] = (0.25, 0.50, 0.75)
K1_DENYLIST: Tuple[str, ...] = (
    "rule_reasoning",
    "python_program",
    "korean_reasoning",
)
NEAR_TIE_ABS = 1e-12
NEAR_TIE_REL = 1e-6
NEAR_TIE_RULE = (
    "Two recalibrated predicted inc_A values are a near-tie iff "
    "abs(a-b) <= max(1e-12, 1e-6 * max(a, b, 1e-12)). After a stable sort "
    "by ascending inc_A then content digest, consecutive items that stay "
    "within that tolerance of the cluster minimum form a cluster; each "
    "cluster is re-sorted by ascending predicted cost_L then content "
    "digest. Final tie-break is the SHA-256 of UTF-8 episode text. Episode "
    "id, split, and input order are never keys."
)
K1_DENSITY_EPS = 1e-18
K1_DENSITY_RULE = (
    "orderK_density ranks eligible episodes by descending "
    "predicted_Q_K / max(recalibrated_predicted_inc_K, 1e-18); "
    "orderK_qk ranks by descending predicted_Q_K. Content digest is the "
    "final tie-break in both."
)
K1_ITEM_CAP_FRAC = 0.02
K1_COUNT_CAP_FRAC = 0.25
MIN_VIEW_N = 20
FAMDOM_SHARE = 0.75
FAMDOM_DUP_CAP = 3
FAMDOM_TARGET_N = 1760
DIRICHLET_ALPHA = 0.5
DIRICHLET_N = 880
DIRICHLET_DRAWS = 1000
HALF_N = 880
HALF_DRAWS = 20
SMALL_SIZES: Tuple[int, ...] = (100, 300, 880)
SMALL_DRAWS = 200
FAMDOM_DRAWS = 200
FAST_DRIFT_COEF = 2.2235
H2_6_EXCEEDANCE_MAX = 0.01
PARENT_F_PINS = {
    "balanced": {"quality": 0.665341, "ratio": 1.375851, "ax31_count": 1320},
    "fast": {"quality": 0.609091, "ratio": 1.019526, "ax31_count": 186},
    "premium": {"quality": 0.68125, "ratio": 1.983961, "ax31_count": 1651},
    "weighted": 0.647614,
}
PARENT_F_EXACT = {
    "balanced": {"quality": "0.665340909091", "ratio": "1.375850604373"},
    "fast": {"quality": "0.609090909091", "ratio": "1.019525788415"},
    "premium": {"quality": "0.68125", "ratio": "1.983960802953"},
    "weighted": "0.647613636364",
}
BACKSTOP_TARGETS = {
    tier: float(OFFICIAL_CAPS[tier]) / float(STRESS_BACKSTOP) for tier in TIERS
}
SELECTION_CRITERION = (
    "Choose (sort rule, q_elig, K ordering) by, in strict priority: "
    "(1) highest weighted Q_oof at the certified operating points using "
    "official weights 0.4/0.3/0.3; (2) larger minimum margin "
    "target - Phi_upper(f*) summed over tiers; (3) simpler variant "
    "(sortA_pred_inc before sortA_pred_inc_per_light; orderK_qk before "
    "orderK_density; larger q_elig last). Quality arrays may enter this "
    "criterion (it is a policy choice, not a cost-safety choice), but "
    "record dev_opened: false and q_a_used: false."
)
COST_HEAD_LOCK = (
    "Reuse the cost certificate layer's mechanically selected cost head: variant "
    "direct_log1p_inc, alpha=300.0, bins=512, with the rank-isotonic "
    "recalibration (research.lab.modeling.rank_recalibration, n_bins=10, "
    "clip=(0.5, 6.0)). Its OOF post-recalibration aggregate ratios were "
    "inc_A 1.029434 and inc_K 1.002749. Do NOT re-run the 12-config grid "
    "and do NOT re-select — that selection is already locked in charter "
    "§10.4 and executed in the cost certificate layer."
)
QUALITY_HEADS_PATH = (
    Path(__file__).resolve().parents[2] / "build" / "quality-heads" / "quality-heads.json"
)
LEGAL_MODEL_IDS = MODEL_IDS
_LIGHT = "ax31-light"
_AX31 = "ax31"
_K1 = "axk1-think"
FLOAT64_TABLE_NOTE = (
    "Dense Phi/Psi/Q tables are exact float64 sums of public costs and "
    "scores via numpy.cumsum on the view's sorted order. The official "
    "Decimal scorer is used only for the selected operating-point quality "
    "numbers and the H2-1/H2-2/H2-3/H2-4/H2-5 gate qualities. The two "
    "paths are compared at the selected OOF operating points and must "
    "agree to within 1e-12."
)

# Local shim: research.lab.prompt_features has no prompt_family. Required change
# (NOT applied): add family_of_text(text) to g_features mirroring
# research.lab.validation.prompt_family. Workaround: classify through the the modeling foundation
# validation helper on a text-only episode stand-in so Train and route()
# share one rule without forking g_features.
class _TextEpisode:
    __slots__ = ("prompt", "messages")

    def __init__(self, text: str) -> None:
        self.prompt = text
        self.messages = ()


def family_of_text(text: str) -> str:
    """Content-only family label. See the g_features shim comment above."""

    return _prompt_family(_TextEpisode(text))


def json_float(value: Any) -> float:
    return float(np.float64(value))


def json_floats(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [json_float(item) for item in array]


def content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_digests(texts: Sequence[str]) -> Tuple[str, ...]:
    return tuple(content_digest(text) for text in texts)


def design_matrix_g_features(texts: Sequence[str], *, bins: int) -> np.ndarray:
    """Stdlib-router design matrix. Calls research.lab.prompt_features.feature_row only."""

    width = int(bins)
    if width not in HASH_BINS:
        raise ValueError(f"hash bins must be one of {HASH_BINS}; got {bins!r}")
    n_struct = len(STRUCTURAL_FEATURE_NAMES)
    matrix = np.zeros((len(texts), 1 + n_struct + width), dtype=np.float64)
    matrix[:, 0] = 1.0
    hash_start = 1 + n_struct
    for row, text in enumerate(texts):
        structural, hashed = feature_row(text, bins=width)
        matrix[row, 1:hash_start] = structural
        for bucket in sorted(hashed):
            matrix[row, hash_start + int(bucket)] = float(hashed[bucket])
    return matrix


def apply_recal_baked(
    pred_inc: np.ndarray, edges: np.ndarray, factors: np.ndarray
) -> np.ndarray:
    predicted = floor_inc(pred_inc)
    index = np.digitize(predicted, np.asarray(edges, dtype=np.float64), right=True)
    last = int(np.asarray(factors).size) - 1
    index = np.clip(index, 0, last)
    return predicted * np.asarray(factors, dtype=np.float64)[index]


def prefix_k(fraction: float, n: int) -> int:
    if int(n) <= 0:
        return 0
    return int(min(int(n), max(0, math.floor(float(fraction) * float(n) + 1e-15))))


def sort_pred_inc(inc_a: np.ndarray, digests: Sequence[str]) -> np.ndarray:
    """Ascending recalibrated predicted inc_A, content digest final tie-break."""

    predicted = np.asarray(inc_a, dtype=np.float64).reshape(-1)
    keys = np.asarray(list(digests))
    if predicted.size != keys.size:
        raise ValueError("sort_pred_inc requires aligned inc_A and digests")
    return np.lexsort((keys, predicted))


def sort_pred_inc_per_light(
    inc_a: np.ndarray, pred_l: np.ndarray, digests: Sequence[str]
) -> np.ndarray:
    """Ascending inc_A with near-ties broken by ascending predicted cost_L."""

    predicted = np.asarray(inc_a, dtype=np.float64).reshape(-1)
    light = np.asarray(pred_l, dtype=np.float64).reshape(-1)
    keys = np.asarray(list(digests))
    n_rows = int(predicted.size)
    if not (light.size == n_rows == keys.size):
        raise ValueError("sort_pred_inc_per_light requires aligned inputs")
    if n_rows == 0:
        return np.zeros(0, dtype=np.int64)
    prelim = np.lexsort((keys, predicted))
    ordered = prelim.copy()
    start = 0
    cluster_min = float(predicted[prelim[0]])
    for stop in range(1, n_rows + 1):
        if stop < n_rows:
            value = float(predicted[prelim[stop]])
            thresh = max(NEAR_TIE_ABS, NEAR_TIE_REL * max(value, cluster_min, NEAR_TIE_ABS))
            if value - cluster_min <= thresh:
                continue
        cluster = prelim[start:stop]
        local = np.lexsort((keys[cluster], light[cluster]))
        ordered[start:stop] = cluster[local]
        if stop < n_rows:
            start = stop
            cluster_min = float(predicted[prelim[stop]])
    return ordered


def sort_order(
    rule: str, inc_a: np.ndarray, pred_l: np.ndarray, digests: Sequence[str]
) -> np.ndarray:
    if rule == "sortA_pred_inc":
        return sort_pred_inc(inc_a, digests)
    if rule == "sortA_pred_inc_per_light":
        return sort_pred_inc_per_light(inc_a, pred_l, digests)
    raise ValueError(f"unknown sort rule {rule!r}")


def phi_view(
    actual_light: np.ndarray,
    actual_ax31: np.ndarray,
    order: np.ndarray,
    grid: np.ndarray = F_GRID_ARRAY,
) -> np.ndarray:
    """Realized budget ratio when upgrading the cheapest-predicted prefix.

    Phi(0) == 1.0. Uses actual costs only (no predicted denominator).
    Vectorized with numpy.cumsum. Monotone non-decreasing when every
    actual AX31 increment is non-negative.
    """

    light = np.asarray(actual_light, dtype=np.float64).reshape(-1)
    ax31 = np.asarray(actual_ax31, dtype=np.float64).reshape(-1)
    ranked = np.asarray(order, dtype=np.int64).reshape(-1)
    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    n_rows = int(light.size)
    light_sum = float(light.sum())
    if n_rows == 0 or light_sum <= 0.0:
        return np.ones(knots.size, dtype=np.float64)
    increment = (ax31 - light)[ranked]
    cumulative = np.cumsum(increment)
    ks = np.clip(
        np.floor(knots * float(n_rows) + 1e-15).astype(np.int64), 0, n_rows
    )
    extra = np.zeros(knots.size, dtype=np.float64)
    mask = ks > 0
    extra[mask] = cumulative[ks[mask] - 1]
    return 1.0 + extra / light_sum


def q_view(
    score_light: np.ndarray,
    score_ax31: np.ndarray,
    order: np.ndarray,
    grid: np.ndarray = F_GRID_ARRAY,
) -> np.ndarray:
    """Mean score at the same prefix. Monotone when all uplifts are >= 0."""

    light = np.asarray(score_light, dtype=np.float64).reshape(-1)
    ax31 = np.asarray(score_ax31, dtype=np.float64).reshape(-1)
    ranked = np.asarray(order, dtype=np.int64).reshape(-1)
    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    n_rows = int(light.size)
    if n_rows == 0:
        return np.zeros(knots.size, dtype=np.float64)
    uplift = (ax31 - light)[ranked]
    cumulative = np.cumsum(uplift)
    base = float(light.sum())
    ks = np.clip(
        np.floor(knots * float(n_rows) + 1e-15).astype(np.int64), 0, n_rows
    )
    extra = np.zeros(knots.size, dtype=np.float64)
    mask = ks > 0
    extra[mask] = cumulative[ks[mask] - 1]
    return (base + extra) / float(n_rows)


def monotonize_upper(values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Enforce non-decreasing envelope via running maximum. Returns (mono, L1 change)."""

    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    mono = np.maximum.accumulate(raw)
    return mono, json_float(float(np.sum(np.abs(mono - raw))))


def select_f_star(phi_upper: np.ndarray, target: float, grid: np.ndarray = F_GRID_ARRAY) -> float:
    """f* = max { f : Phi_upper(f) <= target } on the pinned grid."""

    envelope = np.asarray(phi_upper, dtype=np.float64).reshape(-1)
    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    if envelope.size != knots.size:
        raise ValueError("select_f_star requires Phi_upper aligned with the grid")
    ok = envelope <= float(target) + 1e-15
    if not np.any(ok):
        return 0.0
    return json_float(knots[int(np.flatnonzero(ok)[-1])])


def lookup_phi(phi_upper: np.ndarray, fraction: float, grid: np.ndarray = F_GRID_ARRAY) -> float:
    """Conservative lookup: Phi at the smallest grid point >= fraction."""

    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    envelope = np.asarray(phi_upper, dtype=np.float64).reshape(-1)
    frac = min(1.0, max(0.0, float(fraction)))
    index = int(np.searchsorted(knots, frac, side="left"))
    index = min(index, int(knots.size) - 1)
    return json_float(envelope[index])


def kendall_tau_inversions(left: np.ndarray, right: np.ndarray) -> Tuple[float, int]:
    """Kendall tau-a and discordant-pair count between two rank vectors."""

    rank_a = np.asarray(left, dtype=np.float64).reshape(-1)
    rank_b = np.asarray(right, dtype=np.float64).reshape(-1)
    n_rows = int(rank_a.size)
    if n_rows != int(rank_b.size) or n_rows < 2:
        return 1.0, 0
    delta_a = rank_a[:, None] - rank_a[None, :]
    delta_b = rank_b[:, None] - rank_b[None, :]
    upper = np.triu_indices(n_rows, k=1)
    product = delta_a[upper] * delta_b[upper]
    n_inv = int(np.sum(product < 0.0))
    n_tie = int(np.sum(product == 0.0))
    n_pairs = n_rows * (n_rows - 1) // 2
    n_con = n_pairs - n_inv - n_tie
    tau = float(n_con - n_inv) / float(n_pairs) if n_pairs else 1.0
    return json_float(tau), n_inv


@dataclass(frozen=True)
class QKHead:
    """Q_K only. Q_A coefficients are never stored or applied."""

    bins: int
    alpha: float
    target_form: str
    feature_signature: str
    coef: np.ndarray


def load_q_k_head(path: Path = QUALITY_HEADS_PATH) -> QKHead:
    """Load Q_K from the quality head study quality-heads.json. Never bind the AX31 quality head."""

    resolved = reject_dev_reference(Path(path))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    bins = int(payload["bins_qk"])
    alpha = float(payload["alpha_qk"])
    target_form = str(payload["target_form_qk"])
    signature = str(payload["feature_signature_qk"])
    if target_form != "direct_signed":
        raise ValueError(f"the prefix certificate layer expects Q_K target_form direct_signed; got {target_form!r}")
    if signature != feature_signature(bins):
        raise ValueError("Q_K feature signature mismatch")
    coef_block = payload["coef_qk"]
    if not coef_block:
        raise ValueError("Q_K coefficient block is empty")
    coef = np.asarray(coef_block[0], dtype=np.float64)
    expected_width = 1 + len(STRUCTURAL_FEATURE_NAMES) + bins
    if coef.shape != (expected_width,):
        raise ValueError("Q_K coefficient width does not match bins")
    return QKHead(
        bins=bins,
        alpha=alpha,
        target_form=target_form,
        feature_signature=signature,
        coef=coef,
    )


def predict_q_k(
    texts: Sequence[str], head: QKHead, features: Optional[np.ndarray] = None
) -> np.ndarray:
    matrix = features if features is not None else design_matrix_g_features(
        texts, bins=int(head.bins)
    )
    return ridge_predict(head.coef, matrix)


@dataclass(frozen=True)
class View:
    kind: str
    name: str
    index: np.ndarray
    pred_source: str
    fallback: Optional[str] = None


def _sample_capped(
    rng: np.random.Generator, pool: np.ndarray, count: int, *, cap: int = FAMDOM_DUP_CAP
) -> Tuple[np.ndarray, Optional[str]]:
    """Sample ``count`` ids from ``pool`` without replacement if possible.

    Fallback: with-replacement up to ``cap`` copies of any single id.
    Returns (sample, fallback_tag). If ``count`` exceeds cap * pool, the
    sample is the full capped multiset and the tag records the shortfall.
    """

    ids = np.asarray(pool, dtype=np.int64).reshape(-1)
    need = int(count)
    if need <= 0 or ids.size == 0:
        return np.zeros(0, dtype=np.int64), "empty-pool" if need > 0 else None
    if need <= int(ids.size):
        return rng.choice(ids, size=need, replace=False), None
    max_n = int(cap) * int(ids.size)
    if need > max_n:
        copies = [ids for _ in range(int(cap))]
        return np.concatenate(copies), "dup-cap-shortfall"
    copies_full = need // int(ids.size)
    leftover = need - copies_full * int(ids.size)
    parts = [np.tile(ids, copies_full)]
    if leftover:
        parts.append(rng.choice(ids, size=leftover, replace=False))
    return np.concatenate(parts), "dup-capped"


def _famdom_sizes(family_n: int, other_n: int) -> Tuple[int, int, int, str]:
    """Return (n_focus, n_other, n_batch, fallback) under the cap-3 rule."""

    n_target = int(FAMDOM_TARGET_N)
    n_focus_target = int(round(FAMDOM_SHARE * float(n_target)))
    max_focus = FAMDOM_DUP_CAP * int(family_n)
    max_other = FAMDOM_DUP_CAP * int(other_n)
    fallback = "none"
    if max_focus <= 0:
        return 0, 0, 0, "empty-family"
    if n_focus_target <= max_focus:
        n_focus = n_focus_target
        n_other = n_target - n_focus
        if family_n < n_focus:
            fallback = "focus-dup-capped"
    else:
        n_focus = max_focus
        n_batch = int(math.ceil(float(n_focus) / FAMDOM_SHARE))
        n_other = n_batch - n_focus
        fallback = "dup-cap-shrink-n"
    if n_other > max_other:
        n_other = max_other
        fallback = (
            "dup-cap-shrink-and-other-trim" if fallback == "dup-cap-shrink-n" else "other-cap-trim"
        )
    n_batch = n_focus + n_other
    return n_focus, n_other, n_batch, fallback


def _dirichlet_counts(
    rng: np.random.Generator, n_families: int, size: int, alpha: float
) -> np.ndarray:
    weights = rng.dirichlet(np.full(int(n_families), float(alpha), dtype=np.float64))
    raw = weights * float(size)
    counts = np.floor(raw).astype(np.int64)
    leftover = int(size) - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for step in range(leftover):
        counts[int(order[step])] += 1
    return counts


def build_views(
    families: Sequence[str],
    folds: Sequence[int],
) -> Tuple[Tuple[View, ...], dict[str, Any]]:
    """Pre-registered view catalogue. Degenerate views (n<20) are skipped."""

    fold_ids = np.asarray(list(folds), dtype=np.int64)
    fam = np.asarray(list(families))
    names = tuple(sorted(dict.fromkeys(fam.tolist())))
    n_train = int(fam.size)
    views: list[View] = []
    skipped: list[dict[str, Any]] = []
    planned = {
        "dirichlet": DIRICHLET_DRAWS,
        "famdom": int(len(names) * FAMDOM_DRAWS),
        "half": HALF_DRAWS,
        "lofo": int(len(names)),
        "lofo-combined": int(len(names)),
        "oof-fold": int(FOLDS),
        "small": int(len(SMALL_SIZES) * SMALL_DRAWS),
    }

    def _accept(view: View) -> None:
        if int(view.index.size) < MIN_VIEW_N:
            skipped.append(
                {
                    "fallback": view.fallback,
                    "kind": view.kind,
                    "n": int(view.index.size),
                    "name": view.name,
                    "reason": "n<20",
                }
            )
            return
        views.append(view)

    for fold in range(FOLDS):
        index = np.flatnonzero(fold_ids == fold)
        _accept(View("oof-fold", f"oof-fold-{fold}", index, "oof"))
    for name, index in family_folds(families):
        held = np.asarray(index, dtype=np.int64)
        _accept(View("lofo", f"lofo-{name}", held, "lofo"))
        _accept(View("lofo-combined", f"lofo-combined-{name}", held.copy(), "lofo"))

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
            take, tag = _sample_capped(dir_rng, pools[name], int(count))
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
    for draw in range(HALF_DRAWS):
        chosen = half_rng.choice(universe, size=int(HALF_N), replace=False)
        _accept(View("half", f"half-{draw:02d}", chosen, "oof"))

    small_rng = np.random.default_rng(int(SMALL_SEED))
    for size in SMALL_SIZES:
        for draw in range(SMALL_DRAWS):
            chosen = small_rng.choice(universe, size=int(size), replace=False)
            _accept(View("small", f"small-{size}-{draw:03d}", chosen, "oof"))

    kind_counts = {
        kind: int(sum(1 for view in views if view.kind == kind))
        for kind in (
            "oof-fold",
            "lofo",
            "lofo-combined",
            "famdom",
            "dirichlet",
            "half",
            "small",
        )
    }
    catalogue = {
        "dirichlet_alpha": json_float(DIRICHLET_ALPHA),
        "dirichlet_n": int(DIRICHLET_N),
        "duplication_cap": int(FAMDOM_DUP_CAP),
        "famdom_fallback_by_family": famdom_fallback,
        "famdom_share": json_float(FAMDOM_SHARE),
        "famdom_target_n": int(FAMDOM_TARGET_N),
        "min_view_n": int(MIN_VIEW_N),
        "n_skipped": int(len(skipped)),
        "n_views": int(len(views)),
        "planned": planned,
        "skip_by_kind": {
            kind: int(sum(1 for row in skipped if row["kind"] == kind))
            for kind in planned
        },
        "skipped": skipped,
        "view_kind_counts": kind_counts,
    }
    return tuple(views), catalogue


def k1_mask(
    families: Sequence[str],
    q_k: np.ndarray,
    inc_k: np.ndarray,
    pred_l: np.ndarray,
    *,
    q_elig: float,
    denylist: Sequence[str] = K1_DENYLIST,
    item_cap_frac: float = K1_ITEM_CAP_FRAC,
) -> Tuple[np.ndarray, dict[str, int]]:
    """Eligibility + per-item cap. Prediction-independent cap uses predicted light total."""

    labels = tuple(families)
    pred_q = np.asarray(q_k, dtype=np.float64).reshape(-1)
    pred_inc = np.asarray(inc_k, dtype=np.float64).reshape(-1)
    light = np.asarray(pred_l, dtype=np.float64).reshape(-1)
    n_rows = int(pred_q.size)
    denied = set(denylist)
    fam_ok = np.array([label not in denied for label in labels], dtype=bool)
    q_ok = pred_q > 0.0
    if n_rows == 0:
        return np.zeros(0, dtype=bool), {"item_cap_bind": 0, "q_elig_bind": 0, "denylist_bind": 0}
    threshold = quantile_higher(pred_inc, float(q_elig))
    cost_ok = pred_inc <= float(threshold)
    item_cap = float(item_cap_frac) * float(light.sum())
    cap_ok = pred_inc <= item_cap
    base = fam_ok & q_ok & cost_ok
    chosen = base & cap_ok
    binds = {
        "denylist_bind": int(np.sum((~fam_ok) & q_ok)),
        "item_cap_bind": int(np.sum(fam_ok & q_ok & ~cap_ok)),
        "q_elig_bind": int(np.sum(fam_ok & q_ok & cap_ok & ~cost_ok)),
        "qk_nonpositive_bind": int(np.sum(fam_ok & ~q_ok)),
    }
    return chosen, binds


def order_k1(
    eligible: np.ndarray,
    q_k: np.ndarray,
    inc_k: np.ndarray,
    digests: Sequence[str],
    *,
    rule: str,
    eps: float = K1_DENSITY_EPS,
) -> np.ndarray:
    index = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if index.size == 0:
        return index.astype(np.int64)
    pred_q = np.asarray(q_k, dtype=np.float64).reshape(-1)
    pred_inc = np.asarray(inc_k, dtype=np.float64).reshape(-1)
    keys = np.asarray(list(digests))
    if rule == "orderK_qk":
        primary = -pred_q[index]
    elif rule == "orderK_density":
        primary = -pred_q[index] / np.maximum(pred_inc[index], float(eps))
    else:
        raise ValueError(f"unknown K ordering {rule!r}")
    return index[np.lexsort((keys[index], primary))]


def psi_extras(
    actual_light: np.ndarray,
    actual_ax31: np.ndarray,
    actual_k1: np.ndarray,
    in_prefix: np.ndarray,
    k_order: np.ndarray,
) -> np.ndarray:
    """Per-pick extra actual cost of K1 on top of the AX31 prefix. Vectorized."""

    ranked = np.asarray(k_order, dtype=np.int64).reshape(-1)
    if ranked.size == 0:
        return np.zeros(0, dtype=np.float64)
    light = np.asarray(actual_light, dtype=np.float64).reshape(-1)
    ax31 = np.asarray(actual_ax31, dtype=np.float64).reshape(-1)
    k1 = np.asarray(actual_k1, dtype=np.float64).reshape(-1)
    prefix = np.asarray(in_prefix, dtype=bool).reshape(-1)
    extra_if_ax31 = k1[ranked] - ax31[ranked]
    extra_if_light = k1[ranked] - light[ranked]
    return np.where(prefix[ranked], extra_if_ax31, extra_if_light)


def prefix_mask(order: np.ndarray, k_prefix: int, n: int) -> np.ndarray:
    mask = np.zeros(int(n), dtype=bool)
    take = min(int(k_prefix), int(order.size))
    if take > 0:
        mask[np.asarray(order, dtype=np.int64)[:take]] = True
    return mask


def models_from_masks(upgrade_a: np.ndarray, upgrade_k: np.ndarray) -> Tuple[str, ...]:
    chosen = []
    for ax31, k1 in zip(upgrade_a, upgrade_k):
        if k1:
            chosen.append(_K1)
        elif ax31:
            chosen.append(_AX31)
        else:
            chosen.append(_LIGHT)
    return tuple(chosen)


@dataclass(frozen=True)
class CertifiedPolicy:
    """Deployable monotone-prefix allocator. route() uses g_features only."""

    feature_version: str
    feature_signature: str
    bins: int
    alpha: float
    variant: str
    ridge_coefficients: Mapping[str, Tuple[float, ...]]
    smearing_factors: Mapping[str, float]
    recal_a_edges: Tuple[float, ...]
    recal_a_factors: Tuple[float, ...]
    recal_k_edges: Tuple[float, ...]
    recal_k_factors: Tuple[float, ...]
    qk_bins: int
    qk_alpha: float
    qk_target_form: str
    qk_feature_signature: str
    qk_coefficients: Tuple[float, ...]
    phi_upper: Tuple[float, ...]
    psi_upper: Tuple[float, ...]
    f_star: Mapping[str, float]
    m_star: Mapping[str, int]
    sort_rule: str
    k_order: str
    q_elig: float
    k1_denylist: Tuple[str, ...]
    k1_density_eps: float
    k1_item_cap_frac: float
    k1_count_cap_frac: float
    near_tie_abs: float
    near_tie_rel: float
    operating_targets: Mapping[str, float]
    official_caps: Mapping[str, float]
    stress_backstop: float
    k1_enabled: Mapping[str, bool]
    intercept_policy: str

    def to_dict(self) -> dict[str, Any]:
        return sort_mapping(
            {
                "alpha": json_float(self.alpha),
                "bins": int(self.bins),
                "f_star": {tier: json_float(self.f_star[tier]) for tier in TIERS},
                "feature_signature": self.feature_signature,
                "feature_version": self.feature_version,
                "intercept_policy": self.intercept_policy,
                "k1_count_cap_frac": json_float(self.k1_count_cap_frac),
                "k1_density_eps": json_float(self.k1_density_eps),
                "k1_denylist": list(self.k1_denylist),
                "k1_enabled": {tier: bool(self.k1_enabled[tier]) for tier in TIERS},
                "k1_item_cap_frac": json_float(self.k1_item_cap_frac),
                "k_order": self.k_order,
                "m_star": {tier: int(self.m_star[tier]) for tier in TIERS},
                "near_tie_abs": json_float(self.near_tie_abs),
                "near_tie_rel": json_float(self.near_tie_rel),
                "official_caps": {tier: json_float(self.official_caps[tier]) for tier in TIERS},
                "operating_targets": {
                    tier: json_float(self.operating_targets[tier]) for tier in TIERS
                },
                "phi_upper": json_floats(self.phi_upper),
                "psi_upper": json_floats(self.psi_upper),
                "q_elig": json_float(self.q_elig),
                "qk_alpha": json_float(self.qk_alpha),
                "qk_bins": int(self.qk_bins),
                "qk_coefficients": json_floats(self.qk_coefficients),
                "qk_feature_signature": self.qk_feature_signature,
                "qk_target_form": self.qk_target_form,
                "recal_a_edges": json_floats(self.recal_a_edges),
                "recal_a_factors": json_floats(self.recal_a_factors),
                "recal_k_edges": json_floats(self.recal_k_edges),
                "recal_k_factors": json_floats(self.recal_k_factors),
                "ridge_coefficients": {
                    name: json_floats(values) for name, values in self.ridge_coefficients.items()
                },
                "smearing_factors": {
                    name: json_float(value) for name, value in self.smearing_factors.items()
                },
                "sort_rule": self.sort_rule,
                "stress_backstop": json_float(self.stress_backstop),
                "variant": self.variant,
            }
        )

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "CertifiedPolicy":
        bins = int(payload["bins"])
        if bins not in HASH_BINS:
            raise ValueError("serialized bins are outside the the modeling foundation closed list")
        expected = feature_signature(bins)
        if payload["feature_signature"] != expected:
            raise ValueError("feature signature mismatch")
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError("feature_version mismatch")
        qk_bins = int(payload["qk_bins"])
        if payload["qk_feature_signature"] != feature_signature(qk_bins):
            raise ValueError("Q_K feature signature mismatch")
        return CertifiedPolicy(
            feature_version=str(payload["feature_version"]),
            feature_signature=str(payload["feature_signature"]),
            bins=bins,
            alpha=float(payload["alpha"]),
            variant=str(payload["variant"]),
            ridge_coefficients={
                name: tuple(float(item) for item in payload["ridge_coefficients"][name])
                for name in payload["ridge_coefficients"]
            },
            smearing_factors={
                name: float(value) for name, value in payload["smearing_factors"].items()
            },
            recal_a_edges=tuple(float(item) for item in payload["recal_a_edges"]),
            recal_a_factors=tuple(float(item) for item in payload["recal_a_factors"]),
            recal_k_edges=tuple(float(item) for item in payload["recal_k_edges"]),
            recal_k_factors=tuple(float(item) for item in payload["recal_k_factors"]),
            qk_bins=qk_bins,
            qk_alpha=float(payload["qk_alpha"]),
            qk_target_form=str(payload["qk_target_form"]),
            qk_feature_signature=str(payload["qk_feature_signature"]),
            qk_coefficients=tuple(float(item) for item in payload["qk_coefficients"]),
            phi_upper=tuple(float(item) for item in payload["phi_upper"]),
            psi_upper=tuple(float(item) for item in payload["psi_upper"]),
            f_star={tier: float(payload["f_star"][tier]) for tier in TIERS},
            m_star={tier: int(payload["m_star"][tier]) for tier in TIERS},
            sort_rule=str(payload["sort_rule"]),
            k_order=str(payload["k_order"]),
            q_elig=float(payload["q_elig"]),
            k1_denylist=tuple(str(item) for item in payload["k1_denylist"]),
            k1_density_eps=float(payload["k1_density_eps"]),
            k1_item_cap_frac=float(payload["k1_item_cap_frac"]),
            k1_count_cap_frac=float(payload["k1_count_cap_frac"]),
            near_tie_abs=float(payload["near_tie_abs"]),
            near_tie_rel=float(payload["near_tie_rel"]),
            operating_targets={tier: float(payload["operating_targets"][tier]) for tier in TIERS},
            official_caps={tier: float(payload["official_caps"][tier]) for tier in TIERS},
            stress_backstop=float(payload["stress_backstop"]),
            k1_enabled={tier: bool(payload["k1_enabled"][tier]) for tier in TIERS},
            intercept_policy=str(payload["intercept_policy"]),
        )

    def predict_arrays(self, texts: Sequence[str]) -> dict[str, Any]:
        features = design_matrix_g_features(texts, bins=int(self.bins))
        if int(self.qk_bins) == int(self.bins):
            features_qk = features
        else:
            features_qk = design_matrix_g_features(texts, bins=int(self.qk_bins))
        heads = FittedHeads(
            variant=self.variant,
            alpha=float(self.alpha),
            coefs={
                name: np.asarray(values, dtype=np.float64)
                for name, values in self.ridge_coefficients.items()
            },
            smears={name: float(value) for name, value in self.smearing_factors.items()},
            n_clipped_inc_a=0,
            n_clipped_inc_k=0,
        )
        pred_l, _pred_a, _pred_k, inc_a, inc_k = predict_heads(features, heads)
        recal_a = apply_recal_baked(
            inc_a,
            np.asarray(self.recal_a_edges, dtype=np.float64),
            np.asarray(self.recal_a_factors, dtype=np.float64),
        )
        recal_k = apply_recal_baked(
            inc_k,
            np.asarray(self.recal_k_edges, dtype=np.float64),
            np.asarray(self.recal_k_factors, dtype=np.float64),
        )
        q_k = ridge_predict(np.asarray(self.qk_coefficients, dtype=np.float64), features_qk)
        families = tuple(family_of_text(text) for text in texts)
        digests = content_digests(texts)
        return {
            "digests": digests,
            "families": families,
            "inc_a": recal_a,
            "inc_k": recal_k,
            "pred_l": pred_l,
            "q_k": q_k,
        }

    def allocate(self, tier: str, texts: Sequence[str]) -> Tuple[str, ...]:
        predicted = self.predict_arrays(texts)
        return allocate_from_arrays(
            tier,
            predicted["inc_a"],
            predicted["inc_k"],
            predicted["pred_l"],
            predicted["q_k"],
            predicted["families"],
            predicted["digests"],
            self,
        )

    def route(self, tier: str, texts: Sequence[str]) -> Tuple[str, ...]:
        """Runtime path: feature extraction goes entirely through g_features."""

        return self.allocate(tier, texts)


def _psi_lookup(psi_upper: Sequence[float], count: int) -> float:
    table = np.asarray(list(psi_upper), dtype=np.float64).reshape(-1)
    if table.size == 0:
        return 0.0
    index = min(max(int(count), 0), int(table.size) - 1)
    return json_float(table[index])


def allocate_from_arrays(
    tier: str,
    inc_a: np.ndarray,
    inc_k: np.ndarray,
    pred_l: np.ndarray,
    q_k: np.ndarray,
    families: Sequence[str],
    digests: Sequence[str],
    policy: CertifiedPolicy,
    *,
    force_invariant_fail: bool = False,
) -> Tuple[str, ...]:
    """Deterministic prefix allocator with self-certification and all-light fallback."""

    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    pred_inc_a = np.asarray(inc_a, dtype=np.float64).reshape(-1)
    pred_inc_k = np.asarray(inc_k, dtype=np.float64).reshape(-1)
    light = np.asarray(pred_l, dtype=np.float64).reshape(-1)
    pred_q = np.asarray(q_k, dtype=np.float64).reshape(-1)
    n_rows = int(pred_inc_a.size)
    if n_rows == 0:
        return tuple()
    if not (
        pred_inc_k.size
        == light.size
        == pred_q.size
        == len(families)
        == len(digests)
        == n_rows
    ):
        raise ValueError("allocate_from_arrays requires aligned inputs")
    order = sort_order(policy.sort_rule, pred_inc_a, light, digests)
    f_star = float(policy.f_star[tier])
    k_star = prefix_k(f_star, n_rows)
    upgrade_a = prefix_mask(order, k_star, n_rows)
    upgrade_k = np.zeros(n_rows, dtype=bool)
    if bool(policy.k1_enabled.get(tier, False)) and int(policy.m_star[tier]) > 0:
        eligible, _binds = k1_mask(
            families,
            pred_q,
            pred_inc_k,
            light,
            q_elig=float(policy.q_elig),
            denylist=policy.k1_denylist,
            item_cap_frac=float(policy.k1_item_cap_frac),
        )
        ranked = order_k1(
            eligible,
            pred_q,
            pred_inc_k,
            digests,
            rule=policy.k_order,
            eps=float(policy.k1_density_eps),
        )
        count_cap = int(math.floor(float(policy.k1_count_cap_frac) * float(n_rows)))
        take = min(int(policy.m_star[tier]), int(ranked.size), count_cap)
        if take > 0:
            upgrade_k[ranked[:take]] = True
            upgrade_a[ranked[:take]] = True

    def _accounted(mask_a: np.ndarray, mask_k: np.ndarray) -> Tuple[float, int, int]:
        k_used = int(np.sum(mask_a | mask_k))
        m_used = int(np.sum(mask_k))
        frac = float(k_used) / float(n_rows) if n_rows else 0.0
        phi = lookup_phi(np.asarray(policy.phi_upper, dtype=np.float64), frac)
        psi = _psi_lookup(policy.psi_upper, m_used)
        return phi + psi, k_used, m_used

    target = float(policy.operating_targets[tier])
    accounted, k_used, m_used = _accounted(upgrade_a, upgrade_k)
    if force_invariant_fail:
        accounted = target + 1.0
    if accounted > target + 1e-15 or k_used > k_star or m_used > int(policy.m_star[tier]):
        k_ranked = order_k1(
            upgrade_k,
            pred_q,
            pred_inc_k,
            digests,
            rule=policy.k_order,
            eps=float(policy.k1_density_eps),
        )
        for index in k_ranked[::-1]:
            upgrade_k[int(index)] = False
            accounted, k_used, m_used = _accounted(upgrade_a, upgrade_k)
            if accounted <= target + 1e-15 and k_used <= k_star and m_used <= int(policy.m_star[tier]):
                break
        if accounted > target + 1e-15 or k_used > k_star:
            a_order = order
            for index in a_order[::-1]:
                if upgrade_k[int(index)]:
                    continue
                if not upgrade_a[int(index)]:
                    continue
                upgrade_a[int(index)] = False
                accounted, k_used, m_used = _accounted(upgrade_a, upgrade_k)
                if accounted <= target + 1e-15 and k_used <= k_star:
                    break
        if accounted > target + 1e-15:
            return tuple(_LIGHT for _ in range(n_rows))
    return models_from_masks(upgrade_a, upgrade_k)


def allocate(tier: str, texts: Sequence[str], policy: CertifiedPolicy) -> Tuple[str, ...]:
    return policy.allocate(tier, texts)


def _lofo_cost(
    features: np.ndarray, costs: np.ndarray, families: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LOFO heads + complement-fit recalibration. Uses the cost certificate layer public fit/predict."""

    n_rows = int(features.shape[0])
    pred_l = np.empty(n_rows, dtype=np.float64)
    inc_a = np.empty(n_rows, dtype=np.float64)
    inc_k = np.empty(n_rows, dtype=np.float64)
    actual_a, actual_k = actual_increments(costs)
    for _name, held_index in family_folds(families):
        held = np.zeros(n_rows, dtype=bool)
        held[held_index] = True
        train = ~held
        heads = fit_heads(
            features[train], costs[train], variant=LOCKED_VARIANT, alpha=LOCKED_ALPHA
        )
        _p_l_tr, _a_tr, _k_tr, i_a_tr, i_k_tr = predict_heads(features[train], heads)
        p_l_te, _a_te, _k_te, i_a_te, i_k_te = predict_heads(features[held], heads)
        rec_a = fit_recal(i_a_tr, actual_a[train])
        rec_k = fit_recal(i_k_tr, actual_k[train])
        pred_l[held] = p_l_te
        inc_a[held] = apply_recal(rec_a, i_a_te)
        inc_k[held] = apply_recal(rec_k, i_k_te)
    return pred_l, inc_a, inc_k


def _lofo_qk(
    features: np.ndarray, target: np.ndarray, families: Sequence[str], *, alpha: float
) -> np.ndarray:
    predicted = np.empty(int(features.shape[0]), dtype=np.float64)
    for _name, held_index in family_folds(families):
        held = np.zeros(int(features.shape[0]), dtype=bool)
        held[held_index] = True
        train = ~held
        coef = ridge_fit(features[train], target[train], alpha=float(alpha))
        predicted[held] = ridge_predict(coef, features[held])
    return predicted


def _gather(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.asarray(index, dtype=np.int64)]


def _combo_key(sort_rule: str, q_elig: float, k_order: str) -> str:
    return f"{sort_rule}|q_elig={q_elig:.2f}|{k_order}"


def _select_combo(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["weighted_q_oof"]),
            -float(row["margin_sum"]),
            0 if row["sort_rule"] == "sortA_pred_inc" else 1,
            0 if row["k_order"] == "orderK_qk" else 1,
            -float(row["q_elig"]),
        ),
    )
    return ordered[0]


def _parent_assignments(bundle: TrainBundle) -> dict[str, Tuple[str, ...]]:
    from ossp_router.feasibility_ladder import load_bundled_artifact, make_submission

    artifact = load_bundled_artifact()
    assigned: dict[str, Tuple[str, ...]] = {}
    for tier in TIERS:
        plan = make_submission(bundle.inputs, bundle.policy, artifact, tier)
        assigned[tier] = tuple(decision.model_id for decision in plan.submission.decisions)
    return assigned


def _score_mean(scores: np.ndarray, model_ids: Sequence[str]) -> float:
    columns = np.asarray([MODEL_IDS.index(model_id) for model_id in model_ids], dtype=np.int64)
    rows = np.arange(int(scores.shape[0]), dtype=np.int64)
    return json_float(float(scores[rows, columns].mean()))


def _episode_scores(scores: np.ndarray, model_ids: Sequence[str]) -> np.ndarray:
    columns = np.asarray([MODEL_IDS.index(model_id) for model_id in model_ids], dtype=np.int64)
    rows = np.arange(int(scores.shape[0]), dtype=np.int64)
    return scores[rows, columns]


def _realized_ratio(costs: np.ndarray, model_ids: Sequence[str]) -> float:
    columns = np.asarray([MODEL_IDS.index(model_id) for model_id in model_ids], dtype=np.int64)
    rows = np.arange(int(costs.shape[0]), dtype=np.int64)
    light = float(costs[:, 0].sum())
    if light <= 0.0:
        return float("inf")
    return json_float(float(costs[rows, columns].sum()) / light)


def _count_models(model_ids: Sequence[str]) -> dict[str, int]:
    counts = {model_id: 0 for model_id in MODEL_IDS}
    for model_id in model_ids:
        counts[model_id] += 1
    return counts


def locked_record() -> Mapping[str, Any]:
    return sort_mapping(
        {
            "backstop_targets": {tier: json_float(BACKSTOP_TARGETS[tier]) for tier in TIERS},
            "bootstrap_draws": int(BOOTSTRAP_DRAWS),
            "bootstrap_seed": int(BOOTSTRAP_SEED),
            "cost_head_lock": COST_HEAD_LOCK,
            "dirichlet_alpha": json_float(DIRICHLET_ALPHA),
            "dirichlet_draws": int(DIRICHLET_DRAWS),
            "dirichlet_n": int(DIRICHLET_N),
            "dirichlet_seed": int(DIRICHLET_SEED),
            "duplication_cap": int(FAMDOM_DUP_CAP),
            "f_grid": json_floats(F_GRID),
            "famdom_draws": int(FAMDOM_DRAWS),
            "famdom_seed": int(FAMDOM_SEED),
            "famdom_share": json_float(FAMDOM_SHARE),
            "famdom_target_n": int(FAMDOM_TARGET_N),
            "fast_drift_coef": json_float(FAST_DRIFT_COEF),
            "feature_signature": feature_signature(LOCKED_BINS),
            "feature_version": FEATURE_VERSION,
            "float64_table_note": FLOAT64_TABLE_NOTE,
            "fold_seed": int(FOLD_SEED_PREFIX_CERT),
            "folds": int(FOLDS),
            "h2_6_exceedance_max": json_float(H2_6_EXCEEDANCE_MAX),
            "half_draws": int(HALF_DRAWS),
            "half_n": int(HALF_N),
            "half_seed": int(HALF_SEED),
            "hash_bins_allowed": list(HASH_BINS),
            "intercept_policy": INTERCEPT_POLICY,
            "k1_count_cap_frac": json_float(K1_COUNT_CAP_FRAC),
            "k1_density_eps": json_float(K1_DENSITY_EPS),
            "k1_density_rule": K1_DENSITY_RULE,
            "k1_denylist": list(K1_DENYLIST),
            "k1_item_cap_frac": json_float(K1_ITEM_CAP_FRAC),
            "k_orders": list(K_ORDERS),
            "locked_alpha": json_float(LOCKED_ALPHA),
            "locked_bins": int(LOCKED_BINS),
            "locked_variant": LOCKED_VARIANT,
            "min_view_n": int(MIN_VIEW_N),
            "near_tie_abs": json_float(NEAR_TIE_ABS),
            "near_tie_rel": json_float(NEAR_TIE_REL),
            "near_tie_rule": NEAR_TIE_RULE,
            "official_caps": {tier: json_float(OFFICIAL_CAPS[tier]) for tier in TIERS},
            "operating_targets": {tier: json_float(OPERATING_TARGETS[tier]) for tier in TIERS},
            "parent_f_exact": PARENT_F_EXACT,
            "parent_f_pins": PARENT_F_PINS,
            "q_elig": [json_float(item) for item in Q_ELIG],
            "recal_clip": [json_float(RECAL_CLIP[0]), json_float(RECAL_CLIP[1])],
            "recal_n_bins": int(RECAL_BINS),
            "selection_criterion": SELECTION_CRITERION,
            "small_draws": int(SMALL_DRAWS),
            "small_seed": int(SMALL_SEED),
            "small_sizes": list(SMALL_SIZES),
            "sort_rules": list(SORT_RULES),
            "stress_backstop": json_float(STRESS_BACKSTOP),
            "tier_weights": {tier: json_float(TIER_WEIGHTS[tier]) for tier in TIERS},
        }
    )


def assemble_report(
    *,
    identity: Mapping[str, Any],
    locked: Mapping[str, Any],
    observed: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    decision: str,
) -> dict[str, Any]:
    report = {
        "decision": decision,
        "dev_opened": False,
        "diagnostic": diagnostic,
        "experiment": EXPERIMENT,
        "identity": identity,
        "locked": locked,
        "observed": observed,
        "q_a_used": False,
        "quality_entered_cost_selection": False,
        "report_type": REPORT_TYPE,
        "schema_version": SCHEMA_VERSION,
    }
    if report["dev_opened"] is not False:
        raise RuntimeError("the prefix certificate layer report must assert dev_opened is false")
    if report["quality_entered_cost_selection"] is not False:
        raise RuntimeError("the prefix certificate layer report must assert quality_entered_cost_selection is false")
    if report["q_a_used"] is not False:
        raise RuntimeError("the prefix certificate layer report must assert q_a_used is false")
    return sort_mapping(report)


def decide(gates: Mapping[str, Any]) -> str:
    """No rescue. k1-off is acceptable when only H2-10 fails."""

    ordered = (
        ("h2_1_oof_gain", "h2-1-oof-gain"),
        ("h2_2_fold_wins", "h2-2-fold-wins"),
        ("h2_3_bootstrap", "h2-3-bootstrap"),
        ("h2_4_lofo", "h2-4-lofo"),
        ("h2_5_lofo_worst_family", "h2-5-lofo-worst-family"),
        ("h2_6_heldout", "h2-6-heldout-exceedance"),
        ("h2_7_backstop", "h2-7-backstop"),
        ("h2_8_operating", "h2-8-operating"),
        ("h2_9_k1_certificate", "h2-9-k1-certificate"),
        ("h2_11_fast_drift", "h2-11-fast-drift"),
    )
    for key, slug in ordered:
        if not bool(gates[key]["pass"]):
            return f"record-prefix_cert-close-{slug}"
    if not bool(gates["h2_10_premium_k1"]["pass"]):
        return DECISION_K1_OFF
    return DECISION_PASS


def _envelope_from_matrix(phi_matrix: np.ndarray) -> dict[str, Any]:
    raw_upper = np.max(phi_matrix, axis=0) if phi_matrix.size else np.ones(F_GRID_ARRAY.size)
    mono, change = monotonize_upper(raw_upper)
    p99 = np.quantile(phi_matrix, 0.99, axis=0) if phi_matrix.shape[0] else raw_upper
    p50 = np.quantile(phi_matrix, 0.50, axis=0) if phi_matrix.shape[0] else raw_upper
    return {
        "change": change,
        "monotonized": bool(change > 0.0),
        "phi_p50": p50,
        "phi_p99": p99,
        "phi_upper": mono,
        "phi_upper_raw": raw_upper,
    }


def _argmax_names(phi_matrix: np.ndarray, names: Sequence[str]) -> list[str]:
    chosen = []
    n_pts = int(phi_matrix.shape[1]) if phi_matrix.size else 0
    for col in range(n_pts):
        column = phi_matrix[:, col]
        best = float(column.max()) if column.size else 1.0
        winners = [names[row] for row, value in enumerate(column) if float(value) == best]
        chosen.append(min(winners) if winners else "")
    return chosen


def fit_and_evaluate(bundle: TrainBundle) -> dict[str, Any]:
    """Train-only the prefix certificate layer fit. Q_A is never constructed or applied."""

    folds = group_folds(bundle.episodes, folds=FOLDS, seed=FOLD_SEED_PREFIX_CERT)
    fold_ids = np.asarray(list(folds), dtype=np.int64)
    families = bundle.families
    fam_arr = np.asarray(list(families))
    n_train = int(bundle.costs.shape[0])
    texts = bundle.texts
    digests = content_digests(texts)
    costs = np.asarray(bundle.costs, dtype=np.float64)
    scores = np.asarray(bundle.scores, dtype=np.float64)
    actual_l = costs[:, 0]
    actual_a = costs[:, 1]
    actual_k = costs[:, 2]
    score_l = scores[:, 0]
    score_a = scores[:, 1]
    score_k = scores[:, 2]
    actual_inc_a, actual_inc_k = actual_increments(costs)

    features = feature_matrix(texts, bins=LOCKED_BINS)
    qk_head = load_q_k_head()
    if int(qk_head.bins) != LOCKED_BINS:
        features_qk = feature_matrix(texts, bins=int(qk_head.bins))
    else:
        features_qk = features

    oof_l, _oof_a, _oof_k, oof_inc_a_raw, oof_inc_k_raw, _clip = oof_incremental_costs(
        features, costs, folds, variant=LOCKED_VARIANT, alpha=LOCKED_ALPHA
    )
    rec_a = fit_recal(oof_inc_a_raw, actual_inc_a)
    rec_k = fit_recal(oof_inc_k_raw, actual_inc_k)
    oof_inc_a = apply_recal(rec_a, oof_inc_a_raw)
    oof_inc_k = apply_recal(rec_k, oof_inc_k_raw)
    ratio_a = float(actual_inc_a.sum()) / float(oof_inc_a.sum()) if float(oof_inc_a.sum()) > 0 else None
    ratio_k = float(actual_inc_k.sum()) / float(oof_inc_k.sum()) if float(oof_inc_k.sum()) > 0 else None

    full_heads = fit_heads(features, costs, variant=LOCKED_VARIANT, alpha=LOCKED_ALPHA)
    full_l, _fa, _fk, full_inc_a_raw, full_inc_k_raw = predict_heads(features, full_heads)
    full_rec_a = fit_recal(full_inc_a_raw, actual_inc_a)
    full_rec_k = fit_recal(full_inc_k_raw, actual_inc_k)
    full_inc_a = apply_recal(full_rec_a, full_inc_a_raw)
    _full_inc_k = apply_recal(full_rec_k, full_inc_k_raw)

    oof_ranks = np.empty(n_train, dtype=np.float64)
    full_ranks = np.empty(n_train, dtype=np.float64)
    oof_ranks[sort_pred_inc(oof_inc_a, digests)] = np.arange(n_train, dtype=np.float64)
    full_ranks[sort_pred_inc(full_inc_a, digests)] = np.arange(n_train, dtype=np.float64)
    tau, n_inv = kendall_tau_inversions(oof_ranks, full_ranks)

    qk_target = score_k - score_a
    oof_qk = oof_predict(features_qk, qk_target, folds, alpha=float(qk_head.alpha))
    lofo_l, lofo_inc_a, lofo_inc_k = _lofo_cost(features, costs, families)
    lofo_qk = _lofo_qk(features_qk, qk_target, families, alpha=float(qk_head.alpha))
    _full_qk = predict_q_k(texts, qk_head, features=features_qk)

    views, catalogue = build_views(families, folds)
    view_names = tuple(view.name for view in views)

    def _pred_pack(view: View) -> dict[str, np.ndarray]:
        src_l = oof_l if view.pred_source == "oof" else lofo_l
        src_ia = oof_inc_a if view.pred_source == "oof" else lofo_inc_a
        src_ik = oof_inc_k if view.pred_source == "oof" else lofo_inc_k
        src_q = oof_qk if view.pred_source == "oof" else lofo_qk
        idx = view.index
        return {
            "actual_a": _gather(actual_a, idx),
            "actual_k": _gather(actual_k, idx),
            "actual_l": _gather(actual_l, idx),
            "families": fam_arr[idx],
            "inc_a": _gather(src_ia, idx),
            "inc_k": _gather(src_ik, idx),
            "index": idx,
            "pred_l": _gather(src_l, idx),
            "q_k": _gather(src_q, idx),
            "score_a": _gather(score_a, idx),
            "score_k": _gather(score_k, idx),
            "score_l": _gather(score_l, idx),
        }

    view_packs = [_pred_pack(view) for view in views]
    view_digests = [tuple(digests[int(item)] for item in view.index) for view in views]

    phi_by_sort: dict[str, np.ndarray] = {}
    q_by_sort: dict[str, np.ndarray] = {}
    orders: dict[str, list[np.ndarray]] = {}
    envelopes: dict[str, dict[str, Any]] = {}
    argmax: dict[str, list[str]] = {}
    for rule in SORT_RULES:
        phi_rows = []
        q_rows = []
        order_rows = []
        for pack, digest in zip(view_packs, view_digests):
            order = sort_order(rule, pack["inc_a"], pack["pred_l"], digest)
            order_rows.append(order)
            phi_rows.append(phi_view(pack["actual_l"], pack["actual_a"], order))
            q_rows.append(q_view(pack["score_l"], pack["score_a"], order))
        phi_matrix = np.vstack(phi_rows)
        q_matrix = np.vstack(q_rows)
        phi_by_sort[rule] = phi_matrix
        q_by_sort[rule] = q_matrix
        orders[rule] = order_rows
        envelopes[rule] = _envelope_from_matrix(phi_matrix)
        argmax[rule] = _argmax_names(phi_matrix, view_names)

    fold_positions = [index for index, view in enumerate(views) if view.kind == "oof-fold"]

    def _q_oof(rule: str, fraction: float) -> float:
        col = int(round(float(fraction) * 100.0))
        col = min(max(col, 0), 100)
        return json_float(float(np.mean([q_by_sort[rule][row, col] for row in fold_positions])))

    def _operating_block(rule: str, targets: Mapping[str, float]) -> dict[str, Any]:
        env = envelopes[rule]
        block = {}
        for tier in TIERS:
            f_star = select_f_star(env["phi_upper"], float(targets[tier]))
            col = int(round(f_star * 100.0))
            phi_at = json_float(env["phi_upper"][col])
            block[tier] = {
                "f_star": json_float(f_star),
                "k_star_train": prefix_k(f_star, n_train),
                "margin_official": json_float(float(OFFICIAL_CAPS[tier]) - phi_at),
                "margin_target": json_float(float(targets[tier]) - phi_at),
                "phi_upper": phi_at,
                "q_oof": _q_oof(rule, f_star),
            }
        return block

    combo_rows = []
    combo_psi: dict[str, np.ndarray] = {}
    combo_mstar: dict[str, dict[str, int]] = {}
    guard_bind: dict[str, dict[str, int]] = {}
    m_max_global = 0
    for rule in SORT_RULES:
        internal_op = _operating_block(rule, OPERATING_TARGETS)
        for q_elig in Q_ELIG:
            for k_rule in K_ORDERS:
                key = _combo_key(rule, q_elig, k_rule)
                extras_per_view: list[np.ndarray] = []
                caps_per_view: list[int] = []
                binds_total = {
                    "all_light_fallback": 0,
                    "count_cap_bind": 0,
                    "denylist_bind": 0,
                    "item_cap_bind": 0,
                    "q_elig_bind": 0,
                    "qk_nonpositive_bind": 0,
                }
                eligible_sizes = []
                for pack, digest, order, view in zip(
                    view_packs, view_digests, orders[rule], views
                ):
                    eligible, binds = k1_mask(
                        pack["families"].tolist(),
                        pack["q_k"],
                        pack["inc_k"],
                        pack["pred_l"],
                        q_elig=float(q_elig),
                    )
                    for name, value in binds.items():
                        binds_total[name] = binds_total.get(name, 0) + int(value)
                    n_view = int(pack["actual_l"].size)
                    count_cap = int(math.floor(K1_COUNT_CAP_FRAC * float(n_view)))
                    ranked = order_k1(
                        eligible, pack["q_k"], pack["inc_k"], digest, rule=k_rule
                    )
                    take_cap = min(int(ranked.size), count_cap)
                    if int(ranked.size) > count_cap:
                        binds_total["count_cap_bind"] += 1
                    eligible_sizes.append(int(take_cap))
                    # Psi is certified additively; extras use the Balanced f* prefix
                    # as the representative AX31 base (Fast has no K1; Premium uses
                    # the same extras table — extra K1 cost does not depend on how
                    # far the AX31 prefix extends except via the in-prefix flag).
                    # Compute extras at each tier's f* and take the elementwise max
                    # extra path (conservative: charge as if the pick was light).
                    light_sum = float(pack["actual_l"].sum())
                    extra_if_light = (
                        pack["actual_k"][ranked[:take_cap]] - pack["actual_l"][ranked[:take_cap]]
                    )
                    extra_if_ax31 = (
                        pack["actual_k"][ranked[:take_cap]] - pack["actual_a"][ranked[:take_cap]]
                    )
                    # Conservative extras: max of the two (never a predicted denom).
                    extras = np.maximum(extra_if_light, extra_if_ax31)
                    if light_sum <= 0.0:
                        ratios = np.zeros(int(extras.size), dtype=np.float64)
                    else:
                        ratios = np.cumsum(extras) / light_sum
                    extras_per_view.append(ratios)
                    caps_per_view.append(int(take_cap))
                m_max = int(max(caps_per_view) if caps_per_view else 0)
                m_max_global = max(m_max_global, m_max)
                psi_upper = np.zeros(m_max + 1, dtype=np.float64)
                for ratios, cap in zip(extras_per_view, caps_per_view):
                    padded = np.zeros(m_max + 1, dtype=np.float64)
                    if cap > 0 and ratios.size:
                        padded[1 : cap + 1] = ratios[:cap]
                        if cap < m_max:
                            padded[cap + 1 :] = ratios[cap - 1]
                    psi_upper = np.maximum(psi_upper, padded)
                combo_psi[key] = psi_upper
                guard_bind[key] = binds_total
                m_star = {"fast": 0}
                for tier in ("balanced", "premium"):
                    _f_star = float(internal_op[tier]["f_star"])
                    phi_at = float(internal_op[tier]["phi_upper"])
                    target = float(OPERATING_TARGETS[tier])
                    allowed = int(psi_upper.size) - 1
                    chosen_m = 0
                    for count in range(0, allowed + 1):
                        if phi_at + float(psi_upper[count]) <= target + 1e-15:
                            chosen_m = count
                    m_star[tier] = chosen_m
                combo_mstar[key] = m_star

                # Q_oof at certified points: mean over the 5 OOF folds, float64.
                q_fast = float(internal_op["fast"]["q_oof"])
                q_bal_folds = []
                q_prem_folds = []
                for row, view in zip(fold_positions, [views[i] for i in fold_positions]):
                    pack = view_packs[row]
                    digest = view_digests[row]
                    order = orders[rule][row]
                    for tier, sink, fraction, m_use in (
                        ("balanced", q_bal_folds, float(internal_op["balanced"]["f_star"]), m_star["balanced"]),
                        ("premium", q_prem_folds, float(internal_op["premium"]["f_star"]), m_star["premium"]),
                    ):
                        n_view = int(pack["actual_l"].size)
                        k_use = prefix_k(fraction, n_view)
                        mask_a = prefix_mask(order, k_use, n_view)
                        mask_k = np.zeros(n_view, dtype=bool)
                        if m_use > 0:
                            eligible, _b = k1_mask(
                                pack["families"].tolist(),
                                pack["q_k"],
                                pack["inc_k"],
                                pack["pred_l"],
                                q_elig=float(q_elig),
                            )
                            ranked = order_k1(
                                eligible, pack["q_k"], pack["inc_k"], digest, rule=k_rule
                            )
                            count_cap = int(math.floor(K1_COUNT_CAP_FRAC * float(n_view)))
                            take = min(int(m_use), int(ranked.size), count_cap)
                            if take > 0:
                                mask_k[ranked[:take]] = True
                                mask_a[ranked[:take]] = True
                        chosen = np.where(
                            mask_k, pack["score_k"], np.where(mask_a, pack["score_a"], pack["score_l"])
                        )
                        sink.append(float(chosen.mean()))
                q_bal = json_float(float(np.mean(q_bal_folds))) if q_bal_folds else q_fast
                q_prem = json_float(float(np.mean(q_prem_folds))) if q_prem_folds else q_fast
                weighted = json_float(weighted_final(q_fast, q_bal, q_prem))
                margin_sum = json_float(
                    sum(float(internal_op[tier]["margin_target"]) for tier in TIERS)
                )
                combo_rows.append(
                    {
                        "eligible_pool_max": int(max(eligible_sizes) if eligible_sizes else 0),
                        "eligible_pool_mean": json_float(
                            float(np.mean(eligible_sizes)) if eligible_sizes else 0.0
                        ),
                        "eligible_pool_min": int(min(eligible_sizes) if eligible_sizes else 0),
                        "f_star": {tier: json_float(internal_op[tier]["f_star"]) for tier in TIERS},
                        "guard_binds": binds_total,
                        "k_order": k_rule,
                        "key": key,
                        "m_star": dict(m_star),
                        "margin_sum": margin_sum,
                        "q_elig": json_float(q_elig),
                        "q_oof": {
                            "balanced": json_float(q_bal),
                            "fast": json_float(q_fast),
                            "premium": json_float(q_prem),
                        },
                        "sort_rule": rule,
                        "weighted_q_oof": weighted,
                    }
                )
    selected_row = _select_combo(combo_rows)
    selected_key = str(selected_row["key"])
    selected_sort = str(selected_row["sort_rule"])
    selected_q_elig = float(selected_row["q_elig"])
    selected_k_order = str(selected_row["k_order"])
    selected_env = envelopes[selected_sort]
    selected_phi = selected_env["phi_upper"]
    selected_psi = combo_psi[selected_key]
    selected_mstar = dict(combo_mstar[selected_key])
    selected_mstar["fast"] = 0
    internal_op = _operating_block(selected_sort, OPERATING_TARGETS)
    official_op = _operating_block(selected_sort, OFFICIAL_CAPS)
    backstop_op = _operating_block(selected_sort, BACKSTOP_TARGETS)

    # Build OOF / LOFO allocations at the selected operating points (float64 + official).
    def _route_pack(
        pack: dict[str, np.ndarray],
        digest: Sequence[str],
        order: np.ndarray,
        *,
        k1_on: bool,
    ) -> dict[str, Tuple[str, ...]]:
        n_view = int(pack["actual_l"].size)
        assigned: dict[str, Tuple[str, ...]] = {}
        for tier in TIERS:
            fraction = float(internal_op[tier]["f_star"])
            k_use = prefix_k(fraction, n_view)
            mask_a = prefix_mask(order, k_use, n_view)
            mask_k = np.zeros(n_view, dtype=bool)
            m_use = int(selected_mstar[tier]) if k1_on else 0
            if tier != "fast" and m_use > 0:
                eligible, _b = k1_mask(
                    pack["families"].tolist(),
                    pack["q_k"],
                    pack["inc_k"],
                    pack["pred_l"],
                    q_elig=selected_q_elig,
                )
                ranked = order_k1(
                    eligible, pack["q_k"], pack["inc_k"], digest, rule=selected_k_order
                )
                count_cap = int(math.floor(K1_COUNT_CAP_FRAC * float(n_view)))
                take = min(m_use, int(ranked.size), count_cap)
                if take > 0:
                    mask_k[ranked[:take]] = True
                    mask_a[ranked[:take]] = True
            assigned[tier] = models_from_masks(mask_a, mask_k)
        return assigned

    oof_models: dict[str, list[str]] = {tier: [""] * n_train for tier in TIERS}
    for row, view in zip(fold_positions, [views[i] for i in fold_positions]):
        assigned = _route_pack(
            view_packs[row], view_digests[row], orders[selected_sort][row], k1_on=True
        )
        for local, global_i in enumerate(view.index):
            for tier in TIERS:
                oof_models[tier][int(global_i)] = assigned[tier][local]
    oof_models_t = {tier: tuple(oof_models[tier]) for tier in TIERS}
    if any(item == "" for tier in TIERS for item in oof_models[tier]):
        raise RuntimeError("OOF allocation left an unassigned episode")

    lofo_models: dict[str, list[str]] = {tier: [""] * n_train for tier in TIERS}
    lofo_family_gain: dict[str, dict[str, Any]] = {}
    parent_ids = _parent_assignments(bundle)
    parent_fold_q = {tier: [] for tier in TIERS}
    ours_fold_q = {tier: [] for tier in TIERS}
    for fold in range(FOLDS):
        mask = fold_ids == fold
        for tier in TIERS:
            parent_fold_q[tier].append(
                json_float(float(_episode_scores(scores[mask], tuple(np.asarray(parent_ids[tier])[mask])).mean()))
            )
            ours_fold_q[tier].append(
                json_float(float(_episode_scores(scores[mask], tuple(np.asarray(oof_models_t[tier])[mask])).mean()))
            )

    for name, held in family_folds(families):
        held_idx = np.asarray(held, dtype=np.int64)
        view_i = next(i for i, view in enumerate(views) if view.name == f"lofo-{name}")
        assigned = _route_pack(
            view_packs[view_i], view_digests[view_i], orders[selected_sort][view_i], k1_on=True
        )
        for local, global_i in enumerate(held_idx):
            for tier in TIERS:
                lofo_models[tier][int(global_i)] = assigned[tier][local]
        fam_gain = {}
        for tier in TIERS:
            ours = float(
                _episode_scores(scores[held_idx], tuple(np.asarray(assigned[tier]))).mean()
            )
            parent = float(
                _episode_scores(
                    scores[held_idx], tuple(np.asarray(parent_ids[tier])[held_idx])
                ).mean()
            )
            fam_gain[tier] = {
                "gain": json_float(ours - parent),
                "n": int(held_idx.size),
                "ours": json_float(ours),
                "parent": json_float(parent),
            }
        weighted_ours = weighted_final(
            fam_gain["fast"]["ours"], fam_gain["balanced"]["ours"], fam_gain["premium"]["ours"]
        )
        weighted_parent = weighted_final(
            fam_gain["fast"]["parent"],
            fam_gain["balanced"]["parent"],
            fam_gain["premium"]["parent"],
        )
        lofo_family_gain[name] = {
            "n": int(held_idx.size),
            "per_tier": fam_gain,
            "weighted_gain": json_float(weighted_ours - weighted_parent),
            "weighted_ours": json_float(weighted_ours),
            "weighted_parent": json_float(weighted_parent),
        }
    lofo_models_t = {tier: tuple(lofo_models[tier]) for tier in TIERS}
    if any(item == "" for tier in TIERS for item in lofo_models[tier]):
        raise RuntimeError("LOFO allocation left an unassigned episode")

    official_oof = official_score(bundle.inputs, bundle.outcomes, bundle.policy, oof_models_t)
    official_lofo = official_score(bundle.inputs, bundle.outcomes, bundle.policy, lofo_models_t)
    float_oof = {tier: _score_mean(scores, oof_models_t[tier]) for tier in TIERS}
    float_oof_weighted = json_float(
        weighted_final(float_oof["fast"], float_oof["balanced"], float_oof["premium"])
    )
    official_oof_q = {tier: float(official_oof["tiers"][tier]["quality_score"]) for tier in TIERS}
    official_agree = {
        tier: json_float(abs(float_oof[tier] - official_oof_q[tier])) for tier in TIERS
    }
    official_agree["max"] = json_float(max(official_agree[tier] for tier in TIERS))
    official_agree["within_1e_12"] = bool(official_agree["max"] <= 1e-12)

    parent_weighted = float(PARENT_F_PINS["weighted"])
    oof_gain = {
        tier: json_float(official_oof_q[tier] - float(PARENT_F_PINS[tier]["quality"]))
        for tier in TIERS
    }
    oof_gain["weighted"] = json_float(
        float(official_oof["final_score"]) - parent_weighted
    )
    fold_weighted_ours = [
        weighted_final(ours_fold_q["fast"][i], ours_fold_q["balanced"][i], ours_fold_q["premium"][i])
        for i in range(FOLDS)
    ]
    fold_weighted_parent = [
        weighted_final(
            parent_fold_q["fast"][i], parent_fold_q["balanced"][i], parent_fold_q["premium"][i]
        )
        for i in range(FOLDS)
    ]
    fold_gains = [
        json_float(fold_weighted_ours[i] - fold_weighted_parent[i]) for i in range(FOLDS)
    ]
    fold_wins = int(sum(1 for gain in fold_gains if gain > 0.0))

    ours_ep = (
        TIER_WEIGHTS["fast"] * _episode_scores(scores, oof_models_t["fast"])
        + TIER_WEIGHTS["balanced"] * _episode_scores(scores, oof_models_t["balanced"])
        + TIER_WEIGHTS["premium"] * _episode_scores(scores, oof_models_t["premium"])
    )
    parent_ep = (
        TIER_WEIGHTS["fast"] * _episode_scores(scores, parent_ids["fast"])
        + TIER_WEIGHTS["balanced"] * _episode_scores(scores, parent_ids["balanced"])
        + TIER_WEIGHTS["premium"] * _episode_scores(scores, parent_ids["premium"])
    )
    bootstrap = paired_group_bootstrap(
        ours_ep - parent_ep,
        bundle.group_keys,
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )

    lofo_official_q = {tier: float(official_lofo["tiers"][tier]["quality_score"]) for tier in TIERS}
    lofo_gain = {
        tier: json_float(lofo_official_q[tier] - float(PARENT_F_PINS[tier]["quality"]))
        for tier in TIERS
    }
    lofo_gain["weighted"] = json_float(float(official_lofo["final_score"]) - parent_weighted)
    _family_gains = [lofo_family_gain[name]["weighted_gain"] for name in sorted(lofo_family_gain)]
    n50 = {
        name: lofo_family_gain[name]
        for name in sorted(lofo_family_gain)
        if int(lofo_family_gain[name]["n"]) < 50
    }
    n50_or_more = {
        name: lofo_family_gain[name]
        for name in sorted(lofo_family_gain)
        if int(lofo_family_gain[name]["n"]) >= 50
    }
    worst_family = min(n50_or_more, key=lambda name: n50_or_more[name]["weighted_gain"]) if n50_or_more else None
    worst_family_gain = (
        json_float(n50_or_more[worst_family]["weighted_gain"]) if worst_family else 0.0
    )

    # Per-view realized ratios at the selected operating points.
    view_realized: list[dict[str, Any]] = []
    premium_k1_positive = 0
    h2_9_fail = 0
    h2_7_fail = 0
    h2_8_fail = 0
    for view, pack, digest, order in zip(views, view_packs, view_digests, orders[selected_sort]):
        assigned = _route_pack(pack, digest, order, k1_on=True)
        row: dict[str, Any] = {"kind": view.kind, "n": int(view.index.size), "name": view.name}
        for tier in TIERS:
            cols = np.asarray([MODEL_IDS.index(mid) for mid in assigned[tier]], dtype=np.int64)
            spent = float(np.stack([pack["actual_l"], pack["actual_a"], pack["actual_k"]], axis=1)[
                np.arange(int(pack["actual_l"].size)), cols
            ].sum())
            light_sum = float(pack["actual_l"].sum())
            ratio = float(spent / light_sum) if light_sum > 0.0 else float("inf")
            k1_count = int(sum(1 for mid in assigned[tier] if mid == _K1))
            ax_count = int(sum(1 for mid in assigned[tier] if mid != _LIGHT))
            target = float(OPERATING_TARGETS[tier])
            official = float(OFFICIAL_CAPS[tier])
            row[tier] = {
                "k1_count": k1_count,
                "n_upgraded": ax_count,
                "ratio": json_float(ratio),
            }
            if ratio * float(STRESS_BACKSTOP) > official + 1e-15:
                h2_7_fail += 1
            if ratio > target + 1e-15:
                h2_8_fail += 1
            if tier != "fast" and ratio > target + 1e-15:
                h2_9_fail += 1
        if int(row["premium"]["k1_count"]) > 0:
            premium_k1_positive += 1
        view_realized.append(row)

    n_views = int(len(views))
    h2_10_pass = bool(premium_k1_positive == n_views)

    # H2-6 held-out envelope check at each selected f*.
    fold_checks = []
    family_checks = []
    for fold in range(FOLDS):
        held_name = f"oof-fold-{fold}"
        held_i = view_names.index(held_name)
        others = [i for i, view in enumerate(views) if view.kind == "oof-fold" and view.name != held_name]
        hat = np.max(phi_by_sort[selected_sort][others], axis=0)
        hat, _chg = monotonize_upper(hat)
        for tier in TIERS:
            col = int(round(float(internal_op[tier]["f_star"]) * 100.0))
            held_phi = float(phi_by_sort[selected_sort][held_i, col])
            hat_phi = float(hat[col])
            fold_checks.append(
                {
                    "exceeded": bool(held_phi > hat_phi + 1e-15),
                    "f_star": json_float(internal_op[tier]["f_star"]),
                    "held_phi": json_float(held_phi),
                    "heldout": held_name,
                    "phi_upper_hat": json_float(hat_phi),
                    "tier": tier,
                }
            )
    family_names = tuple(sorted(dict.fromkeys(fam_arr.tolist())))
    for name in family_names:
        prefixes = (f"lofo-{name}", f"lofo-combined-{name}", f"famdom-{name}-")
        held_is = [
            i
            for i, view in enumerate(views)
            if view.name == f"lofo-{name}" or view.name == f"lofo-combined-{name}"
        ]
        others = [
            i
            for i, view in enumerate(views)
            if not any(view.name.startswith(prefix) or view.name == prefix.rstrip("-") for prefix in prefixes)
            and not view.name.startswith(f"famdom-{name}-")
            and view.name not in {f"lofo-{name}", f"lofo-combined-{name}"}
        ]
        if not others or not held_is:
            continue
        hat = np.max(phi_by_sort[selected_sort][others], axis=0)
        hat, _chg = monotonize_upper(hat)
        for held_i in held_is:
            for tier in TIERS:
                col = int(round(float(internal_op[tier]["f_star"]) * 100.0))
                held_phi = float(phi_by_sort[selected_sort][held_i, col])
                hat_phi = float(hat[col])
                family_checks.append(
                    {
                        "exceeded": bool(held_phi > hat_phi + 1e-15),
                        "f_star": json_float(internal_op[tier]["f_star"]),
                        "family": name,
                        "held_phi": json_float(held_phi),
                        "heldout": view_names[held_i],
                        "phi_upper_hat": json_float(hat_phi),
                        "tier": tier,
                    }
                )
    all_h26 = fold_checks + family_checks
    n_h26 = int(len(all_h26))
    n_exceed = int(sum(1 for row in all_h26 if row["exceeded"]))
    exceed_rate = json_float(float(n_exceed) / float(n_h26) if n_h26 else 0.0)
    worst_held = max(all_h26, key=lambda row: float(row["held_phi"])) if all_h26 else None

    oof_fast_ratio = _realized_ratio(costs, oof_models_t["fast"])
    worst_fast = max(float(row["fast"]["ratio"]) for row in view_realized)
    drift_oof = json_float(1.0 + FAST_DRIFT_COEF * (float(oof_fast_ratio) - 1.0))
    drift_worst = json_float(1.0 + FAST_DRIFT_COEF * (worst_fast - 1.0))
    drift_phi = json_float(
        1.0 + FAST_DRIFT_COEF * (float(internal_op["fast"]["phi_upper"]) - 1.0)
    )

    gates = {
        "h2_1_oof_gain": {
            "pass": bool(oof_gain["weighted"] > 0.0),
            "per_tier": oof_gain,
            "threshold": 0.0,
            "value": oof_gain["weighted"],
        },
        "h2_2_fold_wins": {
            "fold_gains": fold_gains,
            "pass": bool(fold_wins >= 4),
            "threshold": ">= 4/5",
            "value": f"{fold_wins}/5",
        },
        "h2_3_bootstrap": {
            "pass": bool(float(bootstrap["q2_5"]) > 0.0),
            "threshold": 0.0,
            "value": bootstrap,
        },
        "h2_4_lofo": {
            "pass": bool(lofo_gain["weighted"] >= 0.0),
            "per_tier": lofo_gain,
            "threshold": 0.0,
            "value": lofo_gain["weighted"],
        },
        "h2_5_lofo_worst_family": {
            "n_lt_50": n50,
            "pass": bool(worst_family_gain >= -0.010),
            "threshold": -0.010,
            "value": worst_family_gain,
            "worst_family": worst_family,
        },
        "h2_6_heldout": {
            "exceedance_rate": exceed_rate,
            "n_checks": n_h26,
            "n_exceeded": n_exceed,
            "pass": bool(exceed_rate <= H2_6_EXCEEDANCE_MAX + 1e-15),
            "threshold": json_float(H2_6_EXCEEDANCE_MAX),
            "worst_held_out": worst_held,
        },
        "h2_7_backstop": {
            "n_fail": int(h2_7_fail),
            "pass": bool(h2_7_fail == 0),
            "threshold": "every view realized * 1.054 <= official limit",
        },
        "h2_8_operating": {
            "n_fail": int(h2_8_fail),
            "pass": bool(h2_8_fail == 0),
            "threshold": "every view realized <= internal operating target",
        },
        "h2_9_k1_certificate": {
            "n_fail": int(h2_9_fail),
            "pass": bool(h2_9_fail == 0),
            "threshold": "Phi_view(f*) + Psi_view(m*) <= target on every view",
        },
        "h2_10_premium_k1": {
            "n_positive": int(premium_k1_positive),
            "n_views": n_views,
            "pass": h2_10_pass,
            "threshold": "Premium K1 count > 0 on every view",
        },
        "h2_11_fast_drift": {
            "oof_realized": json_float(oof_fast_ratio),
            "pass": bool(drift_oof <= 1.25 + 1e-15),
            "phi_upper_value": drift_phi,
            "threshold": 1.25,
            "value": drift_oof,
            "worst_view_value": drift_worst,
        },
    }
    decision = decide(gates)
    k1_enabled = {tier: False for tier in TIERS}
    if decision == DECISION_PASS:
        k1_enabled = {"fast": False, "balanced": True, "premium": True}
    deployed_mstar = dict(selected_mstar)
    if decision != DECISION_PASS:
        deployed_mstar = {"fast": 0, "balanced": 0, "premium": 0}
        k1_enabled = {tier: False for tier in TIERS}

    policy = CertifiedPolicy(
        feature_version=FEATURE_VERSION,
        feature_signature=feature_signature(LOCKED_BINS),
        bins=LOCKED_BINS,
        alpha=LOCKED_ALPHA,
        variant=LOCKED_VARIANT,
        ridge_coefficients={
            name: tuple(json_floats(coef)) for name, coef in full_heads.coefs.items()
        },
        smearing_factors={name: json_float(value) for name, value in full_heads.smears.items()},
        recal_a_edges=tuple(json_floats(full_rec_a.edges)),
        recal_a_factors=tuple(json_floats(full_rec_a.clipped_factors)),
        recal_k_edges=tuple(json_floats(full_rec_k.edges)),
        recal_k_factors=tuple(json_floats(full_rec_k.clipped_factors)),
        qk_bins=int(qk_head.bins),
        qk_alpha=float(qk_head.alpha),
        qk_target_form=qk_head.target_form,
        qk_feature_signature=qk_head.feature_signature,
        qk_coefficients=tuple(json_floats(qk_head.coef)),
        phi_upper=tuple(json_floats(selected_phi)),
        psi_upper=tuple(json_floats(selected_psi)),
        f_star={tier: json_float(internal_op[tier]["f_star"]) for tier in TIERS},
        m_star=deployed_mstar,
        sort_rule=selected_sort,
        k_order=selected_k_order,
        q_elig=selected_q_elig,
        k1_denylist=K1_DENYLIST,
        k1_density_eps=K1_DENSITY_EPS,
        k1_item_cap_frac=K1_ITEM_CAP_FRAC,
        k1_count_cap_frac=K1_COUNT_CAP_FRAC,
        near_tie_abs=NEAR_TIE_ABS,
        near_tie_rel=NEAR_TIE_REL,
        operating_targets=dict(OPERATING_TARGETS),
        official_caps=dict(OFFICIAL_CAPS),
        stress_backstop=float(STRESS_BACKSTOP),
        k1_enabled=k1_enabled,
        intercept_policy=INTERCEPT_POLICY,
    )

    highlight_f = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
    phi_highlight = []
    for frac in highlight_f:
        col = int(round(frac * 100.0))
        phi_highlight.append(
            {
                "argmax_view": argmax[selected_sort][col],
                "f": json_float(frac),
                "phi_p50": json_float(selected_env["phi_p50"][col]),
                "phi_p99": json_float(selected_env["phi_p99"][col]),
                "phi_upper": json_float(selected_phi[col]),
            }
        )

    k_star_vs_parent = {}
    binding_view = {}
    for tier in TIERS:
        k_star = prefix_k(float(internal_op[tier]["f_star"]), n_train)
        parent_k = int(PARENT_F_PINS[tier]["ax31_count"])
        col = int(round(float(internal_op[tier]["f_star"]) * 100.0))
        k_star_vs_parent[tier] = {
            "below_ladder_parent": bool(k_star < parent_k),
            "k_star": int(k_star),
            "parent_k": parent_k,
            "phi_upper": json_float(selected_phi[col]),
        }
        binding_view[tier] = argmax[selected_sort][col]

    oof_counts = {tier: _count_models(oof_models_t[tier]) for tier in TIERS}
    oof_ratios = {tier: _realized_ratio(costs, oof_models_t[tier]) for tier in TIERS}

    observed = {
        "ax31_vs_parent": k_star_vs_parent,
        "binding_view_at_f_star": binding_view,
        "combo_table": combo_rows,
        "dev_opened": False,
        "divergence_oof_vs_fullfit_sort": {
            "kendall_tau": json_float(tau),
            "n_rank_inversions": int(n_inv),
            "n_pairs": int(n_train * (n_train - 1) // 2),
        },
        "float64_vs_official": official_agree,
        "gates": gates,
        "k1": {
            "eligible_pool_max": int(selected_row["eligible_pool_max"]),
            "eligible_pool_mean": selected_row["eligible_pool_mean"],
            "eligible_pool_min": int(selected_row["eligible_pool_min"]),
            "guard_binds": selected_row["guard_binds"],
            "h2_10_holds_every_view": h2_10_pass,
            "m_star_balanced": int(selected_mstar["balanced"]),
            "m_star_premium": int(selected_mstar["premium"]),
            "quality_balanced": selected_row["q_oof"]["balanced"],
            "quality_premium": selected_row["q_oof"]["premium"],
        },
        "monotonization": {
            "l1_change": json_float(selected_env["change"]),
            "needed": bool(selected_env["monotonized"]),
        },
        "oof_operating": {
            "model_counts": oof_counts,
            "official_per_tier_quality": official_oof_q,
            "official_weighted": json_float(float(official_oof["final_score"])),
            "parent_weighted": parent_weighted,
            "realized_ratios": oof_ratios,
            "weighted_float64": float_oof_weighted,
        },
        "operating_points": {
            "backstop_1.054": backstop_op,
            "internal": internal_op,
            "official_limits": official_op,
        },
        "phi_upper_highlight": phi_highlight,
        "q_a_used": False,
        "quality_entered_cost_selection": False,
        "quality_entered_policy_selection": True,
        "selected_combo": {
            "k_order": selected_k_order,
            "key": selected_key,
            "q_elig": json_float(selected_q_elig),
            "sort_rule": selected_sort,
            "weighted_q_oof": selected_row["weighted_q_oof"],
        },
        "view_catalogue": {
            "kind_counts": catalogue["view_kind_counts"],
            "n_skipped": catalogue["n_skipped"],
            "n_views": catalogue["n_views"],
            "planned": catalogue["planned"],
            "skip_by_kind": catalogue["skip_by_kind"],
        },
    }
    diagnostic = {
        "famdom_fallback_by_family": catalogue["famdom_fallback_by_family"],
        "float64_table_note": FLOAT64_TABLE_NOTE,
        "fold_checks_h2_6": fold_checks,
        "imported_modeling_symbols": [
            "family_folds",
            "feature_matrix",
            "group_folds",
            "load_train",
            "official_score",
            "oof_predict",
            "paired_group_bootstrap",
            "quantile_higher",
            "rank_recalibration",
            "ridge_fit",
            "ridge_predict",
        ],
        "imported_cost_cert_symbols": [
            "FittedHeads",
            "actual_increments",
            "apply_recal",
            "fit_heads",
            "fit_recal",
            "oof_incremental_costs",
            "predict_heads",
        ],
        "lofo_family_gain": lofo_family_gain,
        "m_max": int(selected_psi.size) - 1,
        "n_views": n_views,
        "oof_recal_ratios": {
            "inc_A": json_float(ratio_a) if ratio_a is not None else None,
            "inc_K": json_float(ratio_k) if ratio_k is not None else None,
        },
        "official_score_imported": official_score is not None,
        "parent_f_driven": True,
        "qk_head": {
            "alpha": json_float(qk_head.alpha),
            "bins": int(qk_head.bins),
            "feature_signature": qk_head.feature_signature,
            "n_positive_oof": int((oof_qk > 0.0).sum()),
            "target_form": qk_head.target_form,
        },
        "selected_psi_upper_head": json_floats(selected_psi[: min(21, int(selected_psi.size))]),
        "skipped_views": catalogue["skipped"],
        "view_kind_counts": catalogue["view_kind_counts"],
        "worst_view_ratios": {
            tier: json_float(max(float(row[tier]["ratio"]) for row in view_realized))
            for tier in TIERS
        },
    }
    return {
        "decision": decision,
        "diagnostic": diagnostic,
        "observed": observed,
        "policy": policy,
    }


__all__ = (
    "BOOTSTRAP_SEED",
    "CertifiedPolicy",
    "DECISION_K1_OFF",
    "DECISION_PASS",
    "EXPERIMENT",
    "F_GRID_ARRAY",
    "K1_COUNT_CAP_FRAC",
    "K1_DENYLIST",
    "K1_DENSITY_EPS",
    "K1_ITEM_CAP_FRAC",
    "LOCKED_ALPHA",
    "LOCKED_BINS",
    "LOCKED_VARIANT",
    "OPERATING_TARGETS",
    "SORT_RULES",
    "View",
    "allocate",
    "allocate_from_arrays",
    "assemble_report",
    "build_views",
    "content_digest",
    "decide",
    "design_matrix_g_features",
    "family_of_text",
    "fit_and_evaluate",
    "k1_mask",
    "load_q_k_head",
    "load_train",
    "locked_record",
    "monotonize_upper",
    "order_k1",
    "phi_view",
    "prefix_k",
    "q_view",
    "reject_dev_reference",
    "select_f_star",
    "sort_order",
    "sort_pred_inc",
    "sort_pred_inc_per_light",
)
