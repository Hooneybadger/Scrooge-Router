# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""V7 champion-challenger protocol. Phase A seals the contract only.

Independent clean-room implementation of the published tykimdream V7
methodology (Apache-2.0). Competitor artifact weights are not used.
Validation never opens outcomes or fits on public Train/Dev.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ossp_router.heuristic import episode_text, extract_features
from ossp_router.protocol import MODEL_IDS, TIERS, Episode, InputBatch
from research.lab.chuf_frozen_runtime_fidelity import EXPLICIT_FIDELITY_SEEDS
from research.lab.chuf_predicted_cost_phase2 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPLICIT_RISK_SEEDS,
    source_sha256,
)
from research.lab.chuf_tvball_confirmation import (
    EXPECTED_POLICY_SHA256,
    N_FRESH_SEEDS,
    OLD_SEEDS,
    REQUIRED_FAMILIES,
    derive_fresh_seeds as derive_confirmation_seeds,
    epsilon_from_input_paths,
    family_counts_from_inputs,
)
from research.lab.e1_objectives import (
    STRESS_RATIO_CAPS,
    canonical_json_text,
    sha256_text,
    write_json_atomic,
)
from research.lab.modeling import OFFICIAL_CAPS, sort_mapping
from research.lab.public_pool import (
    DEV_INPUTS,
    DEV_OUTCOMES,
    EXPECTED_DEV_INPUTS_SHA256,
    EXPECTED_DEV_OUTCOMES_SHA256,
    EXPECTED_N_DEV,
    EXPECTED_N_PUBLIC,
    EXPECTED_N_TRAIN,
    EXPECTED_TRAIN_INPUTS_SHA256,
    EXPECTED_TRAIN_OUTCOMES_SHA256,
    ROOT,
    TRAIN_INPUTS,
    TRAIN_OUTCOMES,
    sha256_path,
)


PROTOCOL_ID = "v7-challenger-v1"
PROTOCOL_RELATIVE = "research/protocols/v7-challenger.v1.json"
PROTOCOL_PATH = ROOT / PROTOCOL_RELATIVE
REPORT_TYPE = "scrooge-v7-challenger"
SCHEMA_VERSION = 1
SEED_PREFIX = "scrooge-v7-challenger-v1"
FIDELITY_CORE_SHA256 = (
    "1c4b8144378f55779aaf1cf3e78424abd749a93e3d4e33744ccda82304b7cbc9"
)
EXPECTED_PROTOCOL_SHA256 = (
    "f7d5e51f29badb0a88b6f05ce0431b4a61aba6c5e0d9e490aab312c023c30fd2"
)
EXPLICIT_CHALLENGER_SEEDS: Tuple[int, ...] = (
    1726202894,
    252428889,
    1120507837,
    141957400,
    1234496749,
    1353411567,
    1103101561,
    2142214382,
    496053794,
    58658564,
    297610007,
    1888638919,
)
HASH_BINS = 512
FEATURE_VERSION = 1
FIXED_BISECTION_STEPS = 48
PREMIUM_AX31_FILL_SAFETY = 0.65
FAST_K1_BAN_SCORE = -2.0
COST_MONOTONE_EPS = 1e-12
ENSEMBLE_ALPHAS = {
    "fast": (300.0, 500.0, 5000.0),
    "balanced": (10000.0,),
    "premium": (3000.0, 10000.0, 15000.0),
}
TRAIN_BUDGET_TARGETS = {"fast": 1.23, "balanced": 1.95, "premium": 3.85}
TEMPLATE_FOLDS = 5
UNIQUE_ENSEMBLE_ALPHAS: Tuple[float, ...] = tuple(
    sorted({alpha for values in ENSEMBLE_ALPHAS.values() for alpha in values})
)
SIGNAL_NAMES: Tuple[str, ...] = (
    "korean_hangul_10pct",
    "code_marker",
    "math_or_numeric",
    "long_context_gt_8000",
    "short_context_le_100",
    "messages",
)
COMPOSITION_PENALTY = 1.0
SMALL_BATCH_PENALTY = 0.25
STABLE_BATCH_SIZE = 1000
MAX_LEARNED_EPISODES = 6_000
MAX_LEARNED_CHARACTERS = 30_000_000
MAX_LEARNED_MESSAGES = 100_000
MAX_LEARNED_WORK_UNITS = 40_000_000
FALLBACK_ROUTER = "ossp_router.budget_brake_router.make_submission"
REPRODUCTION_TOLERANCE = "exact-decimal-12"
GATE_REPRO_MIN = 0.690
GATE_GROUPED_MEAN = 0.690
GATE_GROUPED_WORST = 0.687
GATE_DELTA_MEAN = 0.015
GATE_DELTA_WORST = 0.010
TV_WORST_MIN = -0.003
NO_REF_DECISION = "record-v7-challenger-no-valid-reference-current-runtime"
REPRO_FAIL_DECISION = "record-v7-challenger-reproduction-fail-current-runtime"
GROUPED_FAIL_DECISION = "record-v7-challenger-grouped-fail-current-runtime"
PASS_DECISION = "record-v7-challenger-pass-await-independent-audit"
OUT_RELATIVE = "build/v7-challenger"
AUDIT_RELATIVE = "build/v7-challenger/episode-audit.json"
FIDELITY_REPORT_RELATIVE = "build/frozen-runtime-fidelity/report.json"
PHASE2_REPORT_RELATIVE = "build/phase2-chuf-predicted-cost/report.json"
CONFIRM_REPORT_RELATIVE = "build/confirm-chuf-tvball/report.json"
E1F_REPORT_RELATIVE = "build/compare-e1f-cost-conditioned-frontier/report.json"
BUDGET_BRAKE_RELATIVE = "src/ossp_router/budget_brake_router.py"
BUDGET_BRAKE_ARTIFACT_RELATIVE = "src/ossp_router/resources/budget-brake-router.v1.json"
FAMILY_GUARD_ARTIFACT_RELATIVE = "src/ossp_router/resources/family-guard-router.v1.json"
REFERENCE_FINAL = "14f9a5e387b774b85e3d1eb8687a558074d1cecc"
REFERENCE_V61 = "d7e15d019f4ce2f5d27e071cd56c6260a486cd7a"
REFERENCE_V7_ADAPTIVE = "88641e89f0f373ac30487bbaedd60f9cb5a8d516"
FALLBACK_DEV_OFFICIAL = "0.669517045455"
FALLBACK_DEV_N_K1 = 16
FALLBACK_DEV_RATIOS = {
    "balanced": "1.396000996251",
    "fast": "1.093011852072",
    "premium": "2.160755720509",
}
FALLBACK_DEV_TIER_QUALITY = {
    "balanced": "0.674431818182",
    "fast": "0.643181818182",
    "premium": "0.699715909091",
}
V7_FAIR_DEV_OFFICIAL = "0.691477272727"
V7_FAIR_DEV_TIERS = {
    "balanced": {
        "budget_ratio": "1.653162122486",
        "model_counts": {"ax31": 785, "ax31-light": 93, "axk1-think": 2},
        "quality_score": "0.689772727273",
    },
    "fast": {
        "budget_ratio": "1.145520783178",
        "model_counts": {"ax31": 293, "ax31-light": 587, "axk1-think": 0},
        "quality_score": "0.659943181818",
    },
    "premium": {
        "budget_ratio": "3.006000526541",
        "model_counts": {"ax31": 665, "ax31-light": 100, "axk1-think": 115},
        "quality_score": "0.735227272727",
    },
}
TRAIN_REFERENCE_RATES = {
    "code_marker": 0.26931818181818185,
    "korean_hangul_10pct": 0.2056818181818182,
    "long_context_gt_8000": 0.09090909090909091,
    "math_or_numeric": 0.4909090909090909,
    "messages": 0.0,
    "short_context_le_100": 0.17784090909090908,
}
SEED_DERIVATION = (
    "digest_i = SHA256(UTF8(PREFIX) + NUL + bytes.fromhex(FIDELITY_CORE) + "
    "i.to_bytes(4,'big')); seed_i = int.from_bytes(digest_i[:4],'big') "
    "& 0x7fffffff; i=0..11. Collision or overlap with old E1F, "
    "confirmation-12, phase2-12, or fidelity-12 fails closed."
)

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_UINT64_MASK = (1 << 64) - 1
_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_NUMBER_RUN = re.compile(r"\d+(?:[.,]\d+)*")
_SPACE = re.compile(r"\s+")
_CHOICE = re.compile(r"(?:^|\n)\s*[A-D][.)]\s", re.MULTILINE)
_FORMAL_REASONING = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|"
    r"증명|유도|정리|보조정리|반례|귀납)\b",
    re.IGNORECASE,
)
_PROGRAM_ANALYSIS = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|"
    r"시간\s*복잡도|공간\s*복잡도|예외|스택\s*추적)\b",
    re.IGNORECASE,
)
_MULTI_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_SIMPLE_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|"
    r"요약|바꾸|번역|나열|추출)\b",
    re.IGNORECASE,
)
_ARITHMETIC_REQUEST = re.compile(
    r"\b(?:calculate|compute|evaluate|round|divided by|minus|plus|"
    r"nearest|what is|계산|반올림|나누|더하|빼기)\b",
    re.IGNORECASE,
)
_LOGIC_RULES = re.compile(
    r"\b(?:if someone|if something|all \w+ people|question:)\b",
    re.IGNORECASE,
)
DENSE_FEATURE_NAMES: Tuple[str, ...] = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_message_count",
    "hangul_ratio",
    "log_code_marker_count",
    "log_math_marker_count",
    "numeric_density",
    "long_context",
    "log_reasoning_marker_count",
    "formal_reasoning",
    "program_analysis",
    "log_multi_constraint_count",
    "simple_transform",
    "log_line_count",
    "ascii_ratio",
    "log_digit_count",
    "log_choice_count",
    "has_question_label",
    "has_assert_hole",
    "has_python_function",
    "has_logic_rules",
    "has_arithmetic_request",
    "has_latex",
    "has_binary_choices",
    "length_le_60",
    "length_le_100",
    "length_101_500",
    "length_501_2000",
    "length_over_2000",
    "length_over_8000",
    "korean_multiple_choice",
    "short_code",
    "short_math_numeric",
    "multi_message",
    "quote_density",
    "newline_density",
    "ends_with_question",
    "contains_answer_placeholder",
    "contains_currency_or_percent",
)
_DENSE_INDEX = {name: index for index, name in enumerate(DENSE_FEATURE_NAMES)}


@dataclass(frozen=True)
class LinearHead:
    intercept: float
    coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class CompiledTier:
    score_heads: Mapping[str, LinearHead]
    log_cost_heads: Mapping[str, LinearHead]


def fnv1a64(value: str) -> int:
    digest = _FNV_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def normalized_template(text: str) -> str:
    return _SPACE.sub(" ", _NUMBER_RUN.sub("<number>", text.casefold())).strip()


def _add_hash(bins: list[float], value: str) -> None:
    digest = fnv1a64(value)
    index = digest & (len(bins) - 1)
    bins[index] += -1.0 if digest & (1 << 63) else 1.0


def hashed_features(text: str, hash_bins: int = HASH_BINS) -> Tuple[float, ...]:
    if hash_bins < 16 or hash_bins & (hash_bins - 1):
        raise ValueError("hash_bins must be a power of two >= 16")
    bins = [0.0] * hash_bins
    previous: deque[str] = deque(maxlen=2)
    for match in _TOKEN.finditer(text):
        token = match.group(0).casefold()
        if token.isdecimal():
            token = "<number>"
        _add_hash(bins, f"w1:{token}")
        if previous:
            _add_hash(bins, f"w2:{previous[-1]}\x1f{token}")
        if len(previous) == 2:
            _add_hash(bins, f"w3:{previous[0]}\x1f{previous[1]}\x1f{token}")
        previous.append(token)
    characters = normalized_template(text)
    if len(characters) > 4_000:
        characters = characters[:3_000] + characters[-1_000:]
    for size in (3, 4):
        for index in range(0, max(0, len(characters) - size + 1), 2):
            _add_hash(bins, f"c{size}:{characters[index : index + size]}")
    norm = math.sqrt(math.fsum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return tuple(bins)


def dense_feature_vector(episode: Episode) -> Tuple[float, ...]:
    base = extract_features(episode)
    text = episode_text(episode)
    characters = len(text)
    nonspace = max(1, sum(not character.isspace() for character in text))
    ascii_count = sum(ord(character) < 128 for character in text)
    digit_count = sum(character.isdigit() for character in text)
    choice_count = len(_CHOICE.findall(text))
    lower = text.casefold()
    math_numeric = base.math_marker_count > 0 or base.numeric_density >= 0.08
    code = base.code_marker_count > 0
    dense = (
        math.log1p(base.character_count),
        math.log1p(base.word_count),
        math.log1p(base.sentence_count),
        math.log1p(base.message_count),
        base.hangul_ratio,
        math.log1p(base.code_marker_count),
        math.log1p(base.math_marker_count),
        base.numeric_density,
        float(base.long_context),
        math.log1p(base.reasoning_marker_count),
        float(bool(_FORMAL_REASONING.search(text))),
        float(bool(_PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(_MULTI_CONSTRAINT.findall(text))),
        float(bool(_SIMPLE_TRANSFORM.search(text))),
        math.log1p(text.count("\n") + 1),
        ascii_count / max(1, characters),
        math.log1p(digit_count),
        math.log1p(choice_count),
        float("question:" in lower),
        float("assert f(" in lower and "??" in text),
        float(bool(re.search(r"(?:^|\n)\s*(?:def|class)\s+", text))),
        float(bool(_LOGIC_RULES.search(text))),
        float(bool(_ARITHMETIC_REQUEST.search(text))),
        float("\\frac" in text or "\\boxed" in text or "$$" in text),
        float(choice_count == 2),
        float(characters <= 60),
        float(characters <= 100),
        float(101 <= characters <= 500),
        float(501 <= characters <= 2_000),
        float(characters > 2_000),
        float(characters > 8_000),
        float(base.hangul_ratio >= 0.1 and choice_count >= 2),
        float(code and characters <= 500),
        float(math_numeric and characters <= 150),
        float(episode.messages is not None and len(episode.messages) > 1),
        (text.count('"') + text.count("'")) / nonspace,
        text.count("\n") / nonspace,
        float(text.rstrip().endswith("?")),
        float("??" in text or "[MASK]" in text),
        float("$" in text or "%" in text),
    )
    if len(dense) != len(DENSE_FEATURE_NAMES):
        raise RuntimeError("dense feature arity drifted")
    return dense


def raw_feature_vector(episode: Episode, hash_bins: int = HASH_BINS) -> Tuple[float, ...]:
    return dense_feature_vector(episode) + hashed_features(episode_text(episode), hash_bins)


def compile_head(
    heads: Sequence[LinearHead],
    means: Sequence[Sequence[float]],
    scales: Sequence[Sequence[float]],
) -> LinearHead:
    count = float(len(heads))
    width = len(means[0])
    coefficients = tuple(
        math.fsum(
            head.coefficients[index] / scale[index]
            for head, scale in zip(heads, scales)
        )
        / count
        for index in range(width)
    )
    intercept = (
        math.fsum(
            head.intercept
            - math.fsum(
                mean * coefficient / scale
                for mean, coefficient, scale in zip(mean_row, head.coefficients, scale)
            )
            for head, mean_row, scale in zip(heads, means, scales)
        )
        / count
    )
    return LinearHead(intercept, coefficients)


def apply_linear(head: LinearHead, values: Sequence[float]) -> float:
    return head.intercept + math.fsum(
        coefficient * value for coefficient, value in zip(head.coefficients, values)
    )


def ensemble_predict_standardized(
    raw: Sequence[float],
    heads: Sequence[LinearHead],
    means: Sequence[Sequence[float]],
    scales: Sequence[Sequence[float]],
) -> float:
    total = 0.0
    for head, mean, scale in zip(heads, means, scales):
        standardized = tuple(
            (value - center) / width for value, center, width in zip(raw, mean, scale)
        )
        total += apply_linear(head, standardized)
    return total / float(len(heads))


def monotonic_costs(costs: Mapping[str, float]) -> dict[str, float]:
    light = float(costs[MODEL_IDS[0]])
    ax31 = max(float(costs[MODEL_IDS[1]]), light * (1.0 + COST_MONOTONE_EPS))
    k1 = max(float(costs[MODEL_IDS[2]]), ax31 * (1.0 + COST_MONOTONE_EPS))
    return {MODEL_IDS[0]: light, MODEL_IDS[1]: ax31, MODEL_IDS[2]: k1}


def clamp_scores(values: Sequence[float]) -> dict[str, float]:
    return {
        model_id: min(1.0, max(0.0, float(values[index])))
        for index, model_id in enumerate(MODEL_IDS)
    }


def predict_from_compiled(
    raw: Sequence[float], compiled: CompiledTier
) -> Tuple[dict[str, float], dict[str, float]]:
    scores = clamp_scores(
        tuple(apply_linear(compiled.score_heads[model_id], raw) for model_id in MODEL_IDS)
    )
    costs = monotonic_costs(
        {
            model_id: math.exp(
                min(50.0, max(-50.0, apply_linear(compiled.log_cost_heads[model_id], raw)))
            )
            for model_id in MODEL_IDS
        }
    )
    return scores, costs


def select_models(
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    *,
    budget_multiplier: float,
    safety_ratio: float,
    steps: int = FIXED_BISECTION_STEPS,
) -> Tuple[Tuple[str, ...], float]:
    if len(scores) != len(costs) or not scores:
        raise ValueError("prediction arrays are empty or misaligned")
    light_id = MODEL_IDS[0]
    light_total = math.fsum(row[light_id] for row in costs)
    cap = light_total * max(1.0, budget_multiplier * safety_ratio)

    def choose(penalty: float) -> Tuple[Tuple[str, ...], float]:
        selected = tuple(
            max(
                MODEL_IDS,
                key=lambda model_id: (
                    score_row[model_id] - penalty * cost_row[model_id] / light_total,
                    -MODEL_IDS.index(model_id),
                ),
            )
            for score_row, cost_row in zip(scores, costs)
        )
        total = math.fsum(row[model_id] for row, model_id in zip(costs, selected))
        return selected, total

    selected, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        selected, total = choose(high)
        while total > cap and high < 2**60:
            low, high = high, high * 2.0
            selected, total = choose(high)
        for _ in range(int(steps)):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high, selected, total = middle, candidate, candidate_total
            else:
                low = middle
    if total > cap:
        selected = tuple(light_id for _ in scores)
        total = light_total
    return selected, total / light_total


def fill_ax31(
    selected: Sequence[str],
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    *,
    budget_multiplier: float,
    safety_ratio: float = PREMIUM_AX31_FILL_SAFETY,
    steps: int = FIXED_BISECTION_STEPS,
) -> Tuple[Tuple[str, ...], float]:
    light_id, ax31_id, _ = MODEL_IDS
    light_total = math.fsum(row[light_id] for row in costs)
    current_total = math.fsum(row[model_id] for row, model_id in zip(costs, selected))
    cap = max(current_total, light_total * budget_multiplier * safety_ratio)

    def choose(penalty: float) -> Tuple[Tuple[str, ...], float]:
        result = []
        for current, score_row, cost_row in zip(selected, scores, costs):
            if current != light_id:
                result.append(current)
                continue
            gain = score_row[ax31_id] - score_row[light_id]
            extra = cost_row[ax31_id] - cost_row[light_id]
            result.append(
                ax31_id if gain - penalty * extra / light_total > 0 else light_id
            )
        total = math.fsum(row[model_id] for row, model_id in zip(costs, result))
        return tuple(result), total

    result, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        result, total = choose(high)
        while total > cap and high < 2**60:
            low, high = high, high * 2.0
            result, total = choose(high)
        for _ in range(int(steps)):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high, result, total = middle, candidate, candidate_total
            else:
                low = middle
    if total > cap:
        return tuple(selected), current_total / light_total
    return result, total / light_total


def ban_fast_k1(scores: Mapping[str, float]) -> dict[str, float]:
    row = dict(scores)
    row[MODEL_IDS[2]] = FAST_K1_BAN_SCORE
    return row


def signal_row(episode: Episode, raw: Sequence[float]) -> Tuple[bool, ...]:
    return (
        raw[_DENSE_INDEX["hangul_ratio"]] >= 0.1,
        raw[_DENSE_INDEX["log_code_marker_count"]] > 0.0,
        raw[_DENSE_INDEX["log_math_marker_count"]] > 0.0
        or raw[_DENSE_INDEX["numeric_density"]] >= 0.08,
        raw[_DENSE_INDEX["length_over_8000"]] > 0.0,
        raw[_DENSE_INDEX["length_le_100"]] > 0.0,
        episode.messages is not None,
    )


def reference_rates(rows: Sequence[Sequence[bool]]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty signal batch")
    return {
        name: math.fsum(float(row[index]) for row in rows) / len(rows)
        for index, name in enumerate(SIGNAL_NAMES)
    }


def safety_multiplier(
    rows: Sequence[Sequence[bool]],
    rates: Mapping[str, float],
    *,
    composition_penalty: float = COMPOSITION_PENALTY,
    small_batch_penalty: float = SMALL_BATCH_PENALTY,
    stable_batch_size: int = STABLE_BATCH_SIZE,
) -> float:
    if not rows:
        raise ValueError("empty adaptive-safety batch")
    observed = reference_rates(rows)
    maximum_shift = max(abs(observed[name] - float(rates[name])) for name in SIGNAL_NAMES)
    size_shift = max(0.0, math.sqrt(stable_batch_size / len(rows)) - 1.0)
    return math.exp(
        -float(composition_penalty) * maximum_shift
        - float(small_batch_penalty) * size_shift
    )


def _token_count_upper_bound(text: str, remaining: int) -> int:
    count = 0
    in_word = False
    for character in text:
        if character.isalnum():
            if not in_word:
                count += 1
                if count > remaining:
                    return count
            in_word = True
        elif character.isspace() or character == "_":
            in_word = False
        else:
            count += 1
            in_word = False
            if count > remaining:
                return count
    return count


def learned_path_allowed(inputs: InputBatch) -> bool:
    if len(inputs.episodes) > MAX_LEARNED_EPISODES:
        return False
    characters = 0
    messages = 0
    work_units = 0
    for episode in inputs.episodes:
        text = episode_text(episode)
        characters += len(text)
        messages += 1 if episode.prompt is not None else len(episode.messages or ())
        if characters > MAX_LEARNED_CHARACTERS or messages > MAX_LEARNED_MESSAGES:
            return False
        remaining = max(0, (MAX_LEARNED_WORK_UNITS - work_units) // 3)
        tokens = _token_count_upper_bound(text, remaining)
        work_units += len(text) + 3 * tokens
        if work_units > MAX_LEARNED_WORK_UNITS:
            return False
    return True


def content_sort_key(episode: Episode) -> Tuple[int, str]:
    if episode.prompt is not None:
        canonical = "prompt\x1f" + episode.prompt
    else:
        assert episode.messages is not None
        canonical = "messages\x1f" + "\x1e".join(
            f"{message.role}\x1f{message.content}" for message in episode.messages
        )
    return fnv1a64(canonical), canonical


def canonical_batch(inputs: InputBatch) -> InputBatch:
    episodes = tuple(sorted(inputs.episodes, key=content_sort_key))
    return InputBatch(
        inputs.schema_version,
        inputs.challenge_id,
        inputs.split,
        episodes,
    )


def restore_input_order(inputs: InputBatch, selected: Sequence[str], canonical: InputBatch) -> Tuple[str, ...]:
    by_id = {
        episode.episode_id: model_id
        for episode, model_id in zip(canonical.episodes, selected)
    }
    return tuple(by_id[episode.episode_id] for episode in inputs.episodes)


def allocate_tier(
    inputs: InputBatch,
    compiled: CompiledTier,
    *,
    tier: str,
    budget_multiplier: float,
    safety_ratio: float,
    fill_safety: float = PREMIUM_AX31_FILL_SAFETY,
    reserve_rows: Sequence[Sequence[bool]] | None = None,
    reference: Mapping[str, float] | None = None,
) -> Tuple[Tuple[str, ...], float]:
    canonical = canonical_batch(inputs)
    scores = []
    costs = []
    signals = []
    cache: dict[Tuple[int, str], Tuple[dict[str, float], dict[str, float], Tuple[bool, ...]]] = {}
    for episode in canonical.episodes:
        key = content_sort_key(episode)
        cached = cache.get(key)
        if cached is None:
            raw = raw_feature_vector(episode)
            score_row, cost_row = predict_from_compiled(raw, compiled)
            cached = (score_row, cost_row, signal_row(episode, raw))
            cache[key] = cached
        score_row, cost_row, signal = cached
        if tier == "fast":
            score_row = ban_fast_k1(score_row)
        scores.append(score_row)
        costs.append(cost_row)
        signals.append(signal)
    multiplier = 1.0
    if reserve_rows is None:
        reserve_rows = signals
    if reference is not None:
        multiplier = safety_multiplier(reserve_rows, reference)
        if multiplier > 1.0:
            raise RuntimeError("V7 reserve must be downward-only")
    selected, ratio = select_models(
        scores,
        costs,
        budget_multiplier=budget_multiplier,
        safety_ratio=safety_ratio * multiplier,
    )
    if tier == "premium":
        selected, ratio = fill_ax31(
            selected,
            scores,
            costs,
            budget_multiplier=budget_multiplier,
            safety_ratio=fill_safety * multiplier,
        )
    return restore_input_order(inputs, selected, canonical), ratio


def route_or_fallback(
    inputs: InputBatch,
    compiled: CompiledTier,
    *,
    tier: str,
    budget_multiplier: float,
    safety_ratio: float,
    fill_safety: float = PREMIUM_AX31_FILL_SAFETY,
    reference: Mapping[str, float] | None = None,
) -> Tuple[Tuple[str, ...], str]:
    if not learned_path_allowed(inputs):
        from ossp_router.budget_brake_router import (
            load_bundled_artifact,
            make_submission as make_brake,
        )
        from ossp_router.protocol import load_bundled_policy

        plan = make_brake(
            inputs, load_bundled_policy(), load_bundled_artifact(), tier
        )
        return (
            tuple(decision.model_id for decision in plan.submission.decisions),
            FALLBACK_ROUTER,
        )
    selected, _ratio = allocate_tier(
        inputs,
        compiled,
        tier=tier,
        budget_multiplier=budget_multiplier,
        safety_ratio=safety_ratio,
        fill_safety=fill_safety,
        reference=reference,
    )
    return selected, "v7-learned"


def expected_score(official_quality: float, q999_by_tier: Mapping[str, float]) -> float:
    overrun = max(
        0.0,
        max(
            (float(q999_by_tier[tier]) - float(OFFICIAL_CAPS[tier]))
            / float(OFFICIAL_CAPS[tier])
            for tier in TIERS
        ),
    )
    return float(official_quality) * (1.0 - min(1.0, overrun))


def expected_score_beats_fallback(
    official_quality: float,
    q999_by_tier: Mapping[str, float],
    *,
    fallback: str = FALLBACK_DEV_OFFICIAL,
) -> bool:
    return official_decimal(str(expected_score(official_quality, q999_by_tier))) > (
        official_decimal(fallback)
    )


def blocked_seeds() -> Tuple[int, ...]:
    return tuple(
        sorted(
            set(OLD_SEEDS)
            | set(derive_confirmation_seeds())
            | set(EXPLICIT_RISK_SEEDS)
            | set(EXPLICIT_FIDELITY_SEEDS)
        )
    )


def derive_fresh_challenger_seeds(
    *,
    n: int = N_FRESH_SEEDS,
    core_sha: str = FIDELITY_CORE_SHA256,
    blocked: Sequence[int] | None = None,
    prefix: str = SEED_PREFIX,
) -> Tuple[int, ...]:
    seeds: list[int] = []
    seen: set[int] = set()
    denied = {int(seed) for seed in (blocked_seeds() if blocked is None else blocked)}
    for index in range(int(n)):
        digest = hashlib.sha256(
            prefix.encode("utf-8")
            + b"\0"
            + bytes.fromhex(core_sha)
            + int(index).to_bytes(4, "big")
        ).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed in seen:
            raise RuntimeError(
                f"challenger seed collision at i={index}: {seed}; fail closed"
            )
        if seed in denied:
            raise RuntimeError(
                f"challenger seed overlaps blocked seed at i={index}: "
                f"{seed}; fail closed"
            )
        seen.add(seed)
        seeds.append(seed)
    if (
        blocked is None
        and prefix == SEED_PREFIX
        and int(n) == N_FRESH_SEEDS
        and core_sha == FIDELITY_CORE_SHA256
        and tuple(seeds) != EXPLICIT_CHALLENGER_SEEDS
    ):
        raise RuntimeError("challenger explicit seed list drifted from derivation")
    return tuple(seeds)


def methodology_snapshot() -> dict[str, Any]:
    return sort_mapping(
        {
            "allocator": {
                "bisection_steps": FIXED_BISECTION_STEPS,
                "canonical_content_sort": True,
                "fast_k1_hard_ban": True,
                "premium_ax31_fill_safety": PREMIUM_AX31_FILL_SAFETY,
                "restore_input_order": True,
                "tie_break": "equal utility prefers cheaper MODEL_IDS index",
            },
            "ensemble_alphas": {tier: list(values) for tier, values in ENSEMBLE_ALPHAS.items()},
            "fallback_router": FALLBACK_ROUTER,
            "features": {
                "dense_names": list(DENSE_FEATURE_NAMES),
                "hash_algorithm": "fnv1a64-signed-word-1-3-char-3-4-stride-2",
                "hash_bins": HASH_BINS,
                "normalization": "casefold; decimal tokens and number-runs -> <number>",
                "window": "char grams use normalized_template; >4000 keeps 3000+last 1000",
            },
            "heads": "per-model score Ridge and log-cost Ridge; tier mean ensemble",
            "no_id_split_order_features": True,
            "reserve": {
                "composition_penalty": COMPOSITION_PENALTY,
                "downward_only": True,
                "formula": (
                    "exp(-1*max|rate-ref| - 0.25*max(0,sqrt(1000/n)-1))"
                ),
                "signal_names": list(SIGNAL_NAMES),
                "small_batch_penalty": SMALL_BATCH_PENALTY,
                "stable_batch_size": STABLE_BATCH_SIZE,
                "train_only_reference_rates": True,
            },
            "runtime_compile": "average (coef/scale) and intercept after centering",
            "train_budget_targets": dict(TRAIN_BUDGET_TARGETS),
            "workload_guard": {
                "max_characters": MAX_LEARNED_CHARACTERS,
                "max_episodes": MAX_LEARNED_EPISODES,
                "max_messages": MAX_LEARNED_MESSAGES,
                "max_work_units": MAX_LEARNED_WORK_UNITS,
            },
        }
    )


def methodology_sha256() -> str:
    return sha256_text(canonical_json_text(methodology_snapshot()))


def official_decimal(value: str) -> Decimal:
    return Decimal(str(value))


def ratio_under_95(ratio: str, tier: str) -> bool:
    return official_decimal(ratio) < official_decimal(str(0.95 * OFFICIAL_CAPS[tier]))


def reproduction_matches_reference(reproduction: Mapping[str, Any]) -> bool:
    if str(reproduction.get("final_score")) != V7_FAIR_DEV_OFFICIAL:
        return False
    if official_decimal(reproduction["final_score"]) < official_decimal(str(GATE_REPRO_MIN)):
        return False
    if reproduction.get("tiers") != V7_FAIR_DEV_TIERS:
        return False
    if reproduction.get("tolerance") not in (None, REPRODUCTION_TOLERANCE):
        return False
    if not bool(reproduction.get("all_hard_passed")):
        return False
    if not bool(reproduction.get("all_ratio_under_95")):
        return False
    return True


def grouped_row_passes(row: Mapping[str, Any]) -> bool:
    return (
        float(row["official_score"]) >= GATE_GROUPED_WORST
        and float(row["delta"]) >= GATE_DELTA_WORST
        and bool(row["pooled_hard_caps_ok"])
        and bool(row["pooled_ratio_under_95_ok"])
        and bool(row["bootstrap_q999_under_95_ok"])
        and bool(row["tv_cost_under_official_ok"])
        and float(row["tv_quality_worst"]) >= TV_WORST_MIN
        and row.get("dirac_family_view") is False
    )


def grouped_aggregate_passes(rows: Sequence[Mapping[str, Any]]) -> bool:
    scores = [float(row["official_score"]) for row in rows]
    deltas = [float(row["delta"]) for row in rows]
    return (
        all(grouped_row_passes(row) for row in rows)
        and math.fsum(scores) / len(scores) >= GATE_GROUPED_MEAN
        and min(scores) >= GATE_GROUPED_WORST
        and math.fsum(deltas) / len(deltas) >= GATE_DELTA_MEAN
        and min(deltas) >= GATE_DELTA_WORST
    )


def quality_thresholds() -> dict[str, Any]:
    return {
        "dirac_family_view": False,
        "expected_score_formula": (
            "official_quality * (1 - min(1, max_t max(0, "
            "(q999_t - official_cap_t)/official_cap_t)))"
        ),
        "grouped_delta_mean": GATE_DELTA_MEAN,
        "grouped_delta_worst": GATE_DELTA_WORST,
        "grouped_mean_score": GATE_GROUPED_MEAN,
        "grouped_worst_score": GATE_GROUPED_WORST,
        "official_caps": dict(OFFICIAL_CAPS),
        "reproduction_min_final": GATE_REPRO_MIN,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "stress_ratio_caps": dict(STRESS_RATIO_CAPS),
        "tv_worst_min": TV_WORST_MIN,
    }


def challenger_gate(
    *,
    comparator_valid: bool,
    reproduction: Mapping[str, Any],
    grouped_rows: Sequence[Mapping[str, Any]],
    expected_ok: bool,
) -> dict[str, Any]:
    if len(grouped_rows) != N_FRESH_SEEDS:
        raise RuntimeError("challenger gate expects 12 grouped seed rows")
    seeds = [int(row["fold_seed"]) for row in grouped_rows]
    if tuple(seeds) != EXPLICIT_CHALLENGER_SEEDS:
        raise RuntimeError("challenger seed list is not the sealed list")
    if any(seed in blocked_seeds() for seed in seeds):
        raise RuntimeError("blocked seeds entered the challenger gate")
    reproduction_ok = reproduction_matches_reference(reproduction)
    grouped_ok = grouped_aggregate_passes(grouped_rows)
    if not comparator_valid:
        decision = NO_REF_DECISION
    elif not reproduction_ok:
        decision = REPRO_FAIL_DECISION
    elif not grouped_ok or not expected_ok:
        decision = GROUPED_FAIL_DECISION
    else:
        decision = PASS_DECISION
    return {
        "decision": decision,
        "grouped_ok": grouped_ok,
        "passed": decision == PASS_DECISION,
        "reproduction_ok": reproduction_ok,
        "runtime_export": False,
        "thresholds": quality_thresholds(),
    }


def build_canonical_protocol() -> dict[str, Any]:
    counts = family_counts_from_inputs()
    epsilon = epsilon_from_input_paths()
    fresh = derive_fresh_challenger_seeds()
    if TRAIN_INPUTS.is_file() and sha256_path(TRAIN_INPUTS) != EXPECTED_TRAIN_INPUTS_SHA256:
        raise RuntimeError("train inputs hash drifted while sealing v7 challenger")
    if DEV_INPUTS.is_file() and sha256_path(DEV_INPUTS) != EXPECTED_DEV_INPUTS_SHA256:
        raise RuntimeError("dev inputs hash drifted while sealing v7 challenger")
    return sort_mapping(
        {
            "blocked_seeds": list(blocked_seeds()),
            "comparator": {
                "fallback_dev": {
                    "n_k1": FALLBACK_DEV_N_K1,
                    "official_final_score": FALLBACK_DEV_OFFICIAL,
                    "ratios": FALLBACK_DEV_RATIOS,
                    "tier_quality": FALLBACK_DEV_TIER_QUALITY,
                },
                "fallback_note": (
                    "Current runtime Dev official pin only. Do not subtract "
                    "it from grouped OOF scores."
                ),
                "grouped": (
                    "same-seed current budget_brake_router.make_submission "
                    "on the same public grouped folds. Same official scorer "
                    "as the candidate. Justified so delta is protocol-aligned "
                    "and is not an exact-cost OOF subtraction."
                ),
                "reproduction": {
                    "final_score": V7_FAIR_DEV_OFFICIAL,
                    "note": (
                        "Published Train-only V7 fair Dev. Exact Decimal "
                        "12-place match required; no hyperparameter search."
                    ),
                    "source": (
                        "tykimdream experiments/results/"
                        "aggressive-v7-fair-dev.json routers.v7"
                    ),
                    "tiers": V7_FAIR_DEV_TIERS,
                    "tolerance": REPRODUCTION_TOLERANCE,
                },
            },
            "decisions": {
                "grouped_fail": GROUPED_FAIL_DECISION,
                "no_valid_reference": NO_REF_DECISION,
                "pass": PASS_DECISION,
                "reproduction_fail": REPRO_FAIL_DECISION,
            },
            "epsilon": epsilon,
            "experiment": PROTOCOL_ID,
            "family_counts": counts,
            "fidelity_core_sha256": FIDELITY_CORE_SHA256,
            "fresh_seeds": list(fresh),
            "methodology": methodology_snapshot(),
            "n_fresh_seeds": N_FRESH_SEEDS,
            "output": {
                "audit_relative": AUDIT_RELATIVE,
                "confirm_report_forbidden": CONFIRM_REPORT_RELATIVE,
                "e1f_report_forbidden": E1F_REPORT_RELATIVE,
                "fidelity_report_forbidden": FIDELITY_REPORT_RELATIVE,
                "phase2_report_forbidden": PHASE2_REPORT_RELATIVE,
                "report_relative": f"{OUT_RELATIVE}/report.json",
            },
            "phase": "A",
            "pins": {
                "budget_brake_artifact_sha256": source_sha256(
                    BUDGET_BRAKE_ARTIFACT_RELATIVE
                ),
                # Frozen at seal time (commit 8d77f4f). The small-batch guard
                # later changed budget_brake_router.py through its own gated
                # protocol, so the live file hash no longer applies here.
                "budget_brake_source_sha256": (
                    "a4b3583d3e94dd41b4b447a57a0d25b0d35b6319e4a16df1091a2a53f778142c"
                ),
                "dev_inputs_sha256": EXPECTED_DEV_INPUTS_SHA256,
                "dev_outcomes_sha256": EXPECTED_DEV_OUTCOMES_SHA256,
                "family_guard_artifact_sha256": source_sha256(
                    FAMILY_GUARD_ARTIFACT_RELATIVE
                ),
                "methodology_sha256": methodology_sha256(),
                "n_dev": EXPECTED_N_DEV,
                "n_public": EXPECTED_N_PUBLIC,
                "n_train": EXPECTED_N_TRAIN,
                "policy_sha256": EXPECTED_POLICY_SHA256,
                "source_relative": "research/lab/v7_challenger.py",
                "train_inputs_sha256": EXPECTED_TRAIN_INPUTS_SHA256,
                "train_outcomes_sha256": EXPECTED_TRAIN_OUTCOMES_SHA256,
            },
            "protocol_id": PROTOCOL_ID,
            "reference": {
                "adaptive_commit": REFERENCE_V7_ADAPTIVE,
                "fair_dev_sha256": (
                    "07a721a83c3aee15419ad5b3f1467574ab865e0ec39927a26c1433b83a7d0f06"
                ),
                "file_sha256": {
                    "aggressive_v5.py": (
                        "4ade9fc11269497aa9454439c8e80a16784be1c8a983cd77470f8aa0eca85d85"
                    ),
                    "aggressive_v6.py": (
                        "094c5bd06fbf9fc19bc75e6950955113037a580bf18ab3accc60ad90c7578e8c"
                    ),
                    "aggressive_v7.py": (
                        "b5e4dc18725371a3ef1c3cbff94bede53366eeb750940f12b93393ebeac1c402"
                    ),
                    "competition.py": (
                        "a8fe3e5c20dfb860e785dc8253f783ccd72b247c4e4aa3482433fbcbfe8db50d"
                    ),
                    "submission.py": (
                        "e1af492f0ce44908ce8c617ee2d0cd737987acc67f83f36d4ff7d07a3ccc307e"
                    ),
                    "train_aggressive_router_v5.py": (
                        "4918605ad449c3483ce1718133d005dc1f7983e0abc5e14b6a0771a9518613b6"
                    ),
                },
                "final_commit": REFERENCE_FINAL,
                "license": "Apache-2.0",
                "license_sha256": (
                    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
                ),
                "published_profile_sha256": (
                    "172ddac2eda27b58d3ef0c5d684e1183a4e86988cd0fc662ed73c9defe2215f2"
                ),
                "repository": (
                    "https://github.com/tykimdream/ossp-2026-llm-router-challenge"
                ),
                "train_reference_rates_published": TRAIN_REFERENCE_RATES,
                "v61_commit": REFERENCE_V61,
            },
            "required_families": list(REQUIRED_FAMILIES),
            "runtime_export": False,
            "schema_version": SCHEMA_VERSION,
            "seed_derivation": {
                "algorithm": SEED_DERIVATION,
                "core_sha256": FIDELITY_CORE_SHA256,
                "fail_closed_on_collision": True,
                "n": N_FRESH_SEEDS,
                "prefix": SEED_PREFIX,
                "skip_digest_on_collision": False,
            },
            "stress": {
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "dirac_family_view": False,
            },
            "thresholds": quality_thresholds(),
            "uses_competitor_artifact_weights": False,
        }
    )


def canonical_protocol_text(protocol: Mapping[str, Any]) -> str:
    return canonical_json_text(sort_mapping(dict(protocol)))


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return sha256_text(canonical_protocol_text(protocol))


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("protocol is not a JSON object")
    if "generated_at" in payload:
        raise RuntimeError("protocol must not contain generated_at")
    return payload


def verify_protocol(
    protocol: Mapping[str, Any],
    expected_sha256: str,
    *,
    train_path: Path = TRAIN_INPUTS,
    dev_path: Path = DEV_INPUTS,
) -> str:
    digest = protocol_sha256(protocol)
    if digest != expected_sha256:
        raise RuntimeError(
            f"protocol sha mismatch: got {digest}, expected {expected_sha256}"
        )
    if protocol.get("methodology") != methodology_snapshot():
        raise RuntimeError("v7 methodology snapshot drifted")
    if tuple(int(seed) for seed in protocol["fresh_seeds"]) != derive_fresh_challenger_seeds():
        raise RuntimeError("sealed challenger seeds drifted")
    if set(protocol["fresh_seeds"]) & set(blocked_seeds()):
        raise RuntimeError("sealed challenger seeds overlap blocked seeds")
    if protocol["epsilon"] != epsilon_from_input_paths(train_path, dev_path):
        raise RuntimeError("sealed epsilon drifted")
    if protocol["uses_competitor_artifact_weights"] is not False:
        raise RuntimeError("competitor weights must stay unused")
    if protocol["thresholds"]["dirac_family_view"] is not False:
        raise RuntimeError("Dirac family view must stay off")
    if protocol["pins"]["methodology_sha256"] != methodology_sha256():
        raise RuntimeError("methodology sha drifted")
    if protocol["comparator"]["reproduction"]["final_score"] != V7_FAIR_DEV_OFFICIAL:
        raise RuntimeError("fair Dev reproduction pin drifted")
    if protocol["comparator"]["fallback_dev"]["official_final_score"] != FALLBACK_DEV_OFFICIAL:
        raise RuntimeError("fallback Dev pin drifted")
    banned = json.dumps(protocol["thresholds"], sort_keys=True)
    if "0.669517" in banned or "0.691477" in banned:
        raise RuntimeError("quality thresholds contain comparator pins")
    return digest


def write_canonical_protocol(path: Path = PROTOCOL_PATH) -> Tuple[dict[str, Any], str]:
    protocol = build_canonical_protocol()
    write_json_atomic(path, protocol)
    return protocol, protocol_sha256(protocol)


def refuse_foreign_output_path(path: Path) -> None:
    text = path.resolve().as_posix()
    if "compare-e1f-cost-conditioned-frontier" in text:
        raise RuntimeError("challenger must not write the E1F report path")
    if "frozen-runtime-fidelity" in text:
        raise RuntimeError("challenger must not write the fidelity report path")
    if "phase2-chuf-predicted-cost" in text:
        raise RuntimeError("challenger must not write the phase2 report path")
    if "confirm-chuf-tvball" in text:
        raise RuntimeError("challenger must not write the confirmation report path")


@dataclass(frozen=True)
class FittedMember:
    alpha: float
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    intercept: Tuple[float, ...]
    coefficients: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class V7Bundle:
    compiled: Mapping[str, CompiledTier]
    safety_ratios: Mapping[str, float]
    reference_rates: Mapping[str, float]
    members: Mapping[float, FittedMember]
    template_folds: int


def outcome_cost(outcome: Any, policy: Any) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    return float(
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )


def feature_matrix(episodes: Sequence[Episode], hash_bins: int = HASH_BINS) -> np.ndarray:
    return np.asarray(
        [raw_feature_vector(episode, hash_bins) for episode in episodes],
        dtype=np.float64,
    )


def target_matrix(inputs: InputBatch, outcomes: Any, policy: Any) -> np.ndarray:
    index = {(item.episode_id, item.model_id): item for item in outcomes.outcomes}
    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    }
    if set(index) != expected:
        raise RuntimeError("training outcome matrix is incomplete")
    rows = []
    for episode in inputs.episodes:
        items = [index[(episode.episode_id, model_id)] for model_id in MODEL_IDS]
        rows.append(
            [float(item.score) for item in items]
            + [math.log(outcome_cost(item, policy)) for item in items]
        )
    return np.asarray(rows, dtype=np.float64)


def template_fold_keys(episodes: Sequence[Episode]) -> np.ndarray:
    return np.asarray(
        [fnv1a64(normalized_template(episode_text(episode))) for episode in episodes],
        dtype=np.uint64,
    )


def ridge_fit_standardized(
    matrix: np.ndarray, targets: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    rows, columns = standardized.shape
    with np.errstate(all="ignore"):
        if rows <= columns:
            system = standardized @ standardized.T + float(alpha) * np.eye(rows)
            coefficients = standardized.T @ np.linalg.solve(system, centered)
        else:
            system = standardized.T @ standardized + float(alpha) * np.eye(columns)
            coefficients = np.linalg.solve(system, standardized.T @ centered)
    if not np.isfinite(coefficients).all():
        raise RuntimeError("ridge coefficients are not finite")
    return mean, scale, intercept, coefficients


def ridge_predict_standardized(
    matrix: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    intercept: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    with np.errstate(all="ignore"):
        result = (matrix - mean) / scale @ coefficients + intercept
    if not np.isfinite(result).all():
        raise RuntimeError("ridge predictions are not finite")
    return result


def ridge_oof(
    matrix: np.ndarray,
    targets: np.ndarray,
    fold_keys: np.ndarray,
    folds: int,
    alpha: float,
) -> np.ndarray:
    predictions = np.empty_like(targets)
    fold_ids = fold_keys % int(folds)
    for fold in range(int(folds)):
        validation = fold_ids == fold
        training = ~validation
        if not validation.any() or not training.any():
            raise ValueError("template-group fold is empty")
        mean, scale, intercept, coefficients = ridge_fit_standardized(
            matrix[training], targets[training], alpha
        )
        predictions[validation] = ridge_predict_standardized(
            matrix[validation], mean, scale, intercept, coefficients
        )
    return predictions


def prediction_arrays(predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.clip(predictions[:, : len(MODEL_IDS)], 0.0, 1.0).copy()
    costs = np.exp(np.clip(predictions[:, len(MODEL_IDS) :], -50.0, 50.0))
    costs[:, 1] = np.maximum(costs[:, 1], costs[:, 0] * (1.0 + COST_MONOTONE_EPS))
    costs[:, 2] = np.maximum(costs[:, 2], costs[:, 1] * (1.0 + COST_MONOTONE_EPS))
    return scores, costs


def select_models_vectorized(
    scores: np.ndarray,
    costs: np.ndarray,
    *,
    tier: str,
    safety_ratio: float,
    budget_multiplier: float,
    fill_safety: float = PREMIUM_AX31_FILL_SAFETY,
    steps: int = FIXED_BISECTION_STEPS,
) -> Tuple[np.ndarray, float]:
    light_total = float(costs[:, 0].sum())
    cap = light_total * max(1.0, budget_multiplier * safety_ratio)
    adjusted = scores.copy()
    if tier == "fast":
        adjusted[:, 2] = FAST_K1_BAN_SCORE

    def choose(penalty: float) -> Tuple[np.ndarray, float]:
        selected = np.argmax(adjusted - penalty * costs / light_total, axis=1)
        total = float(costs[np.arange(len(selected)), selected].sum())
        return selected, total

    selected, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        selected, total = choose(high)
        while total > cap and high < 2**60:
            low, high = high, high * 2.0
            selected, total = choose(high)
        for _ in range(int(steps)):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high, selected, total = middle, candidate, candidate_total
            else:
                low = middle
    if tier == "premium":
        fill_cap = max(total, light_total * budget_multiplier * fill_safety)
        base_selected = selected.copy()
        eligible = base_selected == 0
        gain = adjusted[:, 1] - adjusted[:, 0]
        extra = costs[:, 1] - costs[:, 0]

        def fill(penalty: float) -> Tuple[np.ndarray, float]:
            candidate = base_selected.copy()
            promote = eligible & ((gain - penalty * extra / light_total) > 0.0)
            candidate[promote] = 1
            candidate_total = float(costs[np.arange(len(candidate)), candidate].sum())
            return candidate, candidate_total

        candidate, candidate_total = fill(0.0)
        if candidate_total > fill_cap:
            low, high = 0.0, 1.0
            candidate, candidate_total = fill(high)
            while candidate_total > fill_cap and high < 2**60:
                low, high = high, high * 2.0
                candidate, candidate_total = fill(high)
            for _ in range(int(steps)):
                middle = (low + high) / 2.0
                filled, filled_total = fill(middle)
                if filled_total <= fill_cap:
                    high, candidate, candidate_total = middle, filled, filled_total
                else:
                    low = middle
        if candidate_total <= fill_cap:
            selected, total = candidate, candidate_total
    return selected, total / light_total


def observed_quality_ratio(
    selected: np.ndarray,
    scores: np.ndarray,
    costs: np.ndarray,
    mask: np.ndarray | None = None,
) -> Tuple[float, float]:
    if mask is None:
        mask = np.ones(len(selected), dtype=bool)
    rows = np.flatnonzero(mask)
    quality = float(scores[rows, selected[rows]].mean())
    ratio = float(costs[rows, selected[rows]].sum() / costs[rows, 0].sum())
    return quality, ratio


def calibrate_tier_safety(
    predictions: np.ndarray,
    actual_scores: np.ndarray,
    actual_costs: np.ndarray,
    fold_ids: np.ndarray,
    *,
    tier: str,
    target: float,
    budget_multiplier: float,
) -> Tuple[float, dict[str, Any]]:
    predicted_scores, predicted_costs = prediction_arrays(predictions)
    fold_values = tuple(sorted({int(item) for item in fold_ids}))
    best = None
    for step in range(161):
        safety = 0.2 + step / 200.0
        selected, predicted_ratio = select_models_vectorized(
            predicted_scores,
            predicted_costs,
            tier=tier,
            safety_ratio=safety,
            budget_multiplier=budget_multiplier,
        )
        quality, actual_ratio = observed_quality_ratio(
            selected, actual_scores, actual_costs
        )
        fold_reports = {}
        for fold in fold_values:
            fold_quality, fold_ratio = observed_quality_ratio(
                selected, actual_scores, actual_costs, fold_ids == fold
            )
            fold_reports[str(fold)] = {
                "budget_ratio": fold_ratio,
                "quality": fold_quality,
            }
        worst_ratio = max(item["budget_ratio"] for item in fold_reports.values())
        if actual_ratio > target or worst_ratio > target:
            continue
        rank = (quality, -worst_ratio, -actual_ratio, -safety)
        if best is None or rank > best[0]:
            best = (rank, safety, predicted_ratio, quality, actual_ratio, fold_reports)
    if best is None:
        raise RuntimeError(f"{tier} has no aggressive budget candidate")
    return best[1], {
        "actual_budget_ratio": best[4],
        "folds": best[5],
        "predicted_budget_ratio": best[2],
        "safety_ratio": best[1],
        "tier_score": best[3],
        "worst_fold_budget_ratio": max(
            item["budget_ratio"] for item in best[5].values()
        ),
    }


def _member(
    mean: np.ndarray,
    scale: np.ndarray,
    intercept: np.ndarray,
    coefficients: np.ndarray,
    alpha: float,
) -> FittedMember:
    return FittedMember(
        float(alpha),
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in intercept),
        tuple(tuple(float(value) for value in row) for row in coefficients),
    )


def compile_members(members: Sequence[FittedMember], model_id: str, *, score: bool) -> LinearHead:
    index = MODEL_IDS.index(model_id) if score else len(MODEL_IDS) + MODEL_IDS.index(model_id)
    heads = tuple(
        LinearHead(
            member.intercept[index],
            tuple(row[index] for row in member.coefficients),
        )
        for member in members
    )
    means = tuple(member.mean for member in members)
    scales = tuple(member.scale for member in members)
    return compile_head(heads, means, scales)


def compile_bundle_tiers(members_by_alpha: Mapping[float, FittedMember]) -> dict[str, CompiledTier]:
    compiled = {}
    for tier, alphas in ENSEMBLE_ALPHAS.items():
        members = tuple(members_by_alpha[alpha] for alpha in alphas)
        compiled[tier] = CompiledTier(
            {
                model_id: compile_members(members, model_id, score=True)
                for model_id in MODEL_IDS
            },
            {
                model_id: compile_members(members, model_id, score=False)
                for model_id in MODEL_IDS
            },
        )
    return compiled


def rates_from_episodes(episodes: Sequence[Episode]) -> dict[str, float]:
    rows = tuple(
        signal_row(episode, raw_feature_vector(episode)) for episode in episodes
    )
    return reference_rates(rows)


def fit_v7_bundle(inputs: InputBatch, outcomes: Any, policy: Any) -> V7Bundle:
    matrix = feature_matrix(inputs.episodes)
    targets = target_matrix(inputs, outcomes, policy)
    fold_keys = template_fold_keys(inputs.episodes)
    fold_ids = fold_keys % TEMPLATE_FOLDS
    actual_scores = targets[:, : len(MODEL_IDS)]
    actual_costs = np.exp(targets[:, len(MODEL_IDS) :])
    oof = {
        alpha: ridge_oof(matrix, targets, fold_keys, TEMPLATE_FOLDS, alpha)
        for alpha in UNIQUE_ENSEMBLE_ALPHAS
    }
    safety_ratios = {}
    for tier in TIERS:
        stacked = np.mean(
            np.stack([oof[alpha] for alpha in ENSEMBLE_ALPHAS[tier]], axis=0),
            axis=0,
        )
        safety, _report = calibrate_tier_safety(
            stacked,
            actual_scores,
            actual_costs,
            fold_ids,
            tier=tier,
            target=float(TRAIN_BUDGET_TARGETS[tier]),
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        )
        safety_ratios[tier] = safety
    members = {}
    for alpha in UNIQUE_ENSEMBLE_ALPHAS:
        mean, scale, intercept, coefficients = ridge_fit_standardized(
            matrix, targets, alpha
        )
        members[alpha] = _member(mean, scale, intercept, coefficients, alpha)
    return V7Bundle(
        compile_bundle_tiers(members),
        safety_ratios,
        rates_from_episodes(inputs.episodes),
        members,
        TEMPLATE_FOLDS,
    )


def route_bundle(
    inputs: InputBatch, bundle: V7Bundle, tier: str, policy: Any
) -> Tuple[Tuple[str, ...], str]:
    if not learned_path_allowed(inputs):
        from ossp_router.budget_brake_router import (
            load_bundled_artifact,
            make_submission as make_brake,
        )
        from ossp_router.protocol import load_bundled_policy

        used = policy if policy is not None else load_bundled_policy()
        plan = make_brake(inputs, used, load_bundled_artifact(), tier)
        return (
            tuple(decision.model_id for decision in plan.submission.decisions),
            FALLBACK_ROUTER,
        )
    selected, _ratio = allocate_tier(
        inputs,
        bundle.compiled[tier],
        tier=tier,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=float(bundle.safety_ratios[tier]),
        reference=bundle.reference_rates,
    )
    return selected, "v7-learned"


def brake_models(inputs: InputBatch, policy: Any, artifact: Any, tier: str) -> Tuple[str, ...]:
    from ossp_router.budget_brake_router import make_submission as make_brake

    plan = make_brake(inputs, policy, artifact, tier)
    return tuple(decision.model_id for decision in plan.submission.decisions)


def reproduction_record(official: Mapping[str, Any]) -> dict[str, Any]:
    tiers = {}
    hard = True
    under = True
    for tier in TIERS:
        block = official["tiers"][tier]
        counts = {name: int(block["model_counts"][name]) for name in sorted(block["model_counts"])}
        tiers[tier] = {
            "budget_ratio": str(block["budget_ratio"]),
            "model_counts": counts,
            "quality_score": str(block["quality_score"]),
        }
        hard = hard and bool(block["budget_passed"])
        under = under and ratio_under_95(str(block["budget_ratio"]), tier)
    return {
        "all_hard_passed": hard,
        "all_ratio_under_95": under,
        "final_score": str(official["final_score"]),
        "tiers": sort_mapping(tiers),
        "tolerance": REPRODUCTION_TOLERANCE,
    }


def fallback_dev_reproduced(
    official: Mapping[str, Any], models: Mapping[str, Sequence[str]]
) -> bool:
    n_k1 = int(sum(model == MODEL_IDS[2] for model in models["premium"]))
    if str(official["final_score"]) != FALLBACK_DEV_OFFICIAL:
        return False
    if n_k1 != FALLBACK_DEV_N_K1:
        return False
    for tier in TIERS:
        if str(official["tiers"][tier]["budget_ratio"]) != FALLBACK_DEV_RATIOS[tier]:
            return False
        if str(official["tiers"][tier]["quality_score"]) != FALLBACK_DEV_TIER_QUALITY[tier]:
            return False
    return True


def bundle_sha256(bundle: V7Bundle) -> str:
    payload = {
        "compiled": {
            tier: {
                "log_cost": {
                    model_id: {
                        "coefficients": list(head.coefficients),
                        "intercept": head.intercept,
                    }
                    for model_id, head in bundle.compiled[tier].log_cost_heads.items()
                },
                "score": {
                    model_id: {
                        "coefficients": list(head.coefficients),
                        "intercept": head.intercept,
                    }
                    for model_id, head in bundle.compiled[tier].score_heads.items()
                },
            }
            for tier in TIERS
        },
        "reference_rates": dict(bundle.reference_rates),
        "safety_ratios": dict(bundle.safety_ratios),
    }
    return sha256_text(canonical_json_text(sort_mapping(payload)))


def report_sha256(report: Mapping[str, Any]) -> str:
    if "generated_at" in report:
        raise RuntimeError("report must not contain generated_at")
    return sha256_text(canonical_json_text(sort_mapping(dict(report))))


def decision_core_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return sort_mapping(
        {
            "decision": report["decision"],
            "experiment": report["experiment"],
            "gate": report["gate"],
            "grouped_summary": report["grouped"]["summary"],
            "protocol_sha256": report["protocol_sha256"],
            "report_type": report["report_type"],
            "reproduction": {
                "final_score": report["reproduction"]["final_score"],
                "passed": report["reproduction"]["passed"],
                "tiers": report["reproduction"]["tiers"],
            },
            "schema_version": report["schema_version"],
            "thresholds": report["thresholds"],
        }
    )


def decision_core_sha256(report: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(decision_core_payload(report)))


def _hard_and_ratio95(official: Mapping[str, Any]) -> Tuple[bool, bool]:
    hard = all(bool(official["tiers"][tier]["budget_passed"]) for tier in TIERS)
    under = all(
        ratio_under_95(str(official["tiers"][tier]["budget_ratio"]), tier) for tier in TIERS
    )
    return hard, under


def evaluate_grouped_seed(
    pool: Any,
    policy: Any,
    brake: Any,
    epsilon: float,
) -> Tuple[dict[str, Any], dict[str, Tuple[str, ...]], dict[str, Any]]:
    from research.lab.chuf_predicted_cost_phase2 import tv_cost_worst
    from research.lab.chuf_tvball_confirmation import REQUIRED_FAMILIES, tv_worst
    from research.lab.e1_objectives import score_decisions
    from research.lab.e2_cost_uncertainty import grouped_ratio_bootstrap
    from research.lab.modeling import official_score
    from research.lab.public_pool import subset_inputs, subset_outcomes

    fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
    n_episodes = len(pool.inputs.episodes)
    oof_models = {tier: [""] * n_episodes for tier in TIERS}
    fold_rows = []
    for fold in range(int(fold_ids.max()) + 1):
        train_idx = [index for index, value in enumerate(fold_ids) if value != fold]
        test_idx = [index for index, value in enumerate(fold_ids) if value == fold]
        train_inputs = subset_inputs(pool.inputs, train_idx)
        train_outcomes = subset_outcomes(pool.inputs, pool.outcomes, train_idx)
        test_inputs = subset_inputs(pool.inputs, test_idx)
        test_outcomes = subset_outcomes(pool.inputs, pool.outcomes, test_idx)
        bundle = fit_v7_bundle(train_inputs, train_outcomes, policy)
        candidate = {
            tier: route_bundle(test_inputs, bundle, tier, policy)[0] for tier in TIERS
        }
        comparator = {
            tier: brake_models(test_inputs, policy, brake, tier) for tier in TIERS
        }
        for tier in TIERS:
            for index, model_id in zip(test_idx, candidate[tier]):
                oof_models[tier][index] = model_id
        cand_official = official_score(test_inputs, test_outcomes, policy, candidate)
        base_official = official_score(test_inputs, test_outcomes, policy, comparator)
        cand_hard, cand_ratio = _hard_and_ratio95(cand_official)
        base_hard, base_ratio = _hard_and_ratio95(base_official)
        fold_rows.append(
            {
                "candidate_final": cand_official["final_score"],
                "candidate_hard_ok": cand_hard,
                "candidate_ratio95_ok": cand_ratio,
                "comparator_final": base_official["final_score"],
                "comparator_hard_ok": base_hard,
                "comparator_ratio95_ok": base_ratio,
                "fold": int(fold),
                "n": len(test_idx),
            }
        )
    pooled_models = {tier: tuple(oof_models[tier]) for tier in TIERS}
    if any(not model for models in pooled_models.values() for model in models):
        raise RuntimeError("grouped OOF left an unassigned episode")
    comparator_pooled = {
        tier: brake_models(pool.inputs, policy, brake, tier) for tier in TIERS
    }
    pooled = score_decisions(pool, pooled_models)
    base_pooled = score_decisions(pool, comparator_pooled)
    official_delta = float(pooled["official_final_score"]) - float(
        base_pooled["official_final_score"]
    )
    family_deltas = {}
    for name in REQUIRED_FAMILIES:
        indexes = [
            index for index, family in enumerate(pool.families) if family == name
        ]
        if len(indexes) < 20:
            continue
        family_deltas[name] = float(
            score_decisions(pool, pooled_models, indexes=indexes)["official_final_score"]
        ) - float(
            score_decisions(pool, comparator_pooled, indexes=indexes)[
                "official_final_score"
            ]
        )
    quality_tv = tv_worst(official_delta, epsilon, family_deltas)
    actual = np.asarray(pool.costs, dtype=np.float64)
    q999 = {}
    bootstrap_ok = True
    tv_cost_ok = True
    for tier in TIERS:
        block = grouped_ratio_bootstrap(
            pooled_models[tier],
            actual,
            actual[:, 0],
            pool.group_keys,
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )
        q999[tier] = float(block["q99_9"])
        if q999[tier] >= float(STRESS_RATIO_CAPS[tier]):
            bootstrap_ok = False
        spend = {}
        light = {}
        columns = np.asarray(
            [{"ax31-light": 0, "ax31": 1, "axk1-think": 2}[model] for model in pooled_models[tier]],
            dtype=np.int64,
        )
        selected = actual[np.arange(actual.shape[0]), columns]
        for name in REQUIRED_FAMILIES:
            mask = np.asarray([family == name for family in pool.families])
            spend[name] = float(selected[mask].mean())
            light[name] = float(actual[mask, 0].mean())
        center = {
            name: float(sum(1 for family in pool.families if family == name))
            / float(len(pool.families))
            for name in REQUIRED_FAMILIES
        }
        if tv_cost_worst(center, spend, light, epsilon) > float(OFFICIAL_CAPS[tier]):
            tv_cost_ok = False
    pooled_hard = all(pooled["tiers"][tier]["within_hard_cap"] for tier in TIERS)
    fold_hard = all(bool(row["candidate_hard_ok"]) for row in fold_rows)
    pooled_ratio = all(
        float(pooled["tiers"][tier]["budget_ratio"]) < float(STRESS_RATIO_CAPS[tier])
        for tier in TIERS
    )
    fold_ratio = all(bool(row["candidate_ratio95_ok"]) for row in fold_rows)
    official_score_value = float(pooled["official_final_score"])
    expected = expected_score(official_score_value, q999)
    row = {
        "bootstrap_q999": q999,
        "bootstrap_q999_under_95_ok": bootstrap_ok,
        "comparator_score": float(base_pooled["official_final_score"]),
        "delta": official_delta,
        "dirac_family_view": False,
        "expected_score": expected,
        "fold_hard_caps_ok": fold_hard,
        "fold_ratio_under_95_ok": fold_ratio,
        "fold_seed": int(pool.identity["fold_seed"]),
        "folds": fold_rows,
        "official_score": official_score_value,
        "pooled_hard_caps_ok": pooled_hard and fold_hard,
        "pooled_ratio_under_95_ok": pooled_ratio and fold_ratio,
        "tv_cost_under_official_ok": tv_cost_ok,
        "tv_quality_worst": quality_tv,
    }
    return row, pooled_models, {
        "comparator_final": base_pooled["official_final_score"],
        "expected_score": expected,
        "family_deltas": family_deltas,
        "q999": q999,
    }


def run_challenger(
    protocol: Mapping[str, Any],
    *,
    output: Path,
    audit_output: Path,
) -> dict[str, Any]:
    """Public Train-only reproduction then grouped OOF. Run path only."""

    from ossp_router.budget_brake_router import load_bundled_artifact
    from ossp_router.protocol import load_bundled_policy, load_input, load_outcomes
    from research.lab.e1c_regime_residual import relabel_folds
    from research.lab.modeling import official_score
    from research.lab.public_pool import load_public_pool

    refuse_foreign_output_path(output)
    refuse_foreign_output_path(audit_output)
    if output.exists() or audit_output.exists():
        raise RuntimeError("challenger output exists; refuse overwrite")
    digest = protocol_sha256(protocol)
    verify_protocol(protocol, digest)
    policy = load_bundled_policy()
    brake = load_bundled_artifact()
    train_inputs = load_input(TRAIN_INPUTS)
    train_outcomes = load_outcomes(TRAIN_OUTCOMES)
    dev_inputs = load_input(DEV_INPUTS)
    dev_outcomes = load_outcomes(DEV_OUTCOMES)
    if sha256_path(TRAIN_INPUTS) != EXPECTED_TRAIN_INPUTS_SHA256:
        raise RuntimeError("train inputs hash drifted in the run path")
    if sha256_path(TRAIN_OUTCOMES) != EXPECTED_TRAIN_OUTCOMES_SHA256:
        raise RuntimeError("train outcomes hash drifted in the run path")
    if sha256_path(DEV_INPUTS) != EXPECTED_DEV_INPUTS_SHA256:
        raise RuntimeError("dev inputs hash drifted in the run path")
    if sha256_path(DEV_OUTCOMES) != EXPECTED_DEV_OUTCOMES_SHA256:
        raise RuntimeError("dev outcomes hash drifted in the run path")
    comparator_models = {
        tier: brake_models(dev_inputs, policy, brake, tier) for tier in TIERS
    }
    comparator_official = official_score(
        dev_inputs, dev_outcomes, policy, comparator_models
    )
    comparator_valid = fallback_dev_reproduced(comparator_official, comparator_models)
    bundle = fit_v7_bundle(train_inputs, train_outcomes, policy)
    candidate_dev = {
        tier: route_bundle(dev_inputs, bundle, tier, policy)[0] for tier in TIERS
    }
    reproduction_official = official_score(
        dev_inputs, dev_outcomes, policy, candidate_dev
    )
    reproduction = reproduction_record(reproduction_official)
    reproduction["passed"] = reproduction_matches_reference(reproduction)
    reproduction["artifact_sha256"] = bundle_sha256(bundle)
    reproduction["computed_train_reference_rates"] = dict(bundle.reference_rates)
    reproduction["safety_ratios"] = dict(bundle.safety_ratios)
    pool = load_public_pool()
    epsilon = float(protocol["epsilon"])
    grouped_rows = []
    grouped_models = {}
    grouped_details = {}
    for seed in EXPLICIT_CHALLENGER_SEEDS:
        labeled = relabel_folds(pool, int(seed))
        row, models, details = evaluate_grouped_seed(labeled, policy, brake, epsilon)
        grouped_rows.append(row)
        grouped_models[str(seed)] = {
            tier: list(models[tier]) for tier in TIERS
        }
        grouped_details[str(seed)] = details
    expected_ok = all(
        expected_score_beats_fallback(
            float(row["official_score"]), row["bootstrap_q999"]
        )
        for row in grouped_rows
    )
    gate = challenger_gate(
        comparator_valid=comparator_valid,
        reproduction=reproduction,
        grouped_rows=grouped_rows,
        expected_ok=expected_ok,
    )
    if gate["decision"] == PASS_DECISION:
        reason = (
            "V7 challenger passed reproduction and grouped gates. "
            "This is not a runtime export. Hand off to independent audit only."
        )
    elif gate["decision"] == NO_REF_DECISION:
        reason = (
            "Current-runtime Dev comparator failed pin reproduction. "
            "No valid reference. Keep the current runtime."
        )
    elif gate["decision"] == REPRO_FAIL_DECISION:
        reason = (
            "Train-only Dev reproduction missed the exact published pin. "
            "No hyperparameter search. Keep the current runtime."
        )
    else:
        reason = (
            "Grouped or expected-score gate failed. Keep the current runtime."
        )
    audit = sort_mapping(
        {
            "grouped": {
                seed: {
                    "episode_ids": [
                        episode.episode_id for episode in pool.inputs.episodes
                    ],
                    "models": grouped_models[seed],
                }
                for seed in grouped_models
            },
            "reproduction": {
                "episode_ids": [episode.episode_id for episode in dev_inputs.episodes],
                "models": {tier: list(candidate_dev[tier]) for tier in TIERS},
            },
            "schema_version": SCHEMA_VERSION,
        }
    )
    write_json_atomic(audit_output, audit)
    summary = {
        "expected_ok": expected_ok,
        "mean_delta": math.fsum(float(row["delta"]) for row in grouped_rows)
        / len(grouped_rows),
        "mean_official": math.fsum(float(row["official_score"]) for row in grouped_rows)
        / len(grouped_rows),
        "worst_delta": min(float(row["delta"]) for row in grouped_rows),
        "worst_official": min(float(row["official_score"]) for row in grouped_rows),
        "worst_tv_quality": min(float(row["tv_quality_worst"]) for row in grouped_rows),
    }
    report = sort_mapping(
        {
            "audit": {
                "n_rows": int(EXPECTED_N_DEV)
                + int(EXPECTED_N_PUBLIC) * int(N_FRESH_SEEDS),
                "relative_path": AUDIT_RELATIVE,
                "sha256": sha256_path(audit_output),
            },
            "comparator": {
                "dev_official": comparator_official["final_score"],
                "valid": comparator_valid,
            },
            "decision": gate["decision"],
            "decision_reason": reason,
            "experiment": PROTOCOL_ID,
            "gate": gate,
            "grouped": {
                "details": grouped_details,
                "rows": grouped_rows,
                "summary": summary,
            },
            "protocol_sha256": digest,
            "report_type": REPORT_TYPE,
            "reproduction": reproduction,
            "runtime_export": False,
            "schema_version": SCHEMA_VERSION,
            "thresholds": quality_thresholds(),
        }
    )
    report["decision_core_sha256"] = decision_core_sha256(report)
    if "generated_at" in report:
        raise RuntimeError("report must not contain generated_at")
    return report


def validation_function_names() -> Tuple[str, ...]:
    return (
        "blocked_seeds",
        "build_canonical_protocol",
        "challenger_gate",
        "derive_fresh_challenger_seeds",
        "expected_score",
        "expected_score_beats_fallback",
        "grouped_aggregate_passes",
        "grouped_row_passes",
        "load_protocol",
        "methodology_sha256",
        "methodology_snapshot",
        "protocol_sha256",
        "quality_thresholds",
        "reproduction_matches_reference",
        "verify_protocol",
    )


def assert_validation_path_has_no_outcomes(source: str | None = None) -> None:
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden = {
        "load_outcomes",
        "load_public_pool",
        "run_challenger",
        "oof_chuf_heads",
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in validation_function_names():
        for child in ast.walk(functions[name]):
            if isinstance(child, ast.Name) and child.id in forbidden:
                raise RuntimeError(f"{name} references forbidden {child.id}")
            if isinstance(child, ast.Attribute) and child.attr in forbidden:
                raise RuntimeError(f"{name} references forbidden {child.attr}")


__all__ = (
    "DENSE_FEATURE_NAMES",
    "EXPLICIT_CHALLENGER_SEEDS",
    "EXPECTED_PROTOCOL_SHA256",
    "FALLBACK_DEV_OFFICIAL",
    "GROUPED_FAIL_DECISION",
    "NO_REF_DECISION",
    "OUT_RELATIVE",
    "PASS_DECISION",
    "PROTOCOL_PATH",
    "REPRO_FAIL_DECISION",
    "V7_FAIR_DEV_OFFICIAL",
    "allocate_tier",
    "assert_validation_path_has_no_outcomes",
    "blocked_seeds",
    "build_canonical_protocol",
    "calibrate_tier_safety",
    "challenger_gate",
    "compile_head",
    "decision_core_sha256",
    "dense_feature_vector",
    "derive_fresh_challenger_seeds",
    "expected_score",
    "fill_ax31",
    "hashed_features",
    "learned_path_allowed",
    "load_protocol",
    "protocol_sha256",
    "raw_feature_vector",
    "refuse_foreign_output_path",
    "report_sha256",
    "reproduction_record",
    "ridge_fit_standardized",
    "ridge_oof",
    "ridge_predict_standardized",
    "run_challenger",
    "safety_multiplier",
    "select_models",
    "select_models_vectorized",
    "verify_protocol",
    "write_canonical_protocol",
)
