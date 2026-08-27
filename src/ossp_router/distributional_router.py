# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit-lexicon distributional router.

Serving evaluates compiled boosted trees, predicts mean and upper-tail costs,
uses a low-quantile batch-risk head for Fast/Balanced, and spends through a
canonical concave-prefix queue.  The runtime intentionally needs only the
Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import struct
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .cost_calibrated_router import prompt_family
from .heuristic import episode_text, make_submission as make_heuristic_submission
from .heuristic import write_submission_atomic
from .protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    Message,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_json,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)


ARTIFACT_RESOURCE = "distributional-router.v1.json"
ARTIFACT_TYPE = "scrooge-distributional-router-v1"
FEATURE_CONTRACT = "scrooge-explicit-lexicon-gbt-risk-prefix-v1"
FEATURE_VERSION = "explicit-lexicon-v1"
ROUTER_VERSION = "1.0-distributional-knapsack"
SCHEMA_VERSION = 1
VOCABULARY_SIZE = 1_024

MAX_LEARNED_EPISODES = 6_000
MAX_LEARNED_CHARACTERS = 30_000_000
MAX_LEARNED_MESSAGES = 100_000
SMALL_BATCH_UNIQUE_CUTOFF = 128
SMALL_BATCH_POWER = 0.5
# Train-frozen overlays for unique_count in [1, 127].  The 128+ path keeps
# the bundled family calibration.  Scales follow FAMILY_NAMES order.
SMALL_BATCH_MEAN_SCALES: Tuple[Tuple[float, float, float], ...] = (
    (1.1541784551218242, 1.0155484748810382, 1.0514963565899413),
    (0.6176328432073351, 0.6478308653872528, 0.6979639882744141),
    (1.1398784850239714, 1.175992471047395, 1.140533832003011),
    (1.1561453592957813, 1.1590159967514397, 1.0533802158560723),
    (0.9817093192819162, 0.9875217168938292, 0.9551826851127155),
    (1.053538753725097, 0.9383613988014844, 1.04304079582686),
    (0.8273882259570366, 0.8132039363392366, 0.9983144742409114),
    (0.90475978558204, 1.0181121953764847, 1.0156747409604754),
    (1.159687571618085, 1.0630728591275511, 1.1131132694149781),
    (1.0768912314475494, 1.0805156860500167, 0.8610257963048321),
)
SMALL_BATCH_UPPER_SCALES: Tuple[Tuple[float, float, float], ...] = (
    (1.0848901121902774, 1.0772589155157182, 1.3015089918755758),
    (1.0, 1.0, 1.205470604150946),
    (1.037496753357003, 1.0992972085883426, 1.41386058258802),
    (1.0864421426903625, 2.160897613255152, 1.3972397458230152),
    (1.0120585571479697, 1.0154020023485952, 1.0463726346372089),
    (1.1729280136789426, 1.1722165150573007, 1.1615029895624174),
    (1.0, 1.0, 1.3828049140105283),
    (1.0339518599042492, 1.0600641970571159, 1.1226486973638172),
    (1.2402688309313077, 1.2809568789014045, 1.369168185707051),
    (1.0732993136471245, 1.0166022103969317, 1.1348896446603225),
)
SMALL_BATCH_LIGHT_LOWER_SCALES: Tuple[float, ...] = (
    0.6172541207661515,
    0.3362594383801259,
    0.14314753340463007,
    0.5028072190713874,
    0.5543980698148139,
    0.06212499507727453,
    0.2764677035534054,
    0.17138695017084749,
    0.11228531524866972,
    0.5682342545610666,
)

NUMBER_TOKEN = "<number>"
HEX_TOKEN = "<hex>"
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?|[_A-Za-z][_A-Za-z0-9]*", re.UNICODE)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)*|\d*\.\d+)$")
_HEX = re.compile(r"^(?:0x)?[0-9a-f]{12,}$", re.IGNORECASE)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_CHOICE = re.compile(r"(?:^|\n)\s*(?:\(?[A-Ea-e1-5]\)|[A-Ea-e1-5][.])\s+", re.MULTILINE)
_CODE_LINE = re.compile(
    r"(?:^|\n)\s*(?:def |class |from |import |if |for |while |SELECT |function )",
    re.MULTILINE,
)
_MATH = re.compile(r"\\(?:frac|sum|int|sqrt|boxed)|\$\$|[=+*/^]", re.IGNORECASE)

STRUCTURAL_FEATURE_NAMES: Tuple[str, ...] = (
    "log_chars",
    "log_utf8_bytes",
    "log_words",
    "log_unique_words",
    "log_lines",
    "log_messages",
    "mean_word_length",
    "max_word_length_log",
    "mean_line_length_log",
    "max_line_length_log",
    "letter_fraction",
    "upper_fraction",
    "digit_fraction",
    "space_fraction",
    "punctuation_fraction",
    "symbol_fraction",
    "non_ascii_fraction",
    "hangul_fraction",
    "newline_fraction",
    "quote_fraction",
    "bracket_fraction",
    "operator_fraction",
    "choice_count_log",
    "code_line_count_log",
    "math_marker_count_log",
    "url_count_log",
    "question_count_log",
    "colon_count_log",
    "semicolon_count_log",
    "markdown_heading_count_log",
    "list_item_count_log",
    "number_token_fraction",
)

FAMILY_NAMES: Tuple[str, ...] = (
    "english_multiple_choice",
    "korean_multiple_choice",
    "korean_reasoning",
    "latex_math",
    "long_context",
    "other",
    "python_program",
    "rule_reasoning",
    "symbolic_math",
    "word_problem",
)


def _batch_risk_feature_names() -> Tuple[str, ...]:
    names = [
        "log_batch_size",
        "log_unique_content",
        "unique_content_fraction",
        "largest_family_fraction",
        "second_family_fraction",
        "family_concentration",
        "family_entropy",
        "family_total_variation",
    ]
    names.extend(f"family_fraction:{name}" for name in FAMILY_NAMES)
    for model_id in MODEL_IDS:
        names.extend(
            (
                f"cost_mean_total_ratio:{model_id}",
                f"cost_q90_total_ratio:{model_id}",
                f"item_cost_ratio_mean:{model_id}",
                f"item_cost_ratio_std:{model_id}",
                f"item_cost_ratio_q10:{model_id}",
                f"item_cost_ratio_q50:{model_id}",
                f"item_cost_ratio_q90:{model_id}",
                f"item_cost_ratio_q99:{model_id}",
                f"quality_mean:{model_id}",
                f"quality_std:{model_id}",
                f"quality_q10:{model_id}",
                f"quality_q50:{model_id}",
                f"quality_q90:{model_id}",
            )
        )
    for model_id in MODEL_IDS[1:]:
        names.extend(
            (
                f"uplift_mean:{model_id}",
                f"uplift_std:{model_id}",
                f"uplift_q10:{model_id}",
                f"uplift_q50:{model_id}",
                f"uplift_q90:{model_id}",
                f"density_mean:{model_id}",
                f"density_q10:{model_id}",
                f"density_q50:{model_id}",
                f"density_q90:{model_id}",
            )
        )
    for name in STRUCTURAL_FEATURE_NAMES:
        names.extend((f"structural_mean:{name}", f"structural_std:{name}"))
    return tuple(names)


BATCH_RISK_FEATURE_NAMES = _batch_risk_feature_names()

# feature, threshold, left, right, weighted leaf value, missing-go-left
TreeNode = Tuple[int, float, int, int, float, bool]
Tree = Tuple[TreeNode, ...]


@dataclass(frozen=True)
class TreeHead:
    base: float
    transform: str
    trees: Tuple[Tree, ...]


@dataclass(frozen=True)
class FamilyCalibration:
    family_names: Tuple[str, ...]
    reference_proportions: Tuple[float, ...]
    mean_scales: Tuple[Tuple[float, ...], ...]
    q90_scales: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class TierConfig:
    base_fraction: float
    composition_penalty: float
    risk_reserve: float
    ax31_tail_weight: float
    k1_tail_weight: float


@dataclass(frozen=True)
class FiniteSampleGates:
    min_content_groups: int
    balanced_k1_min_groups: int
    premium_k1_min_groups: int
    premium_k1_max_tv: float


@dataclass(frozen=True)
class DistributionalArtifact:
    policy_id: str
    policy_digest: str
    vocabulary: Tuple[str, ...]
    quality_heads: Mapping[str, TreeHead]
    cost_mean_heads: Mapping[str, TreeHead]
    cost_q50_heads: Mapping[str, TreeHead]
    cost_q90_heads: Mapping[str, TreeHead]
    family_calibration: FamilyCalibration
    risk_heads: Mapping[str, TreeHead]
    tier_config: Mapping[str, TierConfig]
    gates: FiniteSampleGates
    training_summary: Mapping[str, Any]
    certification_summary: Mapping[str, Any]
    experiment: Mapping[str, Any]


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProtocolError(f"{label} must be a positive integer")
    return value


def _tree_head(value: Any, width: int, label: str) -> TreeHead:
    raw = _object(value, label)
    if set(raw) != {"base", "transform", "trees"}:
        raise ProtocolError(f"{label} fields are invalid")
    transform = raw["transform"]
    if transform not in {"identity", "clip01", "positive", "expm1_nonnegative"}:
        raise ProtocolError(f"{label} transform is invalid")
    raw_trees = raw["trees"]
    if not isinstance(raw_trees, list) or not raw_trees:
        raise ProtocolError(f"{label} trees are invalid")
    trees = []
    for tree_index, raw_tree in enumerate(raw_trees):
        if not isinstance(raw_tree, list) or not raw_tree:
            raise ProtocolError(f"{label}.trees[{tree_index}] is invalid")
        nodes = []
        for node_index, raw_node in enumerate(raw_tree):
            if not isinstance(raw_node, list) or len(raw_node) != 6:
                raise ProtocolError(f"{label} tree node is invalid")
            feature, threshold, left, right, leaf_value, missing_left = raw_node
            if (
                isinstance(feature, bool)
                or not isinstance(feature, int)
                or feature < -1
                or feature >= width
            ):
                raise ProtocolError(f"{label} tree feature is invalid")
            if not isinstance(left, int) or not isinstance(right, int):
                raise ProtocolError(f"{label} tree links are invalid")
            if feature == -1:
                if left != -1 or right != -1:
                    raise ProtocolError(f"{label} leaf links are invalid")
            elif not (0 <= left < len(raw_tree) and 0 <= right < len(raw_tree)):
                raise ProtocolError(f"{label} tree links are out of range")
            if not isinstance(missing_left, bool):
                raise ProtocolError(f"{label} missing direction is invalid")
            nodes.append(
                (
                    feature,
                    _number(threshold, f"{label}.threshold"),
                    left,
                    right,
                    _number(leaf_value, f"{label}.leaf"),
                    missing_left,
                )
            )
        trees.append(tuple(nodes))
    return TreeHead(
        _number(raw["base"], f"{label}.base"), transform, tuple(trees)
    )


def _model_heads(
    value: Any, width: int, label: str, expected_transform: str
) -> Mapping[str, TreeHead]:
    raw = _object(value, label)
    if set(raw) != set(MODEL_IDS):
        raise ProtocolError(f"{label} model set is invalid")
    heads = {
        model_id: _tree_head(raw[model_id], width, f"{label}.{model_id}")
        for model_id in MODEL_IDS
    }
    if any(head.transform != expected_transform for head in heads.values()):
        raise ProtocolError(f"{label} transforms are invalid")
    return heads


def _matrix(value: Any, rows: int, columns: int, label: str) -> Tuple[Tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ProtocolError(f"{label} rows are invalid")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ProtocolError(f"{label} columns are invalid")
        parsed = tuple(_number(item, label) for item in row)
        if any(item <= 0.0 for item in parsed):
            raise ProtocolError(f"{label} values must be positive")
        result.append(parsed)
    return tuple(result)


def load_artifact_mapping(value: Any) -> DistributionalArtifact:
    root = _object(value, "distributional artifact")
    expected = {
        "artifact_type",
        "schema_version",
        "feature_contract",
        "feature_version",
        "model_ids",
        "policy_id",
        "policy_sha256",
        "structural_feature_names",
        "batch_risk_feature_names",
        "vocabulary",
        "quality_heads",
        "cost_mean_heads",
        "cost_q50_heads",
        "cost_q90_heads",
        "family_calibration",
        "risk_heads",
        "tier_config",
        "finite_sample_gates",
        "training_summary",
        "certification_summary",
        "experiment",
    }
    if set(root) != expected:
        raise ProtocolError("distributional artifact fields are invalid")
    if (
        root["artifact_type"] != ARTIFACT_TYPE
        or root["schema_version"] != SCHEMA_VERSION
        or root["feature_contract"] != FEATURE_CONTRACT
        or root["feature_version"] != FEATURE_VERSION
    ):
        raise ProtocolError("unsupported distributional artifact")
    if root["model_ids"] != list(MODEL_IDS):
        raise ProtocolError("distributional model contract changed")
    if root["structural_feature_names"] != list(STRUCTURAL_FEATURE_NAMES):
        raise ProtocolError("distributional structural feature contract changed")
    if root["batch_risk_feature_names"] != list(BATCH_RISK_FEATURE_NAMES):
        raise ProtocolError("distributional batch-risk feature contract changed")
    vocabulary = root["vocabulary"]
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != VOCABULARY_SIZE
        or any(not isinstance(term, str) for term in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ProtocolError("distributional vocabulary is invalid")
    item_width = len(STRUCTURAL_FEATURE_NAMES) + len(vocabulary)
    quality = _model_heads(root["quality_heads"], item_width, "quality_heads", "clip01")
    mean = _model_heads(root["cost_mean_heads"], item_width, "cost_mean_heads", "positive")
    q50 = _model_heads(
        root["cost_q50_heads"], item_width, "cost_q50_heads", "expm1_nonnegative"
    )
    q90 = _model_heads(
        root["cost_q90_heads"], item_width, "cost_q90_heads", "expm1_nonnegative"
    )

    calibration_raw = _object(root["family_calibration"], "family_calibration")
    if set(calibration_raw) != {
        "family_names",
        "reference_proportions",
        "mean_scales",
        "q90_scales",
    } or calibration_raw["family_names"] != list(FAMILY_NAMES):
        raise ProtocolError("distributional family calibration contract changed")
    reference_raw = calibration_raw["reference_proportions"]
    if not isinstance(reference_raw, list) or len(reference_raw) != len(FAMILY_NAMES):
        raise ProtocolError("distributional reference composition is invalid")
    reference = tuple(_number(item, "reference_proportions") for item in reference_raw)
    if any(item <= 0.0 for item in reference) or abs(math.fsum(reference) - 1.0) > 1e-9:
        raise ProtocolError("distributional reference composition is invalid")
    calibration = FamilyCalibration(
        FAMILY_NAMES,
        reference,
        _matrix(calibration_raw["mean_scales"], len(FAMILY_NAMES), 3, "mean_scales"),
        _matrix(calibration_raw["q90_scales"], len(FAMILY_NAMES), 3, "q90_scales"),
    )
    risk_raw = _object(root["risk_heads"], "risk_heads")
    if set(risk_raw) != {"fast", "balanced"}:
        raise ProtocolError("distributional risk head set is invalid")
    risk = {
        tier: _tree_head(risk_raw[tier], len(BATCH_RISK_FEATURE_NAMES), f"risk_heads.{tier}")
        for tier in ("fast", "balanced")
    }
    if any(head.transform != "identity" for head in risk.values()):
        raise ProtocolError("distributional risk head transforms are invalid")

    config_raw = _object(root["tier_config"], "tier_config")
    if set(config_raw) != set(TIERS):
        raise ProtocolError("distributional tier config is incomplete")
    configs: Dict[str, TierConfig] = {}
    for tier in TIERS:
        row = _object(config_raw[tier], f"tier_config.{tier}")
        fields = {
            "base_fraction",
            "composition_penalty",
            "risk_reserve",
            "ax31_tail_weight",
            "k1_tail_weight",
        }
        if set(row) != fields:
            raise ProtocolError(f"tier_config.{tier} fields are invalid")
        config = TierConfig(
            *(
                _number(row[name], f"tier_config.{tier}.{name}")
                for name in (
                    "base_fraction",
                    "composition_penalty",
                    "risk_reserve",
                    "ax31_tail_weight",
                    "k1_tail_weight",
                )
            )
        )
        if not 0.0 < config.base_fraction <= 1.0 or min(
            config.composition_penalty,
            config.risk_reserve,
            config.ax31_tail_weight,
            config.k1_tail_weight,
        ) < 0.0:
            raise ProtocolError(f"tier_config.{tier} values are invalid")
        configs[tier] = config

    gates_raw = _object(root["finite_sample_gates"], "finite_sample_gates")
    if set(gates_raw) != {
        "min_content_groups",
        "balanced_k1_min_groups",
        "premium_k1_min_groups",
        "premium_k1_max_tv",
    }:
        raise ProtocolError("distributional finite-sample gates are invalid")
    gates = FiniteSampleGates(
        _positive_int(gates_raw["min_content_groups"], "min_content_groups"),
        _positive_int(
            gates_raw["balanced_k1_min_groups"], "balanced_k1_min_groups"
        ),
        _positive_int(gates_raw["premium_k1_min_groups"], "premium_k1_min_groups"),
        _number(gates_raw["premium_k1_max_tv"], "premium_k1_max_tv"),
    )
    if not 0.0 <= gates.premium_k1_max_tv <= 1.0:
        raise ProtocolError("premium_k1_max_tv is invalid")
    policy_id = root["policy_id"]
    policy_digest = root["policy_sha256"]
    if not isinstance(policy_id, str) or not isinstance(policy_digest, str):
        raise ProtocolError("distributional policy metadata is invalid")
    return DistributionalArtifact(
        policy_id,
        policy_digest,
        tuple(vocabulary),
        quality,
        mean,
        q50,
        q90,
        calibration,
        risk,
        configs,
        gates,
        dict(_object(root["training_summary"], "training_summary")),
        dict(_object(root["certification_summary"], "certification_summary")),
        dict(_object(root["experiment"], "experiment")),
    )


def load_bundled_artifact() -> DistributionalArtifact:
    text = (
        resources.files("ossp_router.resources")
        .joinpath(ARTIFACT_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return load_artifact_mapping(json.loads(text))


def load_artifact_file(path: Path) -> DistributionalArtifact:
    return load_artifact_mapping(load_json(path))


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _canonical_token(token: str) -> str:
    folded = token.casefold()
    if _NUMBER.fullmatch(folded):
        return NUMBER_TOKEN
    if _HEX.fullmatch(folded):
        return HEX_TOKEN
    return folded


def word_tokens(text: str) -> Tuple[str, ...]:
    bounded = text if len(text) <= 24_000 else text[:16_000] + text[-8_000:]
    return tuple(_canonical_token(match.group(0)) for match in _WORD.finditer(bounded))


def lexical_terms(text: str) -> frozenset[str]:
    tokens = word_tokens(text)
    terms = set()
    previous: Optional[str] = None
    for token in tokens:
        terms.add(f"w:{token}")
        if len(token) >= 5:
            terms.add(f"p:{token[:4]}")
            terms.add(f"s:{token[-4:]}")
        if previous is not None:
            terms.add(f"b:{previous}\x1f{token}")
        previous = token
    return frozenset(terms)


def structural_features(episode: Episode) -> Tuple[float, ...]:
    text = episode_text(episode)
    characters = len(text)
    safe_characters = max(1, characters)
    tokens = word_tokens(text)
    token_lengths = [len(token) for token in tokens]
    lines = text.splitlines() or [text]
    line_lengths = [len(line) for line in lines]
    category = Counter(unicodedata.category(character)[0] for character in text)
    letters = category["L"]
    punctuation = category["P"]
    symbols = category["S"]
    digits = sum(character.isdecimal() for character in text)
    spaces = sum(character.isspace() for character in text)
    non_ascii = sum(ord(character) >= 128 for character in text)
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in text)
    upper = sum(character.isupper() for character in text)
    quotes = sum(character in "'\"`“”‘’" for character in text)
    brackets = sum(character in "()[]{}<>" for character in text)
    operators = sum(character in "+-*/=^%|&!" for character in text)
    number_tokens = sum(token == NUMBER_TOKEN for token in tokens)
    message_count = len(episode.messages) if episode.messages is not None else 1
    result = (
        math.log1p(characters),
        math.log1p(len(text.encode("utf-8"))),
        math.log1p(len(tokens)),
        math.log1p(len(set(tokens))),
        math.log1p(len(lines)),
        math.log1p(message_count),
        math.fsum(token_lengths) / max(1, len(token_lengths)),
        math.log1p(max(token_lengths, default=0)),
        math.log1p(math.fsum(line_lengths) / max(1, len(line_lengths))),
        math.log1p(max(line_lengths, default=0)),
        letters / safe_characters,
        upper / safe_characters,
        digits / safe_characters,
        spaces / safe_characters,
        punctuation / safe_characters,
        symbols / safe_characters,
        non_ascii / safe_characters,
        hangul / safe_characters,
        text.count("\n") / safe_characters,
        quotes / safe_characters,
        brackets / safe_characters,
        operators / safe_characters,
        math.log1p(len(_CHOICE.findall(text))),
        math.log1p(len(_CODE_LINE.findall(text))),
        math.log1p(len(_MATH.findall(text))),
        math.log1p(len(_URL.findall(text))),
        math.log1p(text.count("?")),
        math.log1p(text.count(":")),
        math.log1p(text.count(";")),
        math.log1p(sum(line.lstrip().startswith("#") for line in lines)),
        math.log1p(
            sum(
                line.lstrip().startswith(("- ", "* ", "+ "))
                or bool(re.match(r"\s*\d+[.)]\s", line))
                for line in lines
            )
        ),
        number_tokens / max(1, len(tokens)),
    )
    return tuple(_float32(value) for value in result)


def feature_row(episode: Episode, vocabulary: Mapping[str, int]) -> Tuple[float, ...]:
    structural = structural_features(episode)
    return structural + _lexical_feature_row(episode_text(episode), vocabulary)


def _lexical_feature_row(
    text: str, vocabulary: Mapping[str, int]
) -> Tuple[float, ...]:
    lexical = [0.0] * len(vocabulary)
    for term in lexical_terms(text):
        index = vocabulary.get(term)
        if index is not None:
            lexical[index] = 1.0
    return tuple(lexical)


def _predict_tree(tree: Tree, row: Sequence[float]) -> float:
    index = 0
    while True:
        feature, threshold, left, right, leaf, missing_left = tree[index]
        if feature < 0:
            return leaf
        value = row[feature]
        if math.isnan(value):
            index = left if missing_left else right
        else:
            index = left if value <= threshold else right


def predict_head(head: TreeHead, row: Sequence[float]) -> float:
    value = head.base + math.fsum(_predict_tree(tree, row) for tree in head.trees)
    if head.transform == "clip01":
        return min(1.0, max(0.0, value))
    if head.transform == "positive":
        return max(sys.float_info.min, value)
    if head.transform == "expm1_nonnegative":
        return math.expm1(max(0.0, min(value, 50.0)))
    return value


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(max(0.0, variance))


def _batch_features(
    quality: Sequence[Sequence[float]],
    mean_cost: Sequence[Sequence[float]],
    q90_cost: Sequence[Sequence[float]],
    structural: Sequence[Sequence[float]],
    families: Sequence[str],
    tie_keys: Sequence[str],
    calibration: FamilyCalibration,
) -> Tuple[float, ...]:
    count = len(quality)
    family_index = {name: index for index, name in enumerate(FAMILY_NAMES)}
    family_counts = [0] * len(FAMILY_NAMES)
    for family in families:
        family_counts[family_index[family]] += 1
    proportions = [amount / count for amount in family_counts]
    ranked = sorted(proportions)
    positive = [value for value in proportions if value > 0.0]
    unique_count = len(set(tie_keys))
    row = [
        math.log1p(count),
        math.log1p(unique_count),
        unique_count / count,
        ranked[-1],
        ranked[-2],
        math.fsum(value * value for value in proportions),
        -math.fsum(value * math.log(value) for value in positive),
        0.5
        * math.fsum(
            abs(value - reference)
            for value, reference in zip(
                proportions, calibration.reference_proportions
            )
        ),
    ]
    row.extend(proportions)
    generic = [list(values) for values in mean_cost]
    for values, upper in zip(generic, q90_cost):
        values[2] = max(values[2], upper[2])
    light = [max(values[0], sys.float_info.min) for values in mean_cost]
    light_total = math.fsum(light)
    for model_index in range(3):
        ratios = [values[model_index] / base for values, base in zip(generic, light)]
        ratio_mean, ratio_std = _mean_std(ratios)
        scores = [values[model_index] for values in quality]
        score_mean, score_std = _mean_std(scores)
        row.extend(
            (
                math.fsum(values[model_index] for values in mean_cost) / light_total,
                math.fsum(values[model_index] for values in q90_cost) / light_total,
                ratio_mean,
                ratio_std,
                _quantile(ratios, 0.10),
                _quantile(ratios, 0.50),
                _quantile(ratios, 0.90),
                _quantile(ratios, 0.99),
                score_mean,
                score_std,
                _quantile(scores, 0.10),
                _quantile(scores, 0.50),
                _quantile(scores, 0.90),
            )
        )
    for model_index in (1, 2):
        uplift = [values[model_index] - values[0] for values in quality]
        density = [
            gain
            / max(cost[model_index] - cost[0], 1e-12)
            for gain, cost in zip(uplift, generic)
        ]
        uplift_mean, uplift_std = _mean_std(uplift)
        row.extend(
            (
                uplift_mean,
                uplift_std,
                _quantile(uplift, 0.10),
                _quantile(uplift, 0.50),
                _quantile(uplift, 0.90),
                math.fsum(density) / len(density),
                _quantile(density, 0.10),
                _quantile(density, 0.50),
                _quantile(density, 0.90),
            )
        )
    for column in range(len(STRUCTURAL_FEATURE_NAMES)):
        values = [features[column] for features in structural]
        row.extend(_mean_std(values))
    if len(row) != len(BATCH_RISK_FEATURE_NAMES):
        raise RuntimeError("distributional batch feature width drifted")
    return tuple(row)


def _concave_path(
    score: Sequence[float], cost: Sequence[float], allow_k1: bool
) -> Tuple[int, ...]:
    candidates = [(0.0, 0.0, 0)] + [
        (max(0.0, cost[action] - cost[0]), score[action] - score[0], action)
        for action in range(1, 3 if allow_k1 else 2)
    ]
    candidates.sort(key=lambda point: (point[0], -point[1], point[2]))
    points = [(0.0, 0.0, 0)]
    for increment, gain, action in candidates:
        if action == 0 or gain <= points[-1][1] + 1e-15:
            continue
        if increment <= points[-1][0] + 1e-15:
            if points[-1][2] == 0:
                points.append((points[-1][0], gain, action))
            elif gain > points[-1][1] + 1e-15:
                points[-1] = (points[-1][0], gain, action)
            continue
        points.append((increment, gain, action))
    hull = []
    for point in points:
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            left_slope = (middle[1] - left[1]) / max(middle[0] - left[0], 1e-15)
            right_slope = (point[1] - middle[1]) / max(point[0] - middle[0], 1e-15)
            if right_slope <= left_slope + 1e-15:
                break
            hull.pop()
        hull.append(point)
    return tuple(point[2] for point in hull)


def _allocate(
    quality: Sequence[Sequence[float]],
    charges: Sequence[Sequence[float]],
    light_credit: Sequence[float],
    tie_keys: Sequence[str],
    budget_multiplier: float,
    target_fraction: float,
    allow_k1: bool,
) -> Tuple[int, ...]:
    grouped: Dict[str, list[int]] = {}
    for index, key in enumerate(tie_keys):
        grouped.setdefault(key, []).append(index)
    names = sorted(grouped)
    rows = [grouped[name] for name in names]
    multiplicity = [len(indexes) for indexes in rows]
    representatives = [indexes[0] for indexes in rows]
    score = [
        [value * amount for value in quality[index]]
        for index, amount in zip(representatives, multiplicity)
    ]
    cost = [
        [value * amount for value in charges[index]]
        for index, amount in zip(representatives, multiplicity)
    ]
    credit = [
        light_credit[index] * amount
        for index, amount in zip(representatives, multiplicity)
    ]
    selected = [0] * len(rows)
    current_total = math.fsum(values[0] for values in cost)
    cap = budget_multiplier * target_fraction * math.fsum(credit)
    if current_total >= cap:
        return tuple(0 for _ in quality)
    paths = [_concave_path(q, c, allow_k1) for q, c in zip(score, cost)]
    queue = []

    def offer(row: int, step: int) -> None:
        path = paths[row]
        if step + 1 >= len(path):
            return
        source, target = path[step], path[step + 1]
        gain = score[row][target] - score[row][source]
        increment = max(0.0, cost[row][target] - cost[row][source])
        if gain <= 0.0:
            return
        density = gain / max(increment, 1e-15)
        heapq.heappush(
            queue,
            (-density, -gain, increment, names[row], row, step, source, target),
        )

    for row in range(len(rows)):
        offer(row, 0)
    while queue:
        _density, _gain, increment, _key, row, step, source, target = heapq.heappop(queue)
        if selected[row] != source:
            continue
        if current_total + increment > cap + 1e-12:
            break
        selected[row] = target
        current_total += increment
        offer(row, step + 1)
    result = [0] * len(quality)
    for model_index, indexes in zip(selected, rows):
        for index in indexes:
            result[index] = model_index
    return tuple(result)


def _content_key(episode: Episode) -> str:
    return hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest()


def stable_surface_text(text: str) -> str:
    """Normalize encoding and whitespace while preserving choice notation."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))


def stabilize_episode(episode: Episode) -> Episode:
    """Rewrite prompt text without changing the episode ID or roles."""

    if episode.prompt is not None:
        return Episode(
            episode_id=episode.episode_id,
            prompt=stable_surface_text(episode.prompt),
        )
    assert episode.messages is not None
    return Episode(
        episode_id=episode.episode_id,
        messages=tuple(
            Message(message.role, stable_surface_text(message.content))
            for message in episode.messages
        ),
    )


def small_batch_route_enabled(unique_count: int) -> bool:
    """Return whether the two-sided small-batch route owns this batch."""

    return 0 < int(unique_count) < SMALL_BATCH_UNIQUE_CUTOFF


def _apply_family_scales(
    raw_mean: Sequence[Sequence[float]],
    raw_upper: Sequence[Sequence[float]],
    family_ids: Sequence[int],
    mean_scales: Sequence[Sequence[float]],
    upper_scales: Sequence[Sequence[float]],
) -> Tuple[list[Tuple[float, ...]], list[Tuple[float, ...]]]:
    mean: list[Tuple[float, ...]] = []
    q90: list[Tuple[float, ...]] = []
    for expected, upper, family_id in zip(raw_mean, raw_upper, family_ids):
        mean.append(
            tuple(
                expected[index] * mean_scales[family_id][index] for index in range(3)
            )
        )
        q90.append(
            tuple(
                upper[index] * upper_scales[family_id][index] for index in range(3)
            )
        )
    return mean, q90


def _tier_charges(
    mean: Sequence[Sequence[float]],
    q90: Sequence[Sequence[float]],
    config: TierConfig,
) -> list[Tuple[float, float, float]]:
    charges: list[Tuple[float, float, float]] = []
    for expected, upper in zip(mean, q90):
        gap_ax = max(0.0, upper[1] - expected[1])
        gap_k1 = max(0.0, upper[2] - expected[2])
        charges.append(
            (
                expected[0],
                max(expected[0], expected[1] + config.ax31_tail_weight * gap_ax),
                max(expected[0], expected[2] + config.k1_tail_weight * gap_k1),
            )
        )
    return charges


def _small_batch_target_fraction(
    config: TierConfig,
    budget_multiplier: float,
    unique_count: int,
    family_tv: float,
) -> float:
    lower = 1.0 / float(budget_multiplier)
    fallback = min(
        config.base_fraction,
        max(
            lower,
            config.base_fraction * (1.0 - config.composition_penalty * family_tv),
        ),
    )
    share = min(
        1.0,
        max(0.0, unique_count / float(SMALL_BATCH_UNIQUE_CUTOFF)),
    ) ** float(SMALL_BATCH_POWER)
    return float(lower + (fallback - lower) * share)


def _small_batch_surfaces(
    mean: Sequence[Sequence[float]],
    q90: Sequence[Sequence[float]],
    family_ids: Sequence[int],
    config: TierConfig,
) -> Tuple[list[Tuple[float, float, float]], list[float]]:
    charges = _tier_charges(mean, q90, config)
    guarded: list[Tuple[float, float, float]] = []
    light: list[float] = []
    for charge, family_id, expected in zip(charges, family_ids, mean):
        lower = max(
            expected[0] * SMALL_BATCH_LIGHT_LOWER_SCALES[family_id],
            sys.float_info.min,
        )
        light.append(lower)
        guarded.append((lower, max(charge[1], lower), max(charge[2], lower)))
    return guarded, light


def _content_tuple(episode: Episode) -> Tuple[object, ...]:
    if episode.prompt is not None:
        return ("prompt", episode.prompt)
    assert episode.messages is not None
    return ("messages", episode.messages)


def _canonical_batch(inputs: InputBatch) -> InputBatch:
    return InputBatch(
        inputs.schema_version,
        inputs.challenge_id,
        inputs.split,
        tuple(
            sorted(
                inputs.episodes,
                key=lambda episode: (_content_key(episode), episode_text(episode)),
            )
        ),
    )


def learned_path_allowed(inputs: InputBatch) -> bool:
    if len(inputs.episodes) > MAX_LEARNED_EPISODES:
        return False
    characters = 0
    messages = 0
    for episode in inputs.episodes:
        characters += len(episode_text(episode))
        messages += 1 if episode.prompt is not None else len(episode.messages or ())
        if characters > MAX_LEARNED_CHARACTERS or messages > MAX_LEARNED_MESSAGES:
            return False
    return True


def _route_canonical(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: DistributionalArtifact,
    tier: str,
) -> Submission:
    vocabulary = {term: index for index, term in enumerate(artifact.vocabulary)}
    family_lookup = {name: index for index, name in enumerate(FAMILY_NAMES)}
    cache: Dict[
        Tuple[object, ...],
        Tuple[
            Tuple[float, ...],
            Tuple[float, ...],
            Tuple[float, ...],
            Tuple[float, ...],
            str,
            int,
            str,
        ],
    ] = {}
    rows = []
    for episode in inputs.episodes:
        content = _content_tuple(episode)
        predicted = cache.get(content)
        if predicted is None:
            structural = structural_features(episode)
            features = structural + _lexical_feature_row(
                episode_text(episode), vocabulary
            )
            quality = tuple(
                predict_head(artifact.quality_heads[model_id], features)
                for model_id in MODEL_IDS
            )
            raw_mean = tuple(
                predict_head(artifact.cost_mean_heads[model_id], features)
                for model_id in MODEL_IDS
            )
            raw_q50 = tuple(
                predict_head(artifact.cost_q50_heads[model_id], features)
                for model_id in MODEL_IDS
            )
            raw_q90 = tuple(
                predict_head(artifact.cost_q90_heads[model_id], features)
                for model_id in MODEL_IDS
            )
            raw_upper = tuple(
                max(q50_value, q90_value)
                for q50_value, q90_value in zip(raw_q50, raw_q90)
            )
            family = prompt_family(episode)
            family_id = family_lookup[family]
            predicted = (
                structural,
                quality,
                raw_mean,
                raw_upper,
                family,
                family_id,
                _content_key(episode),
            )
            cache[content] = predicted
        rows.append(predicted)
    structural = [row[0] for row in rows]
    quality = [row[1] for row in rows]
    raw_mean = [row[2] for row in rows]
    raw_upper = [row[3] for row in rows]
    families = [row[4] for row in rows]
    family_ids = [row[5] for row in rows]
    tie_keys = [row[6] for row in rows]
    unique_count = len(set(tie_keys))
    family_counts = Counter(families)
    proportions = [family_counts[name] / len(rows) for name in FAMILY_NAMES]
    tv = 0.5 * math.fsum(
        abs(value - reference)
        for value, reference in zip(
            proportions, artifact.family_calibration.reference_proportions
        )
    )
    budget_multiplier = float(policy.tiers[tier].budget_multiplier)
    config = artifact.tier_config[tier]
    fallback = min(
        config.base_fraction,
        max(
            1.0 / budget_multiplier,
            config.base_fraction * (1.0 - config.composition_penalty * tv),
        ),
    )
    if small_batch_route_enabled(unique_count):
        mean, q90 = _apply_family_scales(
            raw_mean,
            raw_upper,
            family_ids,
            SMALL_BATCH_MEAN_SCALES,
            SMALL_BATCH_UPPER_SCALES,
        )
        charges, light = _small_batch_surfaces(mean, q90, family_ids, config)
        selected = _allocate(
            quality,
            charges,
            light,
            tie_keys,
            budget_multiplier,
            _small_batch_target_fraction(
                config, budget_multiplier, unique_count, tv
            ),
            False,
        )
    elif unique_count < artifact.gates.min_content_groups:
        selected = tuple(0 for _ in rows)
    else:
        mean, q90 = _apply_family_scales(
            raw_mean,
            raw_upper,
            family_ids,
            artifact.family_calibration.mean_scales,
            artifact.family_calibration.q90_scales,
        )
        if tier == "premium":
            allow_k1 = (
                unique_count >= artifact.gates.premium_k1_min_groups
                and tv <= artifact.gates.premium_k1_max_tv
            )
            target = fallback if allow_k1 else config.base_fraction
        else:
            risk_features = _batch_features(
                quality,
                mean,
                q90,
                structural,
                families,
                tie_keys,
                artifact.family_calibration,
            )
            risk_fraction = predict_head(artifact.risk_heads[tier], risk_features)
            target = min(
                config.base_fraction,
                max(fallback, risk_fraction - config.risk_reserve),
            )
            allow_k1 = (
                tier == "balanced"
                and unique_count >= artifact.gates.balanced_k1_min_groups
            )
        charges = _tier_charges(mean, q90, config)
        selected = _allocate(
            quality,
            charges,
            [values[0] for values in mean],
            tie_keys,
            budget_multiplier,
            target,
            allow_k1,
        )
    submission = Submission(
        inputs.schema_version,
        inputs.challenge_id,
        policy.policy_id,
        inputs.split,
        tier,
        tuple(
            Decision(episode.episode_id, MODEL_IDS[model_index])
            for episode, model_index in zip(inputs.episodes, selected)
        ),
    )
    return parse_submission(submission_to_dict(submission))


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: DistributionalArtifact,
    tier: str,
) -> Submission:
    if tier not in TIERS:
        raise ProtocolError(f"unknown tier: {tier}")
    if (
        artifact.policy_id != policy.policy_id
        or artifact.policy_digest != policy_sha256(policy)
    ):
        raise ProtocolError("distributional artifact and policy do not match")
    if not learned_path_allowed(inputs):
        return make_heuristic_submission(inputs, policy, tier, strategy="always-light")
    stabilized = InputBatch(
        inputs.schema_version,
        inputs.challenge_id,
        inputs.split,
        tuple(stabilize_episode(episode) for episode in inputs.episodes),
    )
    canonical = _canonical_batch(stabilized)
    routed = _route_canonical(canonical, policy, artifact, tier)
    by_id = {decision.episode_id: decision.model_id for decision in routed.decisions}
    restored = Submission(
        inputs.schema_version,
        inputs.challenge_id,
        policy.policy_id,
        inputs.split,
        tier,
        tuple(
            Decision(episode.episode_id, by_id[episode.episode_id])
            for episode in inputs.episodes
        ),
    )
    return parse_submission(submission_to_dict(restored))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="router-run")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--artifact", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        artifact = load_artifact_file(args.artifact) if args.artifact else load_bundled_artifact()
        submission = make_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"OK: generated {args.tier} distributional submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
