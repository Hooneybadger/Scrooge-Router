# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""Deployable the modeling foundation feature extractor (Python standard library only).

Training code and the eventual container router share this module. The
vector is deterministic: no ``set`` iteration on value-producing paths,
no ``dict`` iteration-order dependence, no reliance on ``hash()``, and
no float reductions whose result depends on accumulation order.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence, Tuple


FEATURE_VERSION = "modeling.v1"
ALLOWED_HASH_BINS: Tuple[int, ...] = (256, 512)

# FNV-1a 32-bit. Sign is taken from bit 31 (a dedicated sign bit).
_FNV1A32_OFFSET = 2_166_136_261
_FNV1A32_PRIME = 16_777_619
_UINT32_MASK = 0xFFFFFFFF
_SIGN_BIT = 1 << 31

# Shared tokenizer: Latin words, Hangul runs, decimal runs, then any other
# non-whitespace symbol. Structural counts and signed hashes consume this
# exact token sequence (the feasibility ladder's runtime win came from sharing one pass).
_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_CHOICE = re.compile(r"(?:^|\n)\s*(?:[A-D][.)]|\([a-e]\))\s", re.IGNORECASE)
_WORD_PROBLEM = re.compile(
    r"\b(?:how many|how much|how long|how far|total|each|costs?|average|"
    r"percent|percentage|left over|altogether)\b",
    re.IGNORECASE,
)

# Ordered structural layout. Count-like entries are stored raw and as
# ``log1p``. Ratios use the documented denominator (never a running mean).
STRUCTURAL_FEATURE_NAMES: Tuple[str, ...] = (
    "char_count",
    "log1p_char_count",
    "ws_norm_char_count",
    "log1p_ws_norm_char_count",
    "token_count",
    "log1p_token_count",
    "mean_token_length",
    "max_token_length",
    "log1p_max_token_length",
    "digit_ratio",
    "uppercase_ratio",
    "punct_ratio",
    "newline_count",
    "log1p_newline_count",
    "hangul_ratio",
    "cjk_other_ratio",
    "latin_ratio",
    "latex_dollar_count",
    "log1p_latex_dollar_count",
    "latex_bracket_count",
    "log1p_latex_bracket_count",
    "latex_frac_count",
    "log1p_latex_frac_count",
    "latex_begin_count",
    "log1p_latex_begin_count",
    "latex_marker_count",
    "log1p_latex_marker_count",
    "has_latex",
    "code_def_count",
    "log1p_code_def_count",
    "code_assert_count",
    "log1p_code_assert_count",
    "code_return_count",
    "log1p_code_return_count",
    "code_import_count",
    "log1p_code_import_count",
    "code_fence_count",
    "log1p_code_fence_count",
    "code_paren_count",
    "log1p_code_paren_count",
    "code_marker_count",
    "log1p_code_marker_count",
    "has_code",
    "mcq_choice_count",
    "log1p_mcq_choice_count",
    "question_colon_count",
    "log1p_question_colon_count",
    "has_question_colon",
    "word_problem_count",
    "log1p_word_problem_count",
    "has_word_problem",
    "question_mark_count",
    "log1p_question_mark_count",
)


def _message_content(message: Any) -> str:
    if isinstance(message, Mapping):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def episode_text_of(episode_like: Any) -> str:
    """Canonical routing-time text. Mirrors ``heuristic.episode_text``.

    Accepts a ``protocol.Episode`` or a plain dict with ``prompt`` and/or
    ``messages``. Prompt wins when present. Messages are concatenated in
    order by ``content`` (not role) and joined with ``\\n``, which is the
    frozen-router / ``prompt_family`` contract. Role is ignored so family
    labels stay consistent with the frozen extractors.
    """

    if isinstance(episode_like, Mapping):
        prompt = episode_like.get("prompt")
        if prompt is not None:
            return prompt if isinstance(prompt, str) else str(prompt)
        messages = episode_like.get("messages") or ()
        return "\n".join(_message_content(message) for message in messages)
    prompt = getattr(episode_like, "prompt", None)
    if prompt is not None:
        return prompt if isinstance(prompt, str) else str(prompt)
    messages = getattr(episode_like, "messages", None) or ()
    return "\n".join(_message_content(message) for message in messages)


def _tokenize(text: str) -> Tuple[str, ...]:
    return tuple(_TOKEN.findall(text))


def _is_hangul(code: int) -> bool:
    return (
        0xAC00 <= code <= 0xD7A3
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xA960 <= code <= 0xA97F
        or 0xD7B0 <= code <= 0xD7FF
    )


def _is_cjk_other(code: int) -> bool:
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
    )


def _is_latin(code: int) -> bool:
    return (
        0x0041 <= code <= 0x005A
        or 0x0061 <= code <= 0x007A
        or 0x00C0 <= code <= 0x024F
    )


def _ratio(count: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(count) / float(denom)


def _count_and_log(count: int) -> Tuple[float, float]:
    value = float(count)
    return value, math.log1p(value)


def _structural_from(text: str, tokens: Sequence[str]) -> Tuple[float, ...]:
    char_count = len(text)
    nonspace = 0
    digit_count = 0
    uppercase_count = 0
    punct_count = 0
    hangul_count = 0
    cjk_other_count = 0
    latin_count = 0
    for character in text:
        if character.isspace():
            continue
        nonspace += 1
        code = ord(character)
        if character.isdigit():
            digit_count += 1
        if character.isupper():
            uppercase_count += 1
        if unicodedata.category(character).startswith("P"):
            punct_count += 1
        if _is_hangul(code):
            hangul_count += 1
        elif _is_cjk_other(code):
            cjk_other_count += 1
        elif _is_latin(code):
            latin_count += 1

    ws_norm = len(" ".join(text.split()))
    token_count = len(tokens)
    if token_count:
        lengths = [len(token) for token in tokens]
        length_sum = 0
        max_token = 0
        for length in lengths:
            length_sum += length
            if length > max_token:
                max_token = length
        mean_token = float(length_sum) / float(token_count)
    else:
        max_token = 0
        mean_token = 0.0

    newline_count = text.count("\n")
    latex_dollar = text.count("$")
    latex_bracket = text.count("\\[")
    latex_frac = text.count("\\frac")
    latex_begin = text.count("\\begin")
    latex_total = latex_dollar + latex_bracket + latex_frac + latex_begin
    code_def = text.count("def ")
    code_assert = text.count("assert ")
    code_return = text.count("return ")
    code_import = text.count("import ")
    code_fence = text.count("```")
    code_paren = text.count("()")
    code_total = (
        code_def + code_assert + code_return + code_import + code_fence + code_paren
    )
    mcq_choice = len(_CHOICE.findall(text))
    question_colon = text.casefold().count("question:")
    word_problem = len(_WORD_PROBLEM.findall(text))
    question_mark = text.count("?")

    char_count_f, log_char = _count_and_log(char_count)
    ws_norm_f, log_ws = _count_and_log(ws_norm)
    token_count_f, log_token = _count_and_log(token_count)
    max_token_f = float(max_token)
    newline_f, log_newline = _count_and_log(newline_count)
    dollar_f, log_dollar = _count_and_log(latex_dollar)
    bracket_f, log_bracket = _count_and_log(latex_bracket)
    frac_f, log_frac = _count_and_log(latex_frac)
    begin_f, log_begin = _count_and_log(latex_begin)
    latex_f, log_latex = _count_and_log(latex_total)
    def_f, log_def = _count_and_log(code_def)
    assert_f, log_assert = _count_and_log(code_assert)
    return_f, log_return = _count_and_log(code_return)
    import_f, log_import = _count_and_log(code_import)
    fence_f, log_fence = _count_and_log(code_fence)
    paren_f, log_paren = _count_and_log(code_paren)
    code_f, log_code = _count_and_log(code_total)
    mcq_f, log_mcq = _count_and_log(mcq_choice)
    qcolon_f, log_qcolon = _count_and_log(question_colon)
    word_f, log_word = _count_and_log(word_problem)
    qmark_f, log_qmark = _count_and_log(question_mark)

    vector = (
        char_count_f,
        log_char,
        ws_norm_f,
        log_ws,
        token_count_f,
        log_token,
        mean_token,
        max_token_f,
        math.log1p(max_token_f),
        _ratio(digit_count, char_count),
        _ratio(uppercase_count, char_count),
        _ratio(punct_count, char_count),
        newline_f,
        log_newline,
        _ratio(hangul_count, nonspace),
        _ratio(cjk_other_count, nonspace),
        _ratio(latin_count, nonspace),
        dollar_f,
        log_dollar,
        bracket_f,
        log_bracket,
        frac_f,
        log_frac,
        begin_f,
        log_begin,
        latex_f,
        log_latex,
        1.0 if latex_total else 0.0,
        def_f,
        log_def,
        assert_f,
        log_assert,
        return_f,
        log_return,
        import_f,
        log_import,
        fence_f,
        log_fence,
        paren_f,
        log_paren,
        code_f,
        log_code,
        1.0 if code_total else 0.0,
        mcq_f,
        log_mcq,
        qcolon_f,
        log_qcolon,
        1.0 if question_colon else 0.0,
        word_f,
        log_word,
        1.0 if word_problem else 0.0,
        qmark_f,
        log_qmark,
    )
    if len(vector) != len(STRUCTURAL_FEATURE_NAMES):
        raise RuntimeError(
            "structural feature width drifted: "
            f"{len(vector)} != {len(STRUCTURAL_FEATURE_NAMES)}"
        )
    return vector


def _require_bins(bins: int) -> int:
    value = int(bins)
    if value not in ALLOWED_HASH_BINS:
        raise ValueError(
            f"hash bins must be one of {ALLOWED_HASH_BINS}; got {bins!r} "
            "(the G-series closed list forbids 1024)"
        )
    return value


def _fnv1a32(value: str) -> int:
    digest = _FNV1A32_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV1A32_PRIME) & _UINT32_MASK
    return digest


def _hash_terms(tokens: Sequence[str]) -> Tuple[str, ...]:
    normalized = []
    for token in tokens:
        folded = token.casefold()
        normalized.append("<number>" if folded.isdecimal() else folded)
    unigrams = tuple(f"u:{token}" for token in normalized)
    bigrams = tuple(
        f"b:{left}\x1f{right}"
        for left, right in zip(normalized, normalized[1:])
    )
    return unigrams + bigrams


def _hash_from(tokens: Sequence[str], bins: int) -> dict[int, float]:
    width = _require_bins(bins)
    mask = width - 1
    counts = [0] * width
    for term in _hash_terms(tokens):
        digest = _fnv1a32(term)
        bucket = digest & mask
        counts[bucket] += -1 if digest & _SIGN_BIT else 1
    # Emit keys in increasing bucket order so iteration is sorted.
    hashed: dict[int, float] = {}
    for bucket, count in enumerate(counts):
        if count == 0:
            continue
        hashed[bucket] = math.copysign(math.log1p(abs(count)), count)
    return hashed


def structural_features(text: str) -> Tuple[float, ...]:
    """Fixed-length structural vector. See ``STRUCTURAL_FEATURE_NAMES``."""

    return _structural_from(text, _tokenize(text))


def hash_features(text: str, *, bins: int) -> dict[int, float]:
    """Sparse signed-FNV-1a 32-bit word unigram/bigram buckets.

    Each term is hashed with FNV-1a 32-bit. The low bits select the
    bucket; bit 31 is the sign. Signed counts are accumulated, then
    ``sign(count) * log1p(|count|)`` is stored. Only nonzero buckets are
    returned; keys are inserted in increasing order.
    """

    return _hash_from(_tokenize(text), bins)


def feature_row(
    text: str, *, bins: int
) -> Tuple[Tuple[float, ...], dict[int, float]]:
    """Shared one-pass tokenization for structural + hashed blocks."""

    tokens = _tokenize(text)
    return _structural_from(text, tokens), _hash_from(tokens, bins)


def feature_signature(bins: int) -> str:
    """SHA-256 pin over feature names, bin width, and ``FEATURE_VERSION``."""

    width = _require_bins(bins)
    payload = "\n".join(
        (FEATURE_VERSION, str(width), *STRUCTURAL_FEATURE_NAMES, "")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if len(structural_features("")) != len(STRUCTURAL_FEATURE_NAMES):
    raise RuntimeError("modeling structural feature width is inconsistent")
