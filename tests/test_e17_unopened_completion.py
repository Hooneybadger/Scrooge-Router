# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E17 unopened-completion contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e17-unopened-completion.v1.json"
PROTOCOL_SHA256 = "2800f4a912cf32b845e01a4d7f104f5c714ed3b6f56f2b5dbbabadad86f18db9"


class E17ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e17-unopened-completion-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertIn("e9-keep-e14", payload["arms"]["candidates"])
        self.assertIn("cond-fast-1.08-0.75", payload["arms"]["candidates"])
        self.assertIn("cond-top2-1.07-0.25", payload["arms"]["candidates"])
        self.assertIn("cond-top3-1.07-0.75", payload["arms"]["candidates"])
        self.assertIn("leftover-e10-0.73", payload["arms"]["candidates"])
        self.assertIn("unify-e9-fast-1.10", payload["arms"]["candidates"])
        self.assertEqual(payload["arms"]["knobs"]["cond-fast-1.08-0.75"]["fast_cap"], 1.08)
        self.assertEqual(payload["arms"]["knobs"]["leftover-e10-0.73"]["e10_threshold"], 0.73)
        self.assertTrue(payload["arms"]["knobs"]["e9-keep-e14"]["unify_premium"])
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e17_unopened_completion import (
                ProtocolError,
                protocol_sha256,
                verify_protocol,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        digest = protocol_sha256(payload)
        self.assertEqual(digest, PROTOCOL_SHA256)
        self.assertEqual(verify_protocol(payload, digest), digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, "0" * 64)
        tampered = json.loads(json.dumps(payload))
        tampered["arms"]["knobs"]["leftover-e10-0.73"]["e10_threshold"] = 0.74
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E17RuleTest(unittest.TestCase):
    def test_unopened_caps_match_named_rules(self) -> None:
        try:
            from research.lab.e17_unopened_completion import (
                ARM_KNOBS,
                CANDIDATE_ARMS,
                ProtocolError,
                arm_caps,
                arm_knobs,
                resolve_fast_cap,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")

        class _Replica:
            shipped_caps = {"fast": 1.11, "balanced": 1.45, "premium": 3.25}

        replica = _Replica()
        mixed = (
            ["korean_multiple_choice"] * 18
            + ["word_problem"] * 16
            + ["other"] * 16
            + ["python_program"] * 16
            + ["rule_reasoning"] * 16
            + ["long_context"] * 18
        )
        dominated = ["word_problem"] * 40
        pair = ["word_problem"] * 20 + ["english_multiple_choice"] * 20
        triple = (
            ["word_problem"] * 20
            + ["english_multiple_choice"] * 20
            + ["rule_reasoning"] * 20
        )
        self.assertEqual(resolve_fast_cap(dominated, "e9-keep-e14"), 1.07)
        self.assertEqual(resolve_fast_cap(pair, "e9-keep-e14"), 1.07)
        self.assertEqual(resolve_fast_cap(mixed, "e9-keep-e14"), 1.11)
        self.assertEqual(resolve_fast_cap(pair, "e9-keep-e13"), 1.11)
        self.assertEqual(resolve_fast_cap(dominated, "e9-keep-e13"), 1.07)
        self.assertEqual(resolve_fast_cap(pair, "e9-keep-e13-e14"), 1.07)
        self.assertEqual(resolve_fast_cap(dominated, "cond-fast-1.08-0.75"), 1.08)
        self.assertEqual(resolve_fast_cap(pair, "cond-fast-1.08-0.75"), 1.11)
        self.assertEqual(resolve_fast_cap(mixed, "cond-top2-1.07-0.25"), 1.07)
        self.assertEqual(resolve_fast_cap(triple, "cond-top3-1.07-0.75"), 1.07)
        self.assertEqual(resolve_fast_cap(mixed, "cond-top3-1.07-0.75"), 1.11)
        self.assertEqual(arm_caps(replica, dominated, "unify-e9-fast-1.10")["fast"], 1.10)
        for arm in CANDIDATE_ARMS:
            self.assertEqual(arm_knobs(arm), ARM_KNOBS[arm])
        with self.assertRaises(ProtocolError):
            arm_knobs("cond-fast-1.07-0.75")

    def test_triple_views_are_complete_not_leftover(self) -> None:
        try:
            from research.lab.serving_replica import (
                top3_family_fraction,
                triple_family_views,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        families = (
            ["word_problem"] * 20
            + ["english_multiple_choice"] * 20
            + ["other"] * 20
            + ["korean_reasoning"] * 10
        )
        digests = [f"{index:04x}" for index in range(len(families))]
        views = triple_family_views(families, digests)
        self.assertEqual(set(views), {"triple:english_multiple_choice+other+word_problem"})
        self.assertEqual(len(views["triple:english_multiple_choice+other+word_problem"]), 60)
        official_like = (
            ["korean_multiple_choice"] * 18
            + ["rule_reasoning"] * 16
            + ["python_program"] * 14
            + ["other"] * 11
            + ["english_multiple_choice"] * 9
            + ["long_context"] * 9
            + ["word_problem"] * 8
            + ["symbolic_math"] * 7
            + ["latex_math"] * 5
            + ["korean_reasoning"] * 3
        )
        self.assertLess(top3_family_fraction(official_like), 0.50)
