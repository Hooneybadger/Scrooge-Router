# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E14 top-2 Fast-cap contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e14-top2-fast-cap.v1.json"
PROTOCOL_SHA256 = "402fa563c4e681f24364cfc695cd274eeb6e5052095b5701ad1b48cc5f8a9e21"


class E14ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e14-top2-fast-cap-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            (
                "cond-top2-1.07-0.75",
                "cond-top2-1.07-0.50",
                "cond-top2-1.05-0.75",
                "cond-top2-1.05-0.50",
            ),
        )
        self.assertEqual(payload["arms"]["knobs"]["cond-top2-1.07-0.75"]["fast_cap"], 1.07)
        self.assertEqual(payload["arms"]["knobs"]["cond-top2-1.07-0.75"]["threshold"], 0.75)
        self.assertEqual(payload["arms"]["knobs"]["cond-top2-1.05-0.50"]["fast_cap"], 1.05)
        self.assertNotIn("cond-top2-1.08-0.75", payload["arms"]["candidates"])
        self.assertNotIn("cond-top2-1.07-0.25", payload["arms"]["candidates"])
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e14_top2_fast_cap import (
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
        tampered["arms"]["knobs"]["cond-top2-1.07-0.75"]["fast_cap"] = 1.08
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E14RuleTest(unittest.TestCase):
    def test_arm_knobs_match_protocol(self) -> None:
        try:
            from research.lab.e14_top2_fast_cap import (
                ARM_KNOBS,
                BASELINE_ARM,
                CANDIDATE_ARMS,
                ProtocolError,
                arm_caps,
                arm_knobs,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")

        class _Replica:
            shipped_caps = {"fast": 1.11, "balanced": 1.45, "premium": 3.25}

        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertIsNone(arm_knobs(BASELINE_ARM).threshold)
        self.assertIsNone(arm_knobs(BASELINE_ARM).fast_cap)
        replica = _Replica()
        mixed = (
            ["korean_multiple_choice"] * 18
            + ["word_problem"] * 16
            + ["other"] * 16
            + ["python_program"] * 16
            + ["rule_reasoning"] * 16
            + ["long_context"] * 18
        )
        pair = ["word_problem"] * 20 + ["english_multiple_choice"] * 20
        dominated = ["word_problem"] * 4
        for arm in CANDIDATE_ARMS:
            knobs = arm_knobs(arm)
            sealed = payload["arms"]["knobs"][arm]
            self.assertEqual(knobs, ARM_KNOBS[arm])
            self.assertEqual(knobs.threshold, sealed["threshold"])
            self.assertEqual(knobs.fast_cap, sealed["fast_cap"])
            idle = arm_caps(replica, mixed, arm)
            self.assertEqual(idle["fast"], 1.11)
            self.assertEqual(idle["balanced"], 1.45)
            paired = arm_caps(replica, pair, arm)
            self.assertEqual(paired["fast"], knobs.fast_cap)
            active = arm_caps(replica, dominated, arm)
            self.assertEqual(active["fast"], knobs.fast_cap)
        shipped_pair = arm_caps(replica, pair, BASELINE_ARM)
        self.assertEqual(shipped_pair["fast"], 1.11)
        shipped_dom = arm_caps(replica, dominated, BASELINE_ARM)
        self.assertEqual(shipped_dom["fast"], 1.07)
        with self.assertRaises(ProtocolError):
            arm_knobs("cond-top2-1.08-0.75")

    def test_public_top2_stays_below_lowest_threshold(self) -> None:
        try:
            from research.lab.serving_replica import top2_family_fraction
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        families = (
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
        self.assertAlmostEqual(top2_family_fraction(families), 0.34)
        self.assertLess(top2_family_fraction(families), 0.50)

    def test_pair_views_are_complete_not_leftover(self) -> None:
        try:
            from research.lab.serving_replica import pair_family_views
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        families = (
            ["word_problem"] * 20
            + ["english_multiple_choice"] * 20
            + ["other"] * 20
            + ["korean_reasoning"] * 10
        )
        digests = [f"{index:04x}" for index in range(len(families))]
        views = pair_family_views(families, digests)
        self.assertEqual(
            set(views),
            {
                "pair:english_multiple_choice+other",
                "pair:english_multiple_choice+word_problem",
                "pair:other+word_problem",
            },
        )
        self.assertEqual(len(views["pair:english_multiple_choice+word_problem"]), 40)
