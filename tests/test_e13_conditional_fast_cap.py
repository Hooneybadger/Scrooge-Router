# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E13 conditional Fast-cap contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e13-conditional-fast-cap.v1.json"
PROTOCOL_SHA256 = "8d390e9185d67c84126cf3cdf5b5c0c6f23c4001e9b25d430a12337f0a4fa17c"


class E13ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e13-conditional-fast-cap-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            (
                "cond-fast-1.07-0.75",
                "cond-fast-1.07-0.50",
                "cond-fast-1.07-0.25",
                "cond-fast-1.05-0.75",
                "cond-fast-1.05-0.50",
                "cond-fast-1.05-0.25",
            ),
        )
        self.assertEqual(payload["arms"]["knobs"]["cond-fast-1.07-0.75"]["fast_cap"], 1.07)
        self.assertEqual(payload["arms"]["knobs"]["cond-fast-1.07-0.75"]["threshold"], 0.75)
        self.assertEqual(payload["arms"]["knobs"]["cond-fast-1.05-0.25"]["fast_cap"], 1.05)
        self.assertNotIn("cond-fast-1.08-0.75", payload["arms"]["candidates"])
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e13_conditional_fast_cap import (
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
        tampered["arms"]["knobs"]["cond-fast-1.07-0.75"]["fast_cap"] = 1.08
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E13RuleTest(unittest.TestCase):
    def test_arm_knobs_match_protocol(self) -> None:
        try:
            from research.lab.e13_conditional_fast_cap import (
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
            ["korean_multiple_choice"] * 20
            + ["word_problem"] * 20
            + ["other"] * 20
            + ["python_program"] * 20
            + ["rule_reasoning"] * 20
        )
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
            active = arm_caps(replica, dominated, arm)
            self.assertEqual(active["fast"], knobs.fast_cap)
            self.assertEqual(active["balanced"], 1.45)
        shipped = arm_caps(replica, dominated, BASELINE_ARM)
        self.assertEqual(shipped["fast"], 1.11)
        with self.assertRaises(ProtocolError):
            arm_knobs("cond-fast-1.08-0.75")

    def test_public_mix_stays_below_lowest_threshold(self) -> None:
        try:
            from research.lab.serving_replica import max_family_fraction
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        families = (
            ["korean_multiple_choice"] * 18
            + ["word_problem"] * 17
            + ["other"] * 16
            + ["python_program"] * 16
            + ["rule_reasoning"] * 16
            + ["long_context"] * 17
        )
        self.assertAlmostEqual(max_family_fraction(families), 0.18)
        self.assertLess(max_family_fraction(families), 0.25)
