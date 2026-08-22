# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E10 conditional Premium-guard contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e10-conditional-premium-guard.v1.json"
PROTOCOL_SHA256 = "b0fe689be7ed10db64fbb05c862e1969cfb378163e54053467c467138da542fa"


class E10ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e10-conditional-premium-guard-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            (
                "cond-parent-0.75",
                "cond-parent-0.50",
                "cond-parent-0.25",
                "cond-parent-denylist-0.75",
                "cond-parent-denylist-0.50",
                "cond-parent-denylist-0.25",
            ),
        )
        self.assertEqual(payload["arms"]["knobs"]["cond-parent-0.75"]["threshold"], 0.75)
        self.assertFalse(payload["arms"]["knobs"]["cond-parent-0.75"]["denylist_other"])
        self.assertTrue(
            payload["arms"]["knobs"]["cond-parent-denylist-0.25"]["denylist_other"]
        )
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)
        self.assertEqual(float(payload["thresholds"]["residual_premium_actual_max"]), 4.0)
        self.assertEqual(
            float(payload["thresholds"]["residual_premium_inflated_max"]), 4.0
        )

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e10_conditional_premium_guard import (
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
        tampered["arms"]["knobs"]["cond-parent-0.75"]["threshold"] = 0.62
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E10RuleTest(unittest.TestCase):
    def test_arm_knobs_match_protocol(self) -> None:
        try:
            from research.lab.e10_conditional_premium_guard import (
                ARM_KNOBS,
                BASELINE_ARM,
                CANDIDATE_ARMS,
                ProtocolError,
                arm_knobs,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertIsNone(arm_knobs(BASELINE_ARM).threshold)
        self.assertFalse(arm_knobs(BASELINE_ARM).guard_parent)
        self.assertFalse(arm_knobs(BASELINE_ARM).denylist_other)
        for arm in CANDIDATE_ARMS:
            knobs = arm_knobs(arm)
            sealed = payload["arms"]["knobs"][arm]
            self.assertEqual(knobs, ARM_KNOBS[arm])
            self.assertEqual(knobs.threshold, sealed["threshold"])
            self.assertEqual(knobs.guard_parent, sealed["guard_parent"])
            self.assertEqual(knobs.denylist_other, sealed["denylist_other"])
        with self.assertRaises(ProtocolError):
            arm_knobs("cond-parent-0.62")

    def test_public_mix_stays_below_lowest_threshold(self) -> None:
        try:
            from research.lab.serving_replica import residual_fraction
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        families = ["other"] + ["math"] * 9
        self.assertAlmostEqual(residual_fraction(families), 0.10)
        self.assertLess(residual_fraction(families), 0.25)
