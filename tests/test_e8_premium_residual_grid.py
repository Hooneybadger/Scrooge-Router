# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E8 Premium residual-parent grid contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e8-premium-residual-grid.v1.json"
PROTOCOL_SHA256 = "32c3a9d18115257a9ca703954d857e8feaada6c7ef8b14f6b91e12d30a596b70"


class E8ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e8-premium-residual-grid-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            ("residual-parent-2.75", "residual-parent-3.00"),
        )
        self.assertEqual(payload["arms"]["multipliers"]["residual-parent-2.75"], 2.75)
        self.assertEqual(payload["arms"]["multipliers"]["residual-parent-3.00"], 3.0)
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)
        self.assertEqual(float(payload["thresholds"]["residual_premium_actual_max"]), 4.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e8_premium_residual_grid import (
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
        tampered["arms"]["multipliers"]["residual-parent-2.75"] = 2.62
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E8RuleTest(unittest.TestCase):
    def test_arm_multipliers_match_protocol_and_clip(self) -> None:
        try:
            from ossp_router.family_guard_router import MULTIPLIER_CLIP
            from research.lab.e8_premium_residual_grid import (
                ARM_MULTIPLIERS,
                BASELINE_ARM,
                CANDIDATE_ARMS,
                ProtocolError,
                arm_parent_guard,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        low, high = MULTIPLIER_CLIP
        self.assertEqual(tuple(payload["arms"]["candidates"]), CANDIDATE_ARMS)
        self.assertFalse(arm_parent_guard(BASELINE_ARM)[0])
        self.assertIsNone(arm_parent_guard(BASELINE_ARM)[1])
        for arm in CANDIDATE_ARMS:
            guard, multiplier = arm_parent_guard(arm)
            self.assertTrue(guard)
            self.assertEqual(multiplier, ARM_MULTIPLIERS[arm])
            self.assertEqual(multiplier, payload["arms"]["multipliers"][arm])
            self.assertGreater(multiplier, 2.5)
            self.assertGreaterEqual(multiplier, low)
            self.assertLessEqual(multiplier, high)
        with self.assertRaises(ProtocolError):
            arm_parent_guard("residual-parent-2.62")


if __name__ == "__main__":
    unittest.main()
