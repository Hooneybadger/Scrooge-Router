# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E7 Premium residual-guard contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e7-premium-residual-guard.v1.json"
PROTOCOL_SHA256 = "f64d172486dbeb871297c28709e651d2c6c4d684827a47b4ee64a2bce7ff436c"


class E7ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e7-premium-residual-guard-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(payload["arms"]["candidate"], "premium-residual-guard")
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)
        self.assertEqual(float(payload["thresholds"]["residual_premium_actual_max"]), 4.0)
        self.assertEqual(
            float(payload["thresholds"]["residual_premium_inflated_max"]), 4.0
        )

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e7_premium_residual_guard import (
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
        tampered["arms"]["candidate"] = "premium-residual-guard-3.0"
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E7HelperTest(unittest.TestCase):
    def test_guard_parent_flag_matches_arms(self) -> None:
        try:
            from research.lab.e7_premium_residual_guard import (
                BASELINE_ARM,
                CANDIDATE_ARM,
                ProtocolError,
                _guard_parent,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        self.assertFalse(_guard_parent(BASELINE_ARM))
        self.assertTrue(_guard_parent(CANDIDATE_ARM))
        with self.assertRaises(ProtocolError):
            _guard_parent("premium-residual-guard-3.0")


if __name__ == "__main__":
    unittest.main()
