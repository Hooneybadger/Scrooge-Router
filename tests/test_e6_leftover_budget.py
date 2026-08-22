# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E6 leftover-budget contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e6-leftover-budget.v1.json"
PROTOCOL_SHA256 = "bc817d9ad1f59e55b01b6a97dc886b2bb1b57979c2f3a980d762c9ad2832762e"


class E6ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e6-leftover-budget-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            (
                "cond-residual-fast",
                "cond-residual-both",
                "static-fast-1.13",
                "premium-brake-3.40",
                "premium-brake-3.55",
            ),
        )
        self.assertGreater(float(payload["thresholds"]["train_delta_min"]), 0.0)
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e6_leftover_budget import (
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
        tampered["arms"]["candidates"] = ["static-fast-1.13"]
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E6RuleTest(unittest.TestCase):
    def test_conditioned_caps_match_protocol(self) -> None:
        try:
            from research.lab.serving_replica import (
                conditioned_balanced_cap,
                conditioned_fast_cap,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        rule = payload["residual_rule"]
        self.assertEqual(conditioned_fast_cap(0.04), rule["fast_caps"]["low"])
        self.assertEqual(conditioned_fast_cap(0.08), rule["fast_caps"]["mid"])
        self.assertEqual(conditioned_fast_cap(0.20), rule["fast_caps"]["high"])
        self.assertEqual(conditioned_balanced_cap(0.04), rule["balanced_caps"]["low"])
        self.assertEqual(conditioned_balanced_cap(0.08), rule["balanced_caps"]["mid"])
        self.assertEqual(conditioned_balanced_cap(0.20), rule["balanced_caps"]["high"])


if __name__ == "__main__":
    unittest.main()
