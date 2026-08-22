# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E9 residual K1-coupling contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e9-residual-k1-coupling.v1.json"
PROTOCOL_SHA256 = "6a974bf0254e337d3b5ee3d4b7ae4115dc986625d9b3b561690f584187c32f9c"


class E9ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e9-residual-k1-coupling-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            (
                "residual-k1-denylist",
                "residual-k1-guard",
                "parent-and-k1-denylist",
                "parent-and-k1-guard",
            ),
        )
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e9_residual_k1_coupling import (
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
        tampered["arms"]["knobs"]["residual-k1-denylist"]["guard_parent"] = True
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E9RuleTest(unittest.TestCase):
    def test_arm_knobs_match_protocol(self) -> None:
        try:
            from research.lab.e9_residual_k1_coupling import (
                ARM_KNOBS,
                BASELINE_ARM,
                CANDIDATE_ARMS,
                ProtocolError,
                arm_knobs,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertFalse(arm_knobs(BASELINE_ARM).guard_parent)
        self.assertFalse(arm_knobs(BASELINE_ARM).denylist_other)
        self.assertFalse(arm_knobs(BASELINE_ARM).guard_brake_k1)
        for arm in CANDIDATE_ARMS:
            knobs = arm_knobs(arm)
            sealed = payload["arms"]["knobs"][arm]
            self.assertEqual(knobs, ARM_KNOBS[arm])
            self.assertEqual(knobs.guard_parent, sealed["guard_parent"])
            self.assertEqual(knobs.denylist_other, sealed["denylist_other"])
            self.assertEqual(knobs.guard_brake_k1, sealed["guard_brake_k1"])
        with self.assertRaises(ProtocolError):
            arm_knobs("residual-parent-2.75")


if __name__ == "__main__":
    unittest.main()
