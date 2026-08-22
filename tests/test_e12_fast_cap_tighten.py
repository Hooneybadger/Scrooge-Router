# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E12 Fast-cap tighten contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e12-fast-cap-tighten.v1.json"
PROTOCOL_SHA256 = "85526cdf4b5c9084028cb81c966fac49d6b5d7e0950a6fba15f130e33f9012bf"


class E12ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e12-fast-cap-tighten-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            ("fast-cap-1.09", "fast-cap-1.07", "fast-cap-1.05"),
        )
        self.assertEqual(payload["arms"]["fast_caps"]["fast-cap-1.09"], 1.09)
        self.assertEqual(payload["arms"]["fast_caps"]["fast-cap-1.07"], 1.07)
        self.assertEqual(payload["arms"]["fast_caps"]["fast-cap-1.05"], 1.05)
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e12_fast_cap_tighten import (
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
        tampered["arms"]["fast_caps"]["fast-cap-1.09"] = 1.04
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E12RuleTest(unittest.TestCase):
    def test_arm_caps_only_change_fast(self) -> None:
        try:
            from research.lab.e12_fast_cap_tighten import (
                ARM_CAPS,
                BASELINE_ARM,
                CANDIDATE_ARMS,
                ProtocolError,
                arm_caps,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")

        class _Replica:
            shipped_caps = {"fast": 1.11, "balanced": 1.45, "premium": 3.25}

        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        replica = _Replica()
        shipped = arm_caps(replica, BASELINE_ARM)
        self.assertEqual(shipped["fast"], 1.11)
        self.assertEqual(shipped["balanced"], 1.45)
        self.assertEqual(shipped["premium"], 3.25)
        for arm in CANDIDATE_ARMS:
            caps = arm_caps(replica, arm)
            self.assertEqual(caps["fast"], ARM_CAPS[arm])
            self.assertEqual(caps["fast"], payload["arms"]["fast_caps"][arm])
            self.assertEqual(caps["balanced"], 1.45)
            self.assertEqual(caps["premium"], 3.25)
        with self.assertRaises(ProtocolError):
            arm_caps(replica, "fast-cap-1.04")
