# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E16 leftover-completion contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e16-leftover-completion.v1.json"
PROTOCOL_SHA256 = "a9fc49a6be542d58ec450ce97774f040f67e2646fd814dc5dfb0e7d634bf9ac8"


class E16ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e16-leftover-completion-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertIn("naked-fast-1.08", payload["arms"]["candidates"])
        self.assertIn("leftover-residual-2.62", payload["arms"]["candidates"])
        self.assertIn("leftover-clip-3.25", payload["arms"]["candidates"])
        self.assertIn("leftover-word-1.25", payload["arms"]["candidates"])
        self.assertIn("leftover-cond-e11-0.50", payload["arms"]["candidates"])
        self.assertIn("leftover-e10-0.50", payload["arms"]["candidates"])
        self.assertEqual(
            tuple(payload["arms"]["leftover_fast_families"]),
            ("english_multiple_choice", "word_problem"),
        )
        self.assertEqual(payload["arms"]["knobs"]["naked-fast-1.08"]["fast_cap"], 1.08)
        self.assertEqual(payload["arms"]["knobs"]["leftover-residual-2.62"]["residual_multiplier"], 2.62)
        self.assertTrue(payload["arms"]["knobs"]["leftover-clip-3.25"]["unclipped"])
        self.assertGreater(
            float(payload["arms"]["knobs"]["leftover-clip-3.25"]["residual_multiplier"]),
            3.0,
        )
        self.assertNotIn("leftover-word-1.08", payload["arms"]["candidates"])
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e16_leftover_completion import (
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
        tampered["arms"]["knobs"]["leftover-residual-2.62"]["residual_multiplier"] = 2.63
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E16RuleTest(unittest.TestCase):
    def test_leftover_knobs_match_protocol(self) -> None:
        try:
            from research.lab.e16_leftover_completion import (
                ARM_KNOBS,
                CANDIDATE_ARMS,
                LEFTOVER_FAST_FAMILIES,
                ProtocolError,
                arm_caps,
                arm_knobs,
                leftover_fast_extras,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")

        class _Replica:
            shipped_caps = {"fast": 1.11, "balanced": 1.45, "premium": 3.25}

        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        replica = _Replica()
        dominated = ["word_problem"] * 40
        mixed = ["korean_multiple_choice"] * 20 + ["other"] * 20
        for arm in CANDIDATE_ARMS:
            knobs = arm_knobs(arm)
            self.assertEqual(knobs, ARM_KNOBS[arm])
            sealed = payload["arms"]["knobs"][arm]
            self.assertEqual(knobs.unify_premium, sealed["unify_premium"])
            self.assertEqual(knobs.unclipped, sealed["unclipped"])
            self.assertEqual(knobs.shipped_switches, sealed["shipped_switches"])
        extras = leftover_fast_extras("leftover-word-2.50")
        self.assertEqual(set(extras), set(LEFTOVER_FAST_FAMILIES))
        self.assertEqual(extras["word_problem"], 2.5)
        self.assertEqual(leftover_fast_extras("naked-fast-1.08"), {})
        naked_dom = arm_caps(replica, dominated, "naked-fast-1.08")
        self.assertEqual(naked_dom["fast"], 1.08)
        word_dom = arm_caps(replica, dominated, "leftover-word-1.25")
        self.assertEqual(word_dom["fast"], 1.11)
        shipped_dom = arm_caps(replica, dominated, "leftover-e10-0.50")
        self.assertEqual(shipped_dom["fast"], 1.07)
        shipped_mix = arm_caps(replica, mixed, "leftover-e10-0.50")
        self.assertEqual(shipped_mix["fast"], 1.11)
        with self.assertRaises(ProtocolError):
            arm_knobs("unify-e9-fast-1.08")
