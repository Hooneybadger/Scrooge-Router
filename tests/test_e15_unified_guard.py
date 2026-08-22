# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E15 unified-guard contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e15-unified-guard.v1.json"
PROTOCOL_SHA256 = "06339e08ed96df5decd5e87b44600eda173c6132a8b92ac51c49e7e012144479"


class E15ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e15-unified-guard-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            (
                "unify-e9",
                "unify-e9-fast-1.09",
                "unify-e9-fast-1.08",
                "unify-e9-fast-1.07",
                "unify-e9-fast-1.05",
                "unify-train-ratios",
            ),
        )
        self.assertEqual(payload["arms"]["knobs"]["unify-e9"]["fast_cap"], 1.11)
        self.assertEqual(payload["arms"]["knobs"]["unify-e9-fast-1.08"]["fast_cap"], 1.08)
        self.assertTrue(payload["arms"]["knobs"]["unify-e9"]["unify_premium"])
        self.assertTrue(payload["arms"]["knobs"]["unify-train-ratios"]["extra_train_ratios"])
        self.assertIn("latex_math", payload["arms"]["train_ratio_extras"])
        self.assertNotIn("word_problem", payload["arms"]["train_ratio_extras"])
        self.assertNotIn("english_multiple_choice", payload["arms"]["train_ratio_extras"])
        for row in payload["arms"]["knobs"].values():
            self.assertNotIn("threshold", row)
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e15_unified_guard import (
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
        tampered["arms"]["knobs"]["unify-e9-fast-1.08"]["fast_cap"] = 1.075
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E15RuleTest(unittest.TestCase):
    def test_always_on_caps_ignore_batch_mix(self) -> None:
        try:
            from research.lab.e15_unified_guard import (
                ARM_KNOBS,
                BASELINE_ARM,
                CANDIDATE_ARMS,
                ProtocolError,
                TRAIN_RATIO_EXTRAS,
                arm_caps,
                arm_knobs,
                extra_multipliers,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")

        class _Replica:
            shipped_caps = {"fast": 1.11, "balanced": 1.45, "premium": 3.25}

        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
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
        residual = ["other"] * 40
        pair = ["word_problem"] * 20 + ["english_multiple_choice"] * 20
        self.assertIsNone(arm_knobs(BASELINE_ARM).fast_cap)
        self.assertTrue(arm_knobs(BASELINE_ARM).shipped_switches)
        shipped_dom = arm_caps(replica, dominated, BASELINE_ARM)
        self.assertEqual(shipped_dom["fast"], 1.07)
        shipped_pair = arm_caps(replica, pair, BASELINE_ARM)
        self.assertEqual(shipped_pair["fast"], 1.11)
        for arm in CANDIDATE_ARMS:
            knobs = arm_knobs(arm)
            sealed = payload["arms"]["knobs"][arm]
            self.assertEqual(knobs, ARM_KNOBS[arm])
            self.assertEqual(knobs.fast_cap, sealed["fast_cap"])
            self.assertEqual(knobs.unify_premium, sealed["unify_premium"])
            self.assertEqual(knobs.extra_train_ratios, sealed["extra_train_ratios"])
            self.assertFalse(knobs.shipped_switches)
            for families in (mixed, dominated, residual, pair):
                caps = arm_caps(replica, families, arm)
                self.assertEqual(caps["fast"], knobs.fast_cap)
                self.assertEqual(caps["balanced"], 1.45)
                self.assertEqual(caps["premium"], 3.25)
        self.assertEqual(set(TRAIN_RATIO_EXTRAS), set(payload["arms"]["train_ratio_extras"]))
        self.assertEqual(extra_multipliers("unify-e9"), {})
        extras = extra_multipliers("unify-train-ratios")
        self.assertIn("latex_math", extras)
        self.assertNotIn("word_problem", extras)
        self.assertNotIn("other", extras)
        with self.assertRaises(ProtocolError):
            arm_knobs("cond-fast-1.07-0.75")

    def test_reprice_only_touches_named_train_ratio_families(self) -> None:
        try:
            from research.lab.e15_unified_guard import (
                TRAIN_RATIO_EXTRAS,
                reprice_fast_balanced,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")

        class _Split:
            families = ("word_problem", "latex_math", "other")
            fb_predictions = (
                (0.1, (1.0, 2.0)),
                (0.1, (1.0, 2.0)),
                (0.1, (1.0, 2.0)),
            )

        priced = reprice_fast_balanced(_Split(), (0, 1, 2), TRAIN_RATIO_EXTRAS)
        self.assertEqual(priced[0][1], (1.0, 2.0))
        self.assertAlmostEqual(priced[1][1][1], 1.0 + 1.0 * TRAIN_RATIO_EXTRAS["latex_math"])
        self.assertEqual(priced[2][1], (1.0, 2.0))
