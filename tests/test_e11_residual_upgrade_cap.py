# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E11 residual upgrade-cap contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e11-residual-upgrade-cap.v1.json"
PROTOCOL_SHA256 = "8b8978c5499a8043997794507d3e0125ae7d0345fecbe1c3d47a2aff6edf4d70"


class E11ProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "e11-residual-upgrade-cap-v1")
        self.assertEqual(payload["arms"]["baseline"], "shipped")
        self.assertEqual(
            tuple(payload["arms"]["candidates"]),
            (
                "residual-upgrade-0.80",
                "residual-upgrade-0.67",
                "residual-upgrade-0.50",
            ),
        )
        self.assertEqual(payload["arms"]["upgrade_caps"]["residual-upgrade-0.80"], 0.80)
        self.assertEqual(payload["arms"]["upgrade_caps"]["residual-upgrade-0.67"], 0.67)
        self.assertEqual(payload["arms"]["upgrade_caps"]["residual-upgrade-0.50"], 0.50)
        self.assertLess(float(payload["thresholds"]["dev_delta_min_exclusive"]), 0.0)
        self.assertEqual(float(payload["thresholds"]["residual_premium_actual_max"]), 4.0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.e11_residual_upgrade_cap import (
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
        tampered["arms"]["upgrade_caps"]["residual-upgrade-0.50"] = 0.38
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class E11RuleTest(unittest.TestCase):
    def test_arm_caps_match_protocol(self) -> None:
        try:
            from research.lab.e11_residual_upgrade_cap import (
                ARM_CAPS,
                BASELINE_ARM,
                CANDIDATE_ARMS,
                ProtocolError,
                arm_upgrade_cap,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertIsNone(arm_upgrade_cap(BASELINE_ARM))
        for arm in CANDIDATE_ARMS:
            self.assertEqual(arm_upgrade_cap(arm), ARM_CAPS[arm])
            self.assertEqual(arm_upgrade_cap(arm), payload["arms"]["upgrade_caps"][arm])
        with self.assertRaises(ProtocolError):
            arm_upgrade_cap("residual-upgrade-0.38")

    def test_demote_keeps_floor_fraction_and_spares_non_residual(self) -> None:
        try:
            from research.lab.serving_replica import (
                AX31,
                K1,
                LIGHT,
                ProtocolError,
                demote_residual_upgrades,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        families = ["other", "other", "other", "other", "math"]
        models = [K1, AX31, AX31, AX31, AX31]
        quality = [0.9, 0.1, 0.2, 0.3, 0.05]
        digests = ["a", "b", "c", "d", "e"]
        out = demote_residual_upgrades(families, models, quality, digests, 0.50)
        self.assertEqual(out[4], AX31)
        residual_upgraded = sum(
            1
            for family, model in zip(families, out)
            if family == "other" and model != LIGHT
        )
        self.assertEqual(residual_upgraded, 2)
        with self.assertRaises(ProtocolError):
            demote_residual_upgrades(families, models, quality, digests, 0.0)
