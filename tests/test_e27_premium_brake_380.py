# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E27 3.80-only confirmation contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e27-premium-brake-380.v1.json"
E27_PREFIX = "scrooge-e27-premium-brake-380-v1"
E27_CORE = "84c2075d35bff7cc5c1dc1297bf03e50ce76c4b4b2f2e6cce6983f290279d914"
E27_SEALED_SEEDS = (258662104, 1081333920, 1404762205, 999955836, 1043997100)
E26_SEEDS = (1080595098, 1829377328, 718291179, 1308133280, 1216182205)


class E27SeedDerivationTest(unittest.TestCase):
    def test_sealed_seeds_match_fail_closed_derivation(self) -> None:
        try:
            from research.lab.e5_brake_conditioned import derive_fresh_seeds
        except ImportError:
            self.skipTest("research E5 stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        derivation = payload["seed_derivation"]
        derived = derive_fresh_seeds(
            str(derivation["prefix"]),
            str(derivation["core_sha256"]),
            int(derivation["n"]),
            [int(value) for value in derivation["forbidden_previous_seeds"]],
        )
        self.assertEqual(derived, E27_SEALED_SEEDS)
        self.assertEqual(
            tuple(int(seed) for seed in payload["fresh_seeds"]), E27_SEALED_SEEDS
        )
        self.assertEqual(str(derivation["core_sha256"]), E27_CORE)
        self.assertEqual(derivation["prefix"], E27_PREFIX)
        forbidden = {int(value) for value in derivation["forbidden_previous_seeds"]}
        self.assertTrue(set(E26_SEEDS).issubset(forbidden))
        self.assertFalse(forbidden & set(payload["fresh_seeds"]))

    def test_collision_fails_closed(self) -> None:
        try:
            from research.lab.e5_brake_conditioned import (
                ProtocolError,
                derive_fresh_seeds,
            )
        except ImportError:
            self.skipTest("research E5 stack is not installed")
        with self.assertRaises(ProtocolError):
            derive_fresh_seeds(E27_PREFIX, E27_CORE, 5, list(E27_SEALED_SEEDS[:1]))


class E27ProtocolVerifyTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.e27_premium_brake_380 import (
                EXPECTED_PROTOCOL_SHA256,
                ProtocolError,
                protocol_sha256,
                verify_protocol,
            )
        except ImportError:
            self.skipTest("research E27 stack is not installed")
        return (
            EXPECTED_PROTOCOL_SHA256,
            ProtocolError,
            protocol_sha256,
            verify_protocol,
        )

    def test_protocol_verifies_against_canonical_sha(self) -> None:
        expected, ProtocolError, protocol_sha256, verify_protocol = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        digest = protocol_sha256(payload)
        self.assertEqual(digest, expected)
        self.assertEqual(verify_protocol(payload, digest), digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, "0" * 64)

    def test_seed_drift_is_rejected(self) -> None:
        _expected, ProtocolError, protocol_sha256, verify_protocol = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        tampered = json.loads(json.dumps(payload))
        tampered["fresh_seeds"] = [1, 2, 3, 4, 5]
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, protocol_sha256(payload))

    def test_primary_is_brake_not_conformal(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["arms"]["primary"]["name"], "premium-brake-3.80")
        self.assertEqual(payload["arms"]["identity_conformal"]["selection_use"], "none")
        self.assertEqual(payload["thresholds"]["full_batch_family_selection_use"], "none")
        self.assertEqual(payload["thresholds"]["buy_brake_ratio"], 3.8)
        self.assertEqual(payload["thresholds"]["dev_delta_min_exclusive"], 0.0)


class BootstrapContractTest(unittest.TestCase):
    def test_positive_gains_keep_q25_positive(self) -> None:
        try:
            import numpy as np

            from research.lab.modeling import paired_group_bootstrap
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        gains = np.concatenate(
            [np.full(8, 0.05, dtype=np.float64), np.zeros(92, dtype=np.float64)]
        )
        groups = tuple(f"g{index}" for index in range(gains.size))
        boot = paired_group_bootstrap(gains, groups, draws=2000, seed=7)
        self.assertGreater(float(boot["q2_5"]), 0.0)

    def test_zero_gains_fail_exclusive_floor(self) -> None:
        try:
            import numpy as np

            from research.lab.modeling import paired_group_bootstrap
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        gains = np.zeros(40, dtype=np.float64)
        groups = tuple(f"g{index}" for index in range(gains.size))
        boot = paired_group_bootstrap(gains, groups, draws=200, seed=3)
        self.assertEqual(float(boot["q2_5"]), 0.0)


class OverwriteAndExportTest(unittest.TestCase):
    def test_assemble_refuses_existing_output(self) -> None:
        try:
            from research.lab.e27_premium_brake_380 import ProtocolError, assemble
        except ImportError:
            self.skipTest("research E27 stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            report = pathlib.Path(tmp) / "report.json"
            audit = pathlib.Path(tmp) / "audit.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(ProtocolError):
                assemble(payload, "0" * 64, output=report, audit_output=audit)

    def test_assemble_refuses_src_paths(self) -> None:
        try:
            from research.lab.e27_premium_brake_380 import ProtocolError, assemble
        except ImportError:
            self.skipTest("research E27 stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        with self.assertRaises(ProtocolError):
            assemble(
                payload,
                "0" * 64,
                output=ROOT / "src" / "ossp_router" / "e27.json",
                audit_output=ROOT / "build" / "run-e27-premium-brake-380" / "audit.json",
            )


if __name__ == "__main__":
    unittest.main()
