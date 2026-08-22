# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Final-fit line contracts: final-public-refit-v1 and data-scale-diagnostic-v1.

Synthetic fixtures only. Public outcomes are never opened and the
runners are not executed through a successful public fit.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFIT_PROTOCOL = ROOT / "research" / "protocols" / "final-public-refit.v1.json"
SCALE_PROTOCOL = ROOT / "research" / "protocols" / "data-scale-diagnostic.v1.json"

SCALE_PREFIX = "scrooge-data-scale-diagnostic-v1"
SCALE_CORE = "727c07e6c7ae9c55fbf6c0d659c2594ee2cfd254f3bde635e4cc811dbbfc85ba"
SCALE_SEALED_SEEDS = (1558982291, 995925147, 1201315901, 850766810, 462283710)


class DataScaleSeedTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.data_scale_diagnostic import derive_fresh_seeds
        except ImportError:
            self.skipTest("numpy / sklearn / research stack is not installed")
        return derive_fresh_seeds

    def test_sealed_seeds_match_derivation(self) -> None:
        derive_fresh_seeds = self._import()
        payload = json.loads(SCALE_PROTOCOL.read_text(encoding="utf-8"))
        derivation = payload["seed_derivation"]
        derived = derive_fresh_seeds(
            str(derivation["prefix"]),
            str(derivation["core_sha256"]),
            int(derivation["n"]),
            [int(value) for value in derivation["forbidden_previous_seeds"]],
        )
        self.assertEqual(derived, SCALE_SEALED_SEEDS)
        self.assertEqual(
            tuple(int(seed) for seed in payload["fresh_seeds"]), SCALE_SEALED_SEEDS
        )
        self.assertEqual(str(derivation["core_sha256"]), SCALE_CORE)

    def test_collision_fails_closed(self) -> None:
        from research.lab.data_scale_diagnostic import ProtocolError

        derive_fresh_seeds = self._import()
        with self.assertRaises(ProtocolError):
            derive_fresh_seeds(SCALE_PREFIX, SCALE_CORE, 5, [SCALE_SEALED_SEEDS[0]])


class FinalFitProtocolVerifyTest(unittest.TestCase):
    def _modules(self):
        try:
            from research.lab.final_public_refit import (
                ProtocolError,
                protocol_sha256,
                verify_protocol,
            )
            from research.lab.data_scale_diagnostic import (
                protocol_sha256 as scale_protocol_sha256,
                verify_protocol as scale_verify_protocol,
            )
        except ImportError:
            self.skipTest("numpy / sklearn / research stack is not installed")
        return (
            ProtocolError,
            protocol_sha256,
            verify_protocol,
            scale_protocol_sha256,
            scale_verify_protocol,
        )

    @staticmethod
    def _bad_identity() -> dict[str, str]:
        return {key: "0" * 64 for key in (
            "train_inputs_sha256",
            "train_outcomes_sha256",
            "dev_inputs_sha256",
            "dev_outcomes_sha256",
            "policy_sha256",
        )}

    def test_refit_protocol_verifies_and_rejects_drift(self) -> None:
        (
            ProtocolError,
            protocol_sha256,
            verify_protocol,
            _scale_sha,
            _scale_verify,
        ) = self._modules()
        payload = json.loads(REFIT_PROTOCOL.read_text(encoding="utf-8"))
        digest = protocol_sha256(payload)
        self.assertEqual(verify_protocol(payload, digest), digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, "0" * 64)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, digest, pool_identity=self._bad_identity())

    def test_scale_protocol_verifies_and_rejects_drift(self) -> None:
        (
            ProtocolError,
            _refit_sha,
            _refit_verify,
            scale_protocol_sha256,
            scale_verify_protocol,
        ) = self._modules()
        payload = json.loads(SCALE_PROTOCOL.read_text(encoding="utf-8"))
        digest = scale_protocol_sha256(payload)
        self.assertEqual(scale_verify_protocol(payload, digest), digest)
        tampered = json.loads(json.dumps(payload))
        tampered["fresh_seeds"] = [1, 2, 3, 4, 5]
        with self.assertRaises(ProtocolError):
            scale_verify_protocol(tampered, digest)

    def test_protocols_declare_complementary_decisions(self) -> None:
        refit = json.loads(REFIT_PROTOCOL.read_text(encoding="utf-8"))
        scale = json.loads(SCALE_PROTOCOL.read_text(encoding="utf-8"))
        self.assertNotEqual(refit["decisions"]["pass"], refit["decisions"]["fail"])
        self.assertNotEqual(scale["decisions"]["pass"], scale["decisions"]["fail"])
        self.assertEqual(refit["output"]["runtime_resource_touched"], False)
        self.assertIn(
            "human approval", refit["accepted_artifact_action"]["on_all_gates_passed"]
        )


if __name__ == "__main__":
    unittest.main()
