# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Challenger-heads protocol contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "challenger-heads.v1.json"

PREFIX = "scrooge-challenger-heads-v1"
CORE = "d44a7401f24105ff3c409b45ad76b43981c448507fc58e2daf1cb615c0ddd48b"
SEALED_SEEDS = (130672035, 733125616, 733043265, 1283801968, 498402659)


class ChallengerHeadsProtocolTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.challenger_heads import (
                ARMS,
                ProtocolError,
                derive_fresh_seeds,
                protocol_sha256,
                verify_protocol,
            )
        except ImportError:
            self.skipTest("numpy / sklearn / research stack is not installed")
        return ARMS, ProtocolError, derive_fresh_seeds, protocol_sha256, verify_protocol

    def test_sealed_seeds_match_derivation(self) -> None:
        ARMS, _error, derive_fresh_seeds, _sha, _verify = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        derivation = payload["seed_derivation"]
        derived = derive_fresh_seeds(
            str(derivation["prefix"]),
            str(derivation["core_sha256"]),
            int(derivation["n"]),
            [int(value) for value in derivation["forbidden_previous_seeds"]],
        )
        self.assertEqual(derived, SEALED_SEEDS)
        self.assertEqual(
            tuple(int(seed) for seed in payload["fresh_seeds"]), SEALED_SEEDS
        )
        self.assertEqual(str(derivation["core_sha256"]), CORE)

    def test_collision_fails_closed(self) -> None:
        _arms, ProtocolError, derive_fresh_seeds, _sha, _verify = self._import()
        with self.assertRaises(ProtocolError):
            derive_fresh_seeds(PREFIX, CORE, 5, [SEALED_SEEDS[0]])

    def test_protocol_verifies_and_rejects_drift(self) -> None:
        ARMS, ProtocolError, _derive, protocol_sha256, verify_protocol = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        digest = protocol_sha256(payload)
        self.assertEqual(verify_protocol(payload, digest), digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, "0" * 64)
        tampered = json.loads(json.dumps(payload))
        tampered["fresh_seeds"] = [1, 2, 3, 4, 5]
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)

    def test_arms_and_decisions_are_well_formed(self) -> None:
        ARMS, _error, _derive, _sha, _verify = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(ARMS), 4)
        self.assertIn(payload["arms"]["baseline"]["name"], ARMS[0])
        self.assertNotEqual(payload["decisions"]["pass"], payload["decisions"]["fail"])
        self.assertEqual(
            float(payload["thresholds"]["mean_delta_min"]), 0.002
        )


if __name__ == "__main__":
    unittest.main()
