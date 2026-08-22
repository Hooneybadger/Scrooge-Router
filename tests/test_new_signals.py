# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""New-signals protocol contracts and extractor units. Synthetic fixtures only."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "new-signals.v1.json"

PREFIX = "scrooge-new-signals-v1"
CORE = "491649947238e5c6d11020eb8dc5c969fe10634a85bde82b9a281a1de0403064"
SEALED_SEEDS = (308984772, 213236528, 2121011317, 970535163, 825752938)


class NewSignalsSeedTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.new_signals import derive_fresh_seeds
        except ImportError:
            self.skipTest("numpy / sklearn / research stack is not installed")
        return derive_fresh_seeds

    def test_sealed_seeds_match_derivation(self) -> None:
        derive_fresh_seeds = self._import()
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
        from research.lab.new_signals import ProtocolError

        derive_fresh_seeds = self._import()
        with self.assertRaises(ProtocolError):
            derive_fresh_seeds(PREFIX, CORE, 5, [SEALED_SEEDS[0]])


class ExtractorTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.new_signals import (
                AST_BLOCK_DIM,
                extract_ast_block,
                extract_choice_block,
                extract_numeric_block,
            )
        except ImportError:
            self.skipTest("numpy / sklearn / research stack is not installed")
        return (
            AST_BLOCK_DIM,
            extract_ast_block,
            extract_choice_block,
            extract_numeric_block,
        )

    def test_ast_block_counts_code(self) -> None:
        ast_dim, ast_block, _choice, _numeric = self._import()
        text = (
            "```python\n"
            "def f(x):\n"
            "    for i in range(3):\n"
            "        if x:\n"
            "            assert i\n"
            "    return [y for y in x]\n"
            "```\n"
        )
        row = ast_block(text)
        self.assertEqual(len(row), ast_dim)
        self.assertEqual(row[0], 1.0)
        self.assertGreater(row[1], 0.0)  # function defs
        self.assertGreater(row[3], 0.0)  # loops
        self.assertGreater(row[7], 0.0)  # comprehensions

    def test_ast_block_zero_on_syntax_error(self) -> None:
        _ast_dim, ast_block, _choice, _numeric = self._import()
        self.assertEqual(ast_block("def broken(:"), (0.0,) * 16)

    def test_numeric_block_detects_magnitudes(self) -> None:
        _ast_dim, _ast, _choice, numeric = self._import()
        row = numeric("There are 3 apples, 2.5 kg, 40% off for $5.")
        self.assertAlmostEqual(row[0], __import__("math").log1p(4), places=9)
        self.assertAlmostEqual(row[1], __import__("math").log10(41.0), places=9)
        self.assertEqual(row[4], 1.0)  # currency detected

    def test_choice_block_flags_option_lists(self) -> None:
        _ast_dim, _ast, choice, _numeric = self._import()
        row = choice("Question?\nA) one\nB) two\nC) three")
        self.assertAlmostEqual(row[0], math.log1p(3), places=9)
        self.assertEqual(row[2], 1.0)
        self.assertAlmostEqual(row[3], math.log1p(3), places=9)


class NewSignalsProtocolVerifyTest(unittest.TestCase):
    def _modules(self):
        try:
            from research.lab.new_signals import (
                ProtocolError,
                protocol_sha256,
                verify_protocol,
            )
        except ImportError:
            self.skipTest("numpy / sklearn / research stack is not installed")
        return ProtocolError, protocol_sha256, verify_protocol

    @staticmethod
    def _bad_identity() -> dict[str, str]:
        return {
            key: "0" * 64
            for key in (
                "train_inputs_sha256",
                "train_outcomes_sha256",
                "dev_inputs_sha256",
                "dev_outcomes_sha256",
                "policy_sha256",
            )
        }

    def test_verifies_and_rejects_drift(self) -> None:
        ProtocolError, protocol_sha256, verify_protocol = self._modules()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        digest = protocol_sha256(payload)
        self.assertEqual(verify_protocol(payload, digest), digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, "0" * 64)
        tampered = json.loads(json.dumps(payload))
        tampered["fresh_seeds"] = [1, 2, 3, 4, 5]
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, digest, pool_identity=self._bad_identity())


if __name__ == "__main__":
    unittest.main()
