# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""CHUF TV-ball confirmation protocol contracts.

Synthetic fixtures only. Inputs may be hashed for epsilon. Public
outcomes are not opened and CHUF is not fitted here.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tempfile
import unittest
from copy import deepcopy

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PUBLIC_TRAIN_INPUTS = ROOT / "data" / "materialized" / "train" / "inputs.json"


def _require_research_stack() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("numpy / research stack is not installed")


def _require_public_inputs(test: unittest.TestCase) -> None:
    if not _PUBLIC_TRAIN_INPUTS.is_file():
        test.skipTest("pinned public Train+Dev files are not materialized")


_require_research_stack()


E1F_CORE = "f4cca0d425b47bda6e42be9b5c11b64e3cf9c57efd2810f11b22b1bd6051ba79"
EXPLICIT_FRESH_SEEDS = (
    1524653244,
    655399222,
    544868342,
    1444644023,
    2132086428,
    292577053,
    1003090710,
    253011083,
    534284889,
    2049515330,
    515672025,
    572878001,
)


class SeedDerivationTests(unittest.TestCase):
    def test_derivation_matches_explicit_list(self) -> None:
        from research.lab.chuf_tvball_confirmation import (
            E1F_DECISION_CORE_SHA256,
            derive_fresh_seeds,
        )

        self.assertEqual(E1F_DECISION_CORE_SHA256, E1F_CORE)
        self.assertEqual(derive_fresh_seeds(), EXPLICIT_FRESH_SEEDS)

    def test_unique_and_no_old_overlap(self) -> None:
        from research.lab.chuf_tvball_confirmation import OLD_SEEDS, derive_fresh_seeds

        seeds = derive_fresh_seeds()
        self.assertEqual(len(seeds), 12)
        self.assertEqual(len(set(seeds)), 12)
        self.assertFalse(set(seeds) & set(OLD_SEEDS))

    def test_collision_with_old_seed_fails_closed(self) -> None:
        from research.lab.chuf_tvball_confirmation import derive_fresh_seeds

        first = derive_fresh_seeds()[0]
        with self.assertRaises(RuntimeError) as caught:
            derive_fresh_seeds(old_seeds=(first,))
        self.assertIn("fail closed", str(caught.exception))
        self.assertIn("overlap", str(caught.exception))


class EpsilonAndTvTests(unittest.TestCase):
    def test_epsilon_from_inputs_matches_pin(self) -> None:
        _require_public_inputs(self)
        from research.lab.chuf_tvball_confirmation import (
            EXPECTED_EPSILON,
            epsilon_from_input_paths,
        )

        value = epsilon_from_input_paths()
        self.assertEqual(value, EXPECTED_EPSILON)
        self.assertEqual(value, 0.014204545454545449)

    def test_tv_closed_form_matches_transport_and_grid(self) -> None:
        from research.lab.chuf_tvball_confirmation import tv_worst

        proportions = (0.50, 0.30, 0.20)
        deltas = (0.02, -0.01, 0.04)
        epsilon = 0.10
        official = sum(p * d for p, d in zip(proportions, deltas))
        closed = tv_worst(
            official,
            epsilon,
            {"a": deltas[0], "b": deltas[1], "c": deltas[2]},
        )
        transported = official + epsilon * (min(deltas) - max(deltas))
        self.assertEqual(closed, transported)

        steps = 80
        best = None
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                mix = (i / steps, j / steps, k / steps)
                tv = 0.5 * sum(abs(a - b) for a, b in zip(mix, proportions))
                if tv > epsilon + 1e-12:
                    continue
                value = sum(a * b for a, b in zip(mix, deltas))
                if best is None or value < best:
                    best = value
        self.assertIsNotNone(best)
        self.assertAlmostEqual(closed, float(best), places=6)


class GateBoundaryTests(unittest.TestCase):
    def _row(self, seed: int, **overrides: object) -> dict:
        from research.lab.chuf_tvball_confirmation import REQUIRED_FAMILIES

        families = {name: 0.004 for name in REQUIRED_FAMILIES}
        families["symbolic_math"] = -0.002
        row = {
            "ax31_identical": True,
            "baseline_caps_ok": True,
            "baseline_fold_caps_ok": True,
            "candidate_caps_ok": True,
            "candidate_fold_caps_ok": True,
            "candidate_quality": 0.691,
            "delta": 0.0022,
            "family_deltas": families,
            "fold_seed": seed,
            "k1_fast_balanced_zero": True,
            "tv_worst": 0.0015,
        }
        row.update(overrides)
        return row

    def _rows(self, **overrides: object) -> list[dict]:
        return [self._row(seed, **overrides) for seed in EXPLICIT_FRESH_SEEDS]

    def test_pass_and_tv_boundary(self) -> None:
        from research.lab.chuf_tvball_confirmation import (
            PASS_DECISION,
            confirmation_gate,
        )

        passed = confirmation_gate(self._rows(tv_worst=-0.003))
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["dirac_failures_diagnostic"], [])
        failed = confirmation_gate(self._rows())
        failed_rows = self._rows()
        failed_rows[0]["tv_worst"] = -0.0031
        failed = confirmation_gate(failed_rows)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["tv_failures"], [EXPLICIT_FRESH_SEEDS[0]])
        self.assertTrue(bool(PASS_DECISION))

    def test_dirac_is_diagnostic_only(self) -> None:
        from research.lab.chuf_tvball_confirmation import (
            REQUIRED_FAMILIES,
            confirmation_gate,
        )

        rows = self._rows()
        families = {name: 0.004 for name in REQUIRED_FAMILIES}
        families["english_multiple_choice"] = -0.01
        rows[0]["family_deltas"] = families
        gate = confirmation_gate(rows)
        self.assertTrue(gate["passed"])
        self.assertEqual(
            gate["dirac_failures_diagnostic"],
            [{"failures": ["english_multiple_choice"], "seed": EXPLICIT_FRESH_SEEDS[0]}],
        )

    def test_mean_worst_abs_and_identity_caps(self) -> None:
        from research.lab.chuf_tvball_confirmation import confirmation_gate

        low_mean = self._rows(delta=0.0019, candidate_quality=0.691)
        self.assertFalse(confirmation_gate(low_mean)["passed"])
        low_worst = self._rows()
        low_worst[3]["delta"] = 0.0009
        self.assertFalse(confirmation_gate(low_worst)["passed"])
        low_abs = self._rows(candidate_quality=0.689)
        self.assertFalse(confirmation_gate(low_abs)["passed"])
        identity = self._rows()
        identity[1]["ax31_identical"] = False
        self.assertEqual(
            confirmation_gate(identity)["identity_failures"],
            [EXPLICIT_FRESH_SEEDS[1]],
        )
        caps = self._rows()
        caps[2]["candidate_fold_caps_ok"] = False
        self.assertEqual(
            confirmation_gate(caps)["cap_failures"],
            [EXPLICIT_FRESH_SEEDS[2]],
        )
        k1 = self._rows()
        k1[4]["k1_fast_balanced_zero"] = False
        self.assertEqual(
            confirmation_gate(k1)["k1_failures"],
            [EXPLICIT_FRESH_SEEDS[4]],
        )

    def test_missing_family_fails(self) -> None:
        from research.lab.chuf_tvball_confirmation import confirmation_gate

        rows = self._rows()
        families = dict(rows[0]["family_deltas"])
        del families["word_problem"]
        rows[0]["family_deltas"] = families
        gate = confirmation_gate(rows)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["family_failures"], [EXPLICIT_FRESH_SEEDS[0]])


class ProtocolHashTests(unittest.TestCase):
    def test_canonical_hash_is_deterministic(self) -> None:
        _require_public_inputs(self)
        from research.lab.chuf_tvball_confirmation import (
            EXPECTED_PROTOCOL_SHA256,
            PROTOCOL_PATH,
            build_canonical_protocol,
            load_protocol,
            protocol_sha256,
            verify_protocol,
        )

        first = build_canonical_protocol()
        second = build_canonical_protocol()
        self.assertEqual(protocol_sha256(first), protocol_sha256(second))
        self.assertEqual(protocol_sha256(first), EXPECTED_PROTOCOL_SHA256)
        sealed = load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol_sha256(sealed), EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(verify_protocol(sealed, EXPECTED_PROTOCOL_SHA256), EXPECTED_PROTOCOL_SHA256)
        self.assertNotIn("generated_at", sealed)

    def test_tamper_rejects(self) -> None:
        from research.lab.chuf_tvball_confirmation import (
            EXPECTED_PROTOCOL_SHA256,
            PROTOCOL_PATH,
            load_protocol,
            protocol_sha256,
            verify_protocol,
        )

        tampered = deepcopy(load_protocol(PROTOCOL_PATH))
        tampered["fresh_seeds"][0] = 1
        self.assertNotEqual(protocol_sha256(tampered), EXPECTED_PROTOCOL_SHA256)
        with self.assertRaises(RuntimeError):
            verify_protocol(tampered, EXPECTED_PROTOCOL_SHA256)


class RunnerRefuseTests(unittest.TestCase):
    def test_wrong_expected_sha_and_architecture_drift(self) -> None:
        from research.experiments.confirm_chuf_tvball import main
        from research.lab.chuf_tvball_confirmation import (
            EXPECTED_PROTOCOL_SHA256,
            PROTOCOL_PATH,
            architecture_snapshot,
            assert_live_architecture,
            load_protocol,
            verify_protocol,
        )

        with tempfile.TemporaryDirectory() as folder:
            isolated = pathlib.Path(folder)
            output = isolated / "report.json"
            audit = isolated / "episode-audit.json"
            with self.assertRaises(RuntimeError) as caught:
                main(
                    [
                        "--protocol",
                        str(PROTOCOL_PATH),
                        "--expected-protocol-sha256",
                        "0" * 64,
                        "--output",
                        str(output),
                        "--audit-output",
                        str(audit),
                    ]
                )
            self.assertIn("protocol sha mismatch", str(caught.exception))
            self.assertFalse(output.exists())
            self.assertFalse(audit.exists())
        dirty = architecture_snapshot()
        dirty["n_cost_bins"] = 99
        with self.assertRaises(RuntimeError):
            assert_live_architecture(
                {
                    "architecture": dirty,
                    "candidate": "chuf-v1",
                    "e1f": load_protocol(PROTOCOL_PATH)["e1f"],
                    "e1f_source_sha256": "0" * 64,
                }
            )
        if _PUBLIC_TRAIN_INPUTS.is_file():
            verify_protocol(load_protocol(PROTOCOL_PATH), EXPECTED_PROTOCOL_SHA256)

    def test_overwrite_and_e1f_path_refused(self) -> None:
        from research.experiments.confirm_chuf_tvball import main
        from research.lab.chuf_tvball_confirmation import (
            EXPECTED_PROTOCOL_SHA256,
            PROTOCOL_PATH,
        )

        e1f_out = ROOT / "build" / "compare-e1f-cost-conditioned-frontier" / "report.json"
        with self.assertRaises(RuntimeError) as caught:
            main(
                [
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--expected-protocol-sha256",
                    EXPECTED_PROTOCOL_SHA256,
                    "--output",
                    str(e1f_out),
                ]
            )
        self.assertIn("must not write the E1F", str(caught.exception))
        with tempfile.TemporaryDirectory() as folder:
            existing = pathlib.Path(folder) / "report.json"
            existing.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError) as overwrite:
                main(
                    [
                        "--protocol",
                        str(PROTOCOL_PATH),
                        "--expected-protocol-sha256",
                        EXPECTED_PROTOCOL_SHA256,
                        "--output",
                        str(existing),
                    ]
                )
            self.assertIn("refuse overwrite", str(overwrite.exception))

    def test_result_path_is_separate(self) -> None:
        from research.lab.chuf_tvball_confirmation import (
            E1F_REPORT_RELATIVE,
            OUT_RELATIVE,
        )

        self.assertEqual(OUT_RELATIVE, "build/confirm-chuf-tvball")
        self.assertEqual(
            E1F_REPORT_RELATIVE,
            "build/compare-e1f-cost-conditioned-frontier/report.json",
        )
        self.assertNotEqual(OUT_RELATIVE, pathlib.Path(E1F_REPORT_RELATIVE).parent.as_posix())


class ContractIsolationTests(unittest.TestCase):
    def test_validation_path_does_not_read_outcomes(self) -> None:
        from research.lab.chuf_tvball_confirmation import (
            assert_validation_path_has_no_outcomes,
        )

        assert_validation_path_has_no_outcomes()

    def test_this_module_does_not_open_outcomes_or_fit(self) -> None:
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "load_outcomes",
            "load_public_pool",
            "evaluate_fresh_seed",
            "oof_chuf_heads",
            "run_confirmation",
        }
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        attrs = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)
        path_hits = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "outcomes" in node.value
            and node.value.endswith(".json")
        ]
        self.assertEqual(path_hits, [])

    def test_confirmation_output_not_created(self) -> None:
        """Phase-state independent: this module never invokes the run path.

        Confirmation artifacts may exist after Phase B. Absence of
        ``build/confirm-chuf-tvball`` is not a global invariant.
        """

        from research.lab.chuf_tvball_confirmation import (
            E1F_REPORT_RELATIVE,
            OUT_RELATIVE,
        )

        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "evaluate_fresh_seed",
            "load_public_pool",
            "oof_chuf_heads",
            "run_confirmation",
        }
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        attrs = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)
        self.assertEqual(OUT_RELATIVE, "build/confirm-chuf-tvball")
        self.assertEqual(
            E1F_REPORT_RELATIVE,
            "build/compare-e1f-cost-conditioned-frontier/report.json",
        )
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(E1F_REPORT_RELATIVE).parent.as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
