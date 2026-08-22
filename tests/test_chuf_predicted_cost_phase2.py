# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""CHUF predicted-cost Phase 2 protocol contracts.

Synthetic fixtures only. Inputs may be hashed for epsilon. Public
outcomes are not opened and Phase 2 is not fitted here.
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


EXPLICIT_RISK_SEEDS = (
    1961852001,
    1797397368,
    1763238305,
    999558656,
    1988874908,
    305408514,
    400818725,
    116341498,
    1592039285,
    215679302,
    1124696458,
    1980863820,
)


class SeedDerivationTests(unittest.TestCase):
    def test_derivation_matches_explicit_list(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            CONFIRMATION_CORE_SHA256,
            derive_fresh_risk_seeds,
        )

        self.assertEqual(
            CONFIRMATION_CORE_SHA256,
            "2acba355a7c6863c4ae1971ba03e135041efcf0f9def9400135262734a569e6d",
        )
        self.assertEqual(derive_fresh_risk_seeds(), EXPLICIT_RISK_SEEDS)

    def test_unique_and_no_blocked_overlap(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            blocked_seeds,
            derive_fresh_risk_seeds,
        )

        seeds = derive_fresh_risk_seeds()
        self.assertEqual(len(seeds), 12)
        self.assertEqual(len(set(seeds)), 12)
        self.assertFalse(set(seeds) & set(blocked_seeds()))
        self.assertEqual(len(blocked_seeds()), 17)

    def test_blocked_overlap_fails_closed(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import derive_fresh_risk_seeds

        first = derive_fresh_risk_seeds()[0]
        with self.assertRaises(RuntimeError) as caught:
            derive_fresh_risk_seeds(blocked=(first,))
        self.assertIn("fail closed", str(caught.exception))


class EpsilonAndTvTests(unittest.TestCase):
    def test_epsilon_from_inputs_matches_pin(self) -> None:
        _require_public_inputs(self)
        from research.lab.chuf_predicted_cost_phase2 import epsilon_from_input_paths

        self.assertEqual(epsilon_from_input_paths(), 0.014204545454545449)

    def test_tv_cost_vertices_are_91_and_match_transport(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            REQUIRED_FAMILIES,
            tv_cost_vertices,
            tv_cost_worst,
        )

        center = {name: 0.1 for name in REQUIRED_FAMILIES}
        spend = {name: 2.0 for name in REQUIRED_FAMILIES}
        spend["other"] = 4.0
        light = {name: 1.0 for name in REQUIRED_FAMILIES}
        points = tv_cost_vertices(center, spend, light, 0.014204545454545449)
        self.assertEqual(len(points), 91)
        worst = tv_cost_worst(center, spend, light, 0.014204545454545449)
        self.assertEqual(worst, max(points))
        tiny_center = {"a": 0.5, "b": 0.3, "c": 0.2}
        tiny_spend = {"a": 1.0, "b": 2.0, "c": 3.0}
        tiny_light = {"a": 1.0, "b": 1.0, "c": 1.0}
        tiny = tv_cost_vertices(tiny_center, tiny_spend, tiny_light, 0.1)
        self.assertEqual(len(tiny), 7)
        moved = dict(tiny_center)
        moved["a"] -= 0.1
        moved["c"] += 0.1
        expected = sum(moved[name] * tiny_spend[name] for name in moved)
        self.assertIn(expected, tiny)


class GateBoundaryTests(unittest.TestCase):
    def _row(self, seed: int, **overrides: object) -> dict:
        row = {
            "bootstrap_q999_under_95_ok": True,
            "fast_balanced_k1_zero": True,
            "fold_seed": seed,
            "fold_slice_hard_caps_ok": True,
            "official_delta": 0.0004,
            "parent_identical": True,
            "pooled_hard_caps_ok": True,
            "pooled_ratio_under_95_ok": True,
            "pred_qa_identical": True,
            "premium_delta": 0.0012,
            "premium_k1_count": 48,
            "tv_cost_under_official_ok": True,
            "tv_quality_worst": -0.003,
        }
        row.update(overrides)
        return row

    def _rows(self, **overrides: object) -> list[dict]:
        return [self._row(seed, **overrides) for seed in EXPLICIT_RISK_SEEDS]

    def test_pass_fail_and_no_valid_reference(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            FAIL_DECISION,
            NO_REF_DECISION,
            PASS_DECISION,
            phase2_gate,
        )

        passed = phase2_gate(self._rows(), self._rows())
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["decision"], PASS_DECISION)
        comparator = self._rows()
        comparator[0]["pooled_hard_caps_ok"] = False
        no_ref = phase2_gate(comparator, self._rows())
        self.assertEqual(no_ref["decision"], NO_REF_DECISION)
        candidate = self._rows()
        candidate[2]["bootstrap_q999_under_95_ok"] = False
        failed = phase2_gate(self._rows(), candidate)
        self.assertEqual(failed["decision"], FAIL_DECISION)
        low_prem = phase2_gate(self._rows(), self._rows(premium_delta=0.0009))
        self.assertEqual(low_prem["decision"], FAIL_DECISION)
        tv_fail = self._rows()
        tv_fail[1]["tv_quality_worst"] = -0.0031
        self.assertEqual(
            phase2_gate(self._rows(), tv_fail)["decision"], FAIL_DECISION
        )

    def test_dirac_and_k1_identity_boundaries(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            FAIL_DECISION,
            PASS_DECISION,
            phase2_gate,
        )

        self.assertEqual(phase2_gate(self._rows(), self._rows())["decision"], PASS_DECISION)
        too_many = self._rows()
        too_many[0]["premium_k1_count"] = 49
        self.assertEqual(
            phase2_gate(self._rows(), too_many)["decision"], FAIL_DECISION
        )
        k1 = self._rows()
        k1[3]["fast_balanced_k1_zero"] = False
        self.assertEqual(phase2_gate(self._rows(), k1)["decision"], FAIL_DECISION)


class ArtifactAndThresholdTests(unittest.TestCase):
    def test_artifact_constants_match_live_snapshot(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            FAMILY_OTHER_MULTIPLIER,
            FAST_CAP,
            PREMIUM_K1_MAX,
            RUNAWAY_FRACTION,
            architecture_snapshot,
            live_artifact_snapshot,
        )

        live = live_artifact_snapshot()
        snap = architecture_snapshot()
        self.assertEqual(snap["artifact"], live)
        self.assertEqual(live["fast_cap"], FAST_CAP)
        self.assertEqual(live["runaway_fraction"], RUNAWAY_FRACTION)
        self.assertEqual(live["multipliers"]["other"], FAMILY_OTHER_MULTIPLIER)
        self.assertEqual(live["count_cap"], PREMIUM_K1_MAX)
        self.assertFalse(live["runaway_light_fraction_used_in_eligibility"])
        self.assertFalse(snap["allocator"]["fold_local_rebuy"])

    def test_thresholds_exclude_dev_and_champion_abs(self) -> None:
        _require_public_inputs(self)
        from research.lab.chuf_predicted_cost_phase2 import build_canonical_protocol

        thresholds = build_canonical_protocol()["thresholds"]
        encoded = str(thresholds)
        self.assertNotIn("0.669", encoded)
        self.assertNotIn("0.69", encoded)
        self.assertNotIn(0.669517045455, thresholds.values())
        self.assertNotIn(0.69, thresholds.values())
        self.assertNotIn(0.690, thresholds.values())


class ProtocolHashTests(unittest.TestCase):
    def test_canonical_hash_is_deterministic(self) -> None:
        _require_public_inputs(self)
        from research.lab.chuf_predicted_cost_phase2 import (
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
        self.assertNotIn("generated_at", sealed)
        self.assertEqual(
            verify_protocol(sealed, EXPECTED_PROTOCOL_SHA256),
            EXPECTED_PROTOCOL_SHA256,
        )

    def test_tamper_rejects(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
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
    def test_wrong_expected_sha_isolated_from_overwrite(self) -> None:
        from research.experiments.run_chuf_predicted_cost_phase2 import main
        from research.lab.chuf_predicted_cost_phase2 import (
            EXPECTED_PROTOCOL_SHA256,
            PROTOCOL_PATH,
            architecture_snapshot,
            assert_live_architecture,
            load_protocol,
            verify_protocol,
        )

        with tempfile.TemporaryDirectory() as folder:
            isolated = pathlib.Path(folder)
            with self.assertRaises(RuntimeError) as caught:
                main(
                    [
                        "--protocol",
                        str(PROTOCOL_PATH),
                        "--expected-protocol-sha256",
                        "0" * 64,
                        "--output",
                        str(isolated / "report.json"),
                        "--audit-output",
                        str(isolated / "episode-audit.json"),
                    ]
                )
            self.assertIn("protocol sha mismatch", str(caught.exception))
            self.assertFalse((isolated / "report.json").exists())
        dirty = architecture_snapshot()
        dirty["artifact"] = dict(dirty["artifact"])
        dirty["artifact"]["fast_cap"] = 9.9
        with self.assertRaises(RuntimeError):
            assert_live_architecture(
                {
                    "architecture": dirty,
                    "pins": load_protocol(PROTOCOL_PATH)["pins"],
                }
            )
        if _PUBLIC_TRAIN_INPUTS.is_file():
            verify_protocol(load_protocol(PROTOCOL_PATH), EXPECTED_PROTOCOL_SHA256)

    def test_overwrite_and_foreign_paths_refused(self) -> None:
        from research.experiments.run_chuf_predicted_cost_phase2 import main
        from research.lab.chuf_predicted_cost_phase2 import (
            EXPECTED_PROTOCOL_SHA256,
            PROTOCOL_PATH,
        )

        e1f = ROOT / "build" / "compare-e1f-cost-conditioned-frontier" / "report.json"
        with self.assertRaises(RuntimeError) as caught:
            main(
                [
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--expected-protocol-sha256",
                    EXPECTED_PROTOCOL_SHA256,
                    "--output",
                    str(e1f),
                ]
            )
        self.assertIn("must not write the E1F", str(caught.exception))
        confirm = ROOT / "build" / "confirm-chuf-tvball" / "report.json"
        with self.assertRaises(RuntimeError) as confirm_caught:
            main(
                [
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--expected-protocol-sha256",
                    EXPECTED_PROTOCOL_SHA256,
                    "--output",
                    str(confirm),
                ]
            )
        self.assertIn("must not write the confirmation", str(confirm_caught.exception))
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


class IsolationTests(unittest.TestCase):
    def test_validation_path_forbids_outcomes_and_run(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            assert_no_fold_local_rebuy,
            assert_validation_path_has_no_outcomes,
        )

        assert_validation_path_has_no_outcomes()
        assert_no_fold_local_rebuy()

    def test_this_module_does_not_open_outcomes_or_fit(self) -> None:
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "evaluate_seed",
            "load_outcomes",
            "load_public_pool",
            "oof_chuf_heads",
            "oof_cost_surfaces",
            "run_phase2",
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)

    def test_phase2_output_absent_and_paths_separate(self) -> None:
        from research.lab.chuf_predicted_cost_phase2 import (
            CONFIRM_REPORT_RELATIVE,
            E1F_REPORT_RELATIVE,
            OUT_RELATIVE,
        )

        self.assertEqual(OUT_RELATIVE, "build/phase2-chuf-predicted-cost")
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(E1F_REPORT_RELATIVE).parent.as_posix(),
        )
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(CONFIRM_REPORT_RELATIVE).parent.as_posix(),
        )
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {"run_phase2", "evaluate_seed", "load_public_pool"}
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)


if __name__ == "__main__":
    unittest.main()
