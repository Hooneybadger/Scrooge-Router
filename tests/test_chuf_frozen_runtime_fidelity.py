# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""CHUF frozen-runtime fidelity protocol contracts.

Synthetic fixtures only. Inputs may be hashed for epsilon. Public
outcomes are not opened and the fidelity runner is not executed here.
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


def _require_live_matches_sealed_architecture(test: unittest.TestCase) -> None:
    from research.lab.chuf_frozen_runtime_fidelity import (
        PROTOCOL_PATH,
        architecture_snapshot,
        load_protocol,
    )

    if load_protocol(PROTOCOL_PATH).get("architecture") != architecture_snapshot():
        test.skipTest("CHUF protocol is sealed against a previous runtime artifact")


_require_research_stack()


EXPLICIT_FIDELITY_SEEDS = (
    1043203741,
    1783423358,
    511394098,
    1329561813,
    1860797546,
    2146174231,
    1250738729,
    1773845641,
    578057546,
    1452987153,
    1985170374,
    1303320403,
)


class SeedDerivationTests(unittest.TestCase):
    def test_derivation_matches_explicit_list(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            PHASE2_CORE_SHA256,
            derive_fresh_fidelity_seeds,
        )

        self.assertEqual(
            PHASE2_CORE_SHA256,
            "b84cc866d24fa36b974abfe44bbe7dbfd581c7555465fc4a60f635767a8e7edd",
        )
        self.assertEqual(derive_fresh_fidelity_seeds(), EXPLICIT_FIDELITY_SEEDS)

    def test_unique_and_no_blocked_overlap(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            blocked_seeds,
            derive_fresh_fidelity_seeds,
        )
        from research.lab.chuf_predicted_cost_phase2 import EXPLICIT_RISK_SEEDS
        from research.lab.chuf_tvball_confirmation import OLD_SEEDS, derive_fresh_seeds

        seeds = derive_fresh_fidelity_seeds()
        blocked = blocked_seeds()
        self.assertEqual(len(seeds), 12)
        self.assertEqual(len(set(seeds)), 12)
        self.assertFalse(set(seeds) & set(blocked))
        self.assertEqual(len(blocked), 29)
        self.assertEqual(
            set(blocked),
            set(OLD_SEEDS) | set(derive_fresh_seeds()) | set(EXPLICIT_RISK_SEEDS),
        )

    def test_blocked_overlap_fails_closed(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import derive_fresh_fidelity_seeds

        first = derive_fresh_fidelity_seeds()[0]
        with self.assertRaises(RuntimeError) as caught:
            derive_fresh_fidelity_seeds(blocked=(first,))
        self.assertIn("fail closed", str(caught.exception))


class ArtifactAndPinTests(unittest.TestCase):
    def test_live_artifact_snapshot_matches_architecture(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            FAMILY_OTHER_MULTIPLIER,
            PREMIUM_K1_MAX,
            architecture_snapshot,
        )
        from research.lab.chuf_predicted_cost_phase2 import (
            FAST_CAP,
            RUNAWAY_FRACTION,
            live_artifact_snapshot,
        )

        live = live_artifact_snapshot()
        snap = architecture_snapshot()
        self.assertEqual(snap["artifact"], live)
        self.assertEqual(live["fast_cap"], FAST_CAP)
        self.assertEqual(live["runaway_fraction"], RUNAWAY_FRACTION)
        self.assertEqual(live["multipliers"]["other"], FAMILY_OTHER_MULTIPLIER)
        self.assertEqual(live["count_cap"], PREMIUM_K1_MAX)
        self.assertFalse(snap["allocator"]["e2_surfaces_in_allocator"])
        self.assertFalse(snap["allocator"]["chuf_r_frozen_refit"])
        self.assertFalse(snap["allocator"]["fold_local_rebuy"])
        self.assertFalse(snap["allocator"]["pooled_public_batch"])
        self.assertTrue(snap["allocator"]["split_local_batch"])

    def test_comparator_pins_are_reproduction_not_thresholds(self) -> None:
        _require_public_inputs(self)
        from research.lab.chuf_frozen_runtime_fidelity import (
            COMPARATOR_PINS,
            build_canonical_protocol,
        )

        protocol = build_canonical_protocol()
        thresholds = protocol["thresholds"]
        repro = protocol["comparator_reproduction"]
        encoded = str(thresholds)
        self.assertNotIn("0.669517", encoded)
        self.assertNotIn("0.658636", encoded)
        self.assertNotIn("0.69", encoded)
        self.assertEqual(repro["dev"]["official_final_score"], "0.669517045455")
        self.assertEqual(repro["train"]["official_final_score"], "0.658636363636")
        self.assertEqual(repro["dev"]["n_k1"], 16)
        self.assertEqual(repro["train"]["n_k1"], 29)
        self.assertIn("not quality-gate", repro["note"])
        self.assertEqual(COMPARATOR_PINS["dev"]["ratios"]["fast"], "1.093011852072")
        self.assertNotIn("0.669517045455", thresholds.values())
        self.assertNotIn("0.658636363636", thresholds.values())
        self.assertNotIn(0.69, thresholds.values())


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_uses_exact_fractions(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            DEV_OFFICIAL_DELTA_MIN,
            TRAIN_OFFICIAL_DELTA_MIN,
            WEIGHTED_OFFICIAL_DELTA_MIN,
            quality_thresholds,
            weighted_official_delta,
        )

        thresholds = quality_thresholds()
        self.assertEqual(TRAIN_OFFICIAL_DELTA_MIN, 3 / 17600)
        self.assertEqual(DEV_OFFICIAL_DELTA_MIN, 3 / 8800)
        self.assertEqual(WEIGHTED_OFFICIAL_DELTA_MIN, 3 / 13200)
        self.assertEqual(thresholds["train_official_delta"], 3 / 17600)
        self.assertEqual(thresholds["dev_official_delta"], 3 / 8800)
        self.assertEqual(thresholds["weighted_official_delta"], 3 / 13200)
        self.assertEqual(thresholds["train_official_delta_fraction"], [3, 17600])
        self.assertEqual(thresholds["dev_official_delta_fraction"], [3, 8800])
        self.assertEqual(thresholds["weighted_official_delta_fraction"], [3, 13200])
        self.assertFalse(thresholds["mean_only_exemption"])
        self.assertGreaterEqual(
            weighted_official_delta(3 / 17600, 3 / 8800),
            3 / 13200,
        )

    def _safety(self) -> dict:
        return {
            "bootstrap_q999_under_95_ok": True,
            "pooled_hard_caps_ok": True,
            "pooled_ratio_under_95_ok": True,
            "tv_cost_under_official_ok": True,
        }

    def _row(self, seed: int, **overrides: object) -> dict:
        row = {
            "balanced_identical": True,
            "dev": self._safety(),
            "dev_official_delta": 3 / 8800,
            "dev_tv_quality_worst": -0.003,
            "fast_balanced_k1_zero": True,
            "fast_identical": True,
            "fold_seed": seed,
            "parent_identical": True,
            "premium_k1_count": 16,
            "train": self._safety(),
            "train_official_delta": 3 / 17600,
            "train_tv_quality_worst": -0.003,
        }
        row.update(overrides)
        return row

    def _rows(self, **overrides: object) -> list[dict]:
        return [self._row(seed, **overrides) for seed in EXPLICIT_FIDELITY_SEEDS]

    def _comparator(self, **overrides: object) -> dict:
        row = {
            "dev": self._safety(),
            "pins_reproduced": True,
            "train": self._safety(),
        }
        row.update(overrides)
        return row

    def test_pass_fail_and_no_valid_reference(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            FAIL_DECISION,
            NO_REF_DECISION,
            PASS_DECISION,
            fidelity_gate,
        )

        passed = fidelity_gate(self._comparator(), self._rows())
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["decision"], PASS_DECISION)
        self.assertFalse(passed["mean_only_exemption"])
        pin_fail = fidelity_gate(
            self._comparator(pins_reproduced=False), self._rows()
        )
        self.assertEqual(pin_fail["decision"], NO_REF_DECISION)
        safety = self._comparator()
        safety["dev"] = dict(safety["dev"], bootstrap_q999_under_95_ok=False)
        self.assertEqual(
            fidelity_gate(safety, self._rows())["decision"], NO_REF_DECISION
        )
        candidate = self._rows()
        candidate[2]["train"] = dict(
            candidate[2]["train"], pooled_hard_caps_ok=False
        )
        self.assertEqual(
            fidelity_gate(self._comparator(), candidate)["decision"], FAIL_DECISION
        )

    def test_every_seed_quality_no_mean_exemption(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            FAIL_DECISION,
            PASS_DECISION,
            fidelity_gate,
        )

        self.assertEqual(
            fidelity_gate(self._comparator(), self._rows())["decision"],
            PASS_DECISION,
        )
        low = self._rows()
        low[0]["train_official_delta"] = (3 / 17600) - 1e-18
        self.assertEqual(
            fidelity_gate(self._comparator(), low)["decision"], FAIL_DECISION
        )
        low_dev = self._rows()
        low_dev[5]["dev_official_delta"] = (3 / 8800) * 0.5
        self.assertEqual(
            fidelity_gate(self._comparator(), low_dev)["decision"], FAIL_DECISION
        )
        tv_fail = self._rows()
        tv_fail[11]["dev_tv_quality_worst"] = -0.0031
        self.assertEqual(
            fidelity_gate(self._comparator(), tv_fail)["decision"], FAIL_DECISION
        )

    def test_identity_and_k1_boundaries(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            FAIL_DECISION,
            fidelity_gate,
        )

        parent = self._rows()
        parent[1]["parent_identical"] = False
        self.assertEqual(
            fidelity_gate(self._comparator(), parent)["decision"], FAIL_DECISION
        )
        k1 = self._rows()
        k1[0]["premium_k1_count"] = 49
        self.assertEqual(
            fidelity_gate(self._comparator(), k1)["decision"], FAIL_DECISION
        )
        fast = self._rows()
        fast[3]["fast_identical"] = False
        self.assertEqual(
            fidelity_gate(self._comparator(), fast)["decision"], FAIL_DECISION
        )


class ProtocolHashTests(unittest.TestCase):
    def test_canonical_hash_is_deterministic(self) -> None:
        _require_public_inputs(self)
        _require_live_matches_sealed_architecture(self)
        from research.lab.chuf_frozen_runtime_fidelity import (
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
        self.assertTrue(sealed["split_local_batch"])
        self.assertFalse(sealed["pooled_public_batch"])
        self.assertFalse(sealed["e2_surfaces_in_allocator"])
        self.assertFalse(sealed["chuf_r_frozen_refit"])
        self.assertFalse(sealed["fold_local_rebuy"])
        self.assertTrue(sealed["stress"]["fold_slice_diagnostic_only"])
        self.assertEqual(sealed["stress"]["bootstrap_draws"], 200)
        self.assertEqual(sealed["stress"]["bootstrap_seed"], 557209147)
        self.assertEqual(sealed["epsilon"], 0.014204545454545449)

    def test_tamper_rejects(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
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
        from research.experiments.run_chuf_frozen_runtime_fidelity import main
        from research.lab.chuf_frozen_runtime_fidelity import (
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
        dirty["allocator"] = dict(dirty["allocator"])
        dirty["allocator"]["e2_surfaces_in_allocator"] = True
        with self.assertRaises(RuntimeError):
            assert_live_architecture(
                {
                    "architecture": dirty,
                    "chuf_r_frozen_refit": False,
                    "e2_surfaces_in_allocator": False,
                    "fold_local_rebuy": False,
                    "pins": load_protocol(PROTOCOL_PATH)["pins"],
                    "pooled_public_batch": False,
                    "split_local_batch": True,
                }
            )
        if _PUBLIC_TRAIN_INPUTS.is_file():
            sealed = load_protocol(PROTOCOL_PATH)
            if sealed.get("architecture") == architecture_snapshot():
                verify_protocol(sealed, EXPECTED_PROTOCOL_SHA256)

    def test_overwrite_and_foreign_paths_refused(self) -> None:
        from research.experiments.run_chuf_frozen_runtime_fidelity import main
        from research.lab.chuf_frozen_runtime_fidelity import (
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
        phase2 = ROOT / "build" / "phase2-chuf-predicted-cost" / "report.json"
        with self.assertRaises(RuntimeError) as phase2_caught:
            main(
                [
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--expected-protocol-sha256",
                    EXPECTED_PROTOCOL_SHA256,
                    "--output",
                    str(phase2),
                ]
            )
        self.assertIn("must not write the phase2", str(phase2_caught.exception))
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
        from research.lab.chuf_frozen_runtime_fidelity import (
            assert_allocator_has_no_e2_surfaces,
            assert_validation_path_has_no_outcomes,
        )

        assert_validation_path_has_no_outcomes()
        assert_allocator_has_no_e2_surfaces()

    def test_this_module_does_not_open_outcomes_or_fit(self) -> None:
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "evaluate_seed",
            "load_outcomes",
            "load_public_pool",
            "load_split_pool",
            "oof_chuf_heads",
            "oof_cost_surfaces",
            "run_fidelity",
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)

    def test_output_path_separate_and_absent_in_phase_a(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import (
            CONFIRM_REPORT_RELATIVE,
            E1F_REPORT_RELATIVE,
            OUT_RELATIVE,
            PHASE2_REPORT_RELATIVE,
        )

        self.assertEqual(OUT_RELATIVE, "build/frozen-runtime-fidelity")
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(E1F_REPORT_RELATIVE).parent.as_posix(),
        )
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(CONFIRM_REPORT_RELATIVE).parent.as_posix(),
        )
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(PHASE2_REPORT_RELATIVE).parent.as_posix(),
        )
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "run_fidelity",
            "evaluate_seed",
            "load_public_pool",
            "load_split_pool",
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)


if __name__ == "__main__":
    unittest.main()
