# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E28 batch-relative runaway guard contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e28-batch-relative-runaway.v1.json"
E28_PREFIX = "scrooge-e28-batch-relative-runaway-v1"
E28_CORE = "7deca711b20d3992bb1dd551a2600542f549601e1f220c4633f3fe303054fc6b"
E28_SEALED_SEEDS = (660580858, 2073661913, 1496370400, 224028423, 2098343477)
E27_SEEDS = (258662104, 1081333920, 1404762205, 999955836, 1043997100)


SHIPPED_RUNAWAY_SHARE = 0.06


def _brake_block(**overrides):
    """Shipped block with ``runaway_share`` dropped unless a test asks for it."""

    from ossp_router import budget_brake_router

    block = dict(budget_brake_router.load_bundled_artifact().budget_brake)
    block.pop("runaway_share", None)
    block.update(overrides)
    return block


def _artifact_value():
    return json.loads(
        (
            ROOT / "src" / "ossp_router" / "resources" / "budget-brake-router.v1.json"
        ).read_text(encoding="utf-8")
    )


class RunawayThresholdTest(unittest.TestCase):
    """The lever must be inert until the artifact opts in."""

    def test_absent_field_returns_the_frozen_absolute(self) -> None:
        from ossp_router import budget_brake_router

        block = _brake_block()
        self.assertNotIn("runaway_share", block)
        for light in (0.05, 1.0, 4.4178, 8.5764, 100.0):
            self.assertEqual(
                budget_brake_router.batch_runaway_threshold(block, light),
                float(block["runaway_absolute"]),
            )

    def test_share_is_never_looser_than_the_absolute(self) -> None:
        from ossp_router import budget_brake_router

        absolute = float(_brake_block()["runaway_absolute"])
        block = _brake_block(runaway_share=0.06)
        for light in (0.1, 1.0, 2.8588, 4.4178, 8.5764, 1000.0):
            self.assertLessEqual(
                budget_brake_router.batch_runaway_threshold(block, light), absolute
            )

    def test_share_binds_only_on_short_batches(self) -> None:
        from ossp_router import budget_brake_router

        absolute = float(_brake_block()["runaway_absolute"])
        block = _brake_block(runaway_share=0.06)
        crossover = absolute / 0.06
        self.assertLess(
            budget_brake_router.batch_runaway_threshold(block, crossover * 0.5),
            absolute,
        )
        self.assertEqual(
            budget_brake_router.batch_runaway_threshold(block, crossover * 2.0),
            absolute,
        )


class BrakeBlockValidationTest(unittest.TestCase):
    def test_optional_field_is_accepted_and_parsed(self) -> None:
        from ossp_router import budget_brake_router

        value = _artifact_value()
        value["budget_brake"]["runaway_share"] = SHIPPED_RUNAWAY_SHARE
        artifact = budget_brake_router.load_artifact_mapping(value)
        self.assertEqual(
            float(artifact.budget_brake["runaway_share"]), SHIPPED_RUNAWAY_SHARE
        )

    def test_block_without_the_field_still_validates(self) -> None:
        from ossp_router import budget_brake_router

        value = _artifact_value()
        value["budget_brake"].pop("runaway_share")
        artifact = budget_brake_router.load_artifact_mapping(value)
        self.assertNotIn("runaway_share", artifact.budget_brake)

    def test_out_of_range_share_is_rejected(self) -> None:
        from ossp_router import budget_brake_router
        from ossp_router.protocol import ProtocolError

        base = _artifact_value()
        for bad in (0.0, -0.1, 1.5, float("inf")):
            value = json.loads(json.dumps(base))
            value["budget_brake"]["runaway_share"] = bad
            with self.assertRaises(ProtocolError):
                budget_brake_router.load_artifact_mapping(value)

    def test_unknown_extra_field_is_still_rejected(self) -> None:
        from ossp_router import budget_brake_router
        from ossp_router.protocol import ProtocolError

        value = _artifact_value()
        value["budget_brake"]["not_a_real_field"] = 1
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(value)

    def test_missing_required_field_is_still_rejected(self) -> None:
        from ossp_router import budget_brake_router
        from ossp_router.protocol import ProtocolError

        value = _artifact_value()
        value["budget_brake"].pop("runaway_absolute")
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(value)


class PromotionUsesTheBatchThresholdTest(unittest.TestCase):
    """A synthetic batch where one increment is huge relative to its own light."""

    def _batch(self):
        absolute = float(_brake_block()["runaway_absolute"])
        # Four cheap rows; the first carries an increment just under the frozen
        # absolute but far above 6% of this batch's predicted light.
        light = 0.10
        costs = [
            (light, light, light + absolute * 0.9),
            (light, light, light + 0.001),
            (light, light, light + 0.001),
            (light, light, light + 0.001),
        ]
        parent = ("ax31",) * 4
        quality = (1.0, 0.9, 0.8, 0.7)
        families = ("word_problem",) * 4
        digests = tuple(f"{index:064x}" for index in range(4))
        return parent, quality, families, costs, digests

    def test_frozen_absolute_admits_the_runaway_row(self) -> None:
        from ossp_router import budget_brake_router

        parent, quality, families, costs, digests = self._batch()
        selected = budget_brake_router.promote_premium_brake(
            parent, quality, families, costs, digests, _brake_block()
        )
        self.assertEqual(selected[0], "axk1-think")

    def test_batch_relative_share_blocks_the_runaway_row(self) -> None:
        from ossp_router import budget_brake_router

        parent, quality, families, costs, digests = self._batch()
        selected = budget_brake_router.promote_premium_brake(
            parent,
            quality,
            families,
            costs,
            digests,
            _brake_block(runaway_share=0.06),
        )
        self.assertEqual(selected[0], "ax31")
        self.assertIn("axk1-think", selected[1:])

    def test_eligibility_helper_agrees_with_the_loop(self) -> None:
        from ossp_router import budget_brake_router

        parent, _quality, families, costs, _digests = self._batch()
        block = _brake_block(runaway_share=0.06)
        eligible = budget_brake_router.eligible_promotion_indices(
            parent, families, costs, block
        )
        self.assertNotIn(0, eligible)
        self.assertEqual(set(eligible), {1, 2, 3})


class E28SeedDerivationTest(unittest.TestCase):
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
        self.assertEqual(derived, E28_SEALED_SEEDS)
        self.assertEqual(
            tuple(int(seed) for seed in payload["fresh_seeds"]), E28_SEALED_SEEDS
        )
        self.assertEqual(str(derivation["core_sha256"]), E28_CORE)
        self.assertEqual(derivation["prefix"], E28_PREFIX)
        forbidden = {int(value) for value in derivation["forbidden_previous_seeds"]}
        self.assertTrue(set(E27_SEEDS).issubset(forbidden))
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
            derive_fresh_seeds(E28_PREFIX, E28_CORE, 5, list(E28_SEALED_SEEDS[:1]))


class E28ProtocolVerifyTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.e28_batch_relative_runaway import (
                EXPECTED_PROTOCOL_SHA256,
                ProtocolError,
                protocol_sha256,
                verify_protocol,
            )
        except ImportError:
            self.skipTest("research E28 stack is not installed")
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

    def test_gates_and_invariants_are_disjoint_and_nonempty(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        gates = set(payload["gates"])
        invariants = set(payload["invariants"])
        self.assertTrue(gates)
        self.assertTrue(invariants)
        self.assertFalse(gates & invariants)

    def test_e27_vacuous_claims_are_not_gates(self) -> None:
        """The three E27 gates that could never fail must not reappear as gates."""
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        gates = " ".join(payload["gates"]).lower()
        for banned in ("predicted_ratio", "conformal", "residual_isolated", "fast_balanced"):
            self.assertNotIn(banned, gates)

    def test_probe_shares_bracket_the_candidate(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        shares = [float(value) for value in payload["falsifiability_probe"]["shares"]]
        candidate = float(payload["thresholds"]["runaway_share"])
        self.assertEqual(candidate, SHIPPED_RUNAWAY_SHARE)
        self.assertNotIn(candidate, shares)
        self.assertTrue(any(value < candidate for value in shares))
        self.assertTrue(any(value > candidate for value in shares))

    def test_brake_ratio_and_count_cap_are_held_fixed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["thresholds"]["brake_ratio"], 3.8)
        self.assertEqual(payload["thresholds"]["count_cap"], 48)
        self.assertEqual(payload["thresholds"]["runaway_share"], 0.06)


class OverwriteAndExportTest(unittest.TestCase):
    def test_assemble_refuses_existing_output(self) -> None:
        try:
            from research.lab.e28_batch_relative_runaway import ProtocolError, assemble
        except ImportError:
            self.skipTest("research E28 stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            report = pathlib.Path(tmp) / "report.json"
            audit = pathlib.Path(tmp) / "audit.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(ProtocolError):
                assemble(payload, "0" * 64, output=report, audit_output=audit)

    def test_assemble_refuses_src_paths(self) -> None:
        try:
            from research.lab.e28_batch_relative_runaway import ProtocolError, assemble
        except ImportError:
            self.skipTest("research E28 stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        with self.assertRaises(ProtocolError):
            assemble(
                payload,
                "0" * 64,
                output=ROOT / "src" / "ossp_router" / "e28.json",
                audit_output=ROOT / "build" / "run-e28-batch-relative-runaway" / "a.json",
            )


if __name__ == "__main__":
    unittest.main()
