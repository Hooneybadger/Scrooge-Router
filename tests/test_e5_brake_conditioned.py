# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E5 brake-conditioned ranking contracts. Synthetic fixtures only.

Public outcomes are never opened. The sealed runner is not executed
through a successful public fit; the harness equivalence is proven on a
synthetic batch against the bundled artifacts.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "e5-brake-conditioned.v1.json"

E5_PREFIX = "scrooge-e5-brake-conditioned-v1"
E5_CORE = "1c4b8144378f55779aaf1cf3e78424abd749a93e3d4e33744ccda82304b7cbc9"
E5_SEALED_SEEDS = (763133369, 68726617, 1988695219, 929558432, 1924177588)


def _episode(episode_id: str, prompt: str):
    from ossp_router.protocol import Episode

    return Episode(episode_id=episode_id, prompt=prompt)


class E5SeedDerivationTest(unittest.TestCase):
    def test_sealed_seeds_match_fail_closed_derivation(self) -> None:
        try:
            from research.lab.e5_brake_conditioned import derive_fresh_seeds
        except ImportError:
            self.skipTest("research E5 stack is not installed")
        forbidden = [20260821, 20260825, 1524653244, 572878001]
        derived = derive_fresh_seeds(E5_PREFIX, E5_CORE, 5, forbidden)
        self.assertEqual(derived, E5_SEALED_SEEDS)

    def test_collision_fails_closed(self) -> None:
        try:
            from research.lab.e5_brake_conditioned import (
                ProtocolError,
                derive_fresh_seeds,
            )
        except ImportError:
            self.skipTest("research E5 stack is not installed")

        self.assertEqual(
            derive_fresh_seeds(E5_PREFIX, E5_CORE, 5, []),
            derive_fresh_seeds(E5_PREFIX, E5_CORE, 5, []),
        )
        with self.assertRaises(ProtocolError):
            derive_fresh_seeds(E5_PREFIX, E5_CORE, 5, list(E5_SEALED_SEEDS[:1]))
        with self.assertRaises(ProtocolError):
            derive_fresh_seeds(E5_PREFIX, E5_CORE, 5, list(E5_SEALED_SEEDS[-1:]))

    def test_protocol_file_agrees_with_derivation(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        derivation = payload["seed_derivation"]
        self.assertEqual(int(derivation["n"]), len(payload["fresh_seeds"]))
        self.assertEqual(
            tuple(int(seed) for seed in payload["fresh_seeds"]), E5_SEALED_SEEDS
        )
        self.assertEqual(str(derivation["core_sha256"]), E5_CORE)
        forbidden = {int(value) for value in derivation["forbidden_previous_seeds"]}
        self.assertFalse(forbidden & set(payload["fresh_seeds"]))
        self.assertEqual(
            derivation["prefix"], "scrooge-e5-brake-conditioned-v1"
        )


class E5ProtocolVerifyTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.e5_brake_conditioned import (
                ProtocolError,
                protocol_sha256,
                verify_protocol,
            )
        except ImportError:
            self.skipTest("research E5 stack is not installed")
        return ProtocolError, protocol_sha256, verify_protocol

    def test_protocol_verifies_against_canonical_sha(self) -> None:
        ProtocolError, protocol_sha256, verify_protocol = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        digest = protocol_sha256(payload)
        self.assertEqual(len(digest), 64)
        self.assertEqual(verify_protocol(payload, digest), digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, "0" * 64)

    def test_seed_drift_is_rejected(self) -> None:
        ProtocolError, protocol_sha256, verify_protocol = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        tampered = json.loads(json.dumps(payload))
        tampered["fresh_seeds"] = [1, 2, 3, 4, 5]
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, protocol_sha256(payload))

    def test_pin_drift_is_rejected(self) -> None:
        ProtocolError, protocol_sha256, verify_protocol = self._import()
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        identity = {
            "train_inputs_sha256": "0" * 64,
            "train_outcomes_sha256": "0" * 64,
            "dev_inputs_sha256": "0" * 64,
            "dev_outcomes_sha256": "0" * 64,
            "policy_sha256": "0" * 64,
        }
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, protocol_sha256(payload), pool_identity=identity)


class WeightedRidgeTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np

            from research.lab.e5_brake_conditioned import _weighted_ridge
        except ImportError:
            self.skipTest("numpy is not installed")
        return np, _weighted_ridge

    def test_exact_fit_without_penalty(self) -> None:
        np, _weighted_ridge = self._import()
        x = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = np.asarray([1.0, 3.0, 5.0, 7.0])
        weights = np.ones(4)
        coefficients = _weighted_ridge(x, y, weights, alpha=0.0)
        self.assertAlmostEqual(float(coefficients[0]), 1.0, places=12)
        self.assertAlmostEqual(float(coefficients[1]), 2.0, places=12)

    def test_weights_change_the_fit(self) -> None:
        np, _weighted_ridge = self._import()
        x = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = np.asarray([0.0, 0.0, 0.0, 10.0])
        uniform = _weighted_ridge(x, y, np.ones(4), alpha=0.0)
        tilted = _weighted_ridge(x, y, np.asarray([1.0, 1.0, 1.0, 50.0]), alpha=0.0)
        self.assertGreater(float(tilted[1]), float(uniform[1]))
        for value in tilted:
            self.assertTrue(math.isfinite(float(value)))


class PromoteInjectionTest(unittest.TestCase):
    """The rank key must be injectable through promote_premium_brake alone."""

    def _block(self):
        from ossp_router import budget_brake_router

        block = dict(budget_brake_router.load_bundled_artifact().budget_brake)
        block["denylist_families"] = []
        # Both halves of the runaway guard have to stand down for this fixture:
        # the frozen absolute and the batch-relative share.
        block["runaway_absolute"] = 100.0
        block.pop("runaway_share", None)
        block["brake_ratio"] = 3.5
        block["count_cap"] = 48
        return block

    def test_density_and_uplift_orders_disagree_as_designed(self) -> None:
        from ossp_router import budget_brake_router

        parent = ["ax31", "ax31", "ax31"]
        families = ["other", "other", "other"]
        digests = ["a", "b", "c"]
        costs = [[1.0, 2.0, 6.0], [1.0, 2.0, 3.0], [1.0, 2.0, 6.4]]
        uplift = [0.9, 0.8, 0.7]
        increments = [4.0, 1.0, 4.4]

        raw = budget_brake_router.promote_premium_brake(
            parent, uplift, families, costs, digests, self._block()
        )
        density = [
            quality / max(increment, 1e-12)
            for quality, increment in zip(uplift, increments)
        ]
        by_density = budget_brake_router.promote_premium_brake(
            parent, density, families, costs, digests, self._block()
        )
        # Budget: parent ax31 spend 6 of 3.5x3=10.5 leaves 4.5 for promotions.
        # Uplift order buys item 0 (+4.0) then cannot afford item 1 (+1.0 would
        # reach 11 > 10.5) nor item 2. Density order buys item 1 (+1.0) first
        # and then cannot afford item 0.
        self.assertEqual(raw, ("axk1-think", "ax31", "ax31"))
        self.assertEqual(by_density, ("ax31", "axk1-think", "ax31"))

    def test_composed_path_reproduces_make_submission(self) -> None:
        from ossp_router import budget_brake_router
        from ossp_router.protocol import InputBatch, load_bundled_policy

        prompts = []
        for index in range(60):
            prompts.append(
                f"Question {index}: compute {index} + {index * 2} and "
                "explain each step of the derivation carefully."
            )
            if index % 5 == 0:
                prompts[-1] += "\n\n```python\ndef g(x):\n    return x + 1\n```"
        episodes = tuple(
            _episode(f"toy-{position}", prompt)
            for position, prompt in enumerate(prompts)
        )
        batch = InputBatch(
            schema_version=1, challenge_id="toy", split="public", episodes=episodes
        )
        policy = load_bundled_policy()
        brake = budget_brake_router.load_bundled_artifact()
        rows = [
            budget_brake_router.premium_prediction_row(episode, policy, brake)
            for episode in episodes
        ]
        runtime = budget_brake_router.make_submission(batch, policy, brake, "premium")
        runtime_models = tuple(
            decision.model_id for decision in runtime.submission.decisions
        )
        composed = budget_brake_router.select_premium_with_brake(
            batch, policy, brake, rows
        )
        self.assertEqual(runtime_models, composed)


class DecisionCoreTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.e5_brake_conditioned import (
                decision_core_payload,
                decision_core_sha256,
            )
        except ImportError:
            self.skipTest("research E5 stack is not installed")
        return decision_core_payload, decision_core_sha256

    def test_core_hash_is_deterministic_and_sensitive(self) -> None:
        _, decision_core_sha256 = self._import()

        def report(decision: str) -> dict:
            return {
                "audit": {"n_rows": 1, "relative_path": "x", "sha256": "a"},
                "candidate_primary": "p",
                "constants": {},
                "decision": decision,
                "decision_reason": "r",
                "equivalence": {"matched": True},
                "experiment": "e5-brake-conditioned-v1",
                "fold_seeds": [1],
                "gate": {"passed": True},
                "pin_dev_replay": {"matched": True},
                "protocol_sha256": "b" * 64,
                "report_type": "t",
                "schema_version": 1,
                "seed_results": {},
                "thresholds": {"mean_delta_min": 0.002},
            }

        first = decision_core_sha256(report("pass"))
        second = decision_core_sha256(report("pass"))
        other = decision_core_sha256(report("fail"))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
