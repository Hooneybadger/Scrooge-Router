# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import unittest

from ossp_router import distributional_router
from ossp_router.protocol import Episode


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from research.lab import distributional_knapsack as knapsack
except ImportError:
    raise unittest.SkipTest("numpy / research stack is not installed")


class DistributionalFeatureContractTest(unittest.TestCase):
    def test_supervised_vocabulary_is_explicit_and_deterministic(self) -> None:
        texts = tuple(
            f"shared prompt cohort-{index % 3} signal-{index // 2}"
            for index in range(12)
        )
        targets = np.column_stack(
            (
                np.linspace(-1.0, 1.0, len(texts)),
                np.linspace(1.0, -1.0, len(texts)),
                np.arange(len(texts), dtype=np.float64) % 3,
            )
        )
        arguments = {
            "size": 12,
            "min_document_frequency": 2,
            "max_document_fraction": 1.0,
        }
        first = knapsack.select_vocabulary(texts, targets, **arguments)
        second = knapsack.select_vocabulary(texts, targets, **arguments)
        observed = set().union(*(knapsack.lexical_terms(text) for text in texts))

        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(set(first) <= observed)
        self.assertTrue(
            all(term.startswith(("w:", "b:", "p:", "s:")) for term in first)
        )

        episodes = tuple(
            Episode(f"episode-{index}", prompt=text)
            for index, text in enumerate(texts)
        )
        matrix = knapsack.feature_matrix(episodes, first)
        self.assertEqual(
            (len(texts), len(knapsack.STRUCTURAL_FEATURE_NAMES) + len(first)),
            matrix.shape,
        )
        self.assertEqual(np.float32, matrix.dtype)

    def test_batch_risk_features_match_standard_library_replica(self) -> None:
        rng = np.random.default_rng(20260830)
        count = 40
        quality = rng.uniform(0.2, 0.95, size=(count, 3))
        light = rng.uniform(0.001, 0.01, size=count)
        mean = np.column_stack(
            (
                light,
                light + rng.uniform(0.001, 0.02, size=count),
                light + rng.uniform(0.01, 0.08, size=count),
            )
        )
        upper = mean + rng.uniform(0.0, 0.03, size=(count, 3))
        structural = rng.normal(
            size=(count, len(knapsack.STRUCTURAL_FEATURE_NAMES))
        ).astype(np.float32)
        families = tuple(
            knapsack.FAMILY_NAMES[index % len(knapsack.FAMILY_NAMES)]
            for index in range(count)
        )
        tie_keys = tuple(f"content-{index // 2}" for index in range(count))
        reference = tuple(1.0 / len(knapsack.FAMILY_NAMES) for _ in knapsack.FAMILY_NAMES)
        lab_calibration = knapsack.FamilyCalibration(
            knapsack.FAMILY_NAMES,
            reference,
            np.ones((len(knapsack.FAMILY_NAMES), 3)),
            np.ones((len(knapsack.FAMILY_NAMES), 3)),
        )
        runtime_calibration = distributional_router.FamilyCalibration(
            distributional_router.FAMILY_NAMES,
            reference,
            tuple((1.0, 1.0, 1.0) for _ in knapsack.FAMILY_NAMES),
            tuple((1.0, 1.0, 1.0) for _ in knapsack.FAMILY_NAMES),
        )
        predictions = knapsack.DistributionalPredictions(quality, mean, upper)

        lab = knapsack.batch_risk_features(
            predictions,
            structural,
            families,
            tie_keys,
            lab_calibration,
        )
        runtime = distributional_router._batch_features(
            quality,
            mean,
            upper,
            structural,
            families,
            tie_keys,
            runtime_calibration,
        )

        self.assertEqual(len(knapsack.BATCH_RISK_FEATURE_NAMES), len(runtime))
        # numpy uses a pairwise reducer while the serving replica uses fsum;
        # both accumulate the same frozen float32 item features in float64.
        np.testing.assert_allclose(lab, runtime, rtol=0.0, atol=2e-8)


class CanonicalAllocatorTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260830)
        count = 79
        light_quality = rng.uniform(0.2, 0.65, size=count)
        ax_uplift = rng.uniform(0.01, 0.25, size=count)
        k1_uplift = rng.uniform(0.0, 0.20, size=count)
        self.quality = np.column_stack(
            (
                light_quality,
                light_quality + ax_uplift,
                light_quality + ax_uplift + k1_uplift,
            )
        )
        light_cost = rng.uniform(0.8, 1.2, size=count)
        ax_cost = light_cost + rng.uniform(0.2, 1.8, size=count)
        k1_cost = ax_cost + rng.uniform(0.5, 4.0, size=count)
        self.cost = np.column_stack((light_cost, ax_cost, k1_cost))
        self.light = light_cost.copy()
        self.keys = tuple(f"key-{index:04d}" for index in range(count))

    def _allocate(self, target: float) -> np.ndarray:
        return knapsack.allocate_priority_queue(
            self.quality,
            self.cost,
            self.light,
            budget_multiplier=3.0,
            target_fraction=target,
            tie_keys=self.keys,
        )

    def test_smaller_envelopes_are_canonical_prefixes(self) -> None:
        small = self._allocate(0.42)
        medium = self._allocate(0.62)
        large = self._allocate(0.86)

        self.assertTrue(np.all(small <= medium))
        self.assertTrue(np.all(medium <= large))
        self.assertGreater(np.count_nonzero(large), np.count_nonzero(small))

    def test_input_order_does_not_change_content_choices(self) -> None:
        expected = self._allocate(0.62)
        permutation = np.random.default_rng(17).permutation(len(self.keys))
        actual = knapsack.allocate_priority_queue(
            self.quality[permutation],
            self.cost[permutation],
            self.light[permutation],
            budget_multiplier=3.0,
            target_fraction=0.62,
            tie_keys=tuple(self.keys[index] for index in permutation),
        )
        restored = np.empty_like(actual)
        restored[permutation] = actual
        np.testing.assert_array_equal(expected, restored)

    def test_duplicate_content_is_an_atomic_upgrade_group(self) -> None:
        quality = np.repeat(self.quality[:8], 2, axis=0)
        cost = np.repeat(self.cost[:8], 2, axis=0)
        light = np.repeat(self.light[:8], 2)
        keys = tuple(f"duplicate-{index // 2}" for index in range(16))
        selected = knapsack.allocate_priority_queue(
            quality,
            cost,
            light,
            budget_multiplier=3.0,
            target_fraction=0.55,
            tie_keys=keys,
        )
        for index in range(0, len(selected), 2):
            self.assertEqual(selected[index], selected[index + 1])

    def test_fast_charge_and_gate_contracts_are_frozen(self) -> None:
        fast = knapsack.DEFAULT_TIER_CONFIG["fast"]
        balanced = knapsack.DEFAULT_TIER_CONFIG["balanced"]
        premium = knapsack.DEFAULT_TIER_CONFIG["premium"]
        self.assertEqual((0.92, 1.2, 0.020, 0.20, 1.0), tuple(fast.__dict__.values()))
        self.assertEqual((0.88, 2.1, 0.030, 0.30, 1.0), tuple(balanced.__dict__.values()))
        self.assertEqual((0.94, 2.8, 0.000, 0.00, 1.25), tuple(premium.__dict__.values()))
        self.assertEqual((128, 350, 600, 0.10), (
            knapsack.MIN_CONTENT_GROUPS,
            knapsack.BALANCED_K1_MIN_GROUPS,
            knapsack.PREMIUM_K1_MIN_GROUPS,
            knapsack.PREMIUM_K1_MAX_TV,
        ))


if __name__ == "__main__":
    unittest.main()
