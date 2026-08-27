# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from ossp_router.protocol import Episode, Message

try:
    import numpy as np
    from research.lab import generalization_followups as followups
    from research.lab.distributional_knapsack import (
        DistributionalPredictions,
        FAMILY_NAMES,
    )
except ImportError:
    raise unittest.SkipTest("numpy / research stack is not installed")


class QualityObjectiveTest(unittest.TestCase):
    def test_blend_uses_direct_adjacent_changes(self) -> None:
        absolute = np.asarray([[0.2, 0.5, 0.4], [0.6, 0.7, 0.9]])
        direct = np.asarray([[0.4, 0.3], [-0.2, 0.1]])

        baseline = followups.blended_quality(absolute, direct, 0.0)
        candidate = followups.blended_quality(absolute, direct, 1.0)

        np.testing.assert_allclose(absolute, baseline)
        np.testing.assert_allclose(
            np.asarray([[0.2, 0.6, 0.9], [0.6, 0.4, 0.5]]), candidate
        )

    def test_fixed_count_keeps_fast_off_think(self) -> None:
        quality = np.column_stack(
            (
                np.zeros(20),
                np.linspace(0.0, 1.0, 20),
                np.linspace(1.0, 2.0, 20),
            )
        )
        keys = tuple(f"key-{index:02d}" for index in range(20))

        fast = followups.fixed_count_actions(quality, "fast", keys)
        premium = followups.fixed_count_actions(quality, "premium", keys)

        self.assertNotIn(2, fast)
        self.assertIn(2, premium)
        self.assertGreater(np.count_nonzero(premium), np.count_nonzero(fast))

    def test_grouped_bootstrap_is_deterministic(self) -> None:
        scores = np.asarray(
            [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
        )
        base = {tier: np.zeros(3, dtype=np.int8) for tier in followups.TIERS}
        candidate = {
            tier: np.asarray([1, 1, 0], dtype=np.int8)
            for tier in followups.TIERS
        }
        keys = ("same", "same", "other")

        first = followups.grouped_bootstrap_quality_delta(
            base, candidate, scores, keys, draws=50, seed=7
        )
        second = followups.grouped_bootstrap_quality_delta(
            base, candidate, scores, keys, draws=50, seed=7
        )

        self.assertEqual(first, second)
        self.assertGreater(first["mean"], 0.0)

    def test_linear_stack_learns_from_primary_predictions(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn is not installed")
        count = 80
        base_delta = np.linspace(-0.2, 0.3, count)
        absolute = np.column_stack(
            (np.full(count, 0.4), 0.4 + base_delta, 0.5 + base_delta)
        )
        direct = np.column_stack((base_delta * 1.8, base_delta * -0.5))
        structural = np.column_stack((base_delta * 0.5, base_delta * 1.2))
        uplifts = {
            "direct_squared": direct,
            "direct_huber": direct * 0.9,
            "direct_structural": structural,
        }
        target_a = np.clip(0.2 + direct[:, 0], 0.0, 1.0)
        target_k = np.clip(target_a + structural[:, 1], 0.0, 1.0)
        scores = np.column_stack((np.full(count, 0.2), target_a, target_k))
        families = tuple(FAMILY_NAMES[index % len(FAMILY_NAMES)] for index in range(count))
        groups = tuple(f"group-{index}" for index in range(count))

        fitted = followups.fit_stacked_uplifts(
            absolute,
            uplifts,
            scores,
            families,
            groups,
            alpha=1.0,
            family_interactions=False,
        )
        predicted = followups.predict_stacked_uplifts(
            fitted, absolute, uplifts, families
        )

        self.assertEqual((count, 2), predicted.shape)
        self.assertGreater(np.corrcoef(predicted[:, 0], scores[:, 1] - scores[:, 0])[0, 1], 0.95)


class CostCalibrationTest(unittest.TestCase):
    def _predictions(self) -> DistributionalPredictions:
        quality = np.full((100, 3), 0.5)
        mean = np.ones((100, 3))
        upper = np.ones((100, 3))
        return DistributionalPredictions(quality, mean, upper)

    def test_global_quantile_raises_underpredicted_upper(self) -> None:
        predictions = self._predictions()
        actual = np.ones((100, 3))
        actual[:20] = 2.0
        families = tuple(FAMILY_NAMES[index % len(FAMILY_NAMES)] for index in range(100))

        calibration = followups.fit_cost_calibration(
            predictions, actual, families, method="global_q90"
        )
        calibrated = followups.apply_cost_calibration(
            predictions, families, calibration
        )

        self.assertTrue(np.all(calibrated.cost_q90 >= calibrated.cost_mean))
        self.assertTrue(np.all(calibration.q90_scales >= 2.0))

    def test_cross_calibration_never_uses_held_fold(self) -> None:
        predictions = self._predictions()
        actual = np.ones((100, 3))
        actual[:20] = 10.0
        folds = np.repeat(np.arange(5), 20)
        families = tuple("other" for _ in range(100))

        calibrated = followups.cross_calibrated_costs(
            predictions,
            actual,
            families,
            folds,
            method="global_q90",
        )

        self.assertTrue(np.allclose(1.0, calibrated.cost_q90[:20]))
        self.assertTrue(np.all(calibrated.cost_q90[20:] >= 10.0))

    def test_light_lower_calibration_uses_observed_lower_tail(self) -> None:
        predictions = self._predictions()
        actual = np.ones((100, 3))
        actual[:10, 0] = 0.25
        families = tuple("other" for _ in range(100))

        calibration = followups.fit_light_lower_calibration(
            predictions,
            actual,
            families,
            fraction=0.05,
        )
        credits = followups.apply_light_lower_calibration(
            predictions,
            families,
            calibration,
        )

        self.assertTrue(np.allclose(0.25, credits))

    def test_two_sided_surface_uses_lower_light_and_full_upper(self) -> None:
        predictions = self._predictions()
        lower = np.full(100, 0.25)

        charges, credits = followups.small_batch_cost_surfaces(
            predictions,
            "balanced",
            light_lower_credit=lower,
            full_upper=True,
        )

        np.testing.assert_allclose(lower, credits)
        np.testing.assert_allclose(lower, charges[:, 0])
        np.testing.assert_allclose(predictions.cost_q90[:, 1:], charges[:, 1:])


class SurfaceAndSmallBatchTest(unittest.TestCase):
    def test_surface_transforms_preserve_episode_shape(self) -> None:
        prompt = Episode("p", prompt="A. first\nB. second")
        messages = Episode(
            "m",
            messages=(Message("user", "(A) first\n(B) second"),),
        )

        changed_prompt = followups.transform_episode(prompt, "choice_labels")
        changed_messages = followups.transform_episode(messages, "choice_labels")

        self.assertEqual("p", changed_prompt.episode_id)
        self.assertIn("(A)", changed_prompt.prompt)
        self.assertEqual("user", changed_messages.messages[0].role)
        self.assertIn("A.", changed_messages.messages[0].content)

    def test_surface_canonicalization_collapses_registered_variants(self) -> None:
        original = "A. first  \nB. second"
        canonical = followups.canonical_surface_text(original)

        for variant in followups.SURFACE_VARIANTS:
            changed = followups.transform_surface_text(original, variant)
            self.assertEqual(
                canonical,
                followups.canonical_surface_text(changed),
                msg=variant,
            )
        self.assertEqual(canonical, followups.canonical_surface_text(canonical))

    def test_stable_normalizer_preserves_choice_notation(self) -> None:
        original = "A. first  \nB. second"
        stable = followups.stable_surface_text(original)

        for variant in ("line_endings", "trailing_space", "unicode_nfc"):
            changed = followups.transform_surface_text(original, variant)
            self.assertEqual(stable, followups.stable_surface_text(changed))
        choice = followups.transform_surface_text(original, "choice_labels")
        self.assertNotEqual(stable, followups.stable_surface_text(choice))

    def test_small_batch_schedule_approaches_normal_cap(self) -> None:
        low = followups.small_batch_target_fraction(
            "fast", 1.25, 1, 0.0, power=2.0
        )
        high = followups.small_batch_target_fraction(
            "fast", 1.25, 127, 0.0, power=2.0
        )

        self.assertGreater(low, 1.0 / 1.25)
        self.assertGreater(high, low)
        self.assertLessEqual(high, 0.92)

    def test_small_batch_route_stops_at_128_unique_items(self) -> None:
        self.assertTrue(followups.small_batch_route_enabled(127))
        self.assertFalse(followups.small_batch_route_enabled(128))
        self.assertFalse(followups.small_batch_route_enabled(0))

    def test_small_batch_views_do_not_need_outcomes(self) -> None:
        count = 140
        predictions = DistributionalPredictions(
            np.full((count, 3), 0.5),
            np.column_stack((np.ones(count), np.full(count, 2.0), np.full(count, 4.0))),
            np.column_stack((np.ones(count), np.full(count, 2.5), np.full(count, 5.0))),
        )
        families = tuple(FAMILY_NAMES[index % len(FAMILY_NAMES)] for index in range(count))
        keys = tuple(f"key-{index}" for index in range(count))

        first = followups.make_small_batch_views(
            families, keys, predictions, sizes=(8,), draws_per_kind=2, seed=11
        )
        second = followups.make_small_batch_views(
            families, keys, predictions, sizes=(8,), draws_per_kind=2, seed=11
        )

        self.assertEqual(first, second)
        self.assertEqual(8, len(first))
        self.assertEqual(
            {"uniform", "single_family", "predicted_tail", "duplicate_tail"},
            {view.kind for view in first},
        )
        duplicate = next(view for view in first if view.kind == "duplicate_tail")
        self.assertLess(len(set(duplicate.indexes)), len(duplicate.indexes))


class Issue39IntegrationTest(unittest.TestCase):
    def test_reconstructed_catalogues_match_frozen_counts(self) -> None:
        from research.lab.cap_certification import build_stress_views
        from research.lab.distributional_knapsack import FAMILY_NAMES
        from research.lab.issue39_integration import build_wide_catalogue

        families = tuple(
            FAMILY_NAMES[index % len(FAMILY_NAMES)] for index in range(880)
        )
        digests = tuple(f"{index:064x}" for index in range(880))
        primary, catalogue = build_stress_views(families)
        wide = build_wide_catalogue(families, digests, seed=2026082204)

        self.assertEqual(5_840, catalogue["n_views"])
        self.assertEqual(5_840, len(primary))
        self.assertEqual(3_435, len(wide))

    def test_cache_route_matches_serving_on_toy_and_normal_batch(self) -> None:
        from ossp_router import distributional_router as serving
        from ossp_router.protocol import Episode, InputBatch, load_bundled_policy, load_input
        from research.lab.issue39_integration import (
            precompute_serving_cache,
            route_from_cache,
        )

        root = __import__("pathlib").Path(__file__).resolve().parents[1]
        artifact = serving.load_bundled_artifact()
        policy = load_bundled_policy()
        toy = load_input(root / "data" / "toy" / "inputs.json")
        toy_cache = precompute_serving_cache(toy.episodes, artifact)
        for tier in serving.TIERS:
            expected = [
                serving.MODEL_IDS.index(decision.model_id)
                for decision in serving.make_submission(
                    toy, policy, artifact, tier
                ).decisions
            ]
            actual = list(
                route_from_cache(
                    toy_cache,
                    range(len(toy.episodes)),
                    artifact=artifact,
                    policy=policy,
                    tier=tier,
                    q95_on_normal=False,
                )
            )
            self.assertEqual(expected, actual, msg=tier)

        episodes = tuple(
            Episode(
                f"normal-{index}",
                prompt=(
                    f"Question: analyze deterministic case {index}.\n"
                    "A. alpha\nB. beta\nC. gamma\nD. delta"
                ),
            )
            for index in range(128)
        )
        inputs = InputBatch(1, "issue39-normal", "test", episodes)
        cache = precompute_serving_cache(episodes, artifact)
        for tier in serving.TIERS:
            expected = [
                serving.MODEL_IDS.index(decision.model_id)
                for decision in serving.make_submission(
                    inputs, policy, artifact, tier
                ).decisions
            ]
            actual = list(
                route_from_cache(
                    cache,
                    range(len(episodes)),
                    artifact=artifact,
                    policy=policy,
                    tier=tier,
                    q95_on_normal=False,
                )
            )
            self.assertEqual(expected, actual, msg=tier)


if __name__ == "__main__":
    unittest.main()
