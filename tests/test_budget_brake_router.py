# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Premium predicted-budget brake overlay.

Fast and Balanced match family_guard_router on mixed batches. A
family-majority Fast batch may tighten the predicted cap. Premium is
the parent two-action set plus the frozen brake loop. No Dev outcome
is read.
"""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

from ossp_router import budget_brake_router, family_guard_router
from ossp_router.feasibility_ladder import _select_premium_configured
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Episode,
    InputBatch,
    ProtocolError,
    load_bundled_policy,
    load_input,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOY_INPUTS = ROOT / "data" / "toy" / "inputs.json"
ARTIFACT = ROOT / "src" / "ossp_router" / "resources" / "budget-brake-router.v1.json"
_AX31 = MODEL_IDS[1]
_K1 = MODEL_IDS[2]


def _block() -> dict[str, object]:
    return {
        "brake_ratio": 3.25,
        "count_cap": 48,
        "denylist_families": [
            "korean_reasoning",
            "python_program",
            "rule_reasoning",
        ],
        "runaway_absolute": 0.17152750745633214,
    }


class ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.value = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_bundled_artifact_loads(self) -> None:
        artifact = budget_brake_router.load_bundled_artifact()
        self.assertEqual(artifact.value["artifact_type"], budget_brake_router.ARTIFACT_TYPE)
        self.assertTrue(artifact.budget_brake["enabled"])
        self.assertEqual(artifact.budget_brake["brake_ratio"], 3.25)
        self.assertEqual(artifact.budget_brake["count_cap"], 48)
        self.assertEqual(
            artifact.family_guard.value["artifact_type"],
            family_guard_router.ARTIFACT_TYPE,
        )

    def test_missing_budget_brake_is_rejected(self) -> None:
        broken = copy.deepcopy(self.value)
        del broken["budget_brake"]
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(broken)

    def test_brake_ratio_outside_range_is_rejected(self) -> None:
        broken = copy.deepcopy(self.value)
        broken["budget_brake"]["brake_ratio"] = 9.0
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(broken)
        broken["budget_brake"]["brake_ratio"] = 1.0
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(broken)

    def test_negative_count_cap_is_rejected(self) -> None:
        broken = copy.deepcopy(self.value)
        broken["budget_brake"]["count_cap"] = -1
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(broken)

    def test_empty_forest_is_rejected(self) -> None:
        broken = copy.deepcopy(self.value)
        broken["budget_brake"]["forest"] = {"n_trees": 0, "trees": []}
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(broken)

    def test_tree_array_length_mismatch_is_rejected(self) -> None:
        broken = copy.deepcopy(self.value)
        tree = broken["budget_brake"]["forest"]["trees"][0]
        tree["left"] = list(tree["left"]) + [-1]
        with self.assertRaises(ProtocolError):
            budget_brake_router.load_artifact_mapping(broken)


class PromotionRuleTest(unittest.TestCase):
    def test_premium_never_exceeds_the_brake(self) -> None:
        parent = [_AX31, _AX31]
        quality = [0.9, 0.8]
        families = ["other", "other"]
        costs = ((1.0, 2.0, 2.15), (1.0, 2.0, 2.05))
        digests = ("b", "a")
        block = _block()
        block["brake_ratio"] = 2.05
        selected = budget_brake_router.promote_premium_brake(
            parent, quality, families, costs, digests, block
        )
        ratio = budget_brake_router.predicted_premium_ratio(selected, costs)
        self.assertLessEqual(ratio, float(block["brake_ratio"]) + 1e-12)
        self.assertEqual(selected[0], _AX31)
        self.assertEqual(selected[1], _K1)

    def test_runaway_row_is_never_promoted(self) -> None:
        parent = [_AX31]
        quality = [0.9]
        families = ["other"]
        costs = ((1.0, 2.0, 2.0 + 0.2),)
        digests = ("aa",)
        selected = budget_brake_router.promote_premium_brake(
            parent, quality, families, costs, digests, _block()
        )
        self.assertEqual(selected, (_AX31,))

    def test_denylist_families_are_never_promoted(self) -> None:
        parent = [_AX31, _AX31, _AX31]
        quality = [0.9, 0.8, 0.7]
        families = ["korean_reasoning", "python_program", "rule_reasoning"]
        costs = ((1.0, 2.0, 2.05), (1.0, 2.0, 2.05), (1.0, 2.0, 2.05))
        digests = ("a", "b", "c")
        selected = budget_brake_router.promote_premium_brake(
            parent, quality, families, costs, digests, _block()
        )
        self.assertEqual(selected, (_AX31, _AX31, _AX31))

    def test_premium_selection_is_id_and_order_invariant(self) -> None:
        parent = [_AX31, _AX31, MODEL_IDS[0]]
        quality = [0.4, 0.9, 0.1]
        families = ["other", "word_problem", "other"]
        costs = ((1.0, 2.0, 2.04), (1.0, 2.0, 2.03), (1.0, 1.0, 1.01))
        digests = ("aa", "bb", "cc")
        first = budget_brake_router.promote_premium_brake(
            parent, quality, families, costs, digests, _block()
        )
        order = (1, 2, 0)
        shuffled = budget_brake_router.promote_premium_brake(
            [parent[i] for i in order],
            [quality[i] for i in order],
            [families[i] for i in order],
            [costs[i] for i in order],
            [digests[i] for i in order],
            _block(),
        )
        first_pairs = sorted(zip(digests, first))
        shuffled_pairs = sorted(zip((digests[i] for i in order), shuffled))
        self.assertEqual(first_pairs, shuffled_pairs)


class SubmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_bundled_policy()
        self.artifact = budget_brake_router.load_bundled_artifact()
        self.parent = family_guard_router.load_bundled_artifact()
        self.inputs = load_input(TOY_INPUTS)

    def test_fast_and_balanced_match_family_guard(self) -> None:
        families = [
            family_guard_router.prompt_family(episode)
            for episode in self.inputs.episodes
        ]
        fast_guard = budget_brake_router.fast_family_composition_guard(families)
        for tier in ("fast", "balanced"):
            with self.subTest(tier=tier):
                overlay = budget_brake_router.make_submission(
                    self.inputs, self.policy, self.artifact, tier
                )
                parent = family_guard_router.make_submission(
                    self.inputs, self.policy, self.parent, tier
                )
                if tier == "balanced" or not fast_guard:
                    self.assertEqual(
                        [d.model_id for d in overlay.submission.decisions],
                        [d.model_id for d in parent.submission.decisions],
                    )
                else:
                    self.assertAlmostEqual(overlay.predicted_cap, 1.07)

    def test_fast_and_balanced_never_pick_k1(self) -> None:
        for tier in ("fast", "balanced"):
            plan = budget_brake_router.make_submission(
                self.inputs, self.policy, self.artifact, tier
            )
            self.assertNotIn(
                _K1, [d.model_id for d in plan.submission.decisions]
            )

    def test_all_tiers_produce_complete_submissions(self) -> None:
        for tier in TIERS:
            with self.subTest(tier=tier):
                plan = budget_brake_router.make_submission(
                    self.inputs, self.policy, self.artifact, tier
                )
                decisions = plan.submission.decisions
                self.assertEqual(
                    [d.episode_id for d in decisions],
                    [e.episode_id for e in self.inputs.episodes],
                )
                for decision in decisions:
                    self.assertIn(decision.model_id, MODEL_IDS)

    def test_premium_predicted_ratio_stays_under_brake(self) -> None:
        plan = budget_brake_router.make_submission(
            self.inputs, self.policy, self.artifact, "premium"
        )
        costs = tuple(row[1] for row in plan.premium_rows)
        models = tuple(d.model_id for d in plan.submission.decisions)
        self.assertLessEqual(
            budget_brake_router.predicted_premium_ratio(models, costs),
            float(self.artifact.budget_brake["brake_ratio"]) + 1e-12,
        )

    def test_flat_forest_matches_tree_walk(self) -> None:
        from ossp_router.cost_calibrated_router import structural_features

        trees = self.artifact.budget_brake["forest"]["trees"]
        clip = self.artifact.budget_brake["clip"]
        n_trees_f = float(len(trees))
        for episode in self.inputs.episodes:
            row = structural_features(episode)
            total = 0.0
            for tree in trees:
                total += budget_brake_router._walk_tree(row, tree)
            mean = total / n_trees_f
            if mean < clip[0]:
                expected = float(clip[0])
            elif mean > clip[1]:
                expected = float(clip[1])
            else:
                expected = mean
            self.assertEqual(
                budget_brake_router.predict_quality_features(row, self.artifact),
                expected,
            )

    def test_module_source_has_no_dynamic_code(self) -> None:
        source = pathlib.Path(budget_brake_router.__file__).read_text(encoding="utf-8")
        self.assertNotIn("exec(", source)
        self.assertNotIn("compile(", source)
        self.assertNotIn("eval(", source)

    def test_premium_id_and_order_do_not_change_pairs(self) -> None:
        shuffled = InputBatch(
            schema_version=self.inputs.schema_version,
            challenge_id="audit-shuffle",
            split=self.inputs.split,
            episodes=tuple(
                Episode(
                    episode_id=f"renamed-{index}",
                    prompt=episode.prompt,
                    messages=episode.messages,
                )
                for index, episode in enumerate(reversed(self.inputs.episodes))
            ),
        )
        base = budget_brake_router.make_submission(
            self.inputs, self.policy, self.artifact, "premium"
        )
        other = budget_brake_router.make_submission(
            shuffled, self.policy, self.artifact, "premium"
        )
        base_pairs = sorted(
            (budget_brake_router.content_digest(episode), decision.model_id)
            for episode, decision in zip(
                self.inputs.episodes, base.submission.decisions
            )
        )
        other_pairs = sorted(
            (budget_brake_router.content_digest(episode), decision.model_id)
            for episode, decision in zip(
                shuffled.episodes, other.submission.decisions
            )
        )
        self.assertEqual(base_pairs, other_pairs)


class PrefilterTest(unittest.TestCase):
    def test_prefilter_skipped_row_is_brake_ineligible(self) -> None:
        parent = [_AX31, _AX31, MODEL_IDS[0], _AX31]
        families = ["other", "python_program", "other", "word_problem"]
        costs = (
            (1.0, 2.0, 2.05),
            (1.0, 2.0, 2.05),
            (1.0, 2.0, 2.05),
            (1.0, 2.0, 2.0 + 0.2),
        )
        block = _block()
        eligible = budget_brake_router.eligible_promotion_indices(
            parent, families, costs, block
        )
        self.assertEqual(eligible, (0,))
        skipped = [index for index in range(len(parent)) if index not in set(eligible)]
        selected = budget_brake_router.promote_premium_brake(
            parent,
            [1.0] * len(parent),
            families,
            costs,
            ("a", "b", "c", "d"),
            block,
        )
        for index in skipped:
            self.assertEqual(selected[index], parent[index])

    def test_prefilter_zero_fill_cannot_change_promoted_set(self) -> None:
        parent = [_AX31, _AX31, MODEL_IDS[0], _AX31, _AX31]
        families = [
            "other",
            "rule_reasoning",
            "word_problem",
            "latex_math",
            "korean_reasoning",
        ]
        costs = (
            (1.0, 2.0, 2.04),
            (1.0, 2.0, 2.03),
            (1.0, 1.0, 1.01),
            (1.0, 2.0, 2.02),
            (1.0, 2.0, 2.01),
        )
        quality = [0.4, 0.9, 0.8, -0.2, 0.7]
        digests = ("aa", "bb", "cc", "dd", "ee")
        block = _block()
        eligible = set(
            budget_brake_router.eligible_promotion_indices(parent, families, costs, block)
        )
        full = budget_brake_router.promote_premium_brake(
            parent, quality, families, costs, digests, block
        )
        sparse = [
            value if index in eligible else 0.0 for index, value in enumerate(quality)
        ]
        filtered = budget_brake_router.promote_premium_brake(
            parent, sparse, families, costs, digests, block
        )
        self.assertEqual(full, filtered)
        for index, model_id in enumerate(full):
            if index not in eligible:
                self.assertEqual(model_id, parent[index])


class PremiumParentGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = budget_brake_router.load_bundled_artifact()
        self.residual = Episode(episode_id="residual", prompt="zzz qqq")
        self.word = Episode(
            episode_id="word", prompt="How many apples are left over?"
        )
        self.costs = (1.0, 2.0, 2.2)

    def test_residual_ax31_increment_uses_shipped_multiplier(self) -> None:
        self.assertEqual(
            family_guard_router.prompt_family(self.residual),
            family_guard_router.RESIDUAL_FAMILY,
        )
        multiplier = family_guard_router.guard_multiplier(
            self.residual, self.artifact.family_guard
        )
        self.assertEqual(multiplier, 2.5)
        guarded = budget_brake_router.guard_premium_parent_costs(
            self.residual, self.costs, self.artifact
        )
        self.assertEqual(guarded, (1.0, 3.5, 2.2))

    def test_non_residual_parent_costs_are_unchanged(self) -> None:
        self.assertEqual(
            family_guard_router.prompt_family(self.word), "word_problem"
        )
        guarded = budget_brake_router.guard_premium_parent_costs(
            self.word, self.costs, self.artifact
        )
        self.assertEqual(guarded, self.costs)

    def test_parent_row_helper_matches_manual_guard(self) -> None:
        policy = load_bundled_policy()
        uplift, raw = budget_brake_router.premium_prediction_row(
            self.residual, policy, self.artifact
        )
        parent_uplift, parent_costs = budget_brake_router.premium_parent_prediction_row(
            self.residual, policy, self.artifact
        )
        self.assertEqual(parent_uplift, uplift)
        self.assertEqual(
            parent_costs,
            budget_brake_router.guard_premium_parent_costs(
                self.residual, raw, self.artifact
            ),
        )
        self.assertEqual(parent_costs[2], raw[2])

    def test_parent_costs_require_three_models(self) -> None:
        with self.assertRaises(ProtocolError):
            budget_brake_router.guard_premium_parent_costs(
                self.residual, (1.0, 2.0), self.artifact
            )

    def test_residual_multiplier_override_stays_inside_clip(self) -> None:
        guarded = budget_brake_router.guard_premium_parent_costs(
            self.residual,
            self.costs,
            self.artifact,
            residual_multiplier=3.0,
        )
        self.assertEqual(guarded, (1.0, 4.0, 2.2))
        unchanged = budget_brake_router.guard_premium_parent_costs(
            self.word,
            self.costs,
            self.artifact,
            residual_multiplier=3.0,
        )
        self.assertEqual(unchanged, self.costs)
        with self.assertRaises(ProtocolError):
            budget_brake_router.guard_premium_parent_costs(
                self.residual,
                self.costs,
                self.artifact,
                residual_multiplier=3.25,
            )

    def test_brake_guard_inflates_residual_k1_only(self) -> None:
        guarded = budget_brake_router.guard_premium_brake_costs(
            self.residual, self.costs, self.artifact
        )
        self.assertEqual(guarded[0], 1.0)
        self.assertEqual(guarded[1], 2.0)
        self.assertAlmostEqual(guarded[2], 2.5)
        unchanged = budget_brake_router.guard_premium_brake_costs(
            self.word, self.costs, self.artifact
        )
        self.assertEqual(unchanged, self.costs)

    def test_residual_denylist_blocks_k1(self) -> None:
        parent = [_AX31]
        selected = budget_brake_router.promote_premium_brake(
            parent,
            [0.9],
            [family_guard_router.RESIDUAL_FAMILY],
            ((1.0, 2.0, 2.05),),
            ("aa",),
            {
                **_block(),
                "denylist_families": list(_block()["denylist_families"])
                + [family_guard_router.RESIDUAL_FAMILY],
            },
        )
        self.assertEqual(selected, (_AX31,))


class ConditionalPremiumGuardTest(unittest.TestCase):
    def test_threshold_is_three_quarters(self) -> None:
        self.assertEqual(
            budget_brake_router.CONDITIONAL_PREMIUM_RESIDUAL_THRESHOLD, 0.75
        )
        residual = family_guard_router.RESIDUAL_FAMILY
        self.assertFalse(
            budget_brake_router.premium_residual_composition_guard(
                [residual] * 2 + ["word_problem"] * 2
            )
        )
        self.assertTrue(
            budget_brake_router.premium_residual_composition_guard(
                [residual] * 3 + ["word_problem"]
            )
        )

    def _batch(self, prompts: list[str]) -> InputBatch:
        return InputBatch(
            schema_version=1,
            challenge_id="toy",
            split="public",
            episodes=tuple(
                Episode(episode_id=f"cond-{index}", prompt=prompt)
                for index, prompt in enumerate(prompts)
            ),
        )

    def test_mixed_batch_stays_on_unguarded_parent(self) -> None:
        policy = load_bundled_policy()
        artifact = budget_brake_router.load_bundled_artifact()
        batch = self._batch(
            [
                "zzz qqq",
                "How many apples are left over?",
                "How many oranges are left over?",
                "How many pears are left over?",
            ]
        )
        families = [
            family_guard_router.prompt_family(episode) for episode in batch.episodes
        ]
        self.assertLess(
            budget_brake_router.residual_fraction(families),
            budget_brake_router.CONDITIONAL_PREMIUM_RESIDUAL_THRESHOLD,
        )
        rows = tuple(
            budget_brake_router.premium_prediction_row(episode, policy, artifact)
            for episode in batch.episodes
        )
        live = budget_brake_router.make_submission(batch, policy, artifact, "premium")
        parent, _ratio = _select_premium_configured(
            batch,
            rows,
            float(artifact.value["predicted_caps"]["premium"]),
            artifact.family_guard.base,
        )
        composed = budget_brake_router.promote_premium_brake(
            parent,
            [
                budget_brake_router.predict_quality(episode, artifact)
                for episode in batch.episodes
            ],
            families,
            [row[1] for row in rows],
            [budget_brake_router.content_digest(episode) for episode in batch.episodes],
            artifact.budget_brake,
        )
        self.assertEqual(
            tuple(decision.model_id for decision in live.submission.decisions),
            composed,
        )

    def test_residual_majority_uses_parent_guard_and_denylist(self) -> None:
        policy = load_bundled_policy()
        artifact = budget_brake_router.load_bundled_artifact()
        batch = self._batch(
            [
                "zzz qqq",
                "zzz qqq extra",
                "zzz qqq more",
                "How many apples are left over?",
            ]
        )
        families = [
            family_guard_router.prompt_family(episode) for episode in batch.episodes
        ]
        self.assertEqual(families.count(family_guard_router.RESIDUAL_FAMILY), 3)
        self.assertTrue(budget_brake_router.premium_residual_composition_guard(families))
        rows = tuple(
            budget_brake_router.premium_prediction_row(episode, policy, artifact)
            for episode in batch.episodes
        )
        live = budget_brake_router.select_premium_with_brake(
            batch, policy, artifact, rows
        )
        unguarded_parent, _ratio = _select_premium_configured(
            batch,
            rows,
            float(artifact.value["predicted_caps"]["premium"]),
            artifact.family_guard.base,
        )
        unguarded = budget_brake_router.promote_premium_brake(
            unguarded_parent,
            [
                budget_brake_router.predict_quality(episode, artifact)
                for episode in batch.episodes
            ],
            families,
            [row[1] for row in rows],
            [budget_brake_router.content_digest(episode) for episode in batch.episodes],
            artifact.budget_brake,
        )
        self.assertNotEqual(live, unguarded)
        for family, model_id in zip(families, live):
            if family == family_guard_router.RESIDUAL_FAMILY:
                self.assertNotEqual(model_id, _K1)


class ConditionalFastCapTest(unittest.TestCase):
    def test_threshold_is_three_quarters(self) -> None:
        self.assertEqual(budget_brake_router.CONDITIONAL_FAST_FAMILY_THRESHOLD, 0.75)
        self.assertEqual(budget_brake_router.CONDITIONAL_FAST_CAP, 1.07)
        self.assertFalse(
            budget_brake_router.fast_family_composition_guard(
                ["word_problem"] * 2 + ["other"] * 2
            )
        )
        self.assertTrue(
            budget_brake_router.fast_family_composition_guard(
                ["word_problem"] * 3 + ["other"]
            )
        )

    def test_family_majority_fast_uses_tight_cap(self) -> None:
        policy = load_bundled_policy()
        artifact = budget_brake_router.load_bundled_artifact()
        parent = family_guard_router.load_bundled_artifact()
        batch = InputBatch(
            schema_version=1,
            challenge_id="toy",
            split="public",
            episodes=tuple(
                Episode(
                    episode_id=f"wp-{index}",
                    prompt=f"How many apples are left over after {index}?",
                )
                for index in range(4)
            ),
        )
        families = [
            family_guard_router.prompt_family(episode) for episode in batch.episodes
        ]
        self.assertTrue(budget_brake_router.fast_family_composition_guard(families))
        live = budget_brake_router.make_submission(batch, policy, artifact, "fast")
        guarded = family_guard_router.make_submission(batch, policy, parent, "fast")
        self.assertAlmostEqual(live.predicted_cap, 1.07)
        self.assertAlmostEqual(guarded.predicted_cap, 1.11)
        balanced = budget_brake_router.make_submission(
            batch, policy, artifact, "balanced"
        )
        parent_balanced = family_guard_router.make_submission(
            batch, policy, parent, "balanced"
        )
        self.assertEqual(
            [d.model_id for d in balanced.submission.decisions],
            [d.model_id for d in parent_balanced.submission.decisions],
        )


if __name__ == "__main__":
    unittest.main()
