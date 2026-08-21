# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Premium predicted-budget brake overlay.

Fast and Balanced must stay bit-identical to family_guard_router. Premium is
the parent two-action set plus the frozen brake loop. No Dev outcome is read.
"""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

from ossp_router import budget_brake_router, family_guard_router
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
        for tier in ("fast", "balanced"):
            with self.subTest(tier=tier):
                overlay = budget_brake_router.make_submission(
                    self.inputs, self.policy, self.artifact, tier
                )
                parent = family_guard_router.make_submission(
                    self.inputs, self.policy, self.parent, tier
                )
                self.assertEqual(
                    [d.model_id for d in overlay.submission.decisions],
                    [d.model_id for d in parent.submission.decisions],
                )

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


if __name__ == "__main__":
    unittest.main()
