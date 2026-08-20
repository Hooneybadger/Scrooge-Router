# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the submitted router's content-only family guard.

Covers artifact validation, invariance to episode ids and input order, and
that the guard only ever raises the accounting cost.
"""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

from ossp_router import family_guard_router, feasibility_ladder
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
ARTIFACT = ROOT / "src" / "ossp_router" / "resources" / "family-guard-router.v1.json"


class FamilyBucketTest(unittest.TestCase):
    def test_buckets_are_content_only(self) -> None:
        cases = {
            "def f(x):\n    return x\nassert f(1) == 1": "python_program",
            "Question:\nA. one\nB. two\nwhich?": "english_multiple_choice",
            "질문이 무엇인가요": "korean_reasoning",
            "Question: explain the rule": "rule_reasoning",
            "How many apples are left over?": "word_problem",
            "compute $x^2$": "latex_math",
            "zzz qqq": "other",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(family_guard_router.prompt_family_text(text), expected)

    def test_long_text_is_long_context(self) -> None:
        self.assertEqual(
            family_guard_router.prompt_family_text("a" * 4_000), "long_context"
        )


class ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.value = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_bundled_artifact_loads(self) -> None:
        artifact = family_guard_router.load_bundled_artifact()
        self.assertTrue(artifact.multipliers)
        low, high = family_guard_router.MULTIPLIER_CLIP
        for multiplier in artifact.multipliers.values():
            self.assertGreaterEqual(multiplier, low)
            self.assertLessEqual(multiplier, high)

    def test_guard_is_pessimism_only(self) -> None:
        artifact = family_guard_router.load_artifact_mapping(copy.deepcopy(self.value))
        for multiplier in artifact.multipliers.values():
            self.assertGreaterEqual(multiplier, 1.0)

    def test_multiplier_outside_clip_is_rejected(self) -> None:
        broken = copy.deepcopy(self.value)
        broken["family_guard"]["multipliers"]["other"] = 9.0
        with self.assertRaises(ProtocolError):
            family_guard_router.load_artifact_mapping(broken)

    def test_missing_guard_is_rejected(self) -> None:
        broken = copy.deepcopy(self.value)
        del broken["family_guard"]
        with self.assertRaises(ProtocolError):
            family_guard_router.load_artifact_mapping(broken)

    def test_base_artifact_still_validates_as_ladder(self) -> None:
        artifact = family_guard_router.load_artifact_mapping(copy.deepcopy(self.value))
        self.assertEqual(
            artifact.base.value["artifact_type"], feasibility_ladder.ARTIFACT_TYPE
        )
        self.assertFalse(artifact.base.value["k1_enabled"])


class SubmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_bundled_policy()
        self.artifact = family_guard_router.load_bundled_artifact()
        self.inputs = load_input(TOY_INPUTS)

    def test_all_tiers_produce_complete_submissions(self) -> None:
        for tier in TIERS:
            with self.subTest(tier=tier):
                plan = family_guard_router.make_submission(
                    self.inputs, self.policy, self.artifact, tier
                )
                decisions = plan.submission.decisions
                self.assertEqual(
                    [d.episode_id for d in decisions],
                    [e.episode_id for e in self.inputs.episodes],
                )
                for decision in decisions:
                    self.assertIn(decision.model_id, MODEL_IDS)

    def test_fast_and_balanced_never_pick_k1(self) -> None:
        for tier in ("fast", "balanced"):
            plan = family_guard_router.make_submission(
                self.inputs, self.policy, self.artifact, tier
            )
            self.assertNotIn(
                MODEL_IDS[2], [d.model_id for d in plan.submission.decisions]
            )

    def test_id_and_order_do_not_change_choices(self) -> None:
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
        for tier in TIERS:
            with self.subTest(tier=tier):
                base = family_guard_router.make_submission(
                    self.inputs, self.policy, self.artifact, tier
                )
                other = family_guard_router.make_submission(
                    shuffled, self.policy, self.artifact, tier
                )
                by_text = {
                    episode.prompt or "\n".join(m.content for m in episode.messages or ()): d.model_id
                    for episode, d in zip(self.inputs.episodes, base.submission.decisions)
                }
                for episode, decision in zip(
                    shuffled.episodes, other.submission.decisions
                ):
                    text = episode.prompt or "\n".join(
                        m.content for m in episode.messages or ()
                    )
                    self.assertEqual(by_text[text], decision.model_id)

    def test_guard_only_raises_accounting_cost(self) -> None:
        for episode in self.inputs.episodes:
            base = feasibility_ladder.predict_fast_balanced_row(
                episode, self.policy, self.artifact.base
            )
            guarded = family_guard_router.guarded_prediction(
                episode, self.policy, self.artifact
            )
            self.assertEqual(base[0], guarded[0])
            self.assertEqual(base[1][0], guarded[1][0])
            self.assertGreaterEqual(guarded[1][1], base[1][1])

    def test_premium_matches_the_ladder_path(self) -> None:
        guarded = family_guard_router.make_submission(
            self.inputs, self.policy, self.artifact, "premium"
        )
        base = feasibility_ladder.make_submission(
            self.inputs, self.policy, self.artifact.base, "premium"
        )
        self.assertEqual(
            [d.model_id for d in guarded.submission.decisions],
            [d.model_id for d in base.submission.decisions],
        )


if __name__ == "__main__":
    unittest.main()
