# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from ossp_router import distributional_router, heuristic
from ossp_router.protocol import Episode, InputBatch, load_bundled_policy, load_input


ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "src/ossp_router/resources/distributional-router.v1.json"
DEV_INPUT = ROOT / "data/materialized/dev/inputs.json"


class DistributionalArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        cls.artifact = distributional_router.load_bundled_artifact()

    def test_compiled_distributional_contract_is_frozen(self) -> None:
        artifact = self.artifact
        self.assertEqual(1_024, len(artifact.vocabulary))
        self.assertLess(ARTIFACT_PATH.stat().st_size, 1_500_000)
        self.assertEqual(
            "distributional-knapsack-v1",
            artifact.experiment["experiment_id"],
        )
        self.assertEqual(
            "356b73e737efc4220490fa47e323009b76be87fd",
            artifact.experiment["base_commit"],
        )
        self.assertEqual(
            "explicit lexicon + distributional tree heads + finite-sample "
            "batch risk + canonical concave-prefix allocation",
            artifact.experiment["design"],
        )
        for model_id in distributional_router.MODEL_IDS:
            self.assertEqual(120, len(artifact.quality_heads[model_id].trees))
            self.assertEqual(160, len(artifact.cost_mean_heads[model_id].trees))
            self.assertEqual(120, len(artifact.cost_q50_heads[model_id].trees))
            self.assertEqual(120, len(artifact.cost_q90_heads[model_id].trees))

    def test_distributional_gates_and_certification_are_frozen(self) -> None:
        self.assertEqual((128, 350, 600, 0.10), (
            self.artifact.gates.min_content_groups,
            self.artifact.gates.balanced_k1_min_groups,
            self.artifact.gates.premium_k1_min_groups,
            self.artifact.gates.premium_k1_max_tv,
        ))
        certification = self.artifact.certification_summary
        self.assertEqual(0, certification["primary_margin_violations"])
        self.assertEqual(0, certification["wide_margin_violations"])
        self.assertEqual(11_680, certification["primary_views"])
        self.assertEqual(20_610, certification["wide_views"])
        self.assertAlmostEqual(
            0.7157670454545454,
            certification["official_dev_weighted_quality"],
        )

    def test_parser_rejects_feature_quantile_and_gate_drift(self) -> None:
        changed = copy.deepcopy(self.value)
        changed["feature_version"] = "changed"
        with self.assertRaisesRegex(Exception, "unsupported distributional artifact"):
            distributional_router.load_artifact_mapping(changed)

        changed = copy.deepcopy(self.value)
        del changed["cost_q50_heads"]
        with self.assertRaisesRegex(Exception, "artifact fields"):
            distributional_router.load_artifact_mapping(changed)

        changed = copy.deepcopy(self.value)
        changed["vocabulary"].pop()
        with self.assertRaisesRegex(Exception, "vocabulary"):
            distributional_router.load_artifact_mapping(changed)

        changed = copy.deepcopy(self.value)
        changed["finite_sample_gates"]["premium_k1_max_tv"] = 1.1
        with self.assertRaisesRegex(Exception, "premium_k1_max_tv"):
            distributional_router.load_artifact_mapping(changed)


class DistributionalRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.toy = load_input(ROOT / "data/toy/inputs.json")
        cls.policy = load_bundled_policy()
        cls.artifact = distributional_router.load_bundled_artifact()

    def test_small_batches_use_the_two_sided_route(self) -> None:
        for tier in distributional_router.TIERS:
            submission = distributional_router.make_submission(
                self.toy, self.policy, self.artifact, tier
            )
            models = {decision.model_id for decision in submission.decisions}
            if tier == "fast":
                self.assertNotIn("axk1-think", models)

    def test_stable_surface_collapses_whitespace_and_line_endings(self) -> None:
        original = "Question: pick one.  \r\nA. yes\r\nB. no  "
        stable = distributional_router.stable_surface_text(original)
        self.assertEqual(
            stable,
            distributional_router.stable_surface_text(
                original.replace("\r\n", "\n") + "   "
            ),
        )
        self.assertIn("A. yes", stable)
        self.assertFalse(stable.endswith(" "))

    def test_small_batch_frozen_scales_follow_family_order(self) -> None:
        self.assertEqual(
            len(distributional_router.FAMILY_NAMES),
            len(distributional_router.SMALL_BATCH_MEAN_SCALES),
        )
        self.assertEqual(
            len(distributional_router.FAMILY_NAMES),
            len(distributional_router.SMALL_BATCH_UPPER_SCALES),
        )
        self.assertEqual(
            len(distributional_router.FAMILY_NAMES),
            len(distributional_router.SMALL_BATCH_LIGHT_LOWER_SCALES),
        )
        for row in distributional_router.SMALL_BATCH_UPPER_SCALES:
            self.assertGreaterEqual(min(row), 1.0)
        self.assertTrue(distributional_router.small_batch_route_enabled(1))
        self.assertTrue(distributional_router.small_batch_route_enabled(127))
        self.assertFalse(distributional_router.small_batch_route_enabled(128))
        self.assertFalse(distributional_router.small_batch_route_enabled(0))

    def test_fast_hard_bans_think_model_on_learned_path(self) -> None:
        episodes = tuple(
            Episode(
                f"fast-{index}",
                prompt=f"Question: solve scenario {index}. A. yes B. no",
            )
            for index in range(160)
        )
        inputs = InputBatch(1, "distributional-fast-test", "test", episodes)
        submission = distributional_router.make_submission(
            inputs, self.policy, self.artifact, "fast"
        )
        self.assertNotIn(
            "axk1-think", {decision.model_id for decision in submission.decisions}
        )

    def test_order_and_episode_ids_do_not_change_content_choices(self) -> None:
        episodes = tuple(
            Episode(
                f"original-{index}",
                prompt=(
                    f"Question: analyze deterministic case {index}.\n"
                    "A. alpha\nB. beta\nC. gamma\nD. delta"
                ),
            )
            for index in range(128)
        )
        inputs = InputBatch(1, "distributional-order-a", "test", episodes)
        expected_submission = distributional_router.make_submission(
            inputs, self.policy, self.artifact, "premium"
        )
        expected = {
            episode.prompt: decision.model_id
            for episode, decision in zip(
                inputs.episodes, expected_submission.decisions
            )
        }
        changed_episodes = tuple(
            Episode(
                f"changed-{index}",
                prompt=episode.prompt,
            )
            for index, episode in enumerate(reversed(episodes))
        )
        changed = InputBatch(1, "distributional-order-b", "holdout", changed_episodes)
        actual_submission = distributional_router.make_submission(
            changed, self.policy, self.artifact, "premium"
        )
        actual = {
            episode.prompt: decision.model_id
            for episode, decision in zip(
                changed.episodes, actual_submission.decisions
            )
        }
        self.assertEqual(expected, actual)
        self.assertNotEqual({"ax31-light"}, set(expected.values()))

    def test_oversized_workload_falls_back_before_learned_routing(self) -> None:
        with (
            mock.patch.object(distributional_router, "MAX_LEARNED_EPISODES", 1),
            mock.patch.object(
                distributional_router,
                "_canonical_batch",
                side_effect=AssertionError("canonicalized before workload guard"),
            ),
        ):
            actual = distributional_router.make_submission(
                self.toy, self.policy, self.artifact, "premium"
            )
        expected = heuristic.make_submission(
            self.toy, self.policy, "premium", strategy="always-light"
        )
        self.assertEqual(expected, actual)

    def test_runtime_has_no_old_champion_random_or_clock_dependency(self) -> None:
        source = inspect.getsource(distributional_router)
        self.assertNotIn("champion_router", source)
        self.assertNotIn("e29_", source)
        self.assertNotIn("e30", source)
        self.assertNotIn("import random", source)
        self.assertNotIn("import time", source)
        self.assertNotIn("time.monotonic", source)

    def test_cli_writes_a_valid_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "balanced.json"
            result = distributional_router.main(
                [
                    "--input",
                    str(ROOT / "data/toy/inputs.json"),
                    "--tier",
                    "balanced",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(0, result)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("balanced", value["tier"])
            self.assertEqual(3, len(value["decisions"]))

    @unittest.skipUnless(DEV_INPUT.is_file(), "materialized Dev input is unavailable")
    def test_materialized_dev_decision_digests_are_locked(self) -> None:
        inputs = load_input(DEV_INPUT)
        expected = {
            "fast": "55bb2bfbe8a63237ae820b64d94d1b925caeaedd2d826068da0e1f18f0d3f7a1",
            "balanced": "c8ea4460a41c34acc5459fbcc6557b7027a618784ddffc7a9642d4cb9a0991ac",
            "premium": "72c33002180a684fbcd9507f342198082ac2e67c3b90c0b3b6181c1ed71d15db",
        }
        for tier, digest in expected.items():
            submission = distributional_router.make_submission(
                inputs, self.policy, self.artifact, tier
            )
            payload = [
                (decision.episode_id, decision.model_id)
                for decision in submission.decisions
            ]
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.assertEqual(digest, hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    unittest.main()
