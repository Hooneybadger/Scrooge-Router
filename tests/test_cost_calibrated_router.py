# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest

from ossp_router import cost_calibrated_router
from ossp_router.heuristic import episode_text
from ossp_router.protocol import TIERS, load_bundled_policy, load_input, parse_input


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hash_regex = _load_module("final_test_hash_regex", ROOT / "baselines/hash_regex.py")


def _changed_batch(original):
    return parse_input(
        {
            "schema_version": original.schema_version,
            "challenge_id": original.challenge_id,
            "split": original.split,
            "episodes": [
                {
                    "episode_id": f"changed-{index}",
                    **(
                        {"prompt": episode.prompt}
                        if episode.prompt is not None
                        else {
                            "messages": [
                                {"role": item.role, "content": item.content}
                                for item in episode.messages or ()
                            ]
                        }
                    ),
                }
                for index, episode in enumerate(reversed(original.episodes))
            ],
        }
    )


def _by_content(inputs, submission):
    decisions = {
        decision.episode_id: decision.model_id for decision in submission.decisions
    }
    return {
        episode_text(episode): decisions[episode.episode_id]
        for episode in inputs.episodes
    }


class CostCalibratedRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_input(ROOT / "data/toy/inputs.json")
        cls.policy = load_bundled_policy()
        cls.artifact = cost_calibrated_router.load_bundled_artifact()

    def test_artifact_disables_k1_and_matches_policy(self) -> None:
        value = self.artifact.value
        self.assertEqual(2, value["schema_version"])
        self.assertFalse(value["k1_enabled"])
        self.assertEqual(self.policy.policy_id, value["policy_id"])
        self.assertEqual(set(TIERS), set(value["tier_kappa_q999"]))
        self.assertEqual(
            "exact-content-sha256-v1",
            value["premium_overlay"]["group_method"],
        )

    def test_runtime_features_match_training_feature_implementation(self) -> None:
        for episode in self.inputs.episodes:
            expected = hash_regex.raw_feature_vector(episode, 256)
            actual = cost_calibrated_router.structural_features(
                episode
            ) + cost_calibrated_router.hashed_features(episode, 256)
            self.assertEqual(expected, actual)

    def test_ids_order_and_repetition_do_not_change_content_decisions(self) -> None:
        changed = _changed_batch(self.inputs)
        for tier in TIERS:
            with self.subTest(tier=tier):
                first = cost_calibrated_router.make_submission(
                    self.inputs, self.policy, self.artifact, tier
                ).submission
                second = cost_calibrated_router.make_submission(
                    self.inputs, self.policy, self.artifact, tier
                ).submission
                reordered = cost_calibrated_router.make_submission(
                    changed, self.policy, self.artifact, tier
                ).submission
                self.assertEqual(first, second)
                self.assertEqual(
                    _by_content(self.inputs, first),
                    _by_content(changed, reordered),
                )
                self.assertNotIn(
                    "axk1-think", {decision.model_id for decision in first.decisions}
                )

    def test_cli_writes_atomic_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "submission.json"
            result = cost_calibrated_router.main(
                [
                    "--input",
                    str(ROOT / "data/toy/inputs.json"),
                    "--tier",
                    "fast",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(0, result)
            self.assertTrue(output.is_file())
            self.assertEqual(0o644, output.stat().st_mode & 0o777)


if __name__ == "__main__":
    unittest.main()
