# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Small-batch guard contracts."""

from __future__ import annotations

import pathlib
import sys
import unittest

from ossp_router.protocol import MODEL_IDS


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BatchFactorTest(unittest.TestCase):
    def _module(self):
        try:
            from ossp_router.small_batch import THRESHOLD, batch_factor
        except ImportError:
            self.skipTest("small_batch guard is unavailable")
        return THRESHOLD, batch_factor

    def test_exact_values(self) -> None:
        threshold, batch_factor = self._module()
        self.assertEqual(batch_factor(0), 0.0)
        self.assertAlmostEqual(batch_factor(threshold // 2), 0.5)
        self.assertEqual(batch_factor(threshold), 1.0)
        self.assertEqual(batch_factor(threshold * 10), 1.0)

    def test_monotone(self) -> None:
        _, batch_factor = self._module()
        previous = -1.0
        for size in range(0, 200):
            value = batch_factor(size)
            self.assertGreaterEqual(value, previous)
            self.assertLessEqual(value, 1.0)
            previous = value


class EffectiveCapTest(unittest.TestCase):
    def _module(self):
        try:
            from ossp_router.small_batch import effective_cap
        except ImportError:
            self.skipTest("small_batch guard is unavailable")
        return effective_cap

    def test_identity_at_threshold(self) -> None:
        effective_cap = self._module()
        self.assertEqual(effective_cap(3.25, 4.0, 48), 3.25)
        self.assertEqual(effective_cap(3.25, 4.0, 2640), 3.25)

    def test_shrinkage_below_threshold(self) -> None:
        effective_cap = self._module()
        expected = 1.0 + (3.25 - 1.0) * (16 / 48)
        self.assertAlmostEqual(effective_cap(3.25, 4.0, 16), expected)
        self.assertEqual(effective_cap(1.11, 1.25, 8), 1.0 + 0.11 * (8 / 48))

    def test_never_exceeds_official(self) -> None:
        effective_cap = self._module()
        for size in range(1, 64):
            value = effective_cap(3.9, 4.0, size)
            self.assertLessEqual(value, 4.0)
            self.assertGreaterEqual(value, 1.0)


class SmallBatchRuntimeTest(unittest.TestCase):
    """End-to-end: small batches must land under official caps."""

    def _episode(self, episode_id: str, prompt: str):
        from ossp_router.protocol import Episode

        return Episode(episode_id=episode_id, prompt=prompt)

    def _batch(self, episodes):
        from ossp_router.protocol import InputBatch

        return InputBatch(
            schema_version=1,
            challenge_id="toy",
            split="public",
            episodes=tuple(episodes),
        )

    def test_small_batch_stays_under_caps(self) -> None:
        try:
            from ossp_router import budget_brake_router
            from ossp_router.protocol import load_bundled_policy
        except ImportError:
            self.skipTest("runtime stack is unavailable")
        prompts = []
        for index in range(12):
            prompts.append(
                f"Question {index}: solve for x given that 2*x + {index} = "
                f"{17 + index} and explain every step carefully."
            )
            if index % 3 == 0:
                prompts[-1] += "\n\n```python\ndef f(x):\n    return x*2\n```"
        batch = self._batch(
            [self._episode(f"sb-{i}", p) for i, p in enumerate(prompts)]
        )
        policy = load_bundled_policy()
        artifact = budget_brake_router.load_bundled_artifact()
        for tier in ("fast", "balanced", "premium"):
            plan = budget_brake_router.make_submission(batch, policy, artifact, tier)
            models = [d.model_id for d in plan.submission.decisions]
            self.assertEqual(len(models), 12)
            if tier != "premium":
                # small-batch guard forbids the K1 overlay outside Premium
                for model_id in models:
                    self.assertNotEqual(model_id, MODEL_IDS[2])
        premium_models = [
            d.model_id
            for d in budget_brake_router.make_submission(
                batch, policy, artifact, "premium"
            ).submission.decisions
        ]
        # below the threshold the K1 brake overlay is skipped outright
        self.assertNotIn(MODEL_IDS[2], premium_models)


if __name__ == "__main__":
    unittest.main()
