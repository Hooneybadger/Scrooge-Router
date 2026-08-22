# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1B quality-model contracts. Numpy/sklearn tests skip without research deps."""

from __future__ import annotations

import pathlib
import sys
import unittest

from ossp_router.protocol import Episode


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _episode(episode_id: str, prompt: str) -> Episode:
    return Episode(episode_id=episode_id, prompt=prompt)


class E1BQualityModelTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from decimal import Decimal
            from ossp_router.protocol import InputBatch, Outcome, OutcomeBatch, load_bundled_policy
            from research.lab.e1b_quality_models import (
                BASELINE_NAME,
                CANDIDATE_ORDER,
                CHAMPION_ABS,
                assemble,
                hashed_quality_matrix,
                oof_all_heads,
                promotion_gate,
            )
            from research.lab.grouped_crossfit import (
                FOLD_SEED,
                assign_balanced_group_folds,
                families_of,
                group_episodes,
            )
            from research.lab.public_pool import PublicPool
        except ImportError:
            self.skipTest("numpy / sklearn / research E1B stack is not installed")
        return {
            "Decimal": Decimal,
            "InputBatch": InputBatch,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "PublicPool": PublicPool,
            "FOLD_SEED": FOLD_SEED,
            "assign_balanced_group_folds": assign_balanced_group_folds,
            "assemble": assemble,
            "BASELINE_NAME": BASELINE_NAME,
            "CANDIDATE_ORDER": CANDIDATE_ORDER,
            "CHAMPION_ABS": CHAMPION_ABS,
            "families_of": families_of,
            "group_episodes": group_episodes,
            "hashed_quality_matrix": hashed_quality_matrix,
            "load_bundled_policy": load_bundled_policy,
            "np": np,
            "oof_all_heads": oof_all_heads,
            "promotion_gate": promotion_gate,
        }

    def _pool(self, mods, scores):
        np = mods["np"]
        prompts = [
            "Exact copy used twice for grouping.",
            "Exact copy used twice for grouping.",
            "Korean question: 다음 중 옳은 것은 무엇입니까? A. 1 B. 2",
            "def f(x):\n    return x + 1\nassert f(2) == ??",
            "How many apples are left over altogether if each costs 3?",
            "Solve $\\frac{1}{2} + \\frac{1}{3}$ using \\begin{align}.",
            "Word problem: how far and how long if the average is 12.",
            "Unrelated long english prompt about music history " * 5,
        ]
        episodes = tuple(
            _episode(f"e1b-{index:02d}", prompt) for index, prompt in enumerate(prompts)
        )
        grouping = mods["group_episodes"](episodes)
        families = mods["families_of"](episodes)
        folds = mods["assign_balanced_group_folds"](
            grouping.group_keys, families, folds=3, seed=mods["FOLD_SEED"]
        )
        policy = mods["load_bundled_policy"]()
        costs = np.asarray(
            [
                [1.0, 1.2, 2.0],
                [1.0, 1.2, 2.0],
                [1.0, 1.3, 2.2],
                [1.0, 1.4, 2.5],
                [1.0, 1.1, 1.8],
                [1.0, 1.25, 2.1],
                [1.0, 1.15, 1.9],
                [1.0, 1.35, 2.4],
            ],
            dtype=np.float64,
        )
        inputs = mods["InputBatch"](
            1, "ossp-2026-llm-router-challenge", "public", episodes
        )
        outcomes = []
        for episode, score_row in zip(episodes, scores):
            for model_index, model_id in enumerate(("ax31-light", "ax31", "axk1-think")):
                outcomes.append(
                    mods["Outcome"](
                        episode_id=episode.episode_id,
                        model_id=model_id,
                        score=mods["Decimal"](str(score_row[model_index])),
                        num_generations=1,
                        input_tokens=10 + model_index,
                        output_tokens=4,
                    )
                )
        splits = ("train", "train", "train", "train", "dev", "dev", "dev", "dev")
        pool = mods["PublicPool"](
            episodes=episodes,
            texts=tuple(prompts),
            families=families,
            languages=tuple(
                "korean" if "문항" in prompt or "다음" in prompt else "non_korean"
                for prompt in prompts
            ),
            length_views=tuple("len_lt_120" for _ in prompts),
            group_keys=grouping.group_keys,
            exact_keys=grouping.exact_keys,
            template_keys=grouping.template_keys,
            folds=folds,
            scores=np.asarray(scores, dtype=np.float64),
            costs=costs,
            light_total=float(costs[:, 0].sum()),
            identity={
                "fold_seed": mods["FOLD_SEED"],
                "folds": 3,
                "n_dev": 4,
                "n_episodes": 8,
                "n_train": 4,
                "split": "public",
            },
            grouping={
                "n_groups": grouping.n_groups,
                "n_exact_groups": grouping.n_exact_groups,
                "blocking": dict(grouping.blocking),
                "group_size_histogram": dict(grouping.group_size_histogram),
                "largest_group": grouping.largest_group,
                "n_jaccard_comparisons": grouping.n_jaccard_comparisons,
                "n_near_duplicate_unions": grouping.n_near_duplicate_unions,
                "n_singleton_groups": grouping.n_singleton_groups,
                "n_template_groups": grouping.n_template_groups,
            },
            fold_table=[],
            inputs=inputs,
            outcomes=mods["OutcomeBatch"](
                1, "ossp-2026-llm-router-challenge", "public", tuple(outcomes)
            ),
            policy=policy,
            split_labels=splits,
        )
        return pool, folds

    def _scores(self, mods):
        return mods["np"].asarray(
            [
                [0.25, 0.50, 0.40],
                [0.25, 0.75, 0.80],
                [0.50, 0.50, 0.25],
                [0.00, 1.00, 0.50],
                [0.75, 0.50, 0.25],
                [0.25, 0.25, 1.00],
                [0.50, 1.00, 1.00],
                [0.00, 0.25, 0.50],
            ],
            dtype=mods["np"].float64,
        )

    def test_hash_features_are_deterministic_and_order_independent(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._pool(mods, self._scores(mods))
        first = mods["hashed_quality_matrix"](pool.episodes)
        second = mods["hashed_quality_matrix"](pool.episodes)
        self.assertTrue(np.allclose(first, second))
        reversed_eps = tuple(reversed(pool.episodes))
        rev = mods["hashed_quality_matrix"](reversed_eps)
        self.assertTrue(np.allclose(first, rev[::-1]))
        self.assertEqual(first.shape[1], 1 + 14 + 256)
        self.assertTrue(np.allclose(first[:, 0], 1.0))

    def test_held_out_labels_do_not_change_that_fold_predictions(self) -> None:
        mods = self._import()
        np = mods["np"]
        scores = self._scores(mods)
        pool, folds = self._pool(mods, scores)
        first = mods["oof_all_heads"](pool)
        mutated = scores.copy()
        held = int(folds[0])
        mask = np.asarray(folds) == held
        mutated[mask] = np.asarray([0.0, 1.0, 0.0])
        second = mods["oof_all_heads"](pool, scores=mutated)
        hybrid = first["hashed_logistic_hybrid"]
        hybrid2 = second["hashed_logistic_hybrid"]
        residual = first["shallow_structural_residual"]
        residual2 = second["shallow_structural_residual"]
        self.assertTrue(np.allclose(hybrid.pred_qa[mask], hybrid2.pred_qa[mask]))
        self.assertTrue(np.allclose(hybrid.prob_qa[mask], hybrid2.prob_qa[mask]))
        self.assertTrue(np.allclose(residual.residual_qa[mask], residual2.residual_qa[mask]))
        self.assertTrue(np.allclose(residual.pred_qa[mask], residual2.pred_qa[mask]))
        self.assertFalse(np.allclose(hybrid.pred_qa[~mask], hybrid2.pred_qa[~mask]))

    def test_logistic_probs_and_trees_are_well_behaved(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._pool(mods, self._scores(mods))
        first = mods["oof_all_heads"](pool)
        second = mods["oof_all_heads"](pool)
        prob = first["hashed_logistic_hybrid"].prob_qa
        self.assertTrue(np.all(np.isfinite(prob)))
        self.assertTrue(np.all((prob >= 0.0) & (prob <= 1.0)))
        self.assertTrue(
            np.allclose(
                first["shallow_structural_residual"].pred_qa,
                second["shallow_structural_residual"].pred_qa,
            )
        )
        self.assertTrue(
            np.allclose(
                first["robust_hashed_hybrid"].prob_qk,
                second["robust_hashed_hybrid"].prob_qk,
            )
        )

    def test_gate_requires_gain_absolute_views_and_caps(self) -> None:
        mods = self._import()
        from ossp_router.protocol import TIERS

        def _result(quality: float, pooled_ok: bool = True, fold_ok: bool = True) -> dict:
            return {
                "fold_caps_ok": fold_ok,
                "pooled": {
                    "quality_weighted": quality,
                    "tiers": {tier: {"within_hard_cap": pooled_ok} for tier in TIERS},
                },
            }

        names = mods["CANDIDATE_ORDER"]
        results = {name: _result(0.6877 if name == mods["BASELINE_NAME"] else 0.6910) for name in names}
        views = {
            name: [
                {
                    "kind": "family",
                    "name": "other",
                    "worse_than_gate": False,
                }
            ]
            for name in names
        }
        passing = mods["promotion_gate"](results, views)
        self.assertTrue(passing["passed"])
        self.assertGreaterEqual(
            passing["candidates"][1]["quality_weighted"], mods["CHAMPION_ABS"]
        )

        low_abs = {name: _result(0.6885 if name != mods["BASELINE_NAME"] else 0.6877) for name in names}
        low_gate = mods["promotion_gate"](low_abs, views)
        self.assertFalse(any(row["pass"] for row in low_gate["candidates"] if row["candidate"] != mods["BASELINE_NAME"]))

        view_fail = {
            name: [{"kind": "language", "name": "korean", "worse_than_gate": name != mods["BASELINE_NAME"]}]
            for name in names
        }
        failed_views = mods["promotion_gate"](results, view_fail)
        self.assertFalse(failed_views["passed"])

        cap_fail = {
            name: _result(0.6910 if name != mods["BASELINE_NAME"] else 0.6877, pooled_ok=False)
            for name in names
        }
        self.assertFalse(mods["promotion_gate"](cap_fail, views)["passed"])

    def test_report_and_audit_are_deterministic(self) -> None:
        mods = self._import()
        pool, _folds = self._pool(mods, self._scores(mods))
        report_a, audit_a = mods["assemble"](pool)
        report_b, audit_b = mods["assemble"](pool)
        self.assertEqual(report_a["decision_core_sha256"], report_b["decision_core_sha256"])
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(report_a["audit"]["n_rows"], 8)
        self.assertNotIn("prompt", audit_a["rows"][0])
        self.assertEqual(report_a["promotion_gate"]["baseline"], mods["BASELINE_NAME"])
        kinds = {row["kind"] for row in report_a["stress_views"][mods["BASELINE_NAME"]]}
        self.assertTrue({"family", "fold", "language", "length", "split"} <= kinds)
        for name in mods["CANDIDATE_ORDER"]:
            self.assertIn(name, report_a["results"])
            fast = report_a["results"][name]["pooled"]["model_counts"]["fast"]
            balanced = report_a["results"][name]["pooled"]["model_counts"]["balanced"]
            self.assertEqual(fast["axk1-think"], 0)
            self.assertEqual(balanced["axk1-think"], 0)


if __name__ == "__main__":
    unittest.main()
