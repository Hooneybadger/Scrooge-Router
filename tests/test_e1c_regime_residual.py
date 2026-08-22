# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1C nested-regime residual contracts. Numpy/sklearn tests skip without research deps."""

from __future__ import annotations

import pathlib
import sys
import unittest

from ossp_router.protocol import Episode, TIERS


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _episode(episode_id: str, prompt: str) -> Episode:
    return Episode(episode_id=episode_id, prompt=prompt)


class E1CRegimeResidualTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from decimal import Decimal
            from ossp_router.heuristic import extract_features
            from ossp_router.protocol import InputBatch, Outcome, OutcomeBatch, load_bundled_policy
            from research.lab.e1c_regime_residual import (
                BASELINE_NAME,
                CANDIDATE_NAME,
                FOLD_SEEDS,
                LAMBDA_GRID,
                assemble,
                oof_regime_heads,
                promotion_gate,
                regime_label,
                regimes_of,
                relabel_folds,
                select_lambdas,
            )
            from research.lab.grouped_crossfit import (
                FOLD_SEED,
                assign_balanced_group_folds,
                families_of,
                group_episodes,
                length_view,
            )
            from research.lab.public_pool import PublicPool
        except ImportError:
            self.skipTest("numpy / sklearn / research E1C stack is not installed")
        return {
            "Decimal": Decimal,
            "FOLD_SEED": FOLD_SEED,
            "FOLD_SEEDS": FOLD_SEEDS,
            "InputBatch": InputBatch,
            "LAMBDA_GRID": LAMBDA_GRID,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "PublicPool": PublicPool,
            "assemble": assemble,
            "assign_balanced_group_folds": assign_balanced_group_folds,
            "BASELINE_NAME": BASELINE_NAME,
            "CANDIDATE_NAME": CANDIDATE_NAME,
            "extract_features": extract_features,
            "families_of": families_of,
            "group_episodes": group_episodes,
            "length_view": length_view,
            "load_bundled_policy": load_bundled_policy,
            "np": np,
            "oof_regime_heads": oof_regime_heads,
            "promotion_gate": promotion_gate,
            "regime_label": regime_label,
            "regimes_of": regimes_of,
            "relabel_folds": relabel_folds,
            "select_lambdas": select_lambdas,
        }

    def _pool(self, mods, scores):
        np = mods["np"]
        long_pad = "long-context-padding-block " * 320
        prompts = [
            "Exact copy used twice for grouping.",
            "Exact copy used twice for grouping.",
            "Korean question: 다음 중 옳은 것은 무엇입니까? A. 1 B. 2",
            "def f(x):\n    return x + 1\nassert f(2) == ??",
            "How many apples are left over altogether if each costs 3?",
            "Solve $\\frac{1}{2} + \\frac{1}{3}$ using \\begin{align}.",
            "Word problem: how far and how long if the average is 12.",
            "Unrelated long english prompt about music history " + long_pad,
            "Second long english history essay for the long regime " + long_pad,
            "Short unrelated prompt about a museum visit in spring.",
            "Another short coding riddle: name the output of 2+2.",
            "Final short geography fact about rivers and lakes.",
        ]
        episodes = tuple(
            _episode(f"e1c-{index:02d}", prompt) for index, prompt in enumerate(prompts)
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
                [1.0, 1.32, 2.3],
                [1.0, 1.18, 2.0],
                [1.0, 1.22, 2.05],
                [1.0, 1.28, 2.15],
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
        splits = ("train",) * 6 + ("dev",) * 6
        pool = mods["PublicPool"](
            episodes=episodes,
            texts=tuple(prompts),
            families=families,
            languages=tuple(
                "korean" if "문항" in prompt or "다음" in prompt else "non_korean"
                for prompt in prompts
            ),
            length_views=tuple(mods["length_view"](prompt) for prompt in prompts),
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
                "n_dev": 6,
                "n_episodes": 12,
                "n_train": 6,
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
                [0.10, 0.40, 0.70],
                [0.60, 0.55, 0.20],
                [0.30, 0.80, 0.90],
                [0.15, 0.35, 0.45],
            ],
            dtype=mods["np"].float64,
        )

    def _seed_row(
        self,
        seed: int,
        *,
        delta: float,
        quality: float,
        baseline: float = 0.6877,
        view_fail: bool = False,
        cap_ok: bool = True,
        fold_ok: bool = True,
    ) -> dict:
        return {
            "baseline": {
                "fold_caps_ok": True,
                "pooled": {
                    "quality_weighted": baseline,
                    "tiers": {tier: {"within_hard_cap": True} for tier in TIERS},
                },
            },
            "candidate": {
                "fold_caps_ok": fold_ok,
                "pooled": {
                    "quality_weighted": quality,
                    "tiers": {tier: {"within_hard_cap": cap_ok} for tier in TIERS},
                },
            },
            "delta": delta,
            "fold_seed": seed,
            "views": [
                {
                    "kind": "family",
                    "name": "other",
                    "worse_than_gate": view_fail,
                },
                {
                    "kind": "language",
                    "name": "korean",
                    "worse_than_gate": False,
                },
                {
                    "kind": "length",
                    "name": "len_ge_8000",
                    "worse_than_gate": False,
                },
            ],
        }

    def test_seeds_and_lambda_grid_are_pre_registered(self) -> None:
        mods = self._import()
        self.assertEqual(mods["FOLD_SEEDS"], (20260821, 20260822, 20260823))
        self.assertEqual(mods["LAMBDA_GRID"], (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_regime_is_runtime_content_only_and_order_invariant(self) -> None:
        mods = self._import()
        short = _episode("short", "short prompt")
        long = _episode("long", "L" * 8000)
        self.assertEqual(mods["regime_label"](short), "short")
        self.assertEqual(mods["regime_label"](long), "long")
        self.assertEqual(
            mods["extract_features"](long).long_context,
            True,
        )
        episodes = (short, long, _episode("mid", "M" * 7999))
        first = mods["regimes_of"](episodes)
        second = mods["regimes_of"](tuple(reversed(episodes)))
        self.assertEqual(first, ("short", "long", "short"))
        self.assertEqual(first, tuple(reversed(second)))

    def test_held_out_labels_do_not_change_that_fold_predictions_or_lambdas(self) -> None:
        mods = self._import()
        np = mods["np"]
        scores = self._scores(mods)
        pool, folds = self._pool(mods, scores)
        first_base, first_cand, first_fits = mods["oof_regime_heads"](pool)
        mutated = scores.copy()
        held = int(folds[0])
        mask = np.asarray(folds) == held
        mutated[mask] = np.asarray([0.0, 1.0, 0.0])
        second_base, second_cand, second_fits = mods["oof_regime_heads"](
            pool, scores=mutated
        )
        self.assertTrue(np.allclose(first_base.pred_qa[mask], second_base.pred_qa[mask]))
        self.assertTrue(np.allclose(first_base.pred_qk[mask], second_base.pred_qk[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qa[mask], second_cand.pred_qa[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qk[mask], second_cand.pred_qk[mask]))
        self.assertTrue(
            np.allclose(first_cand.residual_qa[mask], second_cand.residual_qa[mask])
        )
        self.assertTrue(
            np.allclose(first_cand.residual_qk[mask], second_cand.residual_qk[mask])
        )
        held_first = next(row for row in first_fits if row["fold"] == held)
        held_second = next(row for row in second_fits if row["fold"] == held)
        self.assertEqual(held_first["lambda_short"], held_second["lambda_short"])
        self.assertEqual(held_first["lambda_long"], held_second["lambda_long"])
        self.assertEqual(held_first["clip_qa"], held_second["clip_qa"])
        self.assertEqual(held_first["clip_qk"], held_second["clip_qk"])
        self.assertNotEqual(
            held_first["selection"]["chosen"],
            None,
        )

    def test_inner_lambda_does_not_see_outer_labels(self) -> None:
        mods = self._import()
        np = mods["np"]
        scores = self._scores(mods)
        pool, folds = self._pool(mods, scores)
        _base, _cand, first = mods["oof_regime_heads"](pool)
        mutated = scores.copy()
        held = int(folds[-1])
        mask = np.asarray(folds) == held
        mutated[mask] = np.asarray([1.0, 0.0, 1.0])
        _base2, _cand2, second = mods["oof_regime_heads"](pool, scores=mutated)
        first_row = next(row for row in first if row["fold"] == held)
        second_row = next(row for row in second if row["fold"] == held)
        self.assertEqual(
            first_row["selection"]["chosen"]["lambda_short"],
            second_row["selection"]["chosen"]["lambda_short"],
        )
        self.assertEqual(
            first_row["selection"]["chosen"]["lambda_long"],
            second_row["selection"]["chosen"]["lambda_long"],
        )
        self.assertEqual(first_row["clip_qa"], second_row["clip_qa"])
        self.assertEqual(first_row["clip_qk"], second_row["clip_qk"])

    def test_tie_breaks_to_smaller_lambda(self) -> None:
        mods = self._import()
        np = mods["np"]
        n = 6
        scores = np.asarray([[0.2, 0.4, 0.3]] * n, dtype=np.float64)
        costs = np.asarray([[1.0, 1.2, 2.0]] * n, dtype=np.float64)
        zeros = np.zeros(n, dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        chosen = mods["select_lambdas"](
            scores,
            costs,
            ("other",) * n,
            ("non_korean",) * n,
            ("len_lt_120",) * n,
            ("short",) * n,
            tuple(f"tie-{index}" for index in range(n)),
            ones,
            ones,
            zeros,
            zeros,
            1.0,
            1.0,
        )
        self.assertEqual(chosen["chosen"]["lambda_short"], 0.0)
        self.assertEqual(chosen["chosen"]["lambda_long"], 0.0)

    def test_repeated_seed_aggregate_gate(self) -> None:
        mods = self._import()
        passing = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0022, quality=0.6910),
                self._seed_row(20260822, delta=0.0020, quality=0.6905),
                self._seed_row(20260823, delta=0.0018, quality=0.6902),
            ]
        )
        self.assertTrue(passing["passed"])

        lucky_one = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0040, quality=0.6920),
                self._seed_row(20260822, delta=0.0030, quality=0.6910),
                self._seed_row(20260823, delta=0.0004, quality=0.6902),
            ]
        )
        self.assertFalse(lucky_one["passed"])
        self.assertLess(lucky_one["worst_delta"], 0.001)

        low_abs = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0021, quality=0.6895),
                self._seed_row(20260822, delta=0.0021, quality=0.6896),
                self._seed_row(20260823, delta=0.0021, quality=0.6897),
            ]
        )
        self.assertFalse(low_abs["passed"])

        view_fail = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0021, quality=0.6910, view_fail=True),
                self._seed_row(20260822, delta=0.0021, quality=0.6910),
                self._seed_row(20260823, delta=0.0021, quality=0.6910),
            ]
        )
        self.assertFalse(view_fail["passed"])

        cap_fail = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0021, quality=0.6910, cap_ok=False),
                self._seed_row(20260822, delta=0.0021, quality=0.6910),
                self._seed_row(20260823, delta=0.0021, quality=0.6910),
            ]
        )
        self.assertFalse(cap_fail["passed"])

    def test_report_and_audit_are_deterministic_and_prompt_free(self) -> None:
        mods = self._import()
        pool, _folds = self._pool(mods, self._scores(mods))
        report_a, audit_a = mods["assemble"](pool, seeds=(20260821, 20260822))
        report_b, audit_b = mods["assemble"](pool, seeds=(20260821, 20260822))
        self.assertEqual(report_a["decision_core_sha256"], report_b["decision_core_sha256"])
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(report_a["audit"]["n_rows"], 24)
        self.assertFalse(audit_a["prompt_text_included"])
        first_row = audit_a["seeds"]["20260821"]["rows"][0]
        self.assertNotIn("prompt", first_row)
        self.assertIn("regime", first_row)
        self.assertEqual(report_a["experiment"], "e1c-regime-residual")
        self.assertIn("sequential", report_a["sequential_testing"].lower())
        self.assertFalse(report_a["feature"]["runtime_artifact_changed"])
        for seed in ("20260821", "20260822"):
            kinds = {row["kind"] for row in report_a["seed_results"][seed]["views"]}
            self.assertTrue({"family", "fold", "language", "length", "split"} <= kinds)
            for name in (mods["BASELINE_NAME"], mods["CANDIDATE_NAME"]):
                fast = report_a["seed_results"][seed]["results"][name]["pooled"][
                    "model_counts"
                ]["fast"]
                balanced = report_a["seed_results"][seed]["results"][name]["pooled"][
                    "model_counts"
                ]["balanced"]
                self.assertEqual(fast["axk1-think"], 0)
                self.assertEqual(balanced["axk1-think"], 0)

    def test_relabel_keeps_fold_count_and_changes_assignment(self) -> None:
        mods = self._import()
        pool, folds = self._pool(mods, self._scores(mods))
        other = mods["relabel_folds"](pool, 20260822)
        self.assertEqual(other.identity["folds"], 3)
        self.assertEqual(other.identity["fold_seed"], 20260822)
        self.assertNotEqual(tuple(other.folds), tuple(folds))


if __name__ == "__main__":
    unittest.main()
