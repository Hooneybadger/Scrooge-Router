# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1E ebsq-v1 contracts. Numpy tests skip without research deps."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import unittest

from ossp_router.protocol import Episode, TIERS


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _episode(episode_id: str, prompt: str) -> Episode:
    return Episode(episode_id=episode_id, prompt=prompt)


class E1EStratifiedQuotaTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from decimal import Decimal
            from ossp_router.protocol import InputBatch, Outcome, OutcomeBatch, load_bundled_policy
            from research.lab.e1_objectives import (
                allocate_all_tiers,
                oof_candidate_predictions,
                score_decisions,
            )
            from research.lab.e1c_regime_residual import relabel_folds
            from research.lab.e1e_stratified_quota import (
                BASELINE_NAME,
                BETA_PRIOR_A,
                BETA_PRIOR_B,
                CANDIDATE_NAME,
                EXPECTED_BASELINE_20260821,
                EXPORT_PREVIEW_KEYS,
                FAMILY_DEFINITION,
                FOLD_SEEDS,
                MIN_FAMILY_GROUPS,
                adjacent_uplift,
                assemble,
                binomial_counts,
                export_preview_coefficients,
                fit_fold_posterior,
                global_success_posterior,
                method_of_moments_lambda,
                oof_ebsq_heads,
                predict_from_posterior,
                promotion_gate,
            )
            from research.lab.grouped_crossfit import (
                FOLD_SEED,
                assign_balanced_group_folds,
                families_of,
                group_episodes,
                length_view,
            )
            from research.lab.public_pool import TRAIN_INPUTS, PublicPool
            from research.lab.quality_heads import content_tie_keys
        except ImportError:
            self.skipTest("numpy / research E1E stack is not installed")
        return {
            "Decimal": Decimal,
            "EXPECTED_BASELINE_20260821": EXPECTED_BASELINE_20260821,
            "EXPORT_PREVIEW_KEYS": EXPORT_PREVIEW_KEYS,
            "FAMILY_DEFINITION": FAMILY_DEFINITION,
            "FOLD_SEED": FOLD_SEED,
            "FOLD_SEEDS": FOLD_SEEDS,
            "InputBatch": InputBatch,
            "MIN_FAMILY_GROUPS": MIN_FAMILY_GROUPS,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "PublicPool": PublicPool,
            "TRAIN_INPUTS": TRAIN_INPUTS,
            "adjacent_uplift": adjacent_uplift,
            "allocate_all_tiers": allocate_all_tiers,
            "assemble": assemble,
            "assign_balanced_group_folds": assign_balanced_group_folds,
            "BASELINE_NAME": BASELINE_NAME,
            "BETA_PRIOR_A": BETA_PRIOR_A,
            "BETA_PRIOR_B": BETA_PRIOR_B,
            "binomial_counts": binomial_counts,
            "CANDIDATE_NAME": CANDIDATE_NAME,
            "content_tie_keys": content_tie_keys,
            "export_preview_coefficients": export_preview_coefficients,
            "families_of": families_of,
            "fit_fold_posterior": fit_fold_posterior,
            "global_success_posterior": global_success_posterior,
            "group_episodes": group_episodes,
            "length_view": length_view,
            "load_bundled_policy": load_bundled_policy,
            "method_of_moments_lambda": method_of_moments_lambda,
            "np": np,
            "oof_candidate_predictions": oof_candidate_predictions,
            "oof_ebsq_heads": oof_ebsq_heads,
            "predict_from_posterior": predict_from_posterior,
            "promotion_gate": promotion_gate,
            "relabel_folds": relabel_folds,
            "score_decisions": score_decisions,
        }

    def _pool(self, mods, scores, generations=None):
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
        n_gen = (
            [2] * 8 + [4] * 4
            if generations is None
            else list(generations)
        )
        episodes = tuple(
            _episode(f"e1e-{index:02d}", prompt) for index, prompt in enumerate(prompts)
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
        for episode, score_row, generations_i in zip(episodes, scores, n_gen):
            for model_index, model_id in enumerate(("ax31-light", "ax31", "axk1-think")):
                outcomes.append(
                    mods["Outcome"](
                        episode_id=episode.episode_id,
                        model_id=model_id,
                        score=mods["Decimal"](str(score_row[model_index])),
                        num_generations=int(generations_i),
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
                [0.50, 1.00, 0.50],
                [0.50, 0.50, 1.00],
                [0.00, 0.50, 0.00],
                [1.00, 1.00, 0.50],
                [0.50, 0.00, 0.50],
                [0.00, 1.00, 1.00],
                [1.00, 0.50, 1.00],
                [0.50, 0.50, 0.50],
                [0.25, 0.50, 0.75],
                [0.50, 0.75, 0.50],
                [0.00, 0.25, 0.50],
                [0.75, 1.00, 0.75],
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
        k1_ok: bool = True,
        matched: bool | None = None,
    ) -> dict:
        counts = {"ax31-light": 10, "ax31": 2, "axk1-think": 0 if k1_ok else 3}
        return {
            "baseline": {
                "fold_caps_ok": True,
                "k1_fast_balanced_zero": True,
                "pooled": {
                    "quality_weighted": baseline,
                    "tiers": {
                        tier: {
                            "model_counts": dict(counts),
                            "within_hard_cap": True,
                        }
                        for tier in TIERS
                    },
                },
            },
            "candidate": {
                "fold_caps_ok": fold_ok,
                "k1_fast_balanced_zero": k1_ok,
                "pooled": {
                    "quality_weighted": quality,
                    "tiers": {
                        tier: {
                            "model_counts": dict(counts),
                            "within_hard_cap": cap_ok,
                        }
                        for tier in TIERS
                    },
                },
            },
            "delta": delta,
            "fold_seed": seed,
            "matched_e1_baseline": matched if seed != 20260821 else (
                True if matched is None else matched
            ),
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
            ],
        }

    def test_score_times_n_is_integer_and_model_n_matches(self) -> None:
        mods = self._import()
        pool, _folds = self._pool(mods, self._scores(mods))
        n, k, diagnostic = mods["binomial_counts"](pool)
        self.assertEqual(diagnostic["k_non_integer"], 0)
        self.assertEqual(diagnostic["n_mismatch"], 0)
        self.assertTrue(mods["np"].isin(n, (2, 4)).all())
        self.assertTrue(mods["np"].array_equal(n.max(axis=1), n.min(axis=1)))
        self.assertTrue(mods["np"].array_equal(k, (pool.scores * n).astype(mods["np"].int64)))
        if mods["TRAIN_INPUTS"].is_file():
            from research.lab.public_pool import load_public_pool

            public = load_public_pool()
            public_n, public_k, public_diag = mods["binomial_counts"](public)
            self.assertEqual(public_diag["k_non_integer"], 0)
            self.assertEqual(public_diag["n_mismatch"], 0)
            self.assertEqual(int(public_n.size), 7920)
            self.assertTrue(mods["np"].isin(public_n, (2, 4)).all())
            self.assertEqual(int(public_k.min()), 0)

    def test_held_out_labels_do_not_change_that_fold_posterior(self) -> None:
        mods = self._import()
        np = mods["np"]
        scores = self._scores(mods)
        pool, folds = self._pool(mods, scores)
        n, k, _diag = mods["binomial_counts"](pool)
        first_base, first_cand, first_rows = mods["oof_ebsq_heads"](pool, n=n, k=k)
        mutated_scores = scores.copy()
        mutated_k = k.copy()
        held = int(folds[0])
        mask = np.asarray(folds) == held
        mutated_scores[mask] = np.asarray([0.0, 1.0, 0.0])
        mutated_k[mask] = np.asarray([0, int(n[mask][0, 0]), 0])
        second_base, second_cand, second_rows = mods["oof_ebsq_heads"](
            pool, scores=mutated_scores, n=n, k=mutated_k
        )
        self.assertTrue(np.allclose(first_base.pred_qa[mask], second_base.pred_qa[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qa[mask], second_cand.pred_qa[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qk[mask], second_cand.pred_qk[mask]))
        held_first = next(row for row in first_rows if row["fold"] == held)
        held_second = next(row for row in second_rows if row["fold"] == held)
        self.assertEqual(held_first["p_global"], held_second["p_global"])
        self.assertEqual(held_first["u31_global"], held_second["u31_global"])
        self.assertEqual(held_first["uk_global"], held_second["uk_global"])
        self.assertEqual(held_first["lambda_k"], held_second["lambda_k"])
        self.assertEqual(held_first["families"], held_second["families"])

    def test_ax31_predictions_are_constant_within_fold_without_item_model(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, folds = self._pool(mods, self._scores(mods))
        _base, cand, rows = mods["oof_ebsq_heads"](pool)
        fold_ids = np.asarray(folds)
        for fold in sorted(set(fold_ids.tolist())):
            values = cand.pred_qa[fold_ids == fold]
            self.assertEqual(int(np.unique(np.round(values, decimals=12)).size), 1)
            record = next(row for row in rows if row["fold"] == int(fold))
            self.assertTrue(
                np.allclose(values, max(float(record["u31_global"]), 0.0))
            )
            self.assertFalse(record["ax31_family_shrinkage_used_in_selection"])
        source = (ROOT / "research/lab/e1e_stratified_quota.py").read_text(encoding="utf-8")
        self.assertNotIn("hashed_features", source)
        self.assertNotIn("structural_features", source)
        self.assertNotIn("ExtraTrees", source)
        self.assertNotIn("sklearn", source)
        self.assertNotIn("from ossp_router.heuristic", source)

    def test_k1_family_shrink_low_count_and_unseen_fallback(self) -> None:
        mods = self._import()
        np = mods["np"]
        n_big = 25
        families = ["alpha"] * n_big + ["beta"] * 3
        groups = [f"g-{index}" for index in range(n_big + 3)]
        n = np.full((n_big + 3, 3), 4, dtype=np.int64)
        k = np.zeros((n_big + 3, 3), dtype=np.int64)
        k[:, 0] = 1
        k[:n_big, 1] = 2
        k[:n_big, 2] = 3
        k[n_big:, 1] = 1
        k[n_big:, 2] = 4
        posterior = mods["fit_fold_posterior"](families, groups, n, k)
        p = mods["global_success_posterior"](n, k)
        u31, uk = mods["adjacent_uplift"](p)
        self.assertTrue(np.allclose(posterior.p_global, p))
        self.assertEqual(posterior.u31_global, u31)
        self.assertEqual(posterior.uk_global, uk)

        raw_alpha = 75.0 / 100.0 - 50.0 / 100.0
        raw_beta = 12.0 / 12.0 - 3.0 / 12.0
        var_alpha = (0.5 * 0.5) / 100.0 + (0.75 * 0.25) / 100.0
        var_beta = (0.25 * 0.75) / 12.0 + (1.0 * 0.0) / 12.0
        g = np.asarray([float(n_big), 3.0])
        u = np.asarray([raw_alpha, raw_beta])
        v = np.asarray([var_alpha, var_beta])
        expected_lambda, _mom = mods["method_of_moments_lambda"](u, v, g)
        self.assertTrue(np.isfinite(expected_lambda))
        self.assertAlmostEqual(posterior.lambda_k, expected_lambda)
        w_alpha = n_big / (n_big + expected_lambda)
        self.assertEqual(posterior.family_weights["beta"], 0.0)
        self.assertLess(int(posterior.family_group_counts["beta"]), mods["MIN_FAMILY_GROUPS"])
        self.assertAlmostEqual(posterior.family_weights["alpha"], w_alpha)
        expected_alpha = uk + w_alpha * (raw_alpha - uk)
        self.assertAlmostEqual(posterior.family_uk["alpha"], max(min(expected_alpha, 1.0), -1.0))
        self.assertEqual(posterior.family_uk["beta"], uk)
        self.assertEqual(posterior.u_k_for("unseen-family"), uk)

        qa, qk = mods["predict_from_posterior"](["alpha", "beta", "unseen-family"], posterior)
        self.assertTrue(np.allclose(qa, max(u31, 0.0)))
        self.assertEqual(float(qk[0]), max(posterior.family_uk["alpha"], 0.0))
        self.assertEqual(float(qk[1]), max(uk, 0.0))
        self.assertEqual(float(qk[2]), max(uk, 0.0))
        if posterior.family_weights_31["alpha"] > 0.0:
            self.assertNotEqual(posterior.family_u31["alpha"], posterior.u31_global)
        self.assertTrue(np.allclose(qa, max(posterior.u31_global, 0.0)))

        zero_between = mods["method_of_moments_lambda"](
            np.asarray([0.1, 0.1]),
            np.asarray([0.01, 0.01]),
            np.asarray([20.0, 20.0]),
        )[0]
        self.assertTrue(np.isinf(zero_between))
        single = mods["method_of_moments_lambda"](
            np.asarray([0.2]), np.asarray([0.01]), np.asarray([25.0])
        )[0]
        self.assertTrue(np.isinf(single))

    def test_order_invariance_group_atomicity_and_k1_hierarchy(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, folds = self._pool(mods, self._scores(mods))
        n, k, _diag = mods["binomial_counts"](pool)
        first = mods["fit_fold_posterior"](pool.families, pool.group_keys, n, k)
        order = np.arange(len(pool.episodes))[::-1]
        second = mods["fit_fold_posterior"](
            tuple(pool.families[index] for index in order),
            tuple(pool.group_keys[index] for index in order),
            n[order],
            k[order],
        )
        self.assertTrue(np.allclose(first.p_global, second.p_global))
        self.assertEqual(first.uk_global, second.uk_global)
        self.assertEqual(first.family_uk, second.family_uk)
        by_group = {}
        for group, fold in zip(pool.group_keys, folds):
            if group in by_group:
                self.assertEqual(by_group[group], fold)
            by_group[group] = fold
        _base, cand, _rows = mods["oof_ebsq_heads"](pool)
        ties = mods["content_tie_keys"](pool.texts)
        models = mods["allocate_all_tiers"](
            cand.pred_qa, cand.pred_qk, pool.costs, pool.light_total, ties
        )
        self.assertNotIn("axk1-think", models["fast"])
        self.assertNotIn("axk1-think", models["balanced"])
        for model_id, qa, qk in zip(models["premium"], cand.pred_qa, cand.pred_qk):
            if model_id == "axk1-think":
                self.assertGreater(qk, 0.0)
            if model_id == "ax31":
                self.assertGreater(qa, 0.0)

    def test_seed_tuple_baseline_reproduction_and_gate(self) -> None:
        mods = self._import()
        self.assertEqual(
            mods["FOLD_SEEDS"],
            (20260821, 20260822, 20260823, 20260824, 20260825),
        )
        self.assertEqual(mods["BETA_PRIOR_A"], 0.5)
        self.assertEqual(mods["BETA_PRIOR_B"], 0.5)
        self.assertEqual(mods["MIN_FAMILY_GROUPS"], 20)
        self.assertEqual(mods["BASELINE_NAME"], "baseline_continuous_uplift")
        self.assertEqual(mods["CANDIDATE_NAME"], "ebsq-v1")
        passing = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0022, quality=0.6910),
                self._seed_row(20260822, delta=0.0020, quality=0.6905),
                self._seed_row(20260823, delta=0.0018, quality=0.6902),
                self._seed_row(20260824, delta=0.0021, quality=0.6908),
                self._seed_row(20260825, delta=0.0019, quality=0.6904),
            ]
        )
        self.assertTrue(passing["passed"])
        self.assertTrue(passing["phase1_passed"])
        self.assertFalse(passing["phase2_executed"])
        worst = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0040, quality=0.6920),
                self._seed_row(20260822, delta=0.0030, quality=0.6910),
                self._seed_row(20260823, delta=0.0004, quality=0.6902),
                self._seed_row(20260824, delta=0.0021, quality=0.6908),
                self._seed_row(20260825, delta=0.0021, quality=0.6908),
            ]
        )
        self.assertFalse(worst["passed"])
        unmatched = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0021, quality=0.6910, matched=False),
                self._seed_row(20260822, delta=0.0021, quality=0.6910),
                self._seed_row(20260823, delta=0.0021, quality=0.6910),
                self._seed_row(20260824, delta=0.0021, quality=0.6910),
                self._seed_row(20260825, delta=0.0021, quality=0.6910),
            ]
        )
        self.assertFalse(unmatched["passed"])
        self.assertFalse(unmatched["experiment_valid"])
        if not mods["TRAIN_INPUTS"].is_file():
            self.skipTest("public materialized Train is not present")
        from research.lab.public_pool import load_public_pool
        from research.lab.e1_objectives import current_quality_matrix

        pool = mods["relabel_folds"](load_public_pool(), 20260821)
        features = current_quality_matrix(pool.episodes)
        pred_qa, pred_qk = mods["oof_candidate_predictions"](
            features, pool.scores, pool.folds
        )[mods["BASELINE_NAME"]]
        ties = mods["content_tie_keys"](pool.texts)
        models = mods["allocate_all_tiers"](
            pred_qa, pred_qk, pool.costs, pool.light_total, ties
        )
        scored = mods["score_decisions"](pool, models)
        self.assertEqual(scored["quality_weighted"], mods["EXPECTED_BASELINE_20260821"])

    def test_report_audit_deterministic_without_prompts_or_raw_counts(self) -> None:
        mods = self._import()
        pool, _folds = self._pool(mods, self._scores(mods))
        first_report, first_audit = mods["assemble"](pool, seeds=(20260821, 20260822))
        second_report, second_audit = mods["assemble"](pool, seeds=(20260821, 20260822))
        self.assertEqual(first_report["audit"]["sha256"], second_report["audit"]["sha256"])
        self.assertEqual(
            first_report["decision_core_sha256"], second_report["decision_core_sha256"]
        )
        self.assertEqual(first_audit, second_audit)
        self.assertNotIn("elapsed_s", first_report.get("runtime", {}))
        encoded_audit = json.dumps(first_audit)
        self.assertNotIn("long-context-padding-block", encoded_audit)
        self.assertFalse(first_audit["prompt_text_included"])
        keys = first_report["export_preview"]["coefficients"]
        self.assertEqual(tuple(sorted(keys)), tuple(sorted(mods["EXPORT_PREVIEW_KEYS"])))
        self.assertFalse(first_report["export_preview"]["selection_use"])
        self.assertFalse(first_report["phase2"]["executed"])
        preview = json.dumps(first_report["export_preview"])
        self.assertNotIn('"n"', preview)
        self.assertNotIn('"k"', preview)
        self.assertNotIn("episode_id", preview)
        self.assertEqual(first_report["feature"]["quality_feature_dimension"], 0)
        self.assertEqual(
            first_report["feature"]["ax31_policy"], "global_posterior_cost_quota"
        )
        self.assertEqual(keys["family_definition"], mods["FAMILY_DEFINITION"])
        self.assertEqual(keys["min_family_groups"], 20)

    def test_posteriors_are_finite_and_bounded(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._pool(mods, self._scores(mods))
        _base, cand, rows = mods["oof_ebsq_heads"](pool)
        self.assertTrue(np.all(np.isfinite(cand.pred_qa)))
        self.assertTrue(np.all(np.isfinite(cand.pred_qk)))
        self.assertTrue(np.all((cand.pred_qa >= 0.0) & (cand.pred_qa <= 1.0)))
        self.assertTrue(np.all((cand.pred_qk >= 0.0) & (cand.pred_qk <= 1.0)))
        for row in rows:
            for value in row["p_global"].values():
                self.assertGreater(value, 0.0)
                self.assertLess(value, 1.0)
            self.assertGreaterEqual(row["u31_global"], -1.0)
            self.assertLessEqual(row["u31_global"], 1.0)
            self.assertGreaterEqual(row["uk_global"], -1.0)
            self.assertLessEqual(row["uk_global"], 1.0)
            for family in row["families"]:
                if family["u_k"] is not None:
                    self.assertGreaterEqual(family["u_k"], -1.0)
                    self.assertLessEqual(family["u_k"], 1.0)

    def test_source_imports_only_allowed_runtime_utilities(self) -> None:
        allowed_ossp = {"ossp_router.protocol"}
        forbidden = (
            "family_guard_router",
            "budget_brake_router",
            "feasibility_ladder",
            "hashed_features",
            "ossp_router.resources",
        )
        blocked_modules = ("sklearn", "scipy")
        for relative in (
            "research/lab/e1e_stratified_quota.py",
            "research/experiments/compare_e1e_stratified_quota.py",
        ):
            path = ROOT / relative
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                self.assertNotIn(needle, text, msg=f"{relative} contains {needle}")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("ossp_router."))
                        self.assertFalse(
                            any(
                                alias.name == name or alias.name.startswith(name + ".")
                                for name in blocked_modules
                            )
                        )
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("ossp_router."):
                        self.assertIn(node.module, allowed_ossp)
                    self.assertFalse(
                        any(
                            node.module == name or node.module.startswith(name + ".")
                            for name in blocked_modules
                        )
                    )
            self.assertNotIn("src/ossp_router", text)
