# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1D bhmg-v1 contracts. Numpy tests skip without research deps."""

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


class E1DBhmgTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from decimal import Decimal
            from ossp_router.cost_calibrated_router import structural_features
            from ossp_router.protocol import InputBatch, Outcome, OutcomeBatch, load_bundled_policy
            from research.lab.e1_objectives import (
                current_quality_matrix,
                oof_candidate_predictions,
                score_decisions,
            )
            from research.lab.e1c_regime_residual import relabel_folds
            from research.lab.e1d_bhmg import (
                BASELINE_NAME,
                CANDIDATE_NAME,
                EXPECTED_BASELINE_20260821,
                EXPORT_PREVIEW_KEYS,
                FOLD_SEEDS,
                GTOL,
                LAMBDA_BETA,
                LAMBDA_GAMMA,
                MAX_ITERS,
                N_FREE,
                TAU,
                assemble,
                binomial_counts,
                column_scales,
                export_preview_coefficients,
                fit_bhmg,
                model_probabilities,
                oof_bhmg_heads,
                pack_theta,
                promotion_gate,
                scale_features,
                structural_feature_matrix,
                unpack_theta,
                unscale_fit,
                upgrade_from_probabilities,
                _gradient_hessian,
                _penalized_nll,
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
            self.skipTest("numpy / research E1D stack is not installed")
        return {
            "Decimal": Decimal,
            "EXPECTED_BASELINE_20260821": EXPECTED_BASELINE_20260821,
            "EXPORT_PREVIEW_KEYS": EXPORT_PREVIEW_KEYS,
            "FOLD_SEED": FOLD_SEED,
            "FOLD_SEEDS": FOLD_SEEDS,
            "GTOL": GTOL,
            "InputBatch": InputBatch,
            "LAMBDA_BETA": LAMBDA_BETA,
            "LAMBDA_GAMMA": LAMBDA_GAMMA,
            "MAX_ITERS": MAX_ITERS,
            "N_FREE": N_FREE,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "PublicPool": PublicPool,
            "TAU": TAU,
            "TRAIN_INPUTS": TRAIN_INPUTS,
            "assemble": assemble,
            "assign_balanced_group_folds": assign_balanced_group_folds,
            "BASELINE_NAME": BASELINE_NAME,
            "CANDIDATE_NAME": CANDIDATE_NAME,
            "binomial_counts": binomial_counts,
            "column_scales": column_scales,
            "content_tie_keys": content_tie_keys,
            "current_quality_matrix": current_quality_matrix,
            "export_preview_coefficients": export_preview_coefficients,
            "families_of": families_of,
            "fit_bhmg": fit_bhmg,
            "group_episodes": group_episodes,
            "length_view": length_view,
            "load_bundled_policy": load_bundled_policy,
            "model_probabilities": model_probabilities,
            "np": np,
            "oof_bhmg_heads": oof_bhmg_heads,
            "oof_candidate_predictions": oof_candidate_predictions,
            "pack_theta": pack_theta,
            "_gradient_hessian": _gradient_hessian,
            "_penalized_nll": _penalized_nll,
            "promotion_gate": promotion_gate,
            "relabel_folds": relabel_folds,
            "scale_features": scale_features,
            "score_decisions": score_decisions,
            "structural_feature_matrix": structural_feature_matrix,
            "structural_features": structural_features,
            "unpack_theta": unpack_theta,
            "unscale_fit": unscale_fit,
            "upgrade_from_probabilities": upgrade_from_probabilities,
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
            _episode(f"e1d-{index:02d}", prompt) for index, prompt in enumerate(prompts)
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
        singular: bool = False,
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
            "singular": singular,
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

    def test_frozen_constants_and_seed_tuple(self) -> None:
        mods = self._import()
        self.assertEqual(
            mods["FOLD_SEEDS"],
            (20260821, 20260822, 20260823, 20260824, 20260825),
        )
        self.assertEqual(mods["LAMBDA_BETA"], 10.0)
        self.assertEqual(mods["LAMBDA_GAMMA"], 100.0)
        self.assertEqual(mods["TAU"], 0.25)
        self.assertEqual(mods["MAX_ITERS"], 200)
        self.assertEqual(mods["GTOL"], 1e-8)
        self.assertEqual(mods["BASELINE_NAME"], "baseline_continuous_uplift")
        self.assertEqual(mods["CANDIDATE_NAME"], "bhmg-v1")
        self.assertEqual(mods["EXPECTED_BASELINE_20260821"], 0.6877178030302)

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

    def test_features_are_structural_14d_without_hash(self) -> None:
        mods = self._import()
        pool, _folds = self._pool(mods, self._scores(mods))
        matrix = mods["structural_feature_matrix"](pool.episodes)
        self.assertEqual(matrix.shape, (12, 14))
        for episode, row in zip(pool.episodes, matrix):
            self.assertTrue(
                mods["np"].allclose(row, mods["structural_features"](episode))
            )
        with_intercept = mods["current_quality_matrix"](pool.episodes)
        self.assertTrue(mods["np"].allclose(matrix, with_intercept[:, 1:]))
        constant = mods["np"].full((8, 14), 0.6931471805599453)
        constant[:, 0] = mods["np"].linspace(1.0, 2.0, 8)
        scales = mods["column_scales"](constant)
        self.assertEqual(float(scales[1]), 1.0)
        self.assertGreater(float(scales[0]), 0.0)
        self.assertNotEqual(float(scales[0]), 1.0)
        source = (ROOT / "research/lab/e1d_bhmg.py").read_text(encoding="utf-8")
        self.assertNotIn("hashed_features", source)
        self.assertNotIn("from ossp_router.heuristic", source)

    def test_held_out_labels_do_not_change_that_fold_scale_or_prediction(self) -> None:
        mods = self._import()
        np = mods["np"]
        scores = self._scores(mods)
        pool, folds = self._pool(mods, scores)
        n, k, _diag = mods["binomial_counts"](pool)
        first_base, first_cand, first_fits = mods["oof_bhmg_heads"](pool, n=n, k=k)
        mutated_scores = scores.copy()
        mutated_k = k.copy()
        held = int(folds[0])
        mask = np.asarray(folds) == held
        mutated_scores[mask] = np.asarray([0.0, 1.0, 0.0])
        mutated_k[mask] = np.asarray([0, int(n[mask][0, 0]), 0])
        second_base, second_cand, second_fits = mods["oof_bhmg_heads"](
            pool, scores=mutated_scores, n=n, k=mutated_k
        )
        self.assertTrue(np.allclose(first_base.pred_qa[mask], second_base.pred_qa[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qa[mask], second_cand.pred_qa[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qk[mask], second_cand.pred_qk[mask]))
        self.assertTrue(np.allclose(first_cand.extras["pL"][mask], second_cand.extras["pL"][mask]))
        held_first = next(row for row in first_fits if row["fold"] == held)
        held_second = next(row for row in second_fits if row["fold"] == held)
        self.assertEqual(held_first["scale"], held_second["scale"])
        self.assertEqual(held_first["alpha"], held_second["alpha"])
        self.assertEqual(held_first["beta"], held_second["beta"])
        self.assertEqual(held_first["gamma"], held_second["gamma"])

    def test_gamma_sums_to_zero_across_models(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._pool(mods, self._scores(mods))
        n, k, _diag = mods["binomial_counts"](pool)
        features = mods["structural_feature_matrix"](pool.episodes)
        scales = mods["column_scales"](features)
        fit = mods["unscale_fit"](
            mods["fit_bhmg"](mods["scale_features"](features, scales), n, k),
            scales,
        )
        self.assertFalse(fit.singular)
        self.assertTrue(np.allclose(fit.gamma.sum(axis=0), 0.0))

    def test_export_preview_keys_exact_without_n_or_k(self) -> None:
        mods = self._import()
        np = mods["np"]
        preview = mods["export_preview_coefficients"](
            np.zeros(3), np.zeros(14), np.zeros((3, 14))
        )
        self.assertEqual(tuple(sorted(preview)), tuple(sorted(mods["EXPORT_PREVIEW_KEYS"])))
        encoded = json.dumps(preview)
        self.assertNotIn('"n"', encoded)
        self.assertNotIn('"k"', encoded)
        self.assertNotIn("num_generations", encoded)

    def test_probabilities_finite_deterministic_and_order_invariant(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._pool(mods, self._scores(mods))
        first_base, first_cand, first_fits = mods["oof_bhmg_heads"](pool)
        second_base, second_cand, second_fits = mods["oof_bhmg_heads"](pool)
        for name in ("pL", "pA", "pK"):
            values = first_cand.extras[name]
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
        self.assertTrue(np.allclose(first_cand.pred_qa, second_cand.pred_qa))
        self.assertTrue(np.allclose(first_base.pred_qk, second_base.pred_qk))
        self.assertEqual(first_fits[0]["iters"], second_fits[0]["iters"])
        features = mods["structural_feature_matrix"](pool.episodes)
        reversed_features = mods["structural_feature_matrix"](tuple(reversed(pool.episodes)))
        self.assertTrue(np.allclose(features, reversed_features[::-1]))
        n, k, _diag = mods["binomial_counts"](pool)
        scales = mods["column_scales"](features)
        fit_a = mods["fit_bhmg"](mods["scale_features"](features, scales), n, k)
        order = np.arange(features.shape[0])[::-1]
        fit_b = mods["fit_bhmg"](
            mods["scale_features"](features[order], scales), n[order], k[order]
        )
        self.assertTrue(np.allclose(fit_a.alpha, fit_b.alpha, atol=1e-8, rtol=1e-8))
        self.assertTrue(np.allclose(fit_a.beta, fit_b.beta, atol=1e-8, rtol=1e-8))

    def test_seed_tuple_and_public_baseline_reproduction(self) -> None:
        mods = self._import()
        self.assertEqual(
            mods["FOLD_SEEDS"],
            (20260821, 20260822, 20260823, 20260824, 20260825),
        )
        if not mods["TRAIN_INPUTS"].is_file():
            self.skipTest("public materialized Train is not present")
        from research.lab.public_pool import load_public_pool

        pool = mods["relabel_folds"](load_public_pool(), 20260821)
        features = mods["current_quality_matrix"](pool.episodes)
        pred_qa, pred_qk = mods["oof_candidate_predictions"](
            features, pool.scores, pool.folds
        )[mods["BASELINE_NAME"]]
        ties = mods["content_tie_keys"](pool.texts)
        from research.lab.e1_objectives import allocate_all_tiers

        models = allocate_all_tiers(
            pred_qa, pred_qk, pool.costs, pool.light_total, ties
        )
        scored = mods["score_decisions"](pool, models)
        self.assertEqual(
            scored["quality_weighted"], mods["EXPECTED_BASELINE_20260821"]
        )

    def test_view_mean_worst_and_champion_gate(self) -> None:
        mods = self._import()
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
        self.assertLess(worst["worst_delta"], 0.001)

        low_abs = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0021, quality=0.6895),
                self._seed_row(20260822, delta=0.0021, quality=0.6896),
                self._seed_row(20260823, delta=0.0021, quality=0.6897),
                self._seed_row(20260824, delta=0.0021, quality=0.6898),
                self._seed_row(20260825, delta=0.0021, quality=0.6894),
            ]
        )
        self.assertFalse(low_abs["passed"])

        view_fail = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0021, quality=0.6910, view_fail=True),
                self._seed_row(20260822, delta=0.0021, quality=0.6910),
                self._seed_row(20260823, delta=0.0021, quality=0.6910),
                self._seed_row(20260824, delta=0.0021, quality=0.6910),
                self._seed_row(20260825, delta=0.0021, quality=0.6910),
            ]
        )
        self.assertFalse(view_fail["passed"])

        unmatched = mods["promotion_gate"](
            [
                self._seed_row(
                    20260821, delta=0.0021, quality=0.6910, matched=False
                ),
                self._seed_row(20260822, delta=0.0021, quality=0.6910),
                self._seed_row(20260823, delta=0.0021, quality=0.6910),
                self._seed_row(20260824, delta=0.0021, quality=0.6910),
                self._seed_row(20260825, delta=0.0021, quality=0.6910),
            ]
        )
        self.assertFalse(unmatched["passed"])
        self.assertFalse(unmatched["experiment_valid"])

    def test_report_and_audit_hashes_are_deterministic(self) -> None:
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
        keys = first_report["export_preview"]["coefficients"]
        self.assertEqual(tuple(sorted(keys)), tuple(sorted(mods["EXPORT_PREVIEW_KEYS"])))
        self.assertFalse(first_report["export_preview"]["selection_use"])
        self.assertFalse(first_report["phase2"]["executed"])

    def _tiny_glm_fixture(self, mods):
        """Deterministic 6×14 design with mixed n∈{2,4}. Solver is not refit."""

        np = mods["np"]
        rng = np.random.RandomState(20260821)
        features = rng.normal(scale=0.35, size=(6, 14))
        features[:, 0] = np.linspace(-0.8, 0.8, 6)
        n = np.asarray(
            [[2, 2, 2], [4, 4, 4], [2, 2, 2], [2, 2, 2], [4, 4, 4], [2, 2, 2]],
            dtype=np.int64,
        )
        k = np.asarray(
            [[1, 2, 0], [3, 2, 1], [0, 1, 2], [2, 1, 1], [1, 3, 2], [1, 0, 1]],
            dtype=np.int64,
        )
        theta = mods["pack_theta"](
            np.asarray([-0.25, 0.10, 0.40]),
            0.05 * rng.normal(size=14),
            0.02 * rng.normal(size=14),
            0.02 * rng.normal(size=14),
        )
        return features, n, k, theta

    @staticmethod
    def _independent_penalized_nll(features, n, k, theta, lambda_beta, lambda_gamma):
        """First-principles NLL + γ_K=-(γ_L+γ_A) penalty. Not the lab helper."""

        import numpy as np

        vector = np.asarray(theta, dtype=np.float64).reshape(45)
        alpha = vector[0:3]
        beta = vector[3:17]
        gamma_l = vector[17:31]
        gamma_a = vector[31:45]
        gamma_k = -(gamma_l + gamma_a)
        eta = np.column_stack(
            [
                alpha[0] + features @ (beta + gamma_l),
                alpha[1] + features @ (beta + gamma_a),
                alpha[2] + features @ (beta + gamma_k),
            ]
        )

        def _log_sigmoid(logits):
            values = np.asarray(logits, dtype=np.float64)
            out = np.empty_like(values)
            positive = values >= 0.0
            out[positive] = -np.log1p(np.exp(-values[positive]))
            out[~positive] = values[~positive] - np.log1p(np.exp(values[~positive]))
            return out

        n_float = np.asarray(n, dtype=np.float64)
        k_float = np.asarray(k, dtype=np.float64)
        nll = float(
            np.sum(
                -k_float * _log_sigmoid(eta)
                - (n_float - k_float) * _log_sigmoid(-eta)
            )
        )
        penalty = 0.5 * lambda_beta * float(np.dot(beta, beta)) + 0.5 * lambda_gamma * (
            float(np.dot(gamma_l, gamma_l))
            + float(np.dot(gamma_a, gamma_a))
            + float(np.dot(gamma_k, gamma_k))
        )
        return nll + penalty

    def test_independent_nll_matches_implementation(self) -> None:
        mods = self._import()
        features, n, k, theta = self._tiny_glm_fixture(mods)
        independent = self._independent_penalized_nll(
            features, n, k, theta, mods["LAMBDA_BETA"], mods["LAMBDA_GAMMA"]
        )
        implemented = mods["_penalized_nll"](features, n, k, theta)
        self.assertAlmostEqual(independent, implemented, places=12)

    def test_analytic_gradient_hessian_match_finite_difference(self) -> None:
        mods = self._import()
        np = mods["np"]
        features, n, k, theta = self._tiny_glm_fixture(mods)
        gradient, hessian = mods["_gradient_hessian"](features, n, k, theta)
        step = 1e-6
        fd_grad = np.empty(mods["N_FREE"], dtype=np.float64)
        for index in range(mods["N_FREE"]):
            perturb = np.zeros(mods["N_FREE"], dtype=np.float64)
            perturb[index] = step
            plus = mods["_penalized_nll"](features, n, k, theta + perturb)
            minus = mods["_penalized_nll"](features, n, k, theta - perturb)
            fd_grad[index] = (plus - minus) / (2.0 * step)
        self.assertTrue(np.allclose(gradient, fd_grad, atol=1e-6, rtol=1e-6))

        fd_hess = np.empty((mods["N_FREE"], mods["N_FREE"]), dtype=np.float64)
        for index in range(mods["N_FREE"]):
            perturb = np.zeros(mods["N_FREE"], dtype=np.float64)
            perturb[index] = step
            plus, _unused = mods["_gradient_hessian"](features, n, k, theta + perturb)
            minus, _unused = mods["_gradient_hessian"](features, n, k, theta - perturb)
            fd_hess[:, index] = (plus - minus) / (2.0 * step)
        self.assertTrue(np.allclose(hessian, fd_hess, atol=2e-6, rtol=2e-6))
        self.assertTrue(np.allclose(hessian, hessian.T, atol=1e-12))

    def test_gamma_k_penalty_cross_terms_and_alpha_unpenalized(self) -> None:
        mods = self._import()
        np = mods["np"]
        features = np.zeros((3, 14), dtype=np.float64)
        empty_n = np.zeros((3, 3), dtype=np.int64)
        empty_k = np.zeros((3, 3), dtype=np.int64)
        gamma_l = np.linspace(0.01, 0.14, 14)
        gamma_a = np.linspace(-0.07, 0.06, 14)
        theta = mods["pack_theta"](
            np.asarray([0.3, -0.2, 0.5]),
            np.linspace(-0.04, 0.05, 14),
            gamma_l,
            gamma_a,
        )
        penalty_only = mods["_penalized_nll"](features, empty_n, empty_k, theta)
        shifted = theta.copy()
        shifted[0:3] += np.asarray([1.25, -0.75, 2.0])
        self.assertEqual(
            penalty_only,
            mods["_penalized_nll"](features, empty_n, empty_k, shifted),
        )
        identity = np.eye(14, dtype=np.float64)
        gradient, hessian = mods["_gradient_hessian"](features, empty_n, empty_k, theta)
        self.assertTrue(np.allclose(gradient[0:3], 0.0))
        self.assertTrue(np.allclose(hessian[0:3, 0:3], 0.0))
        self.assertTrue(
            np.allclose(
                gradient[17:31],
                mods["LAMBDA_GAMMA"] * (2.0 * gamma_l + gamma_a),
            )
        )
        self.assertTrue(
            np.allclose(
                gradient[31:45],
                mods["LAMBDA_GAMMA"] * (2.0 * gamma_a + gamma_l),
            )
        )
        self.assertTrue(
            np.allclose(hessian[17:31, 31:45], mods["LAMBDA_GAMMA"] * identity)
        )
        self.assertTrue(
            np.allclose(hessian[31:45, 17:31], mods["LAMBDA_GAMMA"] * identity)
        )
        self.assertTrue(
            np.allclose(hessian[17:31, 17:31], 2.0 * mods["LAMBDA_GAMMA"] * identity)
        )

        features, n, k, data_theta = self._tiny_glm_fixture(mods)
        data_nll = mods["_penalized_nll"](features, n, k, data_theta)
        alpha, beta, gamma_l_d, gamma_a_d = mods["unpack_theta"](data_theta)
        moved_alpha = mods["pack_theta"](alpha + 0.35, beta, gamma_l_d, gamma_a_d)
        self.assertNotAlmostEqual(
            data_nll,
            mods["_penalized_nll"](features, n, k, moved_alpha),
        )
        empty_data = mods["_penalized_nll"](
            features, np.zeros_like(n), np.zeros_like(k), data_theta
        )
        empty_moved = mods["_penalized_nll"](
            features, np.zeros_like(n), np.zeros_like(k), moved_alpha
        )
        self.assertEqual(empty_data, empty_moved)

    def test_upgrade_applies_mu_ge_tau_times_bernoulli_sd(self) -> None:
        mods = self._import()
        np = mods["np"]
        tau = mods["TAU"]
        probabilities = np.asarray(
            [
                [0.30, 0.70, 0.85],
                [0.45, 0.50, 0.52],
                [0.00, 1.00, 1.00],
                [0.40, 0.40, 0.40],
            ],
            dtype=np.float64,
        )
        pred_qa, pred_qk, extras = mods["upgrade_from_probabilities"](probabilities)
        mu31 = probabilities[:, 1] - probabilities[:, 0]
        muk1 = probabilities[:, 2] - probabilities[:, 1]
        s31 = np.sqrt(
            probabilities[:, 1] * (1.0 - probabilities[:, 1])
            + probabilities[:, 0] * (1.0 - probabilities[:, 0])
        )
        sk1 = np.sqrt(
            probabilities[:, 2] * (1.0 - probabilities[:, 2])
            + probabilities[:, 1] * (1.0 - probabilities[:, 1])
        )
        self.assertTrue(np.allclose(extras["mu31"], mu31))
        self.assertTrue(np.allclose(extras["s31"], s31))
        self.assertTrue(np.allclose(extras["muk1"], muk1))
        self.assertTrue(np.allclose(extras["sk1"], sk1))
        expected_qa = np.where(mu31 >= tau * s31, mu31, 0.0)
        expected_qk = np.where(muk1 >= tau * sk1, muk1, 0.0)
        self.assertTrue(np.array_equal(pred_qa, expected_qa))
        self.assertTrue(np.array_equal(pred_qk, expected_qk))
        self.assertGreater(pred_qa[0], 0.0)
        self.assertEqual(pred_qa[1], 0.0)
        self.assertEqual(pred_qa[2], mu31[2])
        self.assertEqual(pred_qa[3], 0.0)
        self.assertNotIn("se", extras)
        self.assertNotIn("n", extras)

        keep = np.asarray([[0.35, 0.55, 0.80]], dtype=np.float64)
        mu = float(keep[0, 1] - keep[0, 0])
        s = float(np.sqrt(keep[0, 1] * (1.0 - keep[0, 1]) + keep[0, 0] * (1.0 - keep[0, 0])))
        tau_eq = mu / s
        on_boundary, _qk, extras_eq = mods["upgrade_from_probabilities"](
            keep, tau=tau_eq
        )
        self.assertEqual(float(on_boundary[0]), mu)
        self.assertEqual(float(extras_eq["s31"][0]), s)
        just_below, _qk, _ex = mods["upgrade_from_probabilities"](
            keep, tau=np.nextafter(tau_eq, tau_eq + 1.0)
        )
        self.assertEqual(float(just_below[0]), 0.0)

    def test_singular_hessian_is_fail_closed(self) -> None:
        mods = self._import()
        np = mods["np"]
        empty = mods["fit_bhmg"](
            np.zeros((0, 14), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
        )
        self.assertTrue(empty.singular)
        self.assertFalse(empty.converged)
        constant = np.full((4, 14), 0.0, dtype=np.float64)
        scales = mods["column_scales"](constant)
        self.assertTrue(np.allclose(scales, 1.0))
        source = (ROOT / "research/lab/e1d_bhmg.py").read_text(encoding="utf-8")
        self.assertIn("fail-closed", source)
        self.assertNotIn("pinv", source)
        self.assertNotIn("lstsq", source)

    def test_source_imports_only_allowed_runtime_utilities(self) -> None:
        allowed_ossp = {
            "ossp_router.cost_calibrated_router",
            "ossp_router.protocol",
        }
        forbidden = (
            "family_guard_router",
            "budget_brake_router",
            "feasibility_ladder",
            "hashed_features",
            "ossp_router.resources",
        )
        blocked_modules = ("sklearn", "scipy")
        for relative in (
            "research/lab/e1d_bhmg.py",
            "research/experiments/compare_e1d_bhmg.py",
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
