# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E1F chuf-v1 contracts. Numpy tests skip without research deps."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import unittest
from collections import Counter
from dataclasses import replace

from ossp_router.protocol import Episode, TIERS


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _episode(episode_id: str, prompt: str) -> Episode:
    return Episode(episode_id=episode_id, prompt=prompt)


class E1FCostConditionedFrontierTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from decimal import Decimal
            from ossp_router.protocol import InputBatch, Outcome, OutcomeBatch, load_bundled_policy
            from research.lab.e1_objectives import (
                allocate_all_tiers,
                current_quality_matrix,
                oof_candidate_predictions,
                score_decisions,
            )
            from research.lab.e1c_regime_residual import relabel_folds
            from research.lab.e1f_cost_conditioned_frontier import (
                BASELINE_NAME,
                BETA_PRIOR_A,
                BETA_PRIOR_B,
                CANDIDATE_NAME,
                COST_EPS,
                EXPECTED_BASELINE_20260821,
                EXPORT_PREVIEW_KEYS,
                FOLD_SEEDS,
                MIN_CELL_GROUPS,
                N_COST_BINS,
                PINNED_PUBLIC_DECISION,
                PINNED_PUBLIC_GATE,
                PINNED_PUBLIC_SEEDS,
                PINNED_PUBLIC_VIEW_FAILURES,
                assemble,
                assign_cost_bins,
                ax31_selections_match,
                binomial_counts,
                cost_bin_edges,
                cost_scalar,
                fit_fold_frontier,
                fold_predicted_points,
                global_success_posterior,
                oof_chuf_heads,
                predict_qk,
                premium_parent_models,
                promotion_gate,
                weighted_isotonic_increasing,
            )
            from research.lab.e2_cost_uncertainty import oof_cost_surfaces
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
            self.skipTest("numpy / research E1F stack is not installed")
        return {
            "Decimal": Decimal,
            "EXPECTED_BASELINE_20260821": EXPECTED_BASELINE_20260821,
            "EXPORT_PREVIEW_KEYS": EXPORT_PREVIEW_KEYS,
            "FOLD_SEED": FOLD_SEED,
            "FOLD_SEEDS": FOLD_SEEDS,
            "InputBatch": InputBatch,
            "MIN_CELL_GROUPS": MIN_CELL_GROUPS,
            "N_COST_BINS": N_COST_BINS,
            "PINNED_PUBLIC_DECISION": PINNED_PUBLIC_DECISION,
            "PINNED_PUBLIC_GATE": PINNED_PUBLIC_GATE,
            "PINNED_PUBLIC_SEEDS": PINNED_PUBLIC_SEEDS,
            "PINNED_PUBLIC_VIEW_FAILURES": PINNED_PUBLIC_VIEW_FAILURES,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "PublicPool": PublicPool,
            "TRAIN_INPUTS": TRAIN_INPUTS,
            "allocate_all_tiers": allocate_all_tiers,
            "assemble": assemble,
            "assign_balanced_group_folds": assign_balanced_group_folds,
            "assign_cost_bins": assign_cost_bins,
            "ax31_selections_match": ax31_selections_match,
            "BASELINE_NAME": BASELINE_NAME,
            "BETA_PRIOR_A": BETA_PRIOR_A,
            "BETA_PRIOR_B": BETA_PRIOR_B,
            "binomial_counts": binomial_counts,
            "CANDIDATE_NAME": CANDIDATE_NAME,
            "COST_EPS": COST_EPS,
            "content_tie_keys": content_tie_keys,
            "cost_bin_edges": cost_bin_edges,
            "cost_scalar": cost_scalar,
            "current_quality_matrix": current_quality_matrix,
            "families_of": families_of,
            "fit_fold_frontier": fit_fold_frontier,
            "fold_predicted_points": fold_predicted_points,
            "global_success_posterior": global_success_posterior,
            "group_episodes": group_episodes,
            "length_view": length_view,
            "load_bundled_policy": load_bundled_policy,
            "np": np,
            "oof_candidate_predictions": oof_candidate_predictions,
            "oof_chuf_heads": oof_chuf_heads,
            "oof_cost_surfaces": oof_cost_surfaces,
            "predict_qk": predict_qk,
            "premium_parent_models": premium_parent_models,
            "promotion_gate": promotion_gate,
            "relabel_folds": relabel_folds,
            "score_decisions": score_decisions,
            "weighted_isotonic_increasing": weighted_isotonic_increasing,
        }

    def _pool(self, mods, scores, generations=None, output_tokens=None):
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
        n_gen = [2] * 8 + [4] * 4 if generations is None else list(generations)
        tokens = [4] * 12 if output_tokens is None else list(output_tokens)
        episodes = tuple(
            _episode(f"e1f-{index:02d}", prompt) for index, prompt in enumerate(prompts)
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
        for episode, score_row, generations_i, out_i in zip(
            episodes, scores, n_gen, tokens
        ):
            for model_index, model_id in enumerate(("ax31-light", "ax31", "axk1-think")):
                outcomes.append(
                    mods["Outcome"](
                        episode_id=episode.episode_id,
                        model_id=model_id,
                        score=mods["Decimal"](str(score_row[model_index])),
                        num_generations=int(generations_i),
                        input_tokens=10 + model_index,
                        output_tokens=int(out_i) + model_index,
                    )
                )
        pool = mods["PublicPool"](
            episodes=episodes,
            texts=tuple(prompts),
            families=families,
            languages=tuple(
                "korean" if "다음" in prompt else "non_korean" for prompt in prompts
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
            split_labels=("train",) * 6 + ("dev",) * 6,
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
        ax31_ok: bool = True,
    ) -> dict:
        counts = {"ax31-light": 10, "ax31": 2, "axk1-think": 0 if k1_ok else 3}
        return {
            "ax31_identical_to_baseline": {"all": ax31_ok},
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
                }
            ],
        }

    def test_heldout_quality_and_actual_cost_isolation(self) -> None:
        mods = self._import()
        np = mods["np"]
        scores = self._scores(mods)
        pool, folds = self._pool(mods, scores)
        n, k, _diag = mods["binomial_counts"](pool)
        first_base, first_cand, first_rows = mods["oof_chuf_heads"](pool, n=n, k=k)
        held = int(folds[0])
        mask = np.asarray(folds) == held
        mutated_scores = scores.copy()
        mutated_k = k.copy()
        mutated_scores[mask] = np.asarray([0.0, 1.0, 0.0])
        mutated_k[mask] = np.asarray([0, int(n[mask][0, 0]), 0])
        second_base, second_cand, second_rows = mods["oof_chuf_heads"](
            pool, scores=mutated_scores, n=n, k=mutated_k
        )
        self.assertTrue(np.allclose(first_base.pred_qa[mask], second_base.pred_qa[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qa[mask], second_cand.pred_qa[mask]))
        self.assertTrue(np.allclose(first_cand.pred_qk[mask], second_cand.pred_qk[mask]))
        held_first = next(row for row in first_rows if row["fold"] == held)
        held_second = next(row for row in second_rows if row["fold"] == held)
        self.assertEqual(held_first["bin_edges"], held_second["bin_edges"])
        self.assertEqual(held_first["c_family"], held_second["c_family"])
        self.assertEqual(held_first["c_cell"], held_second["c_cell"])
        self.assertEqual(held_first["cells"], held_second["cells"])

        tokens = [4] * 12
        for index in np.flatnonzero(mask):
            tokens[int(index)] = 4000
        poisoned, _folds = self._pool(mods, scores, output_tokens=tokens)
        _b3, third_cand, third_rows = mods["oof_chuf_heads"](poisoned, n=n, k=k)
        self.assertTrue(np.allclose(first_cand.pred_qk[mask], third_cand.pred_qk[mask]))
        held_third = next(row for row in third_rows if row["fold"] == held)
        self.assertEqual(held_first["bin_edges"], held_third["bin_edges"])
        self.assertEqual(held_first["uk_global"], held_third["uk_global"])

    def test_inner_oof_cost_ignores_actual_heldout_sentinel(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, folds = self._pool(mods, self._scores(mods))
        surfaces = mods["oof_cost_surfaces"](pool)
        fold_ids = np.asarray(folds)
        held = int(folds[0])
        points = mods["fold_predicted_points"](surfaces, fold_ids, held)
        poisoned = dict(surfaces)
        actual = np.asarray(surfaces["actual_costs"], dtype=np.float64).copy()
        actual[fold_ids == held] = 1.0e9
        poisoned["actual_costs"] = actual
        again = mods["fold_predicted_points"](poisoned, fold_ids, held)
        self.assertTrue(np.allclose(points, again))
        self.assertTrue(np.all(np.isfinite(mods["cost_scalar"](points))))

    def test_quantile_bins_are_deterministic_and_order_invariant(self) -> None:
        mods = self._import()
        np = mods["np"]
        values = np.asarray([0.10, 0.20, 0.20, 0.40, 0.80, 1.20], dtype=np.float64)
        edges = mods["cost_bin_edges"](values)
        self.assertEqual(len(edges), mods["N_COST_BINS"] - 1)
        self.assertTrue(np.allclose(edges, mods["cost_bin_edges"](values[::-1])))
        bins = mods["assign_cost_bins"](values, edges)
        self.assertTrue(np.array_equal(bins, mods["assign_cost_bins"](values, edges)))
        self.assertTrue(np.all((bins >= 0) & (bins < mods["N_COST_BINS"])))
        tied = np.full(8, 0.5)
        tied_edges = mods["cost_bin_edges"](tied)
        tied_bins = mods["assign_cost_bins"](tied, tied_edges)
        self.assertEqual(int(np.unique(tied_bins).size), 1)
        point = np.asarray([[1.0, 1.2, 2.0], [1.0, 1.5, 1.5]], dtype=np.float64)
        scalars = mods["cost_scalar"](point)
        self.assertAlmostEqual(float(scalars[0]), float(np.log1p(0.8 / 1.0)))
        self.assertEqual(float(scalars[1]), 0.0)

    def test_hierarchical_posterior_formula_fallback_and_min_groups(self) -> None:
        mods = self._import()
        np = mods["np"]
        n_cell = 25
        families = ["alpha"] * (2 * n_cell) + ["beta"] * 3
        groups = [f"g-{index}" for index in range(2 * n_cell + 3)]
        bins = np.asarray([0] * n_cell + [3] * n_cell + [1, 1, 1], dtype=np.int64)
        n = np.full((2 * n_cell + 3, 3), 4, dtype=np.int64)
        k = np.zeros((2 * n_cell + 3, 3), dtype=np.int64)
        k[:, 1] = 2
        k[:n_cell, 2] = 2
        k[n_cell : 2 * n_cell, 2] = 3
        k[2 * n_cell :, 2] = 1
        edges = np.asarray([0.25, 0.50, 0.75])
        frontier = mods["fit_fold_frontier"](families, groups, bins, n, k, edges)
        p_g = mods["global_success_posterior"](n, k)
        self.assertTrue(np.allclose(frontier.p_global, p_g))
        n_alpha = 4.0 * 2 * n_cell
        k_a_alpha = 2.0 * 2 * n_cell
        k_k_alpha = 2.0 * n_cell + 3.0 * n_cell
        c_f = frontier.c_family
        p_a_f = (k_a_alpha + c_f * p_g[1]) / (n_alpha + c_f)
        p_k_f = (k_k_alpha + c_f * p_g[2]) / (n_alpha + c_f)
        self.assertAlmostEqual(float(frontier.family_p["alpha"][1]), float(p_a_f))
        self.assertAlmostEqual(float(frontier.family_p["alpha"][2]), float(p_k_f))
        n_cell_trials = 4.0 * n_cell
        p_k_b3 = (3.0 * n_cell + frontier.c_cell * p_k_f) / (
            n_cell_trials + frontier.c_cell
        )
        p_a_b3 = (2.0 * n_cell + frontier.c_cell * p_a_f) / (
            n_cell_trials + frontier.c_cell
        )
        self.assertAlmostEqual(float(frontier.family_bin_raw["alpha"][3]), p_k_b3 - p_a_b3)
        self.assertFalse(bool(frontier.family_bin_fallback["alpha"][0]))
        self.assertTrue(bool(frontier.family_bin_fallback["alpha"][1]))
        self.assertTrue(bool(frontier.family_bin_fallback["beta"][1]))
        self.assertEqual(int(frontier.family_bin_groups["beta"][1]), 3)
        self.assertLess(3, mods["MIN_CELL_GROUPS"])
        qk = mods["predict_qk"](["alpha", "beta", "unseen"], np.asarray([3, 1, 2]), frontier)
        self.assertEqual(float(qk[0]), max(float(frontier.family_isotonic["alpha"][3]), 0.0))
        self.assertEqual(float(qk[1]), max(float(frontier.family_isotonic["beta"][1]), 0.0))
        self.assertEqual(float(qk[2]), max(float(frontier.uk_global), 0.0))

    def test_isotonic_pav_is_nondecreasing_and_weighted(self) -> None:
        mods = self._import()
        np = mods["np"]
        equal = mods["weighted_isotonic_increasing"](
            [0.30, 0.10, 0.20, 0.40], [10.0, 10.0, 10.0, 10.0]
        )
        self.assertTrue(np.allclose(equal, [0.20, 0.20, 0.20, 0.40]))
        self.assertTrue(np.all(np.diff(equal) >= -1e-15))
        weighted = mods["weighted_isotonic_increasing"]([0.40, 0.00], [1.0, 99.0])
        self.assertTrue(np.allclose(weighted, [0.004, 0.004]))
        zero = mods["weighted_isotonic_increasing"](
            [0.90, 0.10, 0.20, 0.30], [0.0, 10.0, 10.0, 10.0]
        )
        self.assertTrue(np.all(np.diff(zero) >= -1e-15))
        self.assertTrue(np.allclose(zero[1:], [0.10, 0.20, 0.30]))
        constant = mods["weighted_isotonic_increasing"](
            [0.11, 0.22, 0.33, 0.44], [0.0, 0.0, 0.0, 0.0]
        )
        self.assertTrue(np.allclose(constant, [0.11, 0.22, 0.33, 0.44]))

    def test_ax31_matches_baseline_and_k1_hierarchy(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._pool(mods, self._scores(mods))
        base, cand, _rows = mods["oof_chuf_heads"](pool)
        self.assertTrue(np.array_equal(base.pred_qa, cand.pred_qa))
        ties = mods["content_tie_keys"](pool.texts)
        identity = mods["ax31_selections_match"](
            base.pred_qa, cand.pred_qa, pool.costs, pool.light_total, ties
        )
        self.assertTrue(identity["fast"])
        self.assertTrue(identity["balanced"])
        self.assertTrue(identity["premium_parent"])
        models = mods["allocate_all_tiers"](
            cand.pred_qa, cand.pred_qk, pool.costs, pool.light_total, ties
        )
        base_models = mods["allocate_all_tiers"](
            base.pred_qa, base.pred_qk, pool.costs, pool.light_total, ties
        )
        self.assertEqual(models["fast"], base_models["fast"])
        self.assertEqual(models["balanced"], base_models["balanced"])
        self.assertEqual(
            mods["premium_parent_models"](
                base.pred_qa, pool.costs, pool.light_total, ties
            ),
            mods["premium_parent_models"](
                cand.pred_qa, pool.costs, pool.light_total, ties
            ),
        )
        self.assertNotIn("axk1-think", models["fast"])
        self.assertNotIn("axk1-think", models["balanced"])
        parent = mods["premium_parent_models"](
            cand.pred_qa, pool.costs, pool.light_total, ties
        )
        for model_id, parent_id, qk in zip(models["premium"], parent, cand.pred_qk):
            if model_id == "axk1-think":
                self.assertEqual(parent_id, "ax31")
                self.assertGreater(qk, 0.0)

    def test_seed_baseline_gate_hash_and_audit_without_prompts(self) -> None:
        mods = self._import()
        self.assertEqual(
            mods["FOLD_SEEDS"],
            (20260821, 20260822, 20260823, 20260824, 20260825),
        )
        self.assertEqual(mods["N_COST_BINS"], 4)
        self.assertEqual(mods["MIN_CELL_GROUPS"], 20)
        self.assertEqual(mods["BETA_PRIOR_A"], 0.5)
        self.assertEqual(mods["BETA_PRIOR_B"], 0.5)
        self.assertEqual(mods["COST_EPS"], 1e-15)
        self.assertEqual(mods["CANDIDATE_NAME"], "chuf-v1")
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
        self.assertFalse(passing["phase2_executed"])
        identity_fail = mods["promotion_gate"](
            [
                self._seed_row(20260821, delta=0.0022, quality=0.6910, ax31_ok=False),
                self._seed_row(20260822, delta=0.0020, quality=0.6905),
                self._seed_row(20260823, delta=0.0018, quality=0.6902),
                self._seed_row(20260824, delta=0.0021, quality=0.6908),
                self._seed_row(20260825, delta=0.0019, quality=0.6904),
            ]
        )
        self.assertFalse(identity_fail["experiment_valid"])
        self.assertFalse(identity_fail["passed"])
        pool, _folds = self._pool(mods, self._scores(mods))
        first_report, first_audit = mods["assemble"](pool, seeds=(20260821,))
        second_report, second_audit = mods["assemble"](pool, seeds=(20260821,))
        self.assertEqual(first_report["audit"]["sha256"], second_report["audit"]["sha256"])
        self.assertEqual(
            first_report["decision_core_sha256"], second_report["decision_core_sha256"]
        )
        self.assertEqual(first_audit, second_audit)
        self.assertFalse(first_audit["prompt_text_included"])
        self.assertNotIn("long-context-padding-block", json.dumps(first_audit))
        keys = first_report["export_preview"]["coefficients"]
        self.assertEqual(tuple(sorted(keys)), tuple(sorted(mods["EXPORT_PREVIEW_KEYS"])))
        self.assertFalse(first_report["export_preview"]["selection_use"])
        preview = json.dumps(first_report["export_preview"])
        self.assertNotIn("episode_id", preview)
        self.assertNotIn('"n"', preview)
        self.assertNotIn('"k"', preview)
        if not mods["TRAIN_INPUTS"].is_file():
            return
        from research.lab.public_pool import load_public_pool

        public = mods["relabel_folds"](load_public_pool(), 20260821)
        features = mods["current_quality_matrix"](public.episodes)
        pred_qa, pred_qk = mods["oof_candidate_predictions"](
            features, public.scores, public.folds
        )[mods["BASELINE_NAME"]]
        ties = mods["content_tie_keys"](public.texts)
        models = mods["allocate_all_tiers"](
            pred_qa, pred_qk, public.costs, public.light_total, ties
        )
        scored = mods["score_decisions"](public, models)
        self.assertEqual(scored["quality_weighted"], mods["EXPECTED_BASELINE_20260821"])

    def _true_oof_extras(self, mods, pool, surfaces=None):
        np = mods["np"]
        fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
        cost_bundle = (
            surfaces if surfaces is not None else mods["oof_cost_surfaces"](pool)
        )
        scalars_out = np.full(fold_ids.shape[0], np.nan, dtype=np.float64)
        bins_out = np.full(fold_ids.shape[0], -1, dtype=np.int64)
        hits = np.zeros(fold_ids.shape[0], dtype=np.int64)
        for fold in range(int(fold_ids.max()) + 1):
            train = fold_ids != int(fold)
            test = fold_ids == int(fold)
            points = mods["fold_predicted_points"](cost_bundle, fold_ids, int(fold))
            scalars = mods["cost_scalar"](points)
            edges = mods["cost_bin_edges"](scalars[train])
            bins = mods["assign_cost_bins"](scalars, edges)
            scalars_out[test] = scalars[test]
            bins_out[test] = bins[test]
            hits[test] += 1
        return scalars_out, bins_out, hits

    def _episode_point_surfaces(self, mods, pool):
        np = mods["np"]
        fold_ids = np.asarray(list(pool.folds), dtype=np.int64)
        point = np.zeros((len(pool.episodes), 3), dtype=np.float64)
        for index, episode in enumerate(pool.episodes):
            suffix = episode.episode_id.rsplit("-", 1)[-1]
            rank = int(suffix)
            point[index] = (
                1.0,
                1.05 + 0.08 * rank,
                1.40 + 0.35 * rank,
            )
        inner = []
        for fold in range(int(fold_ids.max()) + 1):
            train = fold_ids != int(fold)
            inner.append(
                {
                    "fold": int(fold),
                    "point": point[train],
                    "train_index": np.flatnonzero(train),
                }
            )
        return {
            "actual_costs": point.copy(),
            "inner_train": inner,
            "point": point,
        }

    def _permute_pool(self, mods, pool, order):
        np = mods["np"]
        order = np.asarray(order, dtype=np.int64)
        return replace(
            pool,
            episodes=tuple(pool.episodes[int(index)] for index in order),
            texts=tuple(pool.texts[int(index)] for index in order),
            families=tuple(pool.families[int(index)] for index in order),
            languages=tuple(pool.languages[int(index)] for index in order),
            length_views=tuple(pool.length_views[int(index)] for index in order),
            group_keys=tuple(pool.group_keys[int(index)] for index in order),
            exact_keys=tuple(pool.exact_keys[int(index)] for index in order),
            template_keys=tuple(pool.template_keys[int(index)] for index in order),
            folds=tuple(pool.folds[int(index)] for index in order),
            scores=np.asarray(pool.scores, dtype=np.float64)[order],
            costs=np.asarray(pool.costs, dtype=np.float64)[order],
            split_labels=tuple(pool.split_labels[int(index)] for index in order),
        )

    def _aligned_map(self, pool, pred_qk, extras):
        mapped = {}
        for index, episode in enumerate(pool.episodes):
            key = (pool.group_keys[index], int(pool.folds[index]), episode.episode_id)
            mapped[key] = (
                float(extras["r"][index]),
                int(extras["cost_bin"][index]),
                float(pred_qk[index]),
            )
        return mapped

    def test_extras_oof_row_assignment_exactly_once(self) -> None:
        mods = self._import()
        np = mods["np"]
        source = (ROOT / "research/lab/e1f_cost_conditioned_frontier.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('extras["cost_bin"][train]', source)
        self.assertNotIn("extras['cost_bin'][train]", source)
        self.assertNotIn('extras["r"][train]', source)
        self.assertNotIn("extras['r'][train]", source)
        pool, _folds = self._pool(mods, self._scores(mods))
        _base, cand, _rows = mods["oof_chuf_heads"](pool)
        expected_r, expected_bins, hits = self._true_oof_extras(mods, pool)
        self.assertTrue(np.array_equal(hits, np.ones(hits.shape[0], dtype=np.int64)))
        self.assertTrue(np.array_equal(cand.extras["cost_bin"], expected_bins))
        self.assertTrue(np.allclose(cand.extras["r"], expected_r, equal_nan=False))
        self.assertTrue(np.all(cand.extras["cost_bin"] >= 0))
        self.assertTrue(np.all(np.isfinite(cand.extras["r"])))

    def test_row_permutation_preserves_group_fold_aligned_map(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._pool(mods, self._scores(mods))
        surfaces = self._episode_point_surfaces(mods, pool)
        _base, cand, _rows = mods["oof_chuf_heads"](pool, surfaces=surfaces)
        original = self._aligned_map(pool, cand.pred_qk, cand.extras)
        orders = (
            np.arange(len(pool.episodes))[::-1],
            np.asarray([3, 11, 0, 7, 1, 8, 2, 10, 4, 9, 5, 6], dtype=np.int64),
        )
        for order in orders:
            shuffled = self._permute_pool(mods, pool, order)
            surfaces2 = self._episode_point_surfaces(mods, shuffled)
            _b2, cand2, _r2 = mods["oof_chuf_heads"](shuffled, surfaces=surfaces2)
            mapped = self._aligned_map(shuffled, cand2.pred_qk, cand2.extras)
            self.assertEqual(set(original), set(mapped))
            for key, values in original.items():
                self.assertEqual(values[1], mapped[key][1])
                self.assertAlmostEqual(values[0], mapped[key][0])
                self.assertAlmostEqual(values[2], mapped[key][2])

    def test_family_bin_cell_counts_match_audit_true_oof(self) -> None:
        mods = self._import()
        pool, _folds = self._pool(mods, self._scores(mods))
        report, audit = mods["assemble"](pool, seeds=(20260821,))
        rows = audit["seeds"]["20260821"]["rows"]
        counts = Counter((row["family"], int(row["cost_bin"])) for row in rows)
        cells = report["seed_results"]["20260821"]["diagnostics"]["family_bin_k1"]
        self.assertEqual(sum(int(cell["n"]) for cell in cells), len(rows))
        self.assertEqual(len(rows), len(pool.episodes))
        seen = set()
        for cell in cells:
            key = (cell["family"], int(cell["bin"]))
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(int(cell["n"]), counts[key])
        self.assertEqual(seen, set(counts))
        _base, cand, _rows = mods["oof_chuf_heads"](pool)
        for index, row in enumerate(rows):
            self.assertEqual(int(row["cost_bin"]), int(cand.extras["cost_bin"][index]))
            self.assertAlmostEqual(float(row["r"]), float(cand.extras["r"][index]))
            self.assertAlmostEqual(float(row["pred_qk"]), float(cand.pred_qk[index]))

    def test_public_oof_scores_and_gate_pinned_unchanged(self) -> None:
        mods = self._import()
        self.assertEqual(mods["PINNED_PUBLIC_DECISION"], "record-e1f-no-promote")
        self.assertEqual(mods["PINNED_PUBLIC_GATE"]["mean_absolute"], 0.6910246212122401)
        self.assertEqual(mods["PINNED_PUBLIC_GATE"]["mean_delta"], 0.002022727272719971)
        self.assertEqual(
            mods["PINNED_PUBLIC_GATE"]["worst_delta"], 0.0013920454545000016
        )
        if not mods["TRAIN_INPUTS"].is_file():
            return
        from research.lab.public_pool import load_public_pool

        public = load_public_pool()
        if int(public.identity.get("n_episodes", 0)) != 2640:
            return
        report, _audit = mods["assemble"](public, seeds=mods["FOLD_SEEDS"])
        self.assertEqual(report["decision"], mods["PINNED_PUBLIC_DECISION"])
        gate = report["promotion_gate"]
        for key, expected in mods["PINNED_PUBLIC_GATE"].items():
            self.assertEqual(gate[key], expected, msg=key)
        self.assertEqual(gate["view_failures"], mods["PINNED_PUBLIC_VIEW_FAILURES"])
        for seed, pinned in mods["PINNED_PUBLIC_SEEDS"].items():
            block = report["seed_results"][str(seed)]
            self.assertEqual(block["baseline_quality"], pinned["baseline_quality"])
            self.assertEqual(block["candidate_quality"], pinned["candidate_quality"])
            self.assertEqual(block["delta"], pinned["delta"])
            self.assertEqual(block["matched_e1_baseline"], pinned["matched_e1_baseline"])
            cand = block["results"][mods["CANDIDATE_NAME"]]["pooled"]["tiers"]
            for tier in ("fast", "balanced", "premium"):
                self.assertEqual(cand[tier]["model_counts"], pinned[tier])

    def test_source_has_no_new_quality_item_model(self) -> None:
        allowed_ossp = {"ossp_router.protocol"}
        forbidden = (
            "family_guard_router",
            "budget_brake_router",
            "feasibility_ladder",
            "hashed_features",
            "ExtraTrees",
            "ossp_router.resources",
        )
        blocked_modules = ("sklearn", "scipy")
        for relative in (
            "research/lab/e1f_cost_conditioned_frontier.py",
            "research/experiments/compare_e1f_cost_conditioned_frontier.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
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
        source = (ROOT / "research/lab/e1f_cost_conditioned_frontier.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("structural_features", source)
        self.assertIn("oof_cost_surfaces", source)
        self.assertIn("baseline_continuous_uplift", source)
