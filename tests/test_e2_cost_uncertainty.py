# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E2/E3 cost-uncertainty and two-price contracts. Numpy tests skip without research deps."""

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


class TwoPriceAllocatorTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from research.lab.e3_two_price import (
                allocate_two_price,
                predicted_ratio,
                rollback_groups,
                selection_spend,
            )
        except ImportError:
            self.skipTest("numpy / research E2 stack is not installed")
        return np, allocate_two_price, predicted_ratio, rollback_groups, selection_spend

    def test_rollback_respects_settle_cap_and_group_atomicity(self) -> None:
        np, allocate_two_price, predicted_ratio, rollback_groups, selection_spend = (
            self._import()
        )
        pred_qa = np.asarray([0.4, 0.4, 0.2, 0.05], dtype=np.float64)
        pred_qk = np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
        buy = np.asarray(
            [
                [1.0, 1.1, 4.0],
                [1.0, 1.1, 4.0],
                [1.0, 1.2, 4.0],
                [1.0, 1.3, 4.0],
            ],
            dtype=np.float64,
        )
        settle = np.asarray(
            [
                [1.0, 3.0, 8.0],
                [1.0, 3.0, 8.0],
                [1.0, 2.5, 8.0],
                [1.0, 1.4, 8.0],
            ],
            dtype=np.float64,
        )
        groups = ("g-dup", "g-dup", "g-b", "g-c")
        ties = ("a", "b", "c", "d")
        models = allocate_two_price(
            pred_qa, pred_qk, buy, settle, 4.0, 1.5, ties, groups, k1_enabled=False
        )
        spend = selection_spend(models, settle)
        self.assertLessEqual(predicted_ratio(spend, 4.0), 1.5 + 1e-12)
        self.assertEqual(models[0] == "ax31", models[1] == "ax31")

    def test_rollback_and_allocation_ignore_input_order(self) -> None:
        np, allocate_two_price, _ratio, _rollback, _spend = self._import()
        pred_qa = np.asarray([0.3, 0.2, 0.1], dtype=np.float64)
        pred_qk = np.zeros(3, dtype=np.float64)
        buy = np.asarray([[1.0, 1.2, 4.0], [1.0, 1.3, 4.0], [1.0, 1.4, 4.0]])
        settle = np.asarray([[1.0, 1.8, 4.0], [1.0, 1.9, 4.0], [1.0, 2.0, 4.0]])
        groups = ("g1", "g2", "g3")
        ties = ("t1", "t2", "t3")
        forward = allocate_two_price(
            pred_qa, pred_qk, buy, settle, 3.0, 1.5, ties, groups, k1_enabled=False
        )
        order = [2, 0, 1]
        backward = allocate_two_price(
            pred_qa[order],
            pred_qk[order],
            buy[order],
            settle[order],
            3.0,
            1.5,
            tuple(ties[i] for i in order),
            tuple(groups[i] for i in order),
            k1_enabled=False,
        )
        restored = [None, None, None]
        for new_index, old in enumerate(order):
            restored[old] = backward[new_index]
        self.assertEqual(tuple(forward), tuple(restored))

    def test_allocator_has_no_actual_light_parameter(self) -> None:
        try:
            import inspect
            from research.lab.e3_two_price import allocate_single_price, allocate_two_price
        except ImportError:
            self.skipTest("numpy / research E2 stack is not installed")
        self.assertNotIn("actual", inspect.signature(allocate_single_price).parameters)
        self.assertNotIn("light_total", inspect.signature(allocate_two_price).parameters)
        self.assertIn(
            "predicted_light_total",
            inspect.signature(allocate_two_price).parameters,
        )


class E2CostContractTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from decimal import Decimal
            from ossp_router.protocol import (
                InputBatch,
                Outcome,
                OutcomeBatch,
                load_bundled_policy,
            )
            from research.lab.e2_cost_uncertainty import (
                CANDIDATE_ORDER,
                GATE_VIEW_KINDS,
                QUALITY_REFERENCE,
                QUALITY_SIGNAL,
                STRESS_95_CAPS,
                assemble,
                clamp_predicted_costs,
                oof_cost_surfaces,
                predicted_light_total,
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
            self.skipTest("numpy / research E2 stack is not installed")
        return {
            "np": np,
            "Decimal": Decimal,
            "InputBatch": InputBatch,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "load_bundled_policy": load_bundled_policy,
            "CANDIDATE_ORDER": CANDIDATE_ORDER,
            "GATE_VIEW_KINDS": GATE_VIEW_KINDS,
            "QUALITY_REFERENCE": QUALITY_REFERENCE,
            "QUALITY_SIGNAL": QUALITY_SIGNAL,
            "STRESS_95_CAPS": STRESS_95_CAPS,
            "promotion_gate": promotion_gate,
            "assemble": assemble,
            "clamp_predicted_costs": clamp_predicted_costs,
            "oof_cost_surfaces": oof_cost_surfaces,
            "predicted_light_total": predicted_light_total,
            "FOLD_SEED": FOLD_SEED,
            "assign_balanced_group_folds": assign_balanced_group_folds,
            "families_of": families_of,
            "group_episodes": group_episodes,
            "PublicPool": PublicPool,
        }

    def _pool(
        self,
        mods,
        scores,
        costs,
        tokens_in,
        tokens_out,
        prompts,
        splits,
        *,
        light_total=None,
    ):
        np = mods["np"]
        episodes = tuple(
            _episode(f"e2-{index:02d}", prompt) for index, prompt in enumerate(prompts)
        )
        grouping = mods["group_episodes"](episodes)
        families = mods["families_of"](episodes)
        folds = mods["assign_balanced_group_folds"](
            grouping.group_keys, families, folds=3, seed=mods["FOLD_SEED"]
        )
        policy = mods["load_bundled_policy"]()
        inputs = mods["InputBatch"](
            1, "ossp-2026-llm-router-challenge", "public", episodes
        )
        outcomes = []
        for episode, score_row, in_row, out_row in zip(
            episodes, scores, tokens_in, tokens_out
        ):
            for model_index, model_id in enumerate(
                ("ax31-light", "ax31", "axk1-think")
            ):
                outcomes.append(
                    mods["Outcome"](
                        episode_id=episode.episode_id,
                        model_id=model_id,
                        score=mods["Decimal"](str(score_row[model_index])),
                        num_generations=1,
                        input_tokens=int(in_row[model_index]),
                        output_tokens=int(out_row[model_index]),
                    )
                )
        outcome_batch = mods["OutcomeBatch"](
            1, "ossp-2026-llm-router-challenge", "public", tuple(outcomes)
        )
        return mods["PublicPool"](
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
            costs=np.asarray(costs, dtype=np.float64),
            light_total=(
                float(light_total)
                if light_total is not None
                else float(np.asarray(costs)[:, 0].sum())
            ),
            identity={
                "fold_seed": mods["FOLD_SEED"],
                "folds": 3,
                "n_dev": sum(1 for label in splits if label == "dev"),
                "n_episodes": len(episodes),
                "n_train": sum(1 for label in splits if label == "train"),
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
            outcomes=outcome_batch,
            policy=policy,
            split_labels=tuple(splits),
        ), folds

    def _default_pool(self, mods, token_scale: float = 1.0):
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
        scores = np.asarray(
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
            dtype=np.float64,
        )
        tokens_in = np.asarray(
            [
                [20, 22, 30],
                [21, 24, 32],
                [18, 19, 28],
                [40, 48, 80],
                [15, 16, 22],
                [25, 27, 40],
                [16, 18, 24],
                [30, 36, 50],
            ],
            dtype=np.float64,
        ) * token_scale
        tokens_out = tokens_in * 0.4
        policy = mods["load_bundled_policy"]()
        from research.lab.e2_cost_uncertainty import assemble_costs, policy_rates

        costs = assemble_costs(
            np.column_stack(
                [
                    tokens_in[:, 0],
                    tokens_out[:, 0],
                    tokens_in[:, 1],
                    tokens_out[:, 1],
                    tokens_in[:, 2],
                    tokens_out[:, 2],
                ]
            ),
            policy_rates(policy),
        )
        splits = ("train", "train", "train", "train", "dev", "dev", "dev", "dev")
        return self._pool(
            mods, scores, costs, tokens_in, tokens_out, prompts, splits
        )

    def test_predicted_costs_are_monotone(self) -> None:
        mods = self._import()
        np = mods["np"]
        raw = np.asarray([[3.0, 1.0, 0.5], [1.0, 1.2, 0.9]], dtype=np.float64)
        clamped = mods["clamp_predicted_costs"](raw)
        self.assertTrue(np.all(clamped[:, 0] <= clamped[:, 1] + 1e-15))
        self.assertTrue(np.all(clamped[:, 1] <= clamped[:, 2] + 1e-15))
        self.assertTrue(np.all(clamped[:, 1] >= clamped[:, 0] * 1.05 - 1e-12))

    def test_held_out_cost_labels_do_not_change_that_fold_prediction(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, folds = self._default_pool(mods)
        first = mods["oof_cost_surfaces"](pool)
        held = int(folds[0])
        mutated_tokens_in = np.asarray(
            [
                [20, 22, 30],
                [21, 24, 32],
                [18, 19, 28],
                [40, 48, 80],
                [15, 16, 22],
                [25, 27, 40],
                [16, 18, 24],
                [30, 36, 50],
            ],
            dtype=np.float64,
        )
        mutated_tokens_in[np.asarray(folds) == held] = np.asarray([400, 500, 800])
        mutated_out = mutated_tokens_in * 0.4
        from research.lab.e2_cost_uncertainty import assemble_costs, policy_rates

        costs = assemble_costs(
            np.column_stack(
                [
                    mutated_tokens_in[:, 0],
                    mutated_out[:, 0],
                    mutated_tokens_in[:, 1],
                    mutated_out[:, 1],
                    mutated_tokens_in[:, 2],
                    mutated_out[:, 2],
                ]
            ),
            policy_rates(pool.policy),
        )
        prompts = list(pool.texts)
        splits = list(pool.split_labels)
        scores = pool.scores
        other, _folds = self._pool(
            mods, scores, costs, mutated_tokens_in, mutated_out, prompts, splits
        )
        second = mods["oof_cost_surfaces"](other)
        mask = np.asarray(folds) == held
        for key in ("point", "sigma", "conservative", "q90", "q99", "kappa", "denom_scale"):
            self.assertTrue(
                np.allclose(first[key][mask], second[key][mask]),
                msg=f"held-out {key} drifted after mutating held-out labels",
            )
        first_cal = next(row for row in first["fold_calibration"] if row["fold"] == held)
        second_cal = next(row for row in second["fold_calibration"] if row["fold"] == held)
        self.assertEqual(first_cal["kappa"], second_cal["kappa"])
        self.assertEqual(first_cal["denom_scale"], second_cal["denom_scale"])
        self.assertEqual(first_cal["global_smear"], second_cal["global_smear"])
        self.assertFalse(np.allclose(first["point"][~mask], second["point"][~mask]))

    def test_inner_residuals_and_report_are_deterministic(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, _folds = self._default_pool(mods)
        first = mods["oof_cost_surfaces"](pool)
        second = mods["oof_cost_surfaces"](pool)
        self.assertTrue(np.allclose(first["point"], second["point"]))
        self.assertTrue(np.allclose(first["sigma"], second["sigma"]))
        self.assertTrue(np.allclose(first["q90"], second["q90"]))
        report_a, audit_a = mods["assemble"](pool)
        report_b, audit_b = mods["assemble"](pool)
        self.assertEqual(report_a["decision_core_sha256"], report_b["decision_core_sha256"])
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(report_a["quality_signal"], mods["QUALITY_SIGNAL"])
        self.assertFalse(report_a["cost_accuracy"]["light_denominator"]["used_actual_light_in_allocator"])
        kinds = {row["kind"] for row in report_a["stress_views"]["quality"]["point_cost_baseline"]}
        ratio_kinds = {row["kind"] for row in report_a["stress_views"]["ratio"]["point_cost_baseline"]}
        self.assertIn("split", kinds)
        self.assertIn("fold", kinds)
        self.assertTrue({"language", "length"} <= ratio_kinds)
        self.assertEqual(
            tuple(report_a["promotion_gate"]["thresholds"]["gated_ratio_view_kinds"]),
            mods["GATE_VIEW_KINDS"],
        )
        self.assertIn("slice's actual light total", report_a["stress_views"]["ratio_denominator"])
        self.assertIn("candidates", report_a["promotion_gate"])
        self.assertIn("reference_budget_valid", report_a["promotion_gate"])
        self.assertTrue(report_a["promotion_gate"]["quality_ok_is_not_sufficient"])
        self.assertEqual(report_a["promotion_gate"]["baseline"], mods["QUALITY_REFERENCE"])
        self.assertNotIn("prompt", audit_a["rows"][0])
        self.assertEqual(report_a["audit"]["n_rows"], len(pool.episodes))
        for name in mods["CANDIDATE_ORDER"]:
            self.assertIn(name, report_a["results"])

    def test_k1_contract_fast_balanced_off_premium_subset(self) -> None:
        mods = self._import()
        pool, _folds = self._default_pool(mods)
        report, _audit = mods["assemble"](pool)
        for name in mods["CANDIDATE_ORDER"]:
            pooled = report["results"][name]["pooled"]
            self.assertEqual(pooled["model_counts"]["fast"]["axk1-think"], 0)
            self.assertEqual(pooled["model_counts"]["balanced"]["axk1-think"], 0)
            for row in report["results"][name]["per_fold"]:
                self.assertEqual(row["tiers"]["fast"]["model_counts"]["axk1-think"], 0)
                self.assertEqual(row["tiers"]["balanced"]["model_counts"]["axk1-think"], 0)
        from research.lab.e3_two_price import allocate_all_tiers_single_price, allocate_all_tiers_two_price

        np = mods["np"]
        pred_qa = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        pred_qk = np.asarray([0.3, 0.2, 0.05, 0.4], dtype=np.float64)
        costs = np.asarray(
            [[1.0, 1.2, 2.0], [1.0, 1.3, 2.2], [1.0, 1.4, 2.4], [1.0, 1.5, 2.6]]
        )
        settle = costs * 1.2
        ties = ("a", "b", "c", "d")
        groups = ("g1", "g2", "g3", "g4")
        caps = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
        single = allocate_all_tiers_single_price(
            pred_qa, pred_qk, costs, 4.0, caps, ties
        )
        two = allocate_all_tiers_two_price(
            pred_qa, pred_qk, costs, {"fast": settle, "balanced": settle, "premium": settle},
            4.0, caps, ties, groups,
        )
        for models in (single, two):
            self.assertTrue(all(model != "axk1-think" for model in models["fast"]))
            self.assertTrue(all(model != "axk1-think" for model in models["balanced"]))
            k1 = {i for i, model in enumerate(models["premium"]) if model == "axk1-think"}
            ax31 = {
                i
                for i, model in enumerate(models["premium"])
                if model in {"ax31", "axk1-think"}
            }
            self.assertTrue(k1 <= ax31)

    def test_gate_applies_caps_bootstrap_and_family_hard_cap(self) -> None:
        mods = self._import()
        from ossp_router.protocol import MODEL_IDS, TIERS

        def _result(quality: float, pooled_ok: bool, fold_ok: bool) -> dict:
            tiers = {tier: {"within_hard_cap": pooled_ok} for tier in TIERS}
            fold_tiers = {tier: {"within_hard_cap": fold_ok} for tier in TIERS}
            return {
                "pooled": {"quality_weighted": quality, "tiers": tiers},
                "per_fold": [{"tiers": fold_tiers}],
            }

        def _coverage(ok: bool = True) -> dict:
            return {
                "models": {
                    model_id: {
                        "q90_coverage": {"slack_ok": ok},
                        "q99_coverage": {"slack_ok": ok},
                    }
                    for model_id in MODEL_IDS
                }
            }

        names = mods["CANDIDATE_ORDER"]
        base_q = 0.67
        results = {name: _result(base_q, True, True) for name in names}
        views = {
            name: [{"kind": "family", "name": "other", "gated": True, "delta": 0.0}]
            for name in names
        }
        ratio = {
            name: [{"kind": "family", "name": "other", "hard_cap_overrun": False}]
            for name in names
        }
        stress = {
            name: {
                "bootstrap": {
                    tier: {"q99_9": mods["STRESS_95_CAPS"][tier] - 0.01}
                    for tier in TIERS
                }
            }
            for name in names
        }
        passing = mods["promotion_gate"](results, views, ratio, stress, _coverage())
        self.assertTrue(passing["passed"])
        self.assertTrue(passing["reference_budget_valid"])

        busted = {name: _result(base_q, False, True) for name in names}
        self.assertFalse(
            mods["promotion_gate"](busted, views, ratio, stress, _coverage())["passed"]
        )
        self.assertFalse(
            mods["promotion_gate"](busted, views, ratio, stress, _coverage())[
                "reference_budget_valid"
            ]
        )

        fold_fail = {name: _result(base_q, True, False) for name in names}
        self.assertFalse(
            mods["promotion_gate"](fold_fail, views, ratio, stress, _coverage())["passed"]
        )

        high_q99 = {
            name: {
                "bootstrap": {
                    "fast": {"q99_9": mods["STRESS_95_CAPS"]["fast"]},
                    "balanced": {"q99_9": mods["STRESS_95_CAPS"]["balanced"] - 0.01},
                    "premium": {"q99_9": mods["STRESS_95_CAPS"]["premium"] - 0.01},
                }
            }
            for name in names
        }
        gated = mods["promotion_gate"](results, views, ratio, high_q99, _coverage())
        self.assertFalse(gated["passed"])
        self.assertTrue(
            all("bootstrap_q99_9:fast" in row["stress_failures"] for row in gated["candidates"])
        )

        family_ratio = {
            name: [{"kind": "family", "name": "other", "hard_cap_overrun": True}]
            for name in names
        }
        family_gate = mods["promotion_gate"](
            results, views, family_ratio, stress, _coverage()
        )
        self.assertFalse(family_gate["passed"])
        self.assertTrue(
            all(
                "family:other" in row["ratio_view_failures"]
                for row in family_gate["candidates"]
            )
        )
        self.assertTrue(family_gate["quality_ok_is_not_sufficient"])

    def test_assemble_ignores_pool_actual_light_sentinel(self) -> None:
        mods = self._import()
        import inspect
        from research.lab.e2_cost_uncertainty import allocate_candidate
        from research.lab.e3_two_price import (
            allocate_all_tiers_single_price,
            allocate_all_tiers_two_price,
            allocate_single_price,
            allocate_two_price,
        )

        for fn in (
            allocate_candidate,
            allocate_single_price,
            allocate_two_price,
            allocate_all_tiers_single_price,
            allocate_all_tiers_two_price,
            mods["predicted_light_total"],
        ):
            names = inspect.signature(fn).parameters
            self.assertNotIn("actual_light_total", names)
            self.assertNotIn("actual_light", names)
            self.assertNotIn("light_total", names)

        pool, _folds = self._default_pool(mods)
        report_a, audit_a = mods["assemble"](pool)
        from dataclasses import replace

        sentineled = replace(pool, light_total=999999.0)
        self.assertAlmostEqual(sentineled.light_total, 999999.0)
        report_b, audit_b = mods["assemble"](sentineled)
        self.assertNotAlmostEqual(pool.light_total, 999999.0)
        for name in mods["CANDIDATE_ORDER"]:
            self.assertEqual(
                report_a["results"][name]["predicted_light_total"],
                report_b["results"][name]["predicted_light_total"],
            )
            self.assertEqual(
                report_a["results"][name]["pooled"]["model_counts"],
                report_b["results"][name]["pooled"]["model_counts"],
            )
        self.assertEqual(
            [row["selected"] for row in audit_a["rows"]],
            [row["selected"] for row in audit_b["rows"]],
        )
        self.assertFalse(
            report_b["cost_accuracy"]["light_denominator"]["used_actual_light_in_allocator"]
        )
        self.assertNotAlmostEqual(
            report_b["results"]["point_cost_baseline"]["predicted_light_total"],
            999999.0,
        )


if __name__ == "__main__":
    unittest.main()
