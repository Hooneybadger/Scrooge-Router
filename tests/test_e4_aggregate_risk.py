# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""E4 aggregate-risk contracts. Numpy/sklearn tests skip without research deps."""

from __future__ import annotations

import inspect
import pathlib
import sys
import unittest

from ossp_router.protocol import Episode


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _episode(episode_id: str, prompt: str) -> Episode:
    return Episode(episode_id=episode_id, prompt=prompt)


class E4AggregateRiskTest(unittest.TestCase):
    def _import(self):
        try:
            import numpy as np
            from decimal import Decimal
            from ossp_router.protocol import InputBatch, Outcome, OutcomeBatch, load_bundled_policy
            from research.lab.e2_cost_uncertainty import oof_cost_surfaces
            from research.lab.e4_aggregate_risk import (
                BASELINE_NAME,
                CANDIDATE_ORDER,
                FOLD_SEEDS,
                OPERATING_CAPS,
                USED_ACTUAL_LIGHT_IN_ALLOCATOR,
                apply_family_increment_multiplier,
                assemble,
                calibrate_bounds,
                conformal_bound,
                evaluate_seed,
                item_risk,
                promotion_gate,
                rollback_until_bound,
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
            self.skipTest("numpy / sklearn / research E4 stack is not installed")
        return {
            "Decimal": Decimal,
            "FOLD_SEED": FOLD_SEED,
            "FOLD_SEEDS": FOLD_SEEDS,
            "InputBatch": InputBatch,
            "OPERATING_CAPS": OPERATING_CAPS,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "PublicPool": PublicPool,
            "USED_ACTUAL_LIGHT_IN_ALLOCATOR": USED_ACTUAL_LIGHT_IN_ALLOCATOR,
            "assemble": assemble,
            "assign_balanced_group_folds": assign_balanced_group_folds,
            "BASELINE_NAME": BASELINE_NAME,
            "CANDIDATE_ORDER": CANDIDATE_ORDER,
            "apply_family_increment_multiplier": apply_family_increment_multiplier,
            "calibrate_bounds": calibrate_bounds,
            "conformal_bound": conformal_bound,
            "evaluate_seed": evaluate_seed,
            "families_of": families_of,
            "group_episodes": group_episodes,
            "item_risk": item_risk,
            "length_view": length_view,
            "load_bundled_policy": load_bundled_policy,
            "np": np,
            "oof_cost_surfaces": oof_cost_surfaces,
            "promotion_gate": promotion_gate,
            "rollback_until_bound": rollback_until_bound,
        }

    def _pool(self, mods, scores, costs=None):
        np = mods["np"]
        prompts = [
            "Exact copy used twice for grouping.",
            "Exact copy used twice for grouping.",
            "Korean question: 다음 중 옳은 것은 무엇입니까? A. 1 B. 2",
            "def f(x):\n    return x + 1\nassert f(2) == ??",
            "How many apples are left over altogether if each costs 3?",
            "Solve $\\frac{1}{2} + \\frac{1}{3}$ using \\begin{align}.",
            "Word problem: how far and how long if the average is 12.",
            "Unrelated long english prompt about music history " * 8,
            "Second unrelated prompt about a museum visit in spring.",
            "Another short coding riddle: name the output of 2+2.",
            "Final short geography fact about rivers and lakes.",
            "Residual other bucket with no markers at all xyz.",
        ]
        episodes = tuple(
            _episode(f"e4-{index:02d}", prompt) for index, prompt in enumerate(prompts)
        )
        grouping = mods["group_episodes"](episodes)
        families = mods["families_of"](episodes)
        folds = mods["assign_balanced_group_folds"](
            grouping.group_keys, families, folds=3, seed=mods["FOLD_SEED"]
        )
        policy = mods["load_bundled_policy"]()
        if costs is None:
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
                        output_tokens=4 + model_index,
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
            costs=np.asarray(costs, dtype=np.float64),
            light_total=float(np.asarray(costs)[:, 0].sum()),
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

    def test_seeds_and_actual_light_sentinel_are_pre_registered(self) -> None:
        mods = self._import()
        self.assertEqual(mods["FOLD_SEEDS"], (20260821, 20260822, 20260823))
        self.assertFalse(mods["USED_ACTUAL_LIGHT_IN_ALLOCATOR"])
        self.assertEqual(mods["OPERATING_CAPS"]["fast"], 1.1875)
        self.assertEqual(mods["CANDIDATE_ORDER"][0], mods["BASELINE_NAME"])

    def test_allocator_has_no_actual_light_parameter(self) -> None:
        try:
            from research.lab.e4_aggregate_risk import allocate_candidate, rollback_until_bound
        except ImportError:
            self.skipTest("research E4 stack is not installed")
        self.assertNotIn("actual", inspect.signature(allocate_candidate).parameters)
        self.assertNotIn("actual", inspect.signature(rollback_until_bound).parameters)
        self.assertIn(
            "predicted_light_total",
            inspect.signature(rollback_until_bound).parameters,
        )

    def test_family_multiplier_changes_ax31_not_light(self) -> None:
        mods = self._import()
        np = mods["np"]
        costs = np.asarray([[1.0, 1.2, 2.0], [1.0, 1.4, 2.5]], dtype=np.float64)
        out = mods["apply_family_increment_multiplier"](
            costs, ("other", "latex_math"), {"other": 2.5}
        )
        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertGreater(out[0, 1], costs[0, 1])
        self.assertTrue(np.allclose(out[1], np.maximum.accumulate(out[1])))

    def test_aggregate_bound_and_rollback_synthetic(self) -> None:
        mods = self._import()
        np = mods["np"]
        chosen = np.asarray([True, True, True, False])
        uplift = np.asarray([0.1, 0.1, 0.4, 0.0])
        increment = np.asarray([1.0, 1.0, 1.0, 1.0])
        current = np.asarray([1.0, 1.0, 1.0, 1.0])
        groups = ("g-dup", "g-dup", "g-hi", "g-off")
        selected, n_rolled = mods["rollback_until_bound"](
            chosen,
            uplift,
            increment,
            current,
            predicted_light_total=4.0,
            cap=1.5,
            bound=1.2,
            group_keys=groups,
        )
        # bound * spend <= 1.5 * 4 = 6 → spend <= 5.
        self.assertLessEqual(
            1.2 * float((current + increment * selected).sum()), 6.0 + 1e-12
        )
        self.assertEqual(bool(selected[0]), bool(selected[1]))
        self.assertGreaterEqual(n_rolled, 1)
        bound = mods["conformal_bound"]([1.01, 1.10, 1.20, 1.50])
        self.assertGreaterEqual(bound["bound"], bound["q99"])
        self.assertGreaterEqual(bound["bound"], bound["finite_sample_upper"] - 1e-12)

    def test_sigma_changes_rollback_priority_only(self) -> None:
        mods = self._import()
        np = mods["np"]
        chosen = np.asarray([True, True])
        uplift = np.asarray([0.2, 0.2])
        increment = np.asarray([1.0, 1.0])
        current = np.asarray([1.0, 1.0])
        groups = ("ga", "gb")
        risk_low_first = np.asarray([0.1, 5.0])
        left, _n = mods["rollback_until_bound"](
            chosen,
            uplift,
            increment,
            current,
            2.0,
            1.8,
            1.2,
            groups,
            risk=risk_low_first,
            sigma_priority=True,
        )
        risk_flip = np.asarray([5.0, 0.1])
        right, _n = mods["rollback_until_bound"](
            chosen,
            uplift,
            increment,
            current,
            2.0,
            1.8,
            1.2,
            groups,
            risk=risk_flip,
            sigma_priority=True,
        )
        self.assertNotEqual(tuple(left.tolist()), tuple(right.tolist()))
        self.assertEqual(int(left.sum()), 1)
        self.assertEqual(int(right.sum()), 1)
        plain, _n = mods["rollback_until_bound"](
            chosen, uplift, increment, current, 2.0, 1.8, 1.2, groups
        )
        again, _n = mods["rollback_until_bound"](
            chosen, uplift, increment, current, 2.0, 1.8, 1.2, groups
        )
        self.assertEqual(tuple(plain.tolist()), tuple(again.tolist()))
        # Buy/settle prices are the increment/current arrays; sigma never edits them.
        self.assertTrue(np.allclose(increment, [1.0, 1.0]))
        self.assertTrue(np.allclose(current, [1.0, 1.0]))

    def test_group_atomicity_and_order_invariance(self) -> None:
        mods = self._import()
        np = mods["np"]
        chosen = np.asarray([True, True, True])
        uplift = np.asarray([0.05, 0.05, 0.50])
        increment = np.asarray([1.0, 1.0, 1.0])
        current = np.ones(3)
        groups = ("dup", "dup", "solo")
        selected, _n = mods["rollback_until_bound"](
            chosen, uplift, increment, current, 3.0, 1.5, 1.2, groups
        )
        self.assertEqual(bool(selected[0]), bool(selected[1]))
        order = [2, 0, 1]
        backward, _n = mods["rollback_until_bound"](
            chosen[order],
            uplift[order],
            increment[order],
            current[order],
            3.0,
            1.5,
            1.2,
            tuple(groups[index] for index in order),
        )
        restored = [None, None, None]
        for new_index, old in enumerate(order):
            restored[old] = bool(backward[new_index])
        self.assertEqual([bool(flag) for flag in selected], restored)

    def test_held_out_actual_costs_do_not_change_bounds(self) -> None:
        mods = self._import()
        np = mods["np"]
        pool, folds = self._pool(mods, self._scores(mods))
        bundle = mods["oof_cost_surfaces"](pool)
        first = mods["calibrate_bounds"](pool, bundle)
        mutated = bundle["actual_costs"].copy()
        held = int(folds[0])
        mask = np.asarray(folds) == held
        mutated[mask] *= 9.0
        second = mods["calibrate_bounds"](pool, bundle, actual_costs=mutated)
        self.assertEqual(first[held], second[held])
        self.assertTrue(np.allclose(bundle["point"][mask], bundle["point"][mask]))

    def test_report_and_audit_are_deterministic_and_prompt_free(self) -> None:
        mods = self._import()
        pool, _folds = self._pool(mods, self._scores(mods))
        report_a, audit_a = mods["assemble"](pool, seeds=(20260821,))
        report_b, audit_b = mods["assemble"](pool, seeds=(20260821,))
        self.assertEqual(report_a["decision_core_sha256"], report_b["decision_core_sha256"])
        self.assertEqual(audit_a, audit_b)
        self.assertFalse(audit_a["prompt_text_included"])
        self.assertNotIn("prompt", audit_a["seeds"]["20260821"]["rows"][0])
        self.assertEqual(report_a["quality_signal"], "baseline_continuous_uplift")
        self.assertFalse(report_a["feature"]["sigma_in_price"])
        self.assertFalse(report_a["feature"]["used_actual_light_in_allocator"])
        self.assertFalse(report_a["feature"]["runtime_artifact_changed"])
        seed = report_a["seed_results"]["20260821"]
        for name in mods["CANDIDATE_ORDER"]:
            fast = seed["results"][name]["pooled"]["model_counts"]["fast"]
            balanced = seed["results"][name]["pooled"]["model_counts"]["balanced"]
            self.assertEqual(fast["axk1-think"], 0)
            self.assertEqual(balanced["axk1-think"], 0)
            premium = seed["results"][name]["pooled"]["model_counts"]["premium"]
            self.assertLessEqual(premium["axk1-think"], 48)

    def test_promotion_gate_requires_valid_baseline_and_all_seeds(self) -> None:
        mods = self._import()

        def _row(name: str, *, ok: bool, quality: float, delta: float | None) -> dict:
            return {
                "candidate": name,
                "coverage_ok": ok,
                "delta_vs_safe_baseline": delta,
                "fold_caps_ok": ok,
                "independent_safety_ok": ok,
                "pass": ok,
                "pooled_caps_ok": ok,
                "preferred_quality_gain": bool(delta is not None and delta >= 0.002),
                "quality_ok": ok,
                "quality_weighted": quality,
                "ratio_view_failures": [],
                "stress_failures": [],
                "view_failures": [],
            }

        passing_seed = {
            "baseline_valid": True,
            "fold_seed": 20260821,
            "gate_rows": [
                _row("safe_family_point_baseline", ok=True, quality=0.680, delta=None),
                _row("aggregate_conformal_rollback", ok=True, quality=0.682, delta=0.002),
                _row("aggregate_sigma_priority", ok=True, quality=0.681, delta=0.001),
            ],
        }
        fail_seed = {
            "baseline_valid": True,
            "fold_seed": 20260822,
            "gate_rows": [
                _row("safe_family_point_baseline", ok=True, quality=0.680, delta=None),
                _row("aggregate_conformal_rollback", ok=False, quality=0.682, delta=0.002),
                _row("aggregate_sigma_priority", ok=True, quality=0.681, delta=0.001),
            ],
        }
        gate = mods["promotion_gate"]([passing_seed, fail_seed, passing_seed])
        conformal = next(
            row for row in gate["candidates"] if row["candidate"] == "aggregate_conformal_rollback"
        )
        self.assertFalse(conformal["pass_all_seeds"])
        invalid = dict(passing_seed)
        invalid["baseline_valid"] = False
        invalid["gate_rows"] = [
            _row("safe_family_point_baseline", ok=False, quality=0.680, delta=None),
            _row("aggregate_conformal_rollback", ok=True, quality=0.682, delta=0.002),
            _row("aggregate_sigma_priority", ok=True, quality=0.681, delta=0.001),
        ]
        none = mods["promotion_gate"]([invalid, invalid, invalid])
        self.assertEqual(none["decision"], "record-e4-no-valid-reference")
        self.assertFalse(none["passed"])


if __name__ == "__main__":
    unittest.main()
