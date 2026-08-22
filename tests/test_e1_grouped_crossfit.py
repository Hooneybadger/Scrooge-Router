# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Grouped cross-fit and E1 objective contracts. Grouping tests are stdlib-only."""

from __future__ import annotations

import pathlib
import sys
import unittest

from ossp_router.protocol import Episode


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.lab.grouped_crossfit import (  # noqa: E402
    FOLD_SEED,
    FOLDS,
    assign_balanced_group_folds,
    char_ngrams,
    families_of,
    fold_leakage_count,
    group_episodes,
    jaccard,
    near_duplicate_text,
)


def _episode(episode_id: str, prompt: str) -> Episode:
    return Episode(episode_id=episode_id, prompt=prompt)


class GroupedCrossfitTest(unittest.TestCase):
    def test_exact_duplicates_share_a_group(self) -> None:
        prompt = "Solve 2 + 2 and show the integer answer."
        episodes = (
            _episode("a", prompt),
            _episode("b", "A different short prompt."),
            _episode("c", prompt),
        )
        grouping = group_episodes(episodes)
        self.assertEqual(grouping.group_keys[0], grouping.group_keys[2])
        self.assertNotEqual(grouping.group_keys[0], grouping.group_keys[1])
        self.assertEqual(grouping.n_exact_groups, 2)

    def test_near_duplicates_at_jaccard_threshold_share_a_group(self) -> None:
        base = "".join(f"{index:03d}" for index in range(80))
        near = base[:60] + "X" + base[61:]
        left = char_ngrams(near_duplicate_text(base))
        right = char_ngrams(near_duplicate_text(near))
        self.assertGreaterEqual(jaccard(left, right), 0.90)
        episodes = (
            _episode("left", base),
            _episode("right", near),
            _episode("other", "zzzz not similar at all " * 4),
        )
        grouping = group_episodes(episodes)
        self.assertEqual(grouping.group_keys[0], grouping.group_keys[1])
        self.assertNotEqual(grouping.group_keys[0], grouping.group_keys[2])
        self.assertGreaterEqual(grouping.n_near_duplicate_unions, 1)

    def test_length_sorted_window_keeps_jaccard_pair_behind_midsize_distractor(
        self,
    ) -> None:
        left_text = "".join(f"{index:03d}" for index in range(80))
        left_grams = char_ngrams(near_duplicate_text(left_text))
        right_text = left_text
        extra = 0
        while extra < 40:
            candidate = right_text + f"Q{extra:03d}"
            candidate_grams = char_ngrams(near_duplicate_text(candidate))
            if (
                len(candidate_grams) > len(left_grams)
                and jaccard(left_grams, candidate_grams) < 0.90
            ):
                break
            right_text = candidate
            extra += 1
        right_grams = char_ngrams(near_duplicate_text(right_text))
        self.assertLess(len(left_grams), len(right_grams))
        self.assertGreaterEqual(jaccard(left_grams, right_grams), 0.90)

        target = (len(left_grams) + len(right_grams)) // 2
        distractor = ""
        extra = 0
        while len(char_ngrams(near_duplicate_text(distractor))) < target:
            distractor += f"Z{extra:04d}"
            extra += 1
        mid_grams = char_ngrams(near_duplicate_text(distractor))
        self.assertLess(len(left_grams), len(mid_grams))
        self.assertLess(len(mid_grams), len(right_grams))
        self.assertLess(jaccard(left_grams, mid_grams), 0.90)
        self.assertLess(jaccard(right_grams, mid_grams), 0.90)

        episodes = (
            _episode("left", left_text),
            _episode("mid", distractor),
            _episode("right", right_text),
        )
        grouping = group_episodes(episodes)
        self.assertEqual(grouping.group_keys[0], grouping.group_keys[2])
        self.assertNotEqual(grouping.group_keys[0], grouping.group_keys[1])

    def test_template_number_variants_share_a_group(self) -> None:
        episodes = (
            _episode("one", "How many apples are left if you start with 12?"),
            _episode("two", "How many apples are left if you start with 99?"),
            _episode("three", "Completely unrelated python def f(x): return x"),
        )
        grouping = group_episodes(episodes)
        self.assertEqual(grouping.group_keys[0], grouping.group_keys[1])
        self.assertNotEqual(grouping.group_keys[0], grouping.group_keys[2])

    def test_group_and_fold_assignment_ignore_input_order(self) -> None:
        episodes = (
            _episode("x1", "Exact copy one."),
            _episode("x2", "Exact copy one."),
            _episode("y1", "Korean question: 다음 중 옳은 것은? A. 1 B. 2"),
            _episode("z1", "def f(x):\n    return x\nassert f(1) == ??"),
            _episode("z2", "def f(x):\n    return x\nassert f(2) == ??"),
            _episode("w1", "Long unique english reasoning prompt " * 8),
        )
        forward = group_episodes(episodes)
        backward = group_episodes(tuple(reversed(episodes)))
        families = families_of(episodes)
        folds = assign_balanced_group_folds(
            forward.group_keys, families, folds=3, seed=FOLD_SEED
        )
        rev_families = families_of(tuple(reversed(episodes)))
        rev_folds = assign_balanced_group_folds(
            backward.group_keys, rev_families, folds=3, seed=FOLD_SEED
        )
        by_id_group = {}
        by_id_fold = {}
        for episode, key, fold in zip(episodes, forward.group_keys, folds):
            by_id_group[episode.episode_id] = key
            by_id_fold[episode.episode_id] = fold
        for episode, key, fold in zip(
            reversed(episodes), backward.group_keys, rev_folds
        ):
            self.assertEqual(by_id_group[episode.episode_id], key)
            self.assertEqual(by_id_fold[episode.episode_id], fold)

    def test_fold_leakage_is_zero(self) -> None:
        episodes = (
            _episode("a1", "shared exact prompt"),
            _episode("a2", "shared exact prompt"),
            _episode("b1", "second family korean 문항입니다"),
            _episode("c1", "def f(n):\n    return n\nassert f(3) == ??"),
            _episode("d1", "word problem: how many left over altogether"),
            _episode("e1", "latex $\\frac{1}{2}$ and \\begin{align}"),
        )
        grouping = group_episodes(episodes)
        folds = assign_balanced_group_folds(
            grouping.group_keys, families_of(episodes), folds=3, seed=FOLD_SEED
        )
        self.assertEqual(0, fold_leakage_count(grouping.group_keys, folds))
        grouped = {}
        for key, fold in zip(grouping.group_keys, folds):
            grouped.setdefault(key, set()).add(fold)
        self.assertTrue(all(len(values) == 1 for values in grouped.values()))

    def test_fold_leakage_counts_unique_groups(self) -> None:
        keys = ("g-shared", "g-shared", "g-shared", "g-clean", "g-clean")
        folds = (0, 0, 1, 1, 1)
        self.assertEqual(1, fold_leakage_count(keys, folds))
        keys_two = ("a", "a", "b", "b")
        folds_two = (0, 1, 1, 2)
        self.assertEqual(2, fold_leakage_count(keys_two, folds_two))


class E1ObjectiveContractTest(unittest.TestCase):
    def _import_e1(self):
        try:
            import numpy as np
            from ossp_router.protocol import (
                InputBatch,
                Outcome,
                OutcomeBatch,
                load_bundled_policy,
            )
            from research.lab.e1_objectives import (
                BASELINE_NAME,
                CANDIDATE_ORDER,
                GATE_VIEW_KINDS,
                allocate_all_tiers,
                assemble,
                canonical_json_text,
                current_quality_matrix,
                decision_core_payload,
                decision_core_sha256,
                measure,
                oof_candidate_predictions,
                promotion_gate,
                sha256_text,
            )
            from research.lab.public_pool import PublicPool
            from research.lab.quality_heads import content_tie_keys
        except ImportError:
            self.skipTest("numpy / research E1 stack is not installed")
        return {
            "np": np,
            "InputBatch": InputBatch,
            "Outcome": Outcome,
            "OutcomeBatch": OutcomeBatch,
            "load_bundled_policy": load_bundled_policy,
            "BASELINE_NAME": BASELINE_NAME,
            "CANDIDATE_ORDER": CANDIDATE_ORDER,
            "GATE_VIEW_KINDS": GATE_VIEW_KINDS,
            "allocate_all_tiers": allocate_all_tiers,
            "assemble": assemble,
            "canonical_json_text": canonical_json_text,
            "current_quality_matrix": current_quality_matrix,
            "decision_core_payload": decision_core_payload,
            "decision_core_sha256": decision_core_sha256,
            "measure": measure,
            "oof_candidate_predictions": oof_candidate_predictions,
            "promotion_gate": promotion_gate,
            "sha256_text": sha256_text,
            "PublicPool": PublicPool,
            "content_tie_keys": content_tie_keys,
        }

    def _synthetic_pool(self, mods):
        from decimal import Decimal

        np = mods["np"]
        prompts = [
            "Exact copy used twice.",
            "Exact copy used twice.",
            "Korean question: 다음 중 옳은 것은 무엇입니까? A. 1 B. 2 C. 3",
            "def f(x):\n    return x + 1\nassert f(2) == ??",
            "How many apples are left over altogether if each costs 3?",
            "Solve $\\frac{1}{2} + \\frac{1}{3}$ using \\begin{align}.",
            "Word problem: how far and how long if the average is 12.",
            "Unrelated long english prompt about music history " * 6,
        ]
        episodes = tuple(
            _episode(f"syn-{index:02d}", prompt) for index, prompt in enumerate(prompts)
        )
        grouping = group_episodes(episodes)
        families = families_of(episodes)
        folds = assign_balanced_group_folds(
            grouping.group_keys, families, folds=min(FOLDS, 3), seed=FOLD_SEED
        )
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
        costs = np.asarray(
            [
                [1.0, 1.2, 4.0],
                [1.0, 1.3, 5.0],
                [1.0, 1.1, 3.5],
                [1.0, 1.4, 6.0],
                [1.0, 1.2, 4.5],
                [1.0, 1.5, 8.0],
                [1.0, 1.2, 3.0],
                [1.0, 1.1, 2.5],
            ],
            dtype=np.float64,
        )
        policy = mods["load_bundled_policy"]()
        inputs = mods["InputBatch"](
            1, "ossp-2026-llm-router-challenge", "public", episodes
        )
        outcomes = []
        for episode, row in zip(episodes, scores):
            for model_index, model_id in enumerate(
                ("ax31-light", "ax31", "axk1-think")
            ):
                outcomes.append(
                    mods["Outcome"](
                        episode_id=episode.episode_id,
                        model_id=model_id,
                        score=Decimal(str(row[model_index])),
                        num_generations=1,
                        input_tokens=10,
                        output_tokens=4,
                    )
                )
        outcome_batch = mods["OutcomeBatch"](
            1, "ossp-2026-llm-router-challenge", "public", tuple(outcomes)
        )
        split_labels = ("train", "train", "train", "train", "dev", "dev", "dev", "dev")
        return mods["PublicPool"](
            episodes=episodes,
            texts=tuple(prompt for prompt in prompts),
            families=families,
            languages=tuple(
                "korean"
                if "Korean" in prompt or "문항" in prompt or "다음" in prompt
                else "non_korean"
                for prompt in prompts
            ),
            length_views=tuple("len_lt_120" for _ in prompts),
            group_keys=grouping.group_keys,
            exact_keys=grouping.exact_keys,
            template_keys=grouping.template_keys,
            folds=folds,
            scores=scores,
            costs=costs,
            light_total=float(costs[:, 0].sum()),
            identity={
                "fold_seed": FOLD_SEED,
                "folds": 3,
                "n_dev": 4,
                "n_episodes": len(episodes),
                "n_train": 4,
                "split": "public",
            },
            grouping={
                "n_exact_groups": grouping.n_exact_groups,
                "n_groups": grouping.n_groups,
                "n_jaccard_comparisons": grouping.n_jaccard_comparisons,
                "n_near_duplicate_unions": grouping.n_near_duplicate_unions,
                "n_template_groups": grouping.n_template_groups,
                "blocking": dict(grouping.blocking),
                "group_size_histogram": dict(grouping.group_size_histogram),
                "largest_group": grouping.largest_group,
                "n_singleton_groups": grouping.n_singleton_groups,
            },
            fold_table=[],
            inputs=inputs,
            outcomes=outcome_batch,
            policy=policy,
            split_labels=split_labels,
        )

    def test_candidate_predictions_and_report_schema_are_deterministic(self) -> None:
        mods = self._import_e1()
        np = mods["np"]
        pool = self._synthetic_pool(mods)
        features = mods["current_quality_matrix"](pool.episodes)
        first = mods["oof_candidate_predictions"](features, pool.scores, pool.folds)
        second = mods["oof_candidate_predictions"](features, pool.scores, pool.folds)
        for name in mods["CANDIDATE_ORDER"]:
            self.assertTrue(np.allclose(first[name][0], second[name][0]))
            self.assertTrue(np.allclose(first[name][1], second[name][1]))
        tie_keys = mods["content_tie_keys"](pool.texts)
        models_a = mods["allocate_all_tiers"](
            first["baseline_continuous_uplift"][0],
            first["baseline_continuous_uplift"][1],
            pool.costs,
            float(pool.costs[:, 0].sum()),
            tie_keys,
        )
        models_b = mods["allocate_all_tiers"](
            second["baseline_continuous_uplift"][0],
            second["baseline_continuous_uplift"][1],
            pool.costs,
            float(pool.costs[:, 0].sum()),
            tie_keys,
        )
        self.assertEqual(models_a, models_b)

        report_a, audit_a = mods["assemble"](pool)
        report_b, audit_b = mods["assemble"](pool)
        self.assertEqual(report_a["decision_core_sha256"], report_b["decision_core_sha256"])
        self.assertEqual(
            report_a["decision_core_sha256"], mods["decision_core_sha256"](report_a)
        )
        self.assertEqual(audit_a, audit_b)
        required = {
            "allocator",
            "audit",
            "candidates",
            "cost_diagnostic",
            "decision",
            "decision_core_sha256",
            "experiment",
            "feature",
            "fold_table",
            "grouping",
            "identity",
            "limitations",
            "promotion_gate",
            "report_type",
            "results",
            "runtime",
            "schema_version",
            "stress_view_kinds",
            "stress_views",
        }
        self.assertTrue(required.issubset(report_a))
        self.assertEqual(report_a["schema_version"], 2)
        self.assertEqual(report_a["experiment"], "e1-quality-objectives")
        self.assertEqual(
            report_a["audit"]["sha256"],
            mods["sha256_text"](mods["canonical_json_text"](audit_a)),
        )
        self.assertEqual(
            mods["decision_core_payload"](report_a)["audit"]["sha256"],
            report_a["audit"]["sha256"],
        )
        self.assertFalse(audit_a["prompt_text_included"])
        for row in audit_a["rows"]:
            self.assertNotIn("prompt", row)
            self.assertNotIn("text", row)
            self.assertIn("episode_id", row)
            self.assertIn("split", row)
            self.assertIn("group_key", row)
            self.assertIn("fold", row)
        self.assertFalse(report_a["cost_diagnostic"]["clamped"])
        self.assertTrue(report_a["cost_diagnostic"]["shared_across_candidates"])
        for name in mods["CANDIDATE_ORDER"]:
            self.assertIn("pooled", report_a["results"][name])
            self.assertIn("per_fold", report_a["results"][name])
            self.assertIn("quality_weighted", report_a["results"][name]["pooled"])
            self.assertIn("official_final_score", report_a["results"][name]["pooled"])
            kinds = {row["kind"] for row in report_a["stress_views"][name]}
            self.assertTrue(set(mods["GATE_VIEW_KINDS"]).issubset(kinds))
            names = {(row["kind"], row["name"]) for row in report_a["stress_views"][name]}
            self.assertIn(("split", "train"), names)
            self.assertIn(("split", "dev"), names)
            for fold in sorted(set(pool.folds)):
                self.assertIn(("fold", str(fold)), names)
            for tier in ("fast", "balanced", "premium"):
                tier_row = report_a["results"][name]["pooled"]["tiers"][tier]
                self.assertIn("quality_score", tier_row)
                self.assertIn("budget_ratio", tier_row)
                self.assertIn("model_counts", tier_row)
                self.assertIn("tier_score", tier_row)
        self.assertIn("candidates", report_a["promotion_gate"])
        self.assertIn("passed", report_a["promotion_gate"])
        self.assertEqual(
            report_a["promotion_gate"]["thresholds"]["gated_view_kinds"],
            list(mods["GATE_VIEW_KINDS"]),
        )

    def test_oof_held_out_labels_do_not_change_that_fold_prediction(self) -> None:
        mods = self._import_e1()
        np = mods["np"]
        pool = self._synthetic_pool(mods)
        features = mods["current_quality_matrix"](pool.episodes)
        folds = np.asarray(list(pool.folds))
        held = int(folds[0])
        original = mods["oof_candidate_predictions"](features, pool.scores, pool.folds)
        mutated = pool.scores.copy()
        mutated[folds == held] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        changed = mods["oof_candidate_predictions"](features, mutated, pool.folds)
        held_mask = folds == held
        other_mask = ~held_mask
        self.assertTrue(np.any(held_mask))
        self.assertTrue(np.any(other_mask))
        other_changed = False
        for name in mods["CANDIDATE_ORDER"]:
            self.assertTrue(
                np.allclose(original[name][0][held_mask], changed[name][0][held_mask])
            )
            self.assertTrue(
                np.allclose(original[name][1][held_mask], changed[name][1][held_mask])
            )
            if not np.allclose(original[name][0][other_mask], changed[name][0][other_mask]):
                other_changed = True
        self.assertTrue(other_changed)

    def test_split_and_fold_view_failures_block_promotion(self) -> None:
        mods = self._import_e1()
        pool = self._synthetic_pool(mods)
        report = mods["measure"](pool)
        kinds = {row["kind"] for row in report["stress_views"]["direct_adjacent_delta"]}
        self.assertTrue(set(mods["GATE_VIEW_KINDS"]).issubset(kinds))

        def _pooled(quality: float) -> dict:
            return {
                "quality_weighted": quality,
                "tiers": {
                    tier: {"within_hard_cap": True}
                    for tier in ("fast", "balanced", "premium")
                },
            }

        results = {
            "baseline_continuous_uplift": {"pooled": _pooled(0.50)},
            "direct_adjacent_delta": {"pooled": _pooled(0.51)},
            "delta_sign_ridge": {"pooled": _pooled(0.51)},
            "hybrid_magnitude_sign": {"pooled": _pooled(0.51)},
        }
        split_fail = {
            "allocation": "pooled",
            "delta": -0.01,
            "gated": True,
            "kind": "split",
            "n": 40,
            "name": "dev",
            "worse_than_gate": True,
        }
        fold_fail = {
            "allocation": "pooled",
            "delta": -0.01,
            "gated": True,
            "kind": "fold",
            "n": 40,
            "name": "2",
            "worse_than_gate": True,
        }
        ok_view = {
            "allocation": "pooled",
            "delta": 0.0,
            "gated": True,
            "kind": "family",
            "n": 40,
            "name": "other",
            "worse_than_gate": False,
        }
        views = {
            "direct_adjacent_delta": [split_fail, ok_view],
            "delta_sign_ridge": [ok_view],
            "hybrid_magnitude_sign": [fold_fail],
        }
        gate = mods["promotion_gate"](results, views)
        by_name = {row["candidate"]: row for row in gate["candidates"]}
        self.assertTrue(by_name["direct_adjacent_delta"]["quality_ok"])
        self.assertFalse(by_name["direct_adjacent_delta"]["views_ok"])
        self.assertIn("split:dev", by_name["direct_adjacent_delta"]["view_failures"])
        self.assertFalse(by_name["direct_adjacent_delta"]["pass"])
        self.assertTrue(by_name["delta_sign_ridge"]["pass"])
        self.assertFalse(by_name["hybrid_magnitude_sign"]["pass"])
        self.assertIn("fold:2", by_name["hybrid_magnitude_sign"]["view_failures"])
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["recommended"], "delta_sign_ridge")


class PublicPoolSmokeTest(unittest.TestCase):
    def test_pinned_public_pool_hashes_groups_and_has_no_fold_leakage(self) -> None:
        try:
            from research.lab.grouped_crossfit import fold_leakage_count
            from research.lab.public_pool import (
                EXPECTED_DEV_INPUTS_SHA256,
                EXPECTED_DEV_OUTCOMES_SHA256,
                EXPECTED_N_PUBLIC,
                EXPECTED_TRAIN_INPUTS_SHA256,
                EXPECTED_TRAIN_OUTCOMES_SHA256,
                TRAIN_INPUTS,
                load_public_pool,
            )
        except ImportError:
            self.skipTest("numpy / research public-pool stack is not installed")
        if not TRAIN_INPUTS.is_file():
            self.skipTest("pinned public Train+Dev files are not materialized")

        pool = load_public_pool()
        self.assertEqual(pool.identity["n_episodes"], EXPECTED_N_PUBLIC)
        self.assertEqual(pool.identity["n_train"], 1760)
        self.assertEqual(pool.identity["n_dev"], 880)
        self.assertEqual(pool.identity["train_inputs_sha256"], EXPECTED_TRAIN_INPUTS_SHA256)
        self.assertEqual(
            pool.identity["train_outcomes_sha256"], EXPECTED_TRAIN_OUTCOMES_SHA256
        )
        self.assertEqual(pool.identity["dev_inputs_sha256"], EXPECTED_DEV_INPUTS_SHA256)
        self.assertEqual(pool.identity["dev_outcomes_sha256"], EXPECTED_DEV_OUTCOMES_SHA256)
        self.assertEqual(pool.grouping["n_groups"], 2521)
        self.assertEqual(0, fold_leakage_count(pool.group_keys, pool.folds))
        self.assertEqual(len(set(pool.split_labels)), 2)
        self.assertEqual(pool.split_labels.count("train"), 1760)
        self.assertEqual(pool.split_labels.count("dev"), 880)


if __name__ == "__main__":
    unittest.main()
