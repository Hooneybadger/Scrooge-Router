# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""V7 champion-challenger protocol contracts.

Synthetic fixtures only. Inputs may be hashed for epsilon. Public
outcomes are not opened and the challenger runner is not executed
through a successful public fit.
"""

from __future__ import annotations

import ast
import hashlib
import math
import pathlib
import struct
import sys
import tempfile
import unittest
from copy import deepcopy

from ossp_router.protocol import MODEL_IDS, Episode, InputBatch, Message


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PUBLIC_TRAIN_INPUTS = ROOT / "data" / "materialized" / "train" / "inputs.json"


def _require_research_stack() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("numpy / research stack is not installed")


def _require_public_inputs(test: unittest.TestCase) -> None:
    if not _PUBLIC_TRAIN_INPUTS.is_file():
        test.skipTest("pinned public Train+Dev files are not materialized")


_require_research_stack()


EXPLICIT_CHALLENGER_SEEDS = (
    1726202894,
    252428889,
    1120507837,
    141957400,
    1234496749,
    1353411567,
    1103101561,
    2142214382,
    496053794,
    58658564,
    297610007,
    1888638919,
)
GOLD = {
    "e-prompt": {
        "dense": [
            4.189654742026425,
            2.302585092994046,
            1.0986122886681096,
            0.6931471805599453,
            0.0,
            0.0,
            0.6931471805599453,
            0.07692307692307693,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.6094379124341003,
            1.0,
            1.6094379124341003,
            1.0986122886681096,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.057692307692307696,
            0.0,
            0.0,
            0.0,
        ],
        "hash_first8": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.07018624063435965,
            0.0,
            0.1403724812687193,
        ],
        "raw_sha256": (
            "50c50abf835415c956b7dfa898ea16508aaa48d67cadd2bf7941bdceadfa009d"
        ),
        "signal": [False, False, True, False, True, False],
        "text": "Question: What is 2 + 2?\nA) 3\nB) 4\nCalculate the nearest integer.",
    },
    "e-ko": {
        "dense": [
            3.6375861597263857,
            2.4849066497880004,
            1.6094379124341003,
            0.6931471805599453,
            0.7777777777777778,
            0.0,
            0.0,
            0.0,
            0.0,
            0.6931471805599453,
            1.0,
            0.0,
            0.6931471805599453,
            0.0,
            1.6094379124341003,
            0.43243243243243246,
            0.0,
            1.0986122886681096,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1111111111111111,
            0.0,
            0.0,
            0.0,
        ],
        "hash_first8": [0.0, 0.0, 0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0],
        "raw_sha256": (
            "1b619f642d689b03cdddd41c629e3eaedeea4d4eb9ead7a4bf2bb7273d6d98a9"
        ),
        "signal": [True, False, False, False, True, False],
        "text": "다음 중 옳은 것은?\nA. 증명\nB. 반례\n반드시 하나만 고르시오.",
    },
    "e-code": {
        "dense": [
            4.304065093204169,
            2.639057329615259,
            1.3862943611198906,
            0.6931471805599453,
            0.18518518518518517,
            1.3862943611198906,
            0.6931471805599453,
            0.0,
            0.0,
            0.6931471805599453,
            0.0,
            1.0,
            0.0,
            0.0,
            1.9459101490553132,
            0.863013698630137,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.09259259259259259,
            0.0,
            1.0,
            0.0,
        ],
        "hash_first8": [0.0, 0.0, 0.0, -0.1424940999758193, 0.0, 0.0, 0.0, 0.0],
        "raw_sha256": (
            "f2eb4a5a89e2262969bca3a8d375e84fc076677c0e6422599a9ad89f6338469c"
        ),
        "signal": [True, True, True, False, True, False],
        "text": (
            "```python\ndef f(x):\n    assert f(??)\n    return x\n```\n"
            "시간 복잡도 Big-O를 구하시오."
        ),
    },
    "e-long": {
        "dense": [
            9.342858751676328,
            7.496652438168283,
            6.398594934535208,
            0.6931471805599453,
            0.0,
            0.0,
            0.6931471805599453,
            0.0002080083203328133,
            1.0,
            6.398594934535208,
            1.0,
            0.0,
            0.0,
            0.0,
            0.6931471805599453,
            1.0,
            1.0986122886681096,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "hash_first8": [
            0.0,
            -0.00045266402814633414,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "raw_sha256": (
            "01806f1bc33cd9202cc83c77847c583b5f9d4f310fd48ba8dac15027fde68f2a"
        ),
        "signal": [False, False, True, True, False, False],
        "text": ("Prove the theorem. " * 600) + r" $$\frac{1}{2}$$",
    },
    "e-msg": {
        "dense": [
            2.302585092994046,
            1.0986122886681096,
            0.6931471805599453,
            1.0986122886681096,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0986122886681096,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.0,
        ],
        "hash_first8": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "raw_sha256": (
            "62c0fa0a7da17d399a5a869b53543a04b7b140656e97fb2359a46d3e2e3e90a8"
        ),
        "signal": [False, False, False, False, True, True],
    },
}


def _raw_sha256(raw) -> str:
    return hashlib.sha256(
        b"".join(struct.pack("<d", float(value)) for value in raw)
    ).hexdigest()


def _episode(episode_id: str) -> Episode:
    if episode_id == "e-msg":
        return Episode(
            episode_id,
            messages=(Message("user", "short?"), Message("assistant", "ok")),
        )
    return Episode(episode_id, prompt=GOLD[episode_id]["text"])


def _batch(episodes) -> InputBatch:
    return InputBatch(1, "ossp-2026-llm-router-challenge", "dev", tuple(episodes))


def _compiled(scores, log_costs):
    from research.lab.v7_challenger import CompiledTier, LinearHead

    zeros = (0.0,) * (40 + 512)
    return CompiledTier(
        {model: LinearHead(scores[model], zeros) for model in MODEL_IDS},
        {model: LinearHead(log_costs[model], zeros) for model in MODEL_IDS},
    )


def _passing_reproduction() -> dict:
    from research.lab.v7_challenger import (
        REPRODUCTION_TOLERANCE,
        V7_FAIR_DEV_OFFICIAL,
        V7_FAIR_DEV_TIERS,
    )

    return {
        "all_hard_passed": True,
        "all_ratio_under_95": True,
        "final_score": V7_FAIR_DEV_OFFICIAL,
        "tiers": V7_FAIR_DEV_TIERS,
        "tolerance": REPRODUCTION_TOLERANCE,
    }


def _row(seed: int, **overrides: object) -> dict:
    row = {
        "bootstrap_q999_under_95_ok": True,
        "comparator_score": 0.669517045455,
        "delta": 0.021,
        "dirac_family_view": False,
        "fold_seed": seed,
        "official_score": 0.691,
        "pooled_hard_caps_ok": True,
        "pooled_ratio_under_95_ok": True,
        "tv_cost_under_official_ok": True,
        "tv_quality_worst": -0.003,
    }
    row.update(overrides)
    return row


def _rows(**overrides: object) -> list[dict]:
    return [_row(seed, **overrides) for seed in EXPLICIT_CHALLENGER_SEEDS]


class SeedDerivationTests(unittest.TestCase):
    def test_derivation_matches_explicit_list(self) -> None:
        from research.lab.v7_challenger import (
            FIDELITY_CORE_SHA256,
            derive_fresh_challenger_seeds,
        )

        self.assertEqual(
            FIDELITY_CORE_SHA256,
            "1c4b8144378f55779aaf1cf3e78424abd749a93e3d4e33744ccda82304b7cbc9",
        )
        self.assertEqual(derive_fresh_challenger_seeds(), EXPLICIT_CHALLENGER_SEEDS)

    def test_unique_and_no_blocked_overlap(self) -> None:
        from research.lab.chuf_frozen_runtime_fidelity import EXPLICIT_FIDELITY_SEEDS
        from research.lab.chuf_predicted_cost_phase2 import EXPLICIT_RISK_SEEDS
        from research.lab.chuf_tvball_confirmation import OLD_SEEDS, derive_fresh_seeds
        from research.lab.v7_challenger import (
            blocked_seeds,
            derive_fresh_challenger_seeds,
        )

        seeds = derive_fresh_challenger_seeds()
        blocked = blocked_seeds()
        self.assertEqual(len(seeds), 12)
        self.assertEqual(len(set(seeds)), 12)
        self.assertFalse(set(seeds) & set(blocked))
        self.assertEqual(len(blocked), 41)
        self.assertEqual(
            set(blocked),
            set(OLD_SEEDS)
            | set(derive_fresh_seeds())
            | set(EXPLICIT_RISK_SEEDS)
            | set(EXPLICIT_FIDELITY_SEEDS),
        )

    def test_blocked_overlap_fails_closed(self) -> None:
        from research.lab.v7_challenger import derive_fresh_challenger_seeds

        first = derive_fresh_challenger_seeds()[0]
        with self.assertRaises(RuntimeError) as caught:
            derive_fresh_challenger_seeds(blocked=(first,))
        self.assertIn("fail closed", str(caught.exception))


class FeatureParityTests(unittest.TestCase):
    def test_dense_hash_and_signals_match_fixed_vectors(self) -> None:
        from research.lab.v7_challenger import (
            DENSE_FEATURE_NAMES,
            hashed_features,
            raw_feature_vector,
            signal_row,
        )

        self.assertEqual(len(DENSE_FEATURE_NAMES), 40)
        for episode_id, expected in GOLD.items():
            episode = _episode(episode_id)
            raw = raw_feature_vector(episode)
            self.assertEqual(len(raw), 552)
            self.assertEqual(list(raw[:40]), expected["dense"])
            self.assertEqual(list(raw[40:48]), expected["hash_first8"])
            self.assertEqual(_raw_sha256(raw), expected["raw_sha256"])
            self.assertEqual(list(signal_row(episode, raw)), expected["signal"])
            streamed = hashed_features(
                episode.prompt
                if episode.prompt is not None
                else "\n".join(message.content for message in episode.messages or ())
            )
            self.assertEqual(raw[40:], streamed)

    def test_order_and_id_do_not_change_features(self) -> None:
        from research.lab.v7_challenger import raw_feature_vector

        left = Episode("left", prompt=GOLD["e-ko"]["text"])
        right = Episode("right", prompt=GOLD["e-ko"]["text"])
        self.assertEqual(raw_feature_vector(left), raw_feature_vector(right))


class CompileAndAllocatorTests(unittest.TestCase):
    def test_compiled_head_matches_standardized_ensemble(self) -> None:
        from research.lab.v7_challenger import (
            LinearHead,
            apply_linear,
            compile_head,
            ensemble_predict_standardized,
        )

        heads = (
            LinearHead(0.25, (1.0, -0.5, 0.25, 2.0)),
            LinearHead(-0.1, (0.5, 1.5, -1.0, 0.0)),
        )
        means = ((1.0, 2.0, 3.0, 4.0), (0.5, 0.0, -1.0, 2.0))
        scales = ((2.0, 1.0, 4.0, 0.5), (1.0, 2.0, 1.0, 4.0))
        compiled = compile_head(heads, means, scales)
        for raw in ((0.0, 0.0, 0.0, 0.0), (3.0, -1.0, 8.0, 0.25), (1.0, 1.0, 1.0, 1.0)):
            self.assertAlmostEqual(
                apply_linear(compiled, raw),
                ensemble_predict_standardized(raw, heads, means, scales),
                places=12,
            )

    def test_cost_is_light_ax31_k1_monotone(self) -> None:
        from research.lab.v7_challenger import monotonic_costs

        costs = monotonic_costs(
            {MODEL_IDS[0]: 3.0, MODEL_IDS[1]: 1.0, MODEL_IDS[2]: 1.5}
        )
        self.assertLess(costs[MODEL_IDS[0]], costs[MODEL_IDS[1]])
        self.assertLess(costs[MODEL_IDS[1]], costs[MODEL_IDS[2]])

    def test_global_bisection_and_cheaper_tie(self) -> None:
        from research.lab.v7_challenger import FIXED_BISECTION_STEPS, select_models

        self.assertEqual(FIXED_BISECTION_STEPS, 48)
        scores = [
            {MODEL_IDS[0]: 0.1, MODEL_IDS[1]: 0.9, MODEL_IDS[2]: 0.9},
            {MODEL_IDS[0]: 0.1, MODEL_IDS[1]: 0.2, MODEL_IDS[2]: 0.2},
        ]
        costs = [
            {MODEL_IDS[0]: 1.0, MODEL_IDS[1]: 2.0, MODEL_IDS[2]: 4.0},
            {MODEL_IDS[0]: 1.0, MODEL_IDS[1]: 2.0, MODEL_IDS[2]: 4.0},
        ]
        selected, ratio = select_models(
            scores, costs, budget_multiplier=1.25, safety_ratio=1.0
        )
        self.assertNotIn(MODEL_IDS[2], selected)
        self.assertLessEqual(ratio, 1.25)
        tied = [
            {MODEL_IDS[0]: 0.5, MODEL_IDS[1]: 0.5, MODEL_IDS[2]: 0.5},
        ]
        even = [
            {MODEL_IDS[0]: 1.0, MODEL_IDS[1]: 1.0, MODEL_IDS[2]: 1.0},
        ]
        chosen, _ = select_models(tied, even, budget_multiplier=4.0, safety_ratio=1.0)
        self.assertEqual(chosen, (MODEL_IDS[0],))

    def test_fast_k1_ban_fill_and_order_restore(self) -> None:
        from research.lab.v7_challenger import allocate_tier

        compiled = _compiled(
            {
                MODEL_IDS[0]: 0.2,
                MODEL_IDS[1]: 0.4,
                MODEL_IDS[2]: 1.0,
            },
            {
                MODEL_IDS[0]: 0.0,
                MODEL_IDS[1]: math.log(2.0),
                MODEL_IDS[2]: math.log(3.0),
            },
        )
        first = _episode("e-prompt")
        second = _episode("e-ko")
        original = _batch((first, second))
        reversed_batch = _batch(
            (
                Episode("e-ko-swap", prompt=GOLD["e-ko"]["text"]),
                Episode("e-prompt-swap", prompt=GOLD["e-prompt"]["text"]),
            )
        )
        fast, _ = allocate_tier(
            original, compiled, tier="fast", budget_multiplier=1.25, safety_ratio=1.0
        )
        self.assertNotIn(MODEL_IDS[2], fast)
        premium, _ = allocate_tier(
            original, compiled, tier="premium", budget_multiplier=4.0, safety_ratio=1.0
        )
        swapped, _ = allocate_tier(
            reversed_batch,
            compiled,
            tier="premium",
            budget_multiplier=4.0,
            safety_ratio=1.0,
        )
        self.assertEqual(premium[0], swapped[1])
        self.assertEqual(premium[1], swapped[0])

    def test_reserve_formula_and_downward_only(self) -> None:
        from research.lab.v7_challenger import SIGNAL_NAMES, safety_multiplier

        zero = tuple(False for _ in SIGNAL_NAMES)
        rates = {name: 0.0 for name in SIGNAL_NAMES}
        matched = safety_multiplier((zero,) * 1000, rates)
        self.assertEqual(matched, 1.0)
        shifted = safety_multiplier(((True,) + zero[1:],) * 1000, rates)
        self.assertAlmostEqual(shifted, math.exp(-1.0), places=12)
        small = safety_multiplier((zero,) * 250, rates)
        self.assertAlmostEqual(small, math.exp(-0.25), places=12)
        self.assertLessEqual(shifted, 1.0)
        self.assertLessEqual(small, 1.0)

    def test_workload_guard_uses_current_fallback(self) -> None:
        from research.lab.v7_challenger import (
            FALLBACK_ROUTER,
            MAX_LEARNED_EPISODES,
            learned_path_allowed,
            route_or_fallback,
        )

        overflow = _batch(
            tuple(
                Episode(f"guard-{index}", prompt="x")
                for index in range(MAX_LEARNED_EPISODES + 1)
            )
        )
        self.assertFalse(learned_path_allowed(overflow))
        compiled = _compiled(
            {model: 0.1 for model in MODEL_IDS},
            {MODEL_IDS[0]: 0.0, MODEL_IDS[1]: 0.0, MODEL_IDS[2]: 0.0},
        )
        selected, path = route_or_fallback(
            overflow,
            compiled,
            tier="fast",
            budget_multiplier=1.25,
            safety_ratio=1.0,
        )
        self.assertEqual(path, FALLBACK_ROUTER)
        self.assertEqual(len(selected), MAX_LEARNED_EPISODES + 1)
        self.assertTrue(
            learned_path_allowed(_batch((_episode("e-prompt"), _episode("e-ko"))))
        )


class ProtocolAndGateTests(unittest.TestCase):
    def test_canonical_hash_and_pins(self) -> None:
        _require_public_inputs(self)
        from research.lab.v7_challenger import (
            EXPECTED_PROTOCOL_SHA256,
            FALLBACK_DEV_OFFICIAL,
            PROTOCOL_PATH,
            V7_FAIR_DEV_OFFICIAL,
            build_canonical_protocol,
            load_protocol,
            protocol_sha256,
            verify_protocol,
        )

        first = build_canonical_protocol()
        second = build_canonical_protocol()
        self.assertEqual(protocol_sha256(first), protocol_sha256(second))
        self.assertEqual(protocol_sha256(first), EXPECTED_PROTOCOL_SHA256)
        sealed = load_protocol(PROTOCOL_PATH)
        self.assertNotIn("generated_at", sealed)
        self.assertEqual(
            verify_protocol(sealed, EXPECTED_PROTOCOL_SHA256),
            EXPECTED_PROTOCOL_SHA256,
        )
        self.assertEqual(sealed["phase"], "A")
        self.assertFalse(sealed["uses_competitor_artifact_weights"])
        self.assertFalse(sealed["runtime_export"])
        self.assertFalse(sealed["thresholds"]["dirac_family_view"])
        self.assertEqual(
            sealed["comparator"]["reproduction"]["final_score"],
            V7_FAIR_DEV_OFFICIAL,
        )
        self.assertEqual(
            sealed["comparator"]["fallback_dev"]["official_final_score"],
            FALLBACK_DEV_OFFICIAL,
        )
        self.assertEqual(sealed["fresh_seeds"], list(EXPLICIT_CHALLENGER_SEEDS))
        self.assertEqual(sealed["stress"]["bootstrap_draws"], 200)
        self.assertEqual(sealed["stress"]["bootstrap_seed"], 557209147)
        encoded = str(sealed["thresholds"])
        self.assertNotIn("0.669517", encoded)
        self.assertNotIn("0.691477", encoded)

    def test_tamper_rejects(self) -> None:
        from research.lab.v7_challenger import (
            EXPECTED_PROTOCOL_SHA256,
            PROTOCOL_PATH,
            load_protocol,
            protocol_sha256,
            verify_protocol,
        )

        tampered = deepcopy(load_protocol(PROTOCOL_PATH))
        tampered["fresh_seeds"][0] = 1
        self.assertNotEqual(protocol_sha256(tampered), EXPECTED_PROTOCOL_SHA256)
        with self.assertRaises(RuntimeError):
            verify_protocol(tampered, EXPECTED_PROTOCOL_SHA256)

    def test_gate_decisions(self) -> None:
        from research.lab.v7_challenger import (
            GROUPED_FAIL_DECISION,
            NO_REF_DECISION,
            PASS_DECISION,
            REPRO_FAIL_DECISION,
            challenger_gate,
        )

        passed = challenger_gate(
            comparator_valid=True,
            reproduction=_passing_reproduction(),
            grouped_rows=_rows(),
            expected_ok=True,
        )
        self.assertEqual(passed["decision"], PASS_DECISION)
        self.assertFalse(passed["runtime_export"])
        self.assertEqual(
            challenger_gate(
                comparator_valid=False,
                reproduction=_passing_reproduction(),
                grouped_rows=_rows(),
                expected_ok=True,
            )["decision"],
            NO_REF_DECISION,
        )
        broken = dict(_passing_reproduction(), final_score="0.690000000000")
        self.assertEqual(
            challenger_gate(
                comparator_valid=True,
                reproduction=broken,
                grouped_rows=_rows(),
                expected_ok=True,
            )["decision"],
            REPRO_FAIL_DECISION,
        )
        low = _rows()
        low[0]["official_score"] = 0.686
        self.assertEqual(
            challenger_gate(
                comparator_valid=True,
                reproduction=_passing_reproduction(),
                grouped_rows=low,
                expected_ok=True,
            )["decision"],
            GROUPED_FAIL_DECISION,
        )
        self.assertEqual(
            challenger_gate(
                comparator_valid=True,
                reproduction=_passing_reproduction(),
                grouped_rows=_rows(),
                expected_ok=False,
            )["decision"],
            GROUPED_FAIL_DECISION,
        )

    def test_expected_score_beats_fallback_pin(self) -> None:
        from research.lab.modeling import OFFICIAL_CAPS
        from research.lab.v7_challenger import (
            FALLBACK_DEV_OFFICIAL,
            expected_score,
            expected_score_beats_fallback,
        )

        under = {tier: float(OFFICIAL_CAPS[tier]) * 0.9 for tier in OFFICIAL_CAPS}
        self.assertEqual(expected_score(0.669517045455, under), 0.669517045455)
        self.assertFalse(expected_score_beats_fallback(0.669517045455, under))
        self.assertTrue(expected_score_beats_fallback(0.691477272727, under))
        self.assertEqual(FALLBACK_DEV_OFFICIAL, "0.669517045455")


class RunnerRefuseTests(unittest.TestCase):
    def test_wrong_expected_sha_isolated_from_overwrite(self) -> None:
        from research.experiments.run_v7_challenger import main
        from research.lab.v7_challenger import PROTOCOL_PATH

        with tempfile.TemporaryDirectory() as folder:
            isolated = pathlib.Path(folder)
            with self.assertRaises(RuntimeError) as caught:
                main(
                    [
                        "--protocol",
                        str(PROTOCOL_PATH),
                        "--expected-protocol-sha256",
                        "0" * 64,
                        "--output",
                        str(isolated / "report.json"),
                        "--audit-output",
                        str(isolated / "episode-audit.json"),
                    ]
                )
            self.assertIn("protocol sha mismatch", str(caught.exception))
            self.assertFalse((isolated / "report.json").exists())

    def test_overwrite_and_foreign_paths_refused(self) -> None:
        from research.experiments.run_v7_challenger import main
        from research.lab.v7_challenger import EXPECTED_PROTOCOL_SHA256, PROTOCOL_PATH

        e1f = ROOT / "build" / "compare-e1f-cost-conditioned-frontier" / "report.json"
        with self.assertRaises(RuntimeError) as caught:
            main(
                [
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--expected-protocol-sha256",
                    EXPECTED_PROTOCOL_SHA256,
                    "--output",
                    str(e1f),
                ]
            )
        self.assertIn("must not write the E1F", str(caught.exception))
        fidelity = ROOT / "build" / "frozen-runtime-fidelity" / "report.json"
        with self.assertRaises(RuntimeError) as fidelity_caught:
            main(
                [
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--expected-protocol-sha256",
                    EXPECTED_PROTOCOL_SHA256,
                    "--output",
                    str(fidelity),
                ]
            )
        self.assertIn("must not write the fidelity", str(fidelity_caught.exception))
        with tempfile.TemporaryDirectory() as folder:
            existing = pathlib.Path(folder) / "report.json"
            existing.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError) as overwrite:
                main(
                    [
                        "--protocol",
                        str(PROTOCOL_PATH),
                        "--expected-protocol-sha256",
                        EXPECTED_PROTOCOL_SHA256,
                        "--output",
                        str(existing),
                    ]
                )
            self.assertIn("refuse overwrite", str(overwrite.exception))

    def test_run_path_is_explicit_and_not_invoked_here(self) -> None:
        source = (ROOT / "research/lab/v7_challenger.py").read_text(encoding="utf-8")
        start = source.index("def run_challenger")
        stop = source.index("def validation_function_names")
        body = source[start:stop]
        self.assertIn("load_outcomes", body)
        self.assertIn("load_public_pool", body)
        self.assertNotIn("Phase A must not fit", body)


class EngineContractTests(unittest.TestCase):
    def test_high_bit_template_fold_keys_match_python_modulo(self) -> None:
        import numpy as np
        from research.lab.v7_challenger import (
            TEMPLATE_FOLDS,
            fnv1a64,
            normalized_template,
            template_fold_keys,
        )

        texts = (
            "Prove the theorem. Prove the theorem. Prove the theorem. ",
            "messages test",
            "```python\nprint(1)\n```",
            "0xFFFFFFFFFFFFFFFF",
            "counterexample induction 증명",
            "Question: What is 2 + 2?",
        )
        expected = (
            14018410716743434854,
            10725547020758918651,
            13220821949829793962,
            14243964164105175676,
            17999151901840523463,
            7117796106154241903,
        )
        episodes = tuple(
            Episode(f"fold-key-{index}", prompt=text)
            for index, text in enumerate(texts)
        )
        keys = template_fold_keys(episodes)
        self.assertEqual(keys.dtype, np.uint64)
        self.assertEqual(tuple(int(value) for value in keys), expected)
        self.assertTrue(any(value >= 1 << 63 for value in expected[:5]))
        python_mod = tuple(
            fnv1a64(normalized_template(text)) % TEMPLATE_FOLDS for text in texts
        )
        array_mod = tuple(int(value) for value in (keys % TEMPLATE_FOLDS))
        self.assertEqual(array_mod, python_mod)
        self.assertEqual(array_mod, (4, 1, 2, 1, 3, 3))
        again = template_fold_keys(episodes)
        np.testing.assert_array_equal(keys, again)

    def test_ridge_matches_independent_solve(self) -> None:
        import numpy as np
        from research.lab.v7_challenger import (
            ridge_fit_standardized,
            ridge_predict_standardized,
        )

        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(20, 5))
        targets = rng.normal(size=(20, 2))
        alpha = 12.5
        mean, scale, intercept, coefficients = ridge_fit_standardized(
            matrix, targets, alpha
        )
        standardized = (matrix - matrix.mean(axis=0)) / np.where(
            matrix.std(axis=0) > 1e-12, matrix.std(axis=0), 1.0
        )
        centered = targets - targets.mean(axis=0)
        system = standardized.T @ standardized + alpha * np.eye(5)
        expected = np.linalg.solve(system, standardized.T @ centered)
        np.testing.assert_allclose(coefficients, expected, rtol=0.0, atol=1e-12)
        predicted = ridge_predict_standardized(
            matrix, mean, scale, intercept, coefficients
        )
        np.testing.assert_allclose(
            predicted,
            standardized @ expected + targets.mean(axis=0),
            rtol=0.0,
            atol=1e-12,
        )

    def test_ridge_oof_respects_fold_isolation(self) -> None:
        import numpy as np
        from research.lab.v7_challenger import ridge_fit_standardized, ridge_oof

        matrix = np.arange(24, dtype=np.float64).reshape(8, 3)
        targets = np.arange(16, dtype=np.float64).reshape(8, 2)
        fold_keys = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        baseline = ridge_oof(matrix, targets, fold_keys, 2, 4.0)
        mutated = targets.copy()
        mutated[fold_keys % 2 == 0] += 50.0
        isolated = ridge_oof(matrix, mutated, fold_keys, 2, 4.0)
        np.testing.assert_allclose(
            baseline[fold_keys % 2 == 0],
            isolated[fold_keys % 2 == 0],
            rtol=0.0,
            atol=0.0,
        )
        mean, scale, intercept, coefficients = ridge_fit_standardized(
            matrix[fold_keys % 2 == 0], targets[fold_keys % 2 == 0], 4.0
        )
        from research.lab.v7_challenger import ridge_predict_standardized

        held = ridge_predict_standardized(
            matrix[fold_keys % 2 == 1], mean, scale, intercept, coefficients
        )
        np.testing.assert_allclose(
            baseline[fold_keys % 2 == 1], held, rtol=0.0, atol=1e-12
        )

    def test_safety_selection_is_deterministic(self) -> None:
        import numpy as np
        from research.lab.v7_challenger import (
            calibrate_tier_safety,
            select_models_vectorized,
        )

        scores = np.asarray(
            [[0.1, 0.8, 0.9], [0.2, 0.3, 0.4], [0.9, 0.2, 0.1]],
            dtype=np.float64,
        )
        costs = np.asarray(
            [[1.0, 2.0, 4.0], [1.0, 2.0, 4.0], [1.0, 2.0, 4.0]],
            dtype=np.float64,
        )
        first, ratio_a = select_models_vectorized(
            scores, costs, tier="fast", safety_ratio=0.8, budget_multiplier=1.25
        )
        second, ratio_b = select_models_vectorized(
            scores, costs, tier="fast", safety_ratio=0.8, budget_multiplier=1.25
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(ratio_a, ratio_b)
        self.assertNotIn(2, first.tolist())
        predictions = np.concatenate(
            [scores, np.log(costs)],
            axis=1,
        )
        fold_ids = np.asarray([0, 1, 0], dtype=np.int64)
        safety_a, report_a = calibrate_tier_safety(
            predictions,
            scores,
            costs,
            fold_ids,
            tier="balanced",
            target=1.95,
            budget_multiplier=2.0,
        )
        safety_b, report_b = calibrate_tier_safety(
            predictions,
            scores,
            costs,
            fold_ids,
            tier="balanced",
            target=1.95,
            budget_multiplier=2.0,
        )
        self.assertEqual(safety_a, safety_b)
        self.assertEqual(report_a, report_b)

    def test_reproduction_record_uses_exact_strings(self) -> None:
        from research.lab.v7_challenger import (
            V7_FAIR_DEV_OFFICIAL,
            V7_FAIR_DEV_TIERS,
            reproduction_matches_reference,
            reproduction_record,
        )

        official = {
            "final_score": V7_FAIR_DEV_OFFICIAL,
            "tiers": {
                tier: {
                    "budget_passed": True,
                    "budget_ratio": values["budget_ratio"],
                    "model_counts": values["model_counts"],
                    "quality_score": values["quality_score"],
                }
                for tier, values in V7_FAIR_DEV_TIERS.items()
            },
        }
        record = reproduction_record(official)
        self.assertTrue(reproduction_matches_reference(record))
        self.assertEqual(record["final_score"], V7_FAIR_DEV_OFFICIAL)
        self.assertEqual(record["tiers"], V7_FAIR_DEV_TIERS)

    def test_report_hash_is_canonical(self) -> None:
        from research.lab.v7_challenger import (
            NO_REF_DECISION,
            decision_core_sha256,
            report_sha256,
        )

        report = {
            "decision": NO_REF_DECISION,
            "experiment": "v7-challenger-v1",
            "gate": {"passed": False},
            "grouped": {"summary": {"mean_official": 0.0}},
            "protocol_sha256": "ab",
            "report_type": "scrooge-v7-challenger",
            "reproduction": {
                "final_score": "0.0",
                "passed": False,
                "tiers": {},
            },
            "schema_version": 1,
            "thresholds": {"dirac_family_view": False},
        }
        self.assertEqual(report_sha256(report), report_sha256(dict(report)))
        self.assertEqual(
            decision_core_sha256(report), decision_core_sha256(dict(report))
        )
        self.assertNotEqual(
            report_sha256(report),
            report_sha256({**report, "protocol_sha256": "cd"}),
        )
        with self.assertRaises(RuntimeError):
            report_sha256({**report, "generated_at": "no"})


class IsolationTests(unittest.TestCase):
    def test_validation_path_forbids_outcomes_and_run(self) -> None:
        from research.lab.v7_challenger import assert_validation_path_has_no_outcomes

        assert_validation_path_has_no_outcomes()

    def test_this_module_does_not_open_outcomes_or_fit(self) -> None:
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "load_outcomes",
            "load_public_pool",
            "load_split_pool",
            "oof_chuf_heads",
            "run_challenger",
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)

    def test_output_path_separate_and_absent_in_phase_a(self) -> None:
        """Phase-state independent: this module never invokes the run path.

        Challenger artifacts may exist after the sealed public run.
        Absence of ``build/v7-challenger`` is not a global invariant.
        """

        from research.lab.v7_challenger import (
            CONFIRM_REPORT_RELATIVE,
            E1F_REPORT_RELATIVE,
            FIDELITY_REPORT_RELATIVE,
            OUT_RELATIVE,
            PHASE2_REPORT_RELATIVE,
        )

        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "load_outcomes",
            "load_public_pool",
            "load_split_pool",
            "run_challenger",
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(forbidden & names)
        self.assertFalse(forbidden & attrs)
        self.assertEqual(OUT_RELATIVE, "build/v7-challenger")
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(E1F_REPORT_RELATIVE).parent.as_posix(),
        )
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(CONFIRM_REPORT_RELATIVE).parent.as_posix(),
        )
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(PHASE2_REPORT_RELATIVE).parent.as_posix(),
        )
        self.assertNotEqual(
            OUT_RELATIVE,
            pathlib.Path(FIDELITY_REPORT_RELATIVE).parent.as_posix(),
        )

    def test_no_src_writes_from_this_tree(self) -> None:
        from research.lab.v7_challenger import PROTOCOL_PATH

        self.assertTrue(str(PROTOCOL_PATH).endswith("research/protocols/v7-challenger.v1.json"))
        self.assertFalse(str(PROTOCOL_PATH).startswith(str(ROOT / "src")))


if __name__ == "__main__":
    unittest.main()
