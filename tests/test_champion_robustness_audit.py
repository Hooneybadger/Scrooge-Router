# SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
# SPDX-License-Identifier: Apache-2.0

"""Champion robustness audit contracts. Synthetic fixtures only."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "research" / "protocols" / "champion-robustness-audit.v1.json"
PROTOCOL_SHA256 = "457bd4694e6716488bd8615fee1c45bc25943417391e14d4722e1097c34c7cb9"


class ChampionAuditProtocolTest(unittest.TestCase):
    def test_protocol_file_is_sealed(self) -> None:
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"], "champion-robustness-audit-v1")
        self.assertEqual(payload["protocol_id"], "champion-robustness-audit-v1")
        self.assertIs(payload["promotion"], False)
        self.assertIn("conditionally_robust", payload["decisions"])
        self.assertIn("fragile", payload["decisions"])
        self.assertIn("robust", payload["decisions"])
        self.assertGreater(float(payload["thresholds"]["h1_selected_gap_min"]), 0.0)
        self.assertGreater(int(payload["thresholds"]["leftover_unused_min"]), 0)

    def test_protocol_verifies_against_its_canonical_sha(self) -> None:
        try:
            from research.lab.champion_robustness_audit import (
                ProtocolError,
                protocol_sha256,
                verify_protocol,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        digest = protocol_sha256(payload)
        self.assertEqual(digest, PROTOCOL_SHA256)
        self.assertEqual(verify_protocol(payload, digest), digest)
        with self.assertRaises(ProtocolError):
            verify_protocol(payload, "0" * 64)
        tampered = json.loads(json.dumps(payload))
        tampered["promotion"] = True
        with self.assertRaises(ProtocolError):
            verify_protocol(tampered, digest)


class ChampionAuditVerdictTest(unittest.TestCase):
    def _import(self):
        try:
            from research.lab.champion_robustness_audit import decide_axes
            from research.lab.serving_replica import (
                PINNED_DEV_FINAL_SCORE,
                conditioned_fast_cap,
                residual_fraction,
                tvball_worst,
            )
        except ImportError:
            self.skipTest("numpy / research stack is not installed")
        return decide_axes, PINNED_DEV_FINAL_SCORE, conditioned_fast_cap, residual_fraction, tvball_worst

    def test_residual_rule_is_monotone(self) -> None:
        _decide, _pin, fast_cap, residual, _tv = self._import()
        self.assertEqual(residual(["other", "other", "latex_math", "latex_math"]), 0.5)
        self.assertEqual(fast_cap(0.0), 1.13)
        self.assertEqual(fast_cap(0.05), 1.13)
        self.assertEqual(fast_cap(0.06), 1.12)
        self.assertEqual(fast_cap(0.10), 1.12)
        self.assertEqual(fast_cap(0.11), 1.11)

    def test_tvball_formula(self) -> None:
        _decide, _pin, _fast, _residual, tvball = self._import()
        self.assertAlmostEqual(tvball(0.01, [0.02, -0.01]), 0.01 + 0.014204545454545449 * (-0.01 - 0.02))

    def _empty_premium(self) -> dict:
        return {
            "n_eligible_unbought": 0,
            "n_k1": 0,
            "selected_cost_q1_mean": None,
            "selected_cost_q4_mean": None,
            "selected_mean_deltak": 0.0,
            "unbought_mean_deltak": 0.0,
        }

    def _split(self, final: float, *, ruin: bool = False, gap: float = 0.05, pearson: float = 0.2) -> dict:
        official = {
            tier: {
                "budget_passed": not ruin,
                "budget_ratio": 3.9 if ruin and tier == "premium" else 1.09,
                "model_counts": {"ax31-light": 1, "ax31": 1, "axk1-think": 0},
                "near_budget": False,
                "quality_score": 0.6,
                "tier_score": 0.6,
            }
            for tier in ("fast", "balanced", "premium")
        }
        return {
            "determinism_passed": True,
            "fidelity": {
                tier: {"matched": True, "n_mismatch": 0}
                for tier in ("fast", "balanced", "premium")
            },
            "final_score": final,
            "h1": {
                "balanced": {
                    "pearson_pred_vs_realized": pearson,
                    "selected_minus_leftover": gap,
                },
                "fast": {
                    "pearson_pred_vs_realized": pearson,
                    "selected_minus_leftover": gap,
                },
                "premium_k1": {
                    "pearson_pred_vs_realized": 0.2,
                    "selected_minus_leftover": 0.2,
                },
            },
            "h2": {"weighted": {"inversions": [], "tvball_worst_vs_always_light": 0.01}},
            "leftover": {
                "balanced": {"n_positive_pred_left_on_light": 10},
                "fast": {"n_positive_pred_left_on_light": 10},
                "premium": self._empty_premium(),
            },
            "official": official,
            "oracle": {
                tier: {"gap_same_budget": 0.02}
                for tier in ("fast", "balanced", "premium")
            },
            "residual_fraction": 0.08,
            "safety_failures": [],
        }

    def test_decide_axes_robust_and_fragile(self) -> None:
        decide_axes, pinned, _fast, _residual, _tv = self._import()
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        splits = {
            "dev": self._split(pinned, gap=0.05, pearson=0.2),
            "train": self._split(0.67, gap=0.05, pearson=0.2),
        }
        robust = decide_axes(protocol, splits)
        self.assertEqual(robust["overall"], "robust")
        splits["dev"] = self._split(pinned, ruin=True)
        fragile = decide_axes(protocol, splits)
        self.assertEqual(fragile["overall"], "fragile")
        splits["dev"] = self._split(pinned, gap=0.001, pearson=0.01)
        splits["train"] = self._split(0.67, gap=0.001, pearson=0.01)
        conditional = decide_axes(protocol, splits)
        self.assertEqual(conditional["overall"], "conditionally_robust")


if __name__ == "__main__":
    unittest.main()
