# SPDX-FileCopyrightText: Copyright 2026 OSSP LLM Router Challenge contributors
# SPDX-License-Identifier: Apache-2.0

"""the expected score ranking — expected-score frontier over the the policy search candidate rows.

The contest pays ``sum_t w_t * q_t`` but zeroes a tier that busts, so the
quantity a selected should maximise is

    E[score] = sum_t w_t * q_t * (1 - p_t)

with ``p_t`` the inflated ruin rate of tier ``t`` over the drift views.
This reads the the policy search report only; it runs no new routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from research.lab.modeling import TIER_WEIGHTS
from ossp_router.protocol import TIERS


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "build" / "search-policy-space" / "report.json"
OUT = ROOT / "build" / "rank-expected-score"
SHRINKS = (1.0, 0.5, 0.25)


def risk(row: Mapping[str, Any], tier: str) -> float:
    worst = 0.0
    for split in row["splits"].values():
        drift = split["tiers"][tier]["drift"]
        n = max(int(drift["n"]), 1)
        worst = max(worst, int(drift["n_ruin_inflated"]) / n)
    return worst


def expected(row: Mapping[str, Any], shrink: float) -> float:
    dev = row["splits"]["dev"]["tiers"]
    return sum(
        float(TIER_WEIGHTS[t]) * float(dev[t]["quality"]) * (1.0 - shrink * risk(row, t))
        for t in TIERS
    )


def hard_ok(row: Mapping[str, Any]) -> bool:
    checks = row["gates"]["checks"]
    return bool(
        checks["A_full_batch_inflated"]
        and checks["A_near_budget"]
        and checks["C_premium_fixed_composition"]
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = list(report["all_rows"])
    parent = next(r for r in rows if r["name"] == "the feasibility ladder")

    table = []
    for row in rows:
        entry = {
            "name": row["name"],
            "stage": row["stage"],
            "dev_weighted": float(row["dev_weighted"]),
            "hard_ok": hard_ok(row),
            "risk": {t: risk(row, t) for t in TIERS},
            "expected": {f"{s:g}": expected(row, s) for s in SHRINKS},
        }
        table.append(entry)

    base = {f"{s:g}": expected(parent, s) for s in SHRINKS}
    print(f"parent the feasibility ladder dev={parent['dev_weighted']:.6f} E={base}")
    for shrink in SHRINKS:
        key = f"{shrink:g}"
        ranked = sorted(
            (e for e in table if e["hard_ok"]),
            key=lambda e: e["expected"][key],
            reverse=True,
        )
        print(f"\n--- risk shrink {key} (p scaled by {key}) ---")
        for entry in ranked[:8]:
            delta = entry["expected"][key] - base[key]
            print(
                f"  {entry['name']:22} E={entry['expected'][key]:.6f} "
                f"(dev {entry['dev_weighted']:.6f}, dE={delta:+.6f}) "
                f"p={{f:{entry['risk']['fast']:.4f}, b:{entry['risk']['balanced']:.4f}, "
                f"p:{entry['risk']['premium']:.4f}}}"
            )

    payload = {
        "experiment": "the expected score ranking",
        "objective": "E[score] = sum_t w_t * q_t * (1 - shrink * p_t)",
        "shrinks": list(SHRINKS),
        "parent_expected": base,
        "rows": table,
    }
    (OUT / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nDONE the expected score ranking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
