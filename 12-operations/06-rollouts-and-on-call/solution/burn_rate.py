"""Burn rate: how fast a service is spending its error budget, on the recorded incident-1 runs.

    uv run python burn_rate.py

Monsoon's agent class promises time to first token (p95) of 2.0 s. "95% of requests under 2.0 s"
leaves a budget of 5% that may be slower. Burn rate = (share of requests over the limit) / budget:
1 spends the budget exactly over the SLO's period; 14.4 would spend a 30-day budget in about 2 days
(Google's SRE workbook uses 14.4 as the 1-hour page threshold for a 99.9% SLO).
"""
import json
from pathlib import Path

LIMIT_MS, BUDGET = 2000, 0.05
DIR = Path(__file__).resolve().parents[3] / "data" / "recorded" / "incidents"
for run in ("good", "bad"):
    rows = [json.loads(l) for l in (DIR / f"incident1-{run}.jsonl").read_text().splitlines()]
    over = sum(r["ttft_ms"] > LIMIT_MS for r in rows)
    burn = over / len(rows) / BUDGET
    print(f"{run:4}: {over:2} of {len(rows)} over {LIMIT_MS} ms = {over / len(rows) * 100:4.1f}%, burn rate {burn:4.1f}")
