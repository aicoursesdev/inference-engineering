"""Offline check: a measured run becomes a ledger row you can read back."""

from servinglab.ledger import record, rows


def test_row_round_trip(tmp_path):
    f = tmp_path / "ledger.csv"
    record({"lesson": "00.6", "engine": "llama.cpp", "change": "baseline",
            "ttft_ms": 35.9, "tpot_ms": 10.1}, f)
    (row,) = rows(f)
    assert row["change"] == "baseline" and float(row["ttft_ms"]) == 35.9
