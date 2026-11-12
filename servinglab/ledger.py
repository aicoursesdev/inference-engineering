"""The results ledger: one row per measurement, kept in ledger.csv (00.6).

    from servinglab.ledger import record, show

    record({"lesson": "00.6", "engine": "llama.cpp", "model": "Qwen3-0.6B-Q8_0",
            "hardware": "M5-Max", "change": "baseline",
            "ttft_ms": 41, "tpot_ms": 3.2, "tok_per_s": 310})
    show()

Every module adds rows. Change one thing per row and say what in `change`,
so a later reader can tell which change moved which number. The file sits in
the repo's root (ledger.csv, not committed); set LEDGER to put it elsewhere.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

PATH = Path(os.environ.get("LEDGER", Path(__file__).resolve().parent.parent / "ledger.csv"))

COLUMNS = ["when", "lesson", "engine", "model", "hardware", "change",
           "requests", "ttft_ms", "tpot_ms", "tok_per_s", "goodput_rps",
           "note"]


def record(row, path=None):
    """Append one result. `row` is a dict whose keys are in COLUMNS; `when`
    is filled in (local time) if you leave it out."""
    path = Path(path or PATH)
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown ledger columns {sorted(unknown)}; "
                         f"use {COLUMNS}")
    row = {"when": datetime.now().isoformat(timespec="minutes"), **row}
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})


def rows(path=None):
    """Every recorded row, as a list of dicts of strings."""
    path = Path(path or PATH)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def show(path=None, columns=("lesson", "engine", "model", "change",
                             "ttft_ms", "tpot_ms", "tok_per_s", "goodput_rps")):
    """Print the ledger as an aligned table."""
    data = rows(path)
    if not data:
        print("ledger is empty")
        return
    table = [list(columns)] + [[r[c] for c in columns] for r in data]
    widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
    for row in table:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
