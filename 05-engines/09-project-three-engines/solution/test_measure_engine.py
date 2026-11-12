import measure_engine
from servinglab.ledger import record, rows

RUNS = [(20.4, 340.2, 113), (19.4, 326.4, 113), (19.4, 324.2, 113),
        (19.4, 316.1, 113), (20.4, 314.7, 113)]  # the lesson's llama.cpp runs


def make():
    return measure_engine.ledger_row(
        RUNS, "llama.cpp", "0.8.0 build 11960", "Qwen3-0.6B-Q8_0.gguf",
        "Q8_0", "Apple M5 Max laptop, 36 GB", 1.2)


def test_the_row_holds_the_medians_of_the_runs():
    row = make()
    assert row["ttft_ms"] == 19.4 and row["tok_per_s"] == 324.2
    assert row["requests"] == 5 and row["engine"] == "llama.cpp"
    assert abs(row["tpot_ms"] - 1000 / 324.2) < 0.01


def test_the_row_names_the_version_the_file_and_the_quantization():
    row = make()
    assert "0.8.0 build 11960" in row["change"]
    assert "Qwen3-0.6B-Q8_0.gguf" in row["model"] and "Q8_0" in row["model"]


def test_the_note_carries_the_spread_and_the_machine_load():
    note = make()["note"]
    assert "spread 25.5 tok/s" in note and "load 1.2" in note


def test_the_row_is_a_valid_ledger_row(tmp_path):
    path = tmp_path / "ledger.csv"
    record(make(), path=path)
    assert rows(path)[0]["tok_per_s"] == "324.2"
