from servinglab.data import traces
from disagg import Cost, kv_bytes, run_colocated, run_disagg, report

C = Cost()
ONE = [{"id": "x", "arrival_s": 100.0, "customer": "tidewater", "slo_class": "interactive", "input_tokens": 500,
        "output_tokens": 20, "prefix_id": "tw-system", "prefix_tokens": 220}]
DAY = [r for r in traces("mixed") if 12 * 3600 <= r["arrival_s"] < 12 * 3600 + 600]


def test_kv_size_matches_distserve():
    assert abs(kv_bytes(64, 9216, 512) / 2**30 - 1.125) < 1e-9


def test_lone_request_waits_for_nothing():
    r = run_colocated(ONE, 1, C)[0]
    assert abs(r["first_t"] - r["_at"] - C.prefill_s(500)) < 1e-9


def test_transfer_is_bytes_over_bandwidth():
    r = run_disagg(ONE, 1, 1, C, 50e9)[0]
    assert abs(r["transfer_s"] - (0.0005 + 500 * C.kv_tok / 50e9)) < 1e-12


def test_everything_finishes_in_order():
    for reqs in (run_colocated(DAY, 4, C, 60), run_colocated(DAY, 4, C, 60, policy="chunked"), run_disagg(DAY, 1, 3, C, 50e9, 60)):
        assert len(reqs) == len(DAY) and all(r["done_t"] >= r["first_t"] > r["_at"] for r in reqs)
        assert 0 < report(reqs)["attained"] <= 1


def test_a_35000_token_prefill_takes_about_1_7_seconds():
    assert 1.6 < C.prefill_s(35_000) < 1.8


def test_a_decode_step_is_the_slower_of_memory_and_compute():
    one = C.decode_s(1, 1000)
    assert abs(one - ((C.mm_bytes + 1000 * C.kv_tok) / C.bw + C.overhead_s)) < 1e-12   # small batch: memory-bound
    assert C.decode_s(4096, 4096 * 1000) > 20 * one                                  # huge batch: compute grows it


def test_goodput_counts_only_requests_that_met_both_targets():
    def req(first, done):  # arrives at 0; 11 output tokens
        return {"_at": 0.0, "first_t": first, "done_t": done, "output_tokens": 11, "slo_class": "interactive"}
    r = report([req(0.5, 0.5 + 10 * 0.04), req(1.5, 1.5 + 10 * 0.04), req(0.5, 0.5 + 10 * 0.2)])
    assert abs(r["attained"] - 1 / 3) < 1e-9     # the second has a late first token, the third slow tokens
    assert abs(r["goodput_rps"] - 1 / 2.5) < 1e-9   # one good request over the 2.5 s the run lasted
