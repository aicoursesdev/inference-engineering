import three_numbers


def test_the_fixed_benchmark_sends_n_identical_length_requests():
    reqs = three_numbers.fixed(5, tokens_in=64, tokens_out=32)
    assert len(reqs) == 5 and {r["max_tokens"] for r in reqs} == {32}
    assert {len(r["prompt"].split()) for r in reqs} == {64}
