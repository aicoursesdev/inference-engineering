import fleet


def test_self_check_reproduces_lmsys_and_deepseek():
    fleet.self_check()


def test_design_for_200_requests_a_second():
    r = fleet.design(200, in_tok=2000, out_tok=400, cache_hit=0.5)
    assert (r["prefill_nodes"], r["decode_nodes"], r["nodes"]) == (5, 5, 10)
    assert r["usd_per_hour"] == 160
    assert round(r["usd_per_m_output"], 2) == 0.56


def test_more_cache_hits_never_need_more_prefill_nodes():
    a = fleet.design(200, 2000, 400, 0.0)["prefill_nodes"]
    b = fleet.design(200, 2000, 400, 0.9)["prefill_nodes"]
    assert b < a
