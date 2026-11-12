from shrink import per_token


def test_fewer_kv_heads_means_proportionally_less_cache():
    assert per_token(32, 32, 128) == 4 * per_token(32, 8, 128)
