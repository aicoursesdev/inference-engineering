from fair_table import ROWS


def test_bytes_read_per_second_is_what_makes_engines_comparable():
    gbs = {name: tps * size / 1e9 for name, tps, size in ROWS}
    assert round(gbs["llama-server"]) == 187
    assert round(gbs["mlx_lm.server"]) == 139
    # MLX is fastest in tokens per second but not in bytes read per second
    fastest = max(ROWS, key=lambda r: r[1])[0]
    assert fastest == "mlx_lm.server" and gbs[fastest] < gbs["llama-server"]
