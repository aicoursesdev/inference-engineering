import json
import pathlib

from image_tokens import (QWEN25, QWEN3, image_tokens, kv_bytes_per_token,
                          matmul_params, smart_resize)

HERE = pathlib.Path(__file__).parent


def test_sizes_round_to_the_token_grid():
    assert smart_resize(1080, 1920, QWEN25) == (1092, 1932)
    assert smart_resize(1080, 1920, QWEN3) == (1088, 1920)


def test_counts_match_what_a_real_server_reported():
    # llama-server, Qwen2.5-VL-3B: prompt tokens minus the 26-token text
    # prompt, minus 2 start/end markers (results_m5max.txt).
    for (w, h), reported in {(224, 224): 92, (448, 448): 284, (896, 896): 1052,
                             (1280, 720): 1224, (1920, 1080): 2719}.items():
        assert image_tokens(h, w, QWEN25) == reported - 26 - 2


def test_clamping():
    assert image_tokens(10, 10, QWEN25) == 4        # min_pixels 3136
    assert image_tokens(10000, 10000, QWEN25) <= 16384


def test_qwen3_vl_8b_numbers():
    cfg = json.load(open(HERE / "qwen3_vl_8b_config.json"))
    assert kv_bytes_per_token(cfg) == 147456
    assert round(matmul_params(cfg) / 1e9, 2) == 6.95
