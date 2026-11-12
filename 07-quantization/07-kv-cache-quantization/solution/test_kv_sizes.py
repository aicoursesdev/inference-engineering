import pytest

import kv_sizes


def test_qwen3_0_6b_kv_bytes_per_token():
    assert kv_sizes.per_token("Qwen3-0.6B", 16) == 114_688
    assert kv_sizes.per_token("Qwen3-0.6B", 8.5) == pytest.approx(114_688 * 8.5 / 16)


def test_fewer_bits_mean_a_smaller_cache():
    sizes = [kv_sizes.per_token("Qwen3-4B", b) for b in (16, 8.5, 4.5)]
    assert sizes == sorted(sizes, reverse=True)
