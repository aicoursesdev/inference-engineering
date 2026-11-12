import block_bytes


def test_the_lessons_block_sizes():
    assert block_bytes.kv_bytes_per_token("Qwen3-0.6B") == 114_688
    assert block_bytes.kv_bytes_per_token("Qwen3-0.6B") * 256 == 29_360_128
    assert block_bytes.kv_bytes_per_token("Qwen3-4B") == 147_456
