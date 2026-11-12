from servinglab.roofline import predict


def test_recompute_cost_grows_faster_than_linearly():
    a = predict("Qwen3-4B", "L4", prompt=8_000)["prefill"]["seconds"]
    b = predict("Qwen3-4B", "L4", prompt=32_000)["prefill"]["seconds"]
    assert b > 4 * a            # attention adds a squared term on top of the linear matmuls
