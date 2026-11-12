import pytest

from servinglab.data import model_config
from servinglab.roofline import predict, shape


def test_parameter_count_matches_llama_cpp():
    # llama-bench reported 596,049,920 parameters for Qwen3-0.6B (build 8642541b2, Nov 2026)
    assert shape(model_config("Qwen3-0.6B"))["params"] == 596_049_920


def test_parameter_count_of_llama_8b():
    assert shape(model_config("Llama-3.1-8B-Instruct"))["params"] == 8_030_261_248


def test_decode_at_batch_one_is_bandwidth_over_weight_bytes():
    chip = {"bandwidth_gbs": 1000, "fp16_tflops": 1000}
    p = predict("Qwen3-0.6B", chip, context=0)
    weights = shape(model_config("Qwen3-0.6B"))["matmul_params"] * 2
    assert p["decode"]["tokens_per_s"] == pytest.approx(1e12 / weights)
    assert p["decode"]["bound"] == "memory"


def test_batching_crosses_the_ridge_near_the_ridge_point():
    chip = {"bandwidth_gbs": 3350, "fp16_tflops": 989.5}  # an H100: ridge about 295
    bound = lambda b: predict("Qwen3-4B", chip, batch=b, context=0)["decode"]["bound"]
    assert bound(64) == "memory" and bound(600) == "compute"


def test_a_long_prefill_is_compute_bound_and_scales_with_peak():
    chip = {"bandwidth_gbs": 3350, "fp16_tflops": 989.5}
    p = predict("Llama-3.1-8B-Instruct", chip, prompt=2000)["prefill"]
    assert p["bound"] == "compute"
    faster = predict("Llama-3.1-8B-Instruct", {**chip, "fp16_tflops": 1979}, prompt=2000)["prefill"]
    assert faster["seconds"] == pytest.approx(p["seconds"] / 2)


def test_chip_without_flops_is_memory_only_and_overhead_adds_time():
    p = predict("Qwen3-0.6B", {"bandwidth_gbs": 460}, context=0)
    q = predict("Qwen3-0.6B", {"bandwidth_gbs": 460}, context=0, overhead_seconds=0.001)
    assert p["ridge_flops_per_byte"] is None
    assert q["decode"]["seconds"] == pytest.approx(p["decode"]["seconds"] + 0.001)


def test_names_work_and_mixture_of_experts_is_refused():
    assert predict("Qwen3-4B", "T4")["decode"]["tokens_per_s"] > 0
    with pytest.raises(ValueError):
        predict("DeepSeek-V3", "H100-SXM")
