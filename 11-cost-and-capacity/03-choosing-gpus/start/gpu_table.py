"""A price-per-token ceiling for Llama 3.1 8B in 16-bit, per GPU, from datasheets and list prices.

This is Module 02's roofline: a ceiling no real engine reaches. It ranks GPUs on memory and price;
only a benchmark at matched speed (see the lesson) says what you will really get.
"""
from servinglab.cost import cost_per_mtok, hourly_rate
from servinglab.data import hardware, model_config
from servinglab.roofline import predict, shape

BATCH, CONTEXT = 32, 1000
weights_gb = shape(model_config("Llama-3.1-8B-Instruct"))["params"] * 2 / 1e9
kv_gb = BATCH * CONTEXT * shape(model_config("Llama-3.1-8B-Instruct"))["kv_bytes_per_token_f16"] / 1e9
print(f"weights {weights_gb:.1f} GB + KV for {BATCH} x {CONTEXT} tokens {kv_gb:.1f} GB = {weights_gb + kv_gb:.1f} GB\n")
print("GPU            memory  GB/s   fits  ceiling tok/s  $/h (provider)           $ per M output tokens")
for gpu, provider, tier in [("T4", "Modal", None), ("L4", "RunPod", "secure cloud"), ("A100-40GB", "Modal", None),
                            ("A100-80GB-SXM", "RunPod", "secure cloud"), ("H100-SXM", "RunPod", "secure cloud"),
                            ("H200", "RunPod", "secure cloud"), ("B200", "RunPod", "secure cloud")]:
    chip = hardware(gpu)
    rate, _ = hourly_rate(provider, gpu, tier)
    fits = weights_gb + kv_gb < chip["memory_gb"] * 0.9  # keep a tenth for the runtime
    tps = predict("Llama-3.1-8B-Instruct", gpu, batch=BATCH, context=CONTEXT)["decode"]["tokens_per_s"] if chip["fp16_tflops"] else None
    cost = f"{cost_per_mtok(rate, tps):7.3f}" if fits else "      -"
    print(f"{gpu:<14} {chip['memory_gb']:5.0f}  {chip['bandwidth_gbs']:5.0f}  {'yes' if fits else 'no ':<4}  "
          f"{tps:12,.0f}  {rate:5.2f} ({provider} {tier or ''})".ljust(77) + cost)
