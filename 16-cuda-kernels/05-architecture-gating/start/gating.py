"""Which kernel runs on which GPU: a support matrix you can query.

Every row is copied from the project's own page, named in `source`, as it
read on 2026-11-12. "listed" means the page names that compute capability;
anything else is "not listed", which is not proof it fails: look it up.
Compute capabilities (NVIDIA CUDA GPUs page): T4 7.5, A100 8.0, L4 8.9,
H100 9.0, B200 10.0, B300 10.3.
"""

GPUS = {"T4": 7.5, "A100": 8.0, "L4": 8.9, "H100": 9.0, "B200": 10.0}

MATRIX = {
    "FlashAttention 2 (flash-attn)": {
        "listed": {8.0, 8.6, 8.9, 9.0},
        "source": "github.com/Dao-AILab/flash-attention: Ampere, Ada or Hopper"},
    "FlashAttention 3 (beta)": {
        "listed": {9.0},
        "source": "github.com/Dao-AILab/flash-attention: H100 / H800"},
    "FlashAttention 4": {
        "listed": {9.0, 10.0},
        "source": "github.com/Dao-AILab/flash-attention: Hopper and Blackwell, "
                  "H100 and B200 as examples"},
    "FlashInfer (SGLang default)": {
        "minimum": 7.5,
        "source": "docs.sglang.io install page: only supports sm75 and above"},
    "Triton tensor-core tl.dot": {
        "minimum": 8.0,
        "source": "github.com/triton-lang/triton README: NVIDIA 8.0+"},
    "Marlin W4 (IST-DASLab repo)": {
        "minimum": 8.0,
        "source": "github.com/IST-DASLab/marlin: 8.0+, not yet optimised for Hopper"},
    "vLLM FP8 W8A8 (llm-compressor)": {
        "listed": {8.9, 9.0},
        "source": "docs.vllm.ai quantization table: Ada and Hopper (no Blackwell column)"},
    "MXFP4 / NVFP4 tensor cores": {
        "listed": {10.0, 10.3},
        "source": "perplexity.ai/hub/blog CuTeDSL post: Blackwell adds them"},
}


def supported(kernel, cc):
    """'yes' or 'not listed'."""
    row = MATRIX[kernel]
    if "minimum" in row:
        return "yes" if cc >= row["minimum"] else "not listed"
    return "yes" if cc in row["listed"] else "not listed"


def table():
    head = "".join(f"{g:>7}" for g in GPUS)
    lines = [f"{'':<34}{head}"]
    for kernel in MATRIX:
        cells = "".join(f"{'yes' if supported(kernel, cc) == 'yes' else '-':>7}"
                        for cc in GPUS.values())
        lines.append(f"{kernel:<34}{cells}")
    return "\n".join(lines)


def this_gpu():
    """The compute capability of the GPU PyTorch sees, e.g. 8.0, or None."""
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return major + minor / 10
    except Exception:
        return None


if __name__ == "__main__":
    print(table())
    cc = this_gpu()
    print("this machine:", cc if cc is not None else "no CUDA GPU visible")
