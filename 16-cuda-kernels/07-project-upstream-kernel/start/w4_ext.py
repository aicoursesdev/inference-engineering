"""Build the 4-bit matvec kernel as a PyTorch extension (needs nvcc and a GPU).
The kernel is the one from lesson 16.4; this is only the binding."""
from pathlib import Path

CPP = "torch::Tensor w4_matvec(torch::Tensor W, torch::Tensor scales, torch::Tensor x);"

CUDA = r"""
#include <torch/extension.h>
#include <cstdint>
#define GROUP 128
__global__ void w4_kernel(const uint32_t* W, const float* scales, const float* x,
                          float* y, int N, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= N) return;
    const uint32_t* w = W + (size_t)row * (K / 8);
    const float* s = scales + (size_t)row * (K / GROUP);
    float acc = 0.f;
    for (int j = lane; j < K / 8; j += 32) {
        uint32_t packed = w[j];
        int k0 = j * 8;
        float scale = s[k0 / GROUP];
#pragma unroll
        for (int i = 0; i < 8; i++)
            acc += (((int)((packed >> (4 * i)) & 0xF) - 8) * scale) * x[k0 + i];
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[row] = acc;
}
torch::Tensor w4_matvec(torch::Tensor W, torch::Tensor scales, torch::Tensor x) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kInt32 && W.is_contiguous());
    TORCH_CHECK(x.dtype() == torch::kFloat32 && scales.dtype() == torch::kFloat32);
    int N = W.size(0), K = W.size(1) * 8;
    auto y = torch::empty({N}, x.options());
    w4_kernel<<<(N + 7) / 8, 256>>>((const uint32_t*)W.data_ptr<int32_t>(),
        scales.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), N, K);
    return y;
}
"""


def load():
    from torch.utils.cpp_extension import load_inline
    return load_inline(name="w4_ext", cpp_sources=[CPP], cuda_sources=[CUDA],
                       functions=["w4_matvec"], extra_cuda_cflags=["-O3"])
