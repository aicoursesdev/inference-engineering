// A 4-bit weight, fp32 activation matrix-vector product: y = W x, the shape of
// one decode step through one layer (batch 1).
//   W      packed (N, K/8) uint32, nibble i of a word = bits 4i..4i+3
//   scales (N, K/GROUP) float;  q in 0..15 stands for (q - 8) * scale
// Build: nvcc -O3 -arch=sm_80 -o w4_matvec w4_matvec.cu && ./w4_matvec
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define GROUP 128

// One warp per output row; 8 warps (rows) per 256-thread block.
__global__ void w4_matvec(const uint32_t* W, const float* scales,
                          const float* x, float* y, int N, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= N) return;                  // all 32 lanes share a row: safe

    const uint32_t* w = W + (size_t)row * (K / 8);
    const float* s = scales + (size_t)row * (K / GROUP);
    float acc = 0.f;
    // Lanes read neighbouring words: a warp loads 128 contiguous bytes.
    for (int j = lane; j < K / 8; j += 32) {
        uint32_t packed = w[j];            // 8 weights in one 4-byte load
        int k0 = j * 8;
        float scale = s[k0 / GROUP];       // 8 | GROUP: one group per word
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int q = (packed >> (4 * i)) & 0xF;           // unpack in a register
            acc += ((q - 8) * scale) * x[k0 + i];        // dequantize, multiply
        }
    }
    for (int off = 16; off > 0; off >>= 1)               // sum the 32 partials
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[row] = acc;
}

int main() {
    const int N = 14336, K = 4096;         // one Llama 3.1 8B MLP matrix
    size_t words = (size_t)N * K / 8, groups = (size_t)N * K / GROUP;
    uint32_t* hW = (uint32_t*)malloc(words * 4);
    float *hs = (float*)malloc(groups * 4), *hx = (float*)malloc(K * 4);
    float* hy = (float*)malloc(N * 4);
    srand(0);
    for (size_t i = 0; i < words; i++) hW[i] = ((uint32_t)rand() << 16) ^ rand();
    for (size_t i = 0; i < groups; i++) hs[i] = 0.01f * (1 + rand() % 5);
    for (int i = 0; i < K; i++) hx[i] = (rand() % 200 - 100) / 100.f;

    uint32_t* dW; float *ds, *dx, *dy;
    cudaMalloc(&dW, words * 4); cudaMalloc(&ds, groups * 4);
    cudaMalloc(&dx, K * 4); cudaMalloc(&dy, N * 4);
    cudaMemcpy(dW, hW, words * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs, groups * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, K * 4, cudaMemcpyHostToDevice);

    int threads = 256, blocks = (N + 8 - 1) / 8;
    w4_matvec<<<blocks, threads>>>(dW, ds, dx, dy, N, K);
    cudaMemcpy(hy, dy, N * 4, cudaMemcpyDeviceToHost);

    for (int r = 0; r < N; r += 1111) {                  // spot-check on the CPU
        double ref = 0;
        for (int j = 0; j < K / 8; j++)
            for (int i = 0; i < 8; i++)
                ref += (((hW[(size_t)r * (K / 8) + j] >> (4 * i)) & 0xF) - 8) *
                       (double)hs[(size_t)r * (K / GROUP) + (j * 8) / GROUP] * hx[j * 8 + i];
        if (fabs(ref - hy[r]) > 1e-2 * (1 + fabs(ref))) { printf("wrong at row %d\n", r); return 1; }
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < 100; r++) w4_matvec<<<blocks, threads>>>(dW, ds, dx, dy, N, K);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= 100;
    double bytes = words * 4.0 + groups * 4.0;           // fp32 scales here
    printf("ok: %.1f us, %.0f GB/s of weight traffic\n", ms * 1e3, bytes / (ms * 1e-3) / 1e9);
    return 0;
}
