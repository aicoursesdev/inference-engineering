// MoE down-projection for ONE decoded token, "one warp per output value":
//   out[h] = sum over the token's k experts e of  weight_e * (W_down[e][h][:] . a[e][:])
// A simplified fp32 sketch of the idea Cursor describes in its "warp decode"
// post (cursor.com/blog/warp-decode); it is not Cursor's kernel.
// Build: nvcc -O3 -arch=sm_80 -o warp_decode warp_decode.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// a: (K, I) the k experts' hidden activations; W: (E, H, I); ids, w: (K)
__global__ void warp_decode_down(const float* a, const float* W, const int* ids,
                                 const float* w, float* out, int K, int I, int H) {
    int h = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // my output value
    int lane = threadIdx.x % 32;
    if (h >= H) return;
    float acc = 0.f;                                  // one fp32 running total
    for (int s = 0; s < K; s++) {                     // the token's experts
        const float* row = W + ((size_t)ids[s] * H + h) * I;
        float part = 0.f;
        for (int i = lane; i < I; i += 32)            // lanes read neighbours
            part += row[i] * a[(size_t)s * I + i];
        acc += w[s] * part;                           // fold the routing weight in
    }
    for (int off = 16; off > 0; off >>= 1)            // add the 32 lanes' totals
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) out[h] = acc;
}

int main() {
    const int E = 32, K = 4, I = 2880, H = 2880;      // gpt-oss-20b's shapes
    size_t wn = (size_t)E * H * I;
    float *hW = (float*)malloc(wn * 4), *ha = (float*)malloc(K * I * 4);
    for (size_t i = 0; i < wn; i++) hW[i] = ((rand() % 200) - 100) / 1000.f;
    for (int i = 0; i < K * I; i++) ha[i] = ((rand() % 200) - 100) / 100.f;
    int hids[K] = {5, 1, 17, 30};
    float hw[K] = {.4f, .3f, .2f, .1f}, *hout = (float*)malloc(H * 4);

    float *dW, *da, *dw, *dout; int* dids;
    cudaMalloc(&dW, wn * 4); cudaMalloc(&da, K * I * 4); cudaMalloc(&dw, K * 4);
    cudaMalloc(&dids, K * 4); cudaMalloc(&dout, H * 4);
    cudaMemcpy(dW, hW, wn * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(da, ha, K * I * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw, K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dids, hids, K * 4, cudaMemcpyHostToDevice);

    warp_decode_down<<<(H + 7) / 8, 256>>>(da, dW, dids, dw, dout, K, I, H);
    cudaMemcpy(hout, dout, H * 4, cudaMemcpyDeviceToHost);
    for (int h = 0; h < H; h += 313) {                // spot-check on the CPU
        double ref = 0;
        for (int s = 0; s < K; s++) {
            double p = 0;
            for (int i = 0; i < I; i++) p += hW[((size_t)hids[s] * H + h) * I + i] * ha[s * I + i];
            ref += hw[s] * p;
        }
        if (fabs(ref - hout[h]) > 1e-3 * (1 + fabs(ref))) { printf("wrong at %d\n", h); return 1; }
    }
    printf("ok\n");
    return 0;
}
