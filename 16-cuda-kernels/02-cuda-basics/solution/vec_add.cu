// Build:  nvcc -O3 -arch=sm_80 -o vec_add vec_add.cu     (A100; sm_90 for H100)
// Run:    ./vec_add
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "vec_add_kernel.cuh"

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

int main() {
    const int n = 1 << 26;                       // 67,108,864 floats
    const size_t bytes = n * sizeof(float);
    float *x = (float*)malloc(bytes), *y = (float*)malloc(bytes);
    float *out = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { x[i] = i % 100; y[i] = 1.0f; }

    float *dx, *dy, *dout;
    CHECK(cudaMalloc(&dx, bytes));
    CHECK(cudaMalloc(&dy, bytes));
    CHECK(cudaMalloc(&dout, bytes));
    CHECK(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;      // round up
    vec_add<<<blocks, threads>>>(dx, dy, dout, n);       // warm-up launch
    CHECK(cudaGetLastError());                           // launch errors
    CHECK(cudaDeviceSynchronize());                      // errors while running

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    const int reps = 20;
    CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++) vec_add<<<blocks, threads>>>(dx, dy, dout, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= reps;

    CHECK(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++)
        if (out[i] != x[i] + 1.0f) { printf("wrong at %d\n", i); return 1; }

    // Two reads and one write of 4 bytes per element.
    printf("ok: %.3f ms, %.1f GB/s\n", ms, 3.0 * bytes / (ms * 1e-3) / 1e9);
    cudaFree(dx); cudaFree(dy); cudaFree(dout);
    return 0;
}
