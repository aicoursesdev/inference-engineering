// Three transposes of an n x n float matrix: the same result, three memory
// patterns.   nvcc -O3 -arch=sm_80 -o transpose transpose.cu && ./transpose
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32

// Reads are coalesced (lanes walk x), but the write out[x * n + y] jumps by n
// floats between lanes: each lane lands in its own 32-byte sector.
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) out[x * n + y] = in[y * n + x];
}

// Stage the tile in shared memory so both the read and the write are
// coalesced. PAD = 1 shifts each row by one bank so the column read is free.
template <int PAD>
__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + PAD];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();                                  // the whole tile is in
    int ox = blockIdx.y * TILE + threadIdx.x;         // the block lands at (by, bx)
    int oy = blockIdx.x * TILE + threadIdx.y;
    if (ox < n && oy < n) out[oy * n + ox] = tile[threadIdx.x][threadIdx.y];
}

template <typename K>
float time_ms(K kernel, dim3 grid, dim3 block, const float* in, float* out, int n) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    kernel<<<grid, block>>>(in, out, n);              // warm-up
    cudaEventRecord(a);
    for (int r = 0; r < 20; r++) kernel<<<grid, block>>>(in, out, n);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / 20;
}

int main() {
    const int n = 8192;
    const size_t bytes = (size_t)n * n * sizeof(float);
    float *h = (float*)malloc(bytes), *h2 = (float*)malloc(bytes);
    for (size_t i = 0; i < (size_t)n * n; i++) h[i] = (float)(i % 1000);
    float *in, *out;
    cudaMalloc(&in, bytes); cudaMalloc(&out, bytes);
    cudaMemcpy(in, h, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    float ms[3] = {
        time_ms(transpose_naive, grid, block, in, out, n),
        time_ms(transpose_tiled<0>, grid, block, in, out, n),
        time_ms(transpose_tiled<1>, grid, block, in, out, n)};
    const char* names[3] = {"naive", "tiled (bank conflicts)", "tiled + pad"};

    cudaMemcpy(h2, out, bytes, cudaMemcpyDeviceToHost);   // last run: padded
    for (int i = 0; i < n; i += 997)
        for (int j = 0; j < n; j += 991)
            if (h2[(size_t)j * n + i] != h[(size_t)i * n + j]) { printf("wrong\n"); return 1; }
    for (int k = 0; k < 3; k++)                           // one read + one write
        printf("%-24s %7.3f ms  %7.1f GB/s\n", names[k], ms[k],
               2.0 * bytes / (ms[k] * 1e-3) / 1e9);
    return 0;
}
