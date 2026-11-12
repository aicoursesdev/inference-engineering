// Runs vec_add_kernel.cuh on the CPU, one "thread" at a time, so you can
// check the index arithmetic with no GPU:  c++ -std=c++17 cpu_check.cpp && ./a.out
#include <cstdio>
#include <vector>

struct dim3 { unsigned x = 1, y = 1, z = 1; };
static dim3 threadIdx, blockIdx, blockDim, gridDim;
#define __global__
#include "vec_add_kernel.cuh"

int main() {
    const int n = 1000, threads = 256;
    const int blocks = (n + threads - 1) / threads;
    std::vector<float> x(n), y(n, 1.0f), out(n, -1.0f);
    for (int i = 0; i < n; i++) x[i] = i;
    blockDim.x = threads; gridDim.x = blocks;
    for (unsigned b = 0; b < gridDim.x; b++)
        for (unsigned t = 0; t < blockDim.x; t++) {
            blockIdx.x = b; threadIdx.x = t;
            vec_add(x.data(), y.data(), out.data(), n);
        }
    int bad = 0;
    for (int i = 0; i < n; i++) bad += out[i] != x[i] + 1.0f;
    printf("%d blocks of %d threads = %d threads for %d elements; wrong: %d\n",
           blocks, threads, blocks * threads, n, bad);
    return bad != 0;
}
