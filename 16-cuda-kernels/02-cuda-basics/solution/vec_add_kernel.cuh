// The kernel alone, so the CPU check (cpu_check.cpp) and the GPU program
// (vec_add.cu) compile the very same lines.
__global__ void vec_add(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // my number in the whole grid
    if (i < n) {                                     // the last block may overhang
        out[i] = x[i] + y[i];
    }
}
