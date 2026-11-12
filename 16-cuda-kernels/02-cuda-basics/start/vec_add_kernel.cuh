// The kernel alone, so the CPU check (cpu_check.cpp) and the GPU program
// (vec_add.cu) compile the very same lines.
__global__ void vec_add(const float* x, const float* y, float* out, int n) {
    // TODO 16.2: find this thread's number in the whole grid (blockIdx.x * blockDim.x + threadIdx.x),
    // guard it against n, then write out[i] = x[i] + y[i]; see the lesson's 'The kernel' section
}
