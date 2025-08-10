// cuda_bench.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

struct CudaBenchResult {
    float avg_ms;   // kernel-only average per iteration
    float gbps;     // memory throughput (2 reads + 1 write)
    float gflops;   // 1 add per element
};

// Kernel type MUST match your kernel: __global__ void K(float* C, const float* A, const float* B, int M, int N)
using MatAddKernel = void (*)(float*, const float*, const float*, int, int);

// Benchmark a matrix-add style kernel. Stream is in the LAUNCH CONFIG, not kernel params.
CudaBenchResult benchmark_matrix_kernel(MatAddKernel kernel,
                                        float* dC,
                                        const float* dA,
                                        const float* dB,
                                        int M, int N,
                                        dim3 grid, dim3 block,
                                        int warmup_iters = 10,
                                        int timed_iters = 100,
                                        cudaStream_t stream = 0);
