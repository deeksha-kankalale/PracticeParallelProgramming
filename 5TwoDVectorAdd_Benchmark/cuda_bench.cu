// cuda_bench.cu
#include "cuda_bench.cuh"
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

CudaBenchResult benchmark_matrix_kernel(MatAddKernel kernel,
                                        float* dC,
                                        const float* dA,
                                        const float* dB,
                                        int M, int N,
                                        dim3 grid, dim3 block,
                                        int warmup_iters,
                                        int timed_iters,
                                        cudaStream_t stream)
{
    // Warmup (launch stream in config; DO NOT pass stream as a kernel argument)
    for (int i = 0; i < warmup_iters; i++) {
        kernel<<<grid, block, 0, stream>>>(dC, dA, dB, M, N);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed loop with events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; i++) {
        kernel<<<grid, block, 0, stream>>>(dC, dA, dB, M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const float avg_ms = ms_total / timed_iters;
    const double elems = 1.0 * M * N;
    const double bytes = elems * sizeof(float);
    const double bytes_moved = 3.0 * bytes; // A read + B read + C write

    CudaBenchResult res;
    res.avg_ms = avg_ms;
    res.gbps   = float((bytes_moved / avg_ms) * 1e-6);
    res.gflops = float((elems        / avg_ms) * 1e-6);
    return res;
}
