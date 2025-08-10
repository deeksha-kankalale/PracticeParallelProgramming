/* Author Deeksha Prakash Kankalale
   Date : August 4 2025
   Description : CUDA matrix add + separate benchmark module
*/
#include <cstdio>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_bench.cuh"

#define ROW_SIZE     100
#define COLUMN_SIZE  100
#define SQUARE_MATRIX (ROW_SIZE * COLUMN_SIZE)

__global__ void vectorAdd2D(float* C, const float* A, const float* B, int M, int N){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (x < N && y < M){
        int idx = y * N + x; // row-major
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    const int M = ROW_SIZE;
    const int N = COLUMN_SIZE;
    const size_t bytes = (size_t)M * N * sizeof(float);

    std::vector<float> A(M*N), B(M*N), C(M*N);

    for(int r=0; r<M ; r++){
        for(int c=0; c<N; c++){
            A[r*N + c] = float(r*N + c);
            B[r*N + c] = 2.0f * float(r*N + c);
        }
    }

    float *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 8);
    dim3 gridDim( (N + blockDim.x - 1) / blockDim.x,
                  (M + blockDim.y - 1) / blockDim.y );

    // Pass function pointer, NOT a kernel launch
    CudaBenchResult result =
        benchmark_matrix_kernel(&vectorAdd2D, d_C, d_A, d_B, M, N,
                                gridDim, blockDim, /*warmup*/10, /*iters*/100);

    std::cout << "Kernel avg: " << result.avg_ms << " ms  |  "
              << result.gbps << " GB/s  |  "
              << result.gflops << " GFLOP/s\n";

    // Copy back once (optional correctness spot-check)
    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Print a small 10x10 corner
    int pr = std::min(10, M), pc = std::min(10, N);
    for (int r=0; r<pr; r++){
        for(int c=0; c<pc; c++){
            std::cout << C[r*N + c] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
