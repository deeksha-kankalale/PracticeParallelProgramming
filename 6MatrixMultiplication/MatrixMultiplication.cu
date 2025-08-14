/* Author: Deeksha Prakash Kankalale
   Date: August 13, 2025
   Description: CUDA 10x10 matrix multiplication (no tiling)
*/
#include <cstdio>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int SIZE = 10;                 // fixed 10x10
constexpr int ELEMS = SIZE * SIZE;

__global__ void matmul10(float* C, const float* A, const float* B) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < SIZE && y < SIZE) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < SIZE; ++k) {
            sum += A[y * SIZE + k] * B[k * SIZE + x];
        }
        C[y * SIZE + x] = sum;
    }
}

int main() {
    const size_t bytes = ELEMS * sizeof(float);

    // Host buffers
    std::vector<float> A(ELEMS), B(ELEMS), C(ELEMS);

    // Simple init so you can sanity-check results
    for (int r = 0; r < SIZE; ++r) {
        for (int c = 0; c < SIZE; ++c) {
            A[r * SIZE + c] = float(r + 1);        // row value
            B[r * SIZE + c] = float(c + 1);        // col value
        }
    }

    // Device buffers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    // One thread per output element; 10x10 fits in a single block
    dim3 blockDim(SIZE, SIZE);
    dim3 gridDim(1, 1);

    matmul10<<<gridDim, blockDim>>>(d_C, d_A, d_B);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Print full 10x10 result
    for (int r = 0; r < SIZE; ++r) {
        for (int c = 0; c < SIZE; ++c) {
            std::cout << C[r * SIZE + c] << (c + 1 == SIZE ? '\n' : ' ');
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
