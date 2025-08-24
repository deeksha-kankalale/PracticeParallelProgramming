#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include<cstdlib> // for atoi
using namespace std;

__global__ void MatrixMultiplicationNaive (float *A, float *B, float *C, int N){
    int m = blockIdx.x * blockDim.x + threadIdx.x; //WIDTH is the dimension of block.
    int n = blockIdx.y * blockDim.y + threadIdx.y; 
    // check if the threads are out of bound from matrix boundary
    if(m<N && n<N){
        C[m*N+n]=0.0f;
        // each thread computes dot product of one element of C array
        for (int k = 0; k<N; k++){
            C[m*N+n]+= A[m*N+k] * B[k*N+n];
        }
    } 
}


int main (int argc, char** argv){
    int N = 10; // default to 10 if user has not input a value
    if(argc>1){
        int tmp = atoi(argv[1]);
        if(tmp>0) N=tmp;
    }

    vector<float> A(N * N), B(N * N), C(N * N);
    
    //populate the A and B vector
    for (int m=0; m<N; m++){
        for(int n=0; n<N; n++){
            A[m*N+n] = m;
            B[m*N+n] = n;

        }
    }
    
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    // Device buffers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    //copy to device memory
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    // kernel call
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,(N + block.y - 1) / block.y,1);

    MatrixMultiplicationNaive<<<grid,block>>>(d_A,d_B,d_C,N);

    cudaDeviceSynchronize();

    cudaMemcpy(C.data(),d_C,bytes,cudaMemcpyDeviceToHost);

    // Print top-left up to 10×10 (clamped to N if N<10)
    int showR = std::min(N, 10); 
    int showC = std::min(N, 10);
    for (int r = 0; r < showR; ++r) {
        for (int c = 0; c < showC; ++c) {
            std::cout << C[r * N + c] << (c + 1 == showC ? '\n' : ' ');
        }
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}