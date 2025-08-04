/* Author Deeksha Prakash Kankalale
   Date : August 3 2025
   Description : CPP host code with vector 1D. 

   LeetGPU : Question 1
*/

#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;

__global__ void VectorAdd(float *a, float *b, float* c, size_t sizeV)
{
    int tid = threadIdx.x +blockIdx.x * blockDim.x;
    c[tid]= a[tid] + b[tid];
}

int main()
{
    vector<float> A(100),B(100),C(100); //defined vectors in cpp
    size_t sizeVector = 100;    // choose the size of vectors.

    //fill host vector A and B
    for(int i=0; i<sizeVector; i++){
        A[i]=i;
        B[i]=i*2;
    }
    
    //Allocate device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeVector * sizeof(float));
    cudaMalloc((void**)&d_B, sizeVector * sizeof(float));
    cudaMalloc((void**)&d_C, sizeVector * sizeof(float));

    //populate the device memory with host memory
    cudaMemcpy(d_A, A.data(), sizeVector*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeVector*sizeof(float), cudaMemcpyHostToDevice);

    VectorAdd<<<1,100>>>(d_A, d_B, d_C, sizeVector);
    cudaDeviceSynchronize();
    //copy the result vector from device to host
    cudaMemcpy(C.data(), d_C, sizeVector*sizeof(float), cudaMemcpyDeviceToHost);

    //print the result

    for(int j=0; j<sizeVector; j++){
        cout<<A[j]<<" + "<<B[j]<<" = "<<C[j]<<endl;
    }
     
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
