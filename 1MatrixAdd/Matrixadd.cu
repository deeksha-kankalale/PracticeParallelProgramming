#include<stdio.h>
#include<cuda_runtime.h>

__global__ void MatrixAdd(const float* A, const float* B, float* C,int N){
    int tid = threadIdx.x + blockIdx.x *blockDim.x;
    if(tid<N){
        C[tid] = A[tid] + B[tid];
    }
}

int main (){  
    // add constant size array
    const int N = 10;
    size_t size = N * sizeof(float);

    //allocate host memory
    float h_A[N], h_B[N], h_C[N];

    for (int i=0; i<N; i++){
        h_A[i]=i;
        h_B[i]=2*i;
    }

    //allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    //copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)/(threadsPerBlock);
    MatrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    //Copy results back to host
     cudaMemcpy(h_C,d_C, size, cudaMemcpyDeviceToHost);

    //Print the output from the host

    for (int i=0; i<N ; i++){
        printf(" A[%d] %f + B[%d] %f = c[%d] %f \n" ,i,h_A[i], i, h_B[i],i, h_C[i]);
    }
    // cuda free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;

}