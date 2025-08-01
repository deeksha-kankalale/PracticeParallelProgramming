#include<stdio.h>

__global__ void hello_from_gpu(){
    printf("Hello from GPU thread %d \n", threadIdx.x);
}

int main (){  
    hello_from_gpu<<<1,5>>>();
    cudaDeviceSynchronize();
    return 0;

}