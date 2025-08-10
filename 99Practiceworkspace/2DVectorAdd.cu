/* Author Deeksha Prakash Kankalale
   Date : August 4 2025
   Description : CPP host code with vector 2D. 
*/

#include<stdio.h>
#include<iostream>
#include<vector>
#include <cuda_runtime.h>

#define SQUARE_MATRIX 10000
#define ROW_SIZE 100
#define COLUMN_SIZE 100 

using namespace std;

__global__ void vectorAdd2D (float* A, float* B, float* C, int width, int height){
   // create square thread
   int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
   int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
   // safer to check x and y bound seperately if((tid_y * width + tid_x) < total_size)
   if (tid_x<width && tid_y<height)
   {
      C[tid_y *width + tid_x] = A[tid_y *width + tid_x] + B[tid_y *width + tid_x];
   }
}


int main(){
   //writing a row major code.
   int width = COLUMN_SIZE;
   int height = ROW_SIZE;
   //cant use vector<float> as this will not be contiguous block of memory
   float A[SQUARE_MATRIX], B[SQUARE_MATRIX], C[SQUARE_MATRIX];

   for(int r=0; r<ROW_SIZE ; r++){
      for(int c=0; c<COLUMN_SIZE; c++){
         A[r*width+c] = r * width + c; 
         B[r*width+c] = 2 * (r * width + c);
      }
   }
   
   //Allocate device memory
   float *d_A, *d_B, * d_C;
   cudaMalloc((void**)&d_A, SQUARE_MATRIX * sizeof(float)); 
   cudaMalloc((void**)&d_B, SQUARE_MATRIX * sizeof(float));
   cudaMalloc((void**)&d_C, SQUARE_MATRIX * sizeof(float));

   //Populate the device memory with host memory
   cudaMemcpy(d_A, A, SQUARE_MATRIX*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B, SQUARE_MATRIX *sizeof(float), cudaMemcpyHostToDevice);

   //calculate the number of 
   dim3 blockDim(32, 32);        // 32 threads in x, 32 threads in y per block
   dim3 gridDim(4,4);           // 4 blocks in x, 4 blocks in y

   //call the kernel
   vectorAdd2D<<<gridDim,blockDim>>>(d_A,d_B,d_C, ROW_SIZE, COLUMN_SIZE);

   //sync threads
   cudaDeviceSynchronize();

   //copy the result back to host memory from device memory
   cudaMemcpy(C, d_C, SQUARE_MATRIX*sizeof(float), cudaMemcpyDeviceToHost);

   //print the output in the host memory
   for (int r=0; r<10; r++){
      for(int c=0; c<10; c++){
         cout << C[r*width+c]<<" ";
      }
      cout<<"\n";
   }
   
   //free allocated device memory
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

}