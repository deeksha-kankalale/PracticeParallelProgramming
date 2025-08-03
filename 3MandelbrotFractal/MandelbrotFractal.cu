/*
Author : Deeksha Prakash Kankalale
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
using namespace cv;


__device__ int Mandelbrot (float x, float y, int width, int height){
    // define complex window
    const float xmin = -1.5f;
    const float xmax = 1.5f;
    const float ymin = -1.5f;
    const float ymax = 1.5f;
    const int max_iter = 200;

    // create variable to convert x and y to the constant C
    float C_real = xmin + x *(xmax - xmin)/width; //constant real
    float C_imag = ymin + y *(ymax - ymin)/height; //constant imaginary

    float z_real = 0;
    float z_imag = 0;

    int i;
    for (i=0; i<max_iter; i++){
        // implement z^2 (of z =  z^2 + C)
        z_real = z_real * z_real - z_imag * z_imag; // a^2 - b^2
        z_imag = 2.0f * z_real * z_imag; // 2ab
        // add Mandelbrot set constant
        z_real = z_real + C_real;
        z_imag = z_imag + C_imag;
    if ((z_real * z_real + z_imag * z_imag) > 4.0f) {
        break;
    }
    }

    return i;
}

__global__ void Kernel(unsigned char *pixelcolor, int width, int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    int iter = Mandelbrot (x, y, width, height);


    // map iteration count from julia set to pixel colour(0-255)

    int colour = (iter * 255)/200; //assuming max iter is 200

    // compute 1D offset 

    int offset = (y*width + x)*4;

    //write RGB

    pixelcolor[offset + 0] = 255 - colour; //R
    pixelcolor[offset + 1] = 0 ;     //G
    pixelcolor[offset + 2] = colour; //B
    pixelcolor[offset + 3] = 100;    // transparency of the pixel
}

int main (){  
    // 
    int width = 800;
    int height = 800;

    size_t  image_size = width * height * 4; // 4 values per pixel (RGBA)

    unsigned char *d_bitmap; // device buffer
    cudaError_t err = cudaMalloc((void**)&d_bitmap, image_size); //allocate device buffer
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    //kernel launch
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((width+15)/16, (height+15)/16);
    Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_bitmap, width, height);
    cudaDeviceSynchronize();

    //Copy results back to host
    unsigned char *h_bitmap = new unsigned char[image_size];
    cudaMemcpy(h_bitmap, d_bitmap, image_size, cudaMemcpyDeviceToHost);

    // Save image as PPM file
    FILE *fp = fopen("Mandelbrot.ppm", "wb");
    if (!fp) {
        printf("Failed to open file for writing!\n");
        return -1;
    }

    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // Write pixel data (RGB only, ignore alpha)
    for (int i = 0; i < width * height; i++) {
        fwrite(&h_bitmap[i * 4], 1, 3, fp);
    }

    fclose(fp);
    printf("Image saved as Mandelbrot.ppm\n");

    Mat image(height, width, CV_8UC4, h_bitmap);
    imshow("Mandelbrot Set", image);
    waitKey(0);


    // Free host memory


    delete[] h_bitmap;
    // cuda free memory 
    cudaFree(d_bitmap);
    return 0;

}