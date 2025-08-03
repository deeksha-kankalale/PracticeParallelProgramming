/*
Author : Deeksha Prakash Kankalale

1. define height and width of the image
2. define complex plane bounds (xmin = -1.5 and xmax = 1.5, ymin = -1.5, ymax =1.5 )
3. Max iterations = 200
4. Device function Julia input x,y, width and height, maps to (real,imaginary) 
 run Julia fractal (z = z*z + c) until escape/max iteration, return iteration count
5. Kernel function -> compute iteration count using the device function and assign colour.
6. main function define the number of threads, grids and launch the kernel

The background is the iterations that escaped the julia set.
The design happens when the pixels dont escape.
*/

#include<stdio.h>
#include<cuda_runtime.h>
#include <opencv2/opencv.hpp>


__device__ int Julia (int x, int y , int width, int height){
    const float C_real = -0.8f;
    const float C_imag = 0.156f;
    // define complex window 
    const float xmin = -1.5f;
    const float xmax = 1.5f;
    const float ymin = -1.5f;
    const float ymax = 1.5f;
    const int max_iter = 1000; //increases the number of iterations - so the number of escapes will increase. 
    
    
    // create variable to convert x and y to complex number
    float real = xmin + x * (xmax -xmin)/width; // new real
    float imag = ymin + y * (ymax - ymin)/height; // new imaginary

    int i;
    for (i= 0; i<max_iter; i++){
        // implementing z = z*z + C -> Formula (a-b)^2= a^2 - b^2 + 2ab
        float z_real = real * real - imag * imag; //a^2 - b^2
        float z_imag = 2.0f * real * imag; // 2ab
        //add julia set constant
        real = z_real + C_real;
        imag = z_imag + C_imag;
        // escape condition is |z|^2

        if(real* real + imag*imag > 4.0f){
           break;
        }
    }
    return i;
}
/*
Global Image (6x6 pixels):

P00 P01 P02 P03 P04 P05
P10 P11 P12 P13 P14 P15
P20 P21 P22 P23 P24 P25
P30 P31 P32 P33 P34 P35
P40 P41 P42 P43 P44 P45
P50 P51 P52 P53 P54 P55

Block(0,0): covers pixels
P00 P01
P10 P11

Block(1,0): covers pixels
P02 P03
P12 P13

Block(2,0): covers pixels
P04 P05
P14 P15

... and so on

*/
__global__ void Kernel(unsigned char *pixelcolor, int width, int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    int iter = Julia (x, y, width, height);


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
    FILE *fp = fopen("julia.ppm", "wb");
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
    printf("Image saved as julia.ppm\n");

    cv::Mat image(height, width, CV_8UC4, h_bitmap);
    cv::imshow("Julia Set", image);
    cv::waitKey(0);


    // Free host memory


    delete[] h_bitmap;
    // cuda free memory 
    cudaFree(d_bitmap);
    return 0;

}