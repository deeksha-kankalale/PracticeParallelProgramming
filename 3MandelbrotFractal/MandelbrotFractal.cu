/*
Author : Deeksha Prakash Kankalale
*/

#include <stdio.h>
#include <cuda_runtime.h>


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
        // add julia set constant
    if ((z_real * z_real + z_imag * z_imag) > 4.0f) {

        break;
    }

    return i;

}