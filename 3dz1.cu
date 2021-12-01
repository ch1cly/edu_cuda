#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>


#define N 100


__global__ void dzeta(float *s,float*c)
{

    c[threadIdx.x] = 1.f / powf(float(threadIdx.x + 1), *s);


}


int main()
{

    float x = 2;
    float c[N] = { 0 };

    
    float* dev_x = 0;
    float* dev_c = 0;

    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(float));



    cudaStatus = cudaMalloc((void**)&dev_x, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 3;
    }

    // Copy input vectors from host memory to GPU buffers.
  
    cudaStatus = cudaMemcpy(dev_c, c, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 5;
    }

    cudaStatus = cudaMemcpy(dev_x, &x,  sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 6;
    }

  
    dzeta << <1, N >> > (dev_x, dev_c);

   
    
    cudaStatus = cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 7;
    }


    double sum = 0;
    for (int i = 0; i < N; ++i) {
       // printf("a = %f\n", a[i]);
        sum += c[i];
    }
    printf("dzeta = %f\n", sum);


    cudaFree(dev_c);
   // cudaFree(dev_a);
    cudaFree(dev_x);



    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 10;
    }



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 11;
    }

    return 0;
}