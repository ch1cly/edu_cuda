
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 10


__global__ void calc(float *c,const float* a,const float* b)
{
    int i = threadIdx.x;
    c[i] = a[i]*b[i];
}

int main(){

    float a[] = {1,1,1,1,1,1,1,1,1,1};
    float b[] = {2,2,2,2,2,2,2,2,2,2};
    float c [N];
    float* ca,*cb,*cc;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&ca, N * sizeof(float));
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&cb, N * sizeof(float));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&cc, N * sizeof(float));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }


    cudaStatus = cudaMemcpy(ca, a, N * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(cb, b, N * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

 

    calc <<<1, N >> > (cc,ca,cb);
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(c, cc, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    float ansv=0;
    for (int i = 0; i < N; ++i) {
        ansv += c[i];
    }

    printf("ansv if %f\n",ansv);
    cudaFree(ca);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
