
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 100


__global__ void calc(float *c)
{
    int i = threadIdx.x;
    c[i] = __expf(float(i)/float(100));
}

int main(){

    float a[N];
    float* ca;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&ca, N * sizeof(float));
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    calc <<<1, N >> > (ca);
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(a, ca, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    for (int i = 0; i < N; ++i) {
        printf("err is %f\n", abs(exp(float(i)/float(100)) - a[i]));
    }
    cudaFree(ca);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
