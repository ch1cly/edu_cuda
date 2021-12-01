
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 100


__global__ void integral(double *a)
{
    int i = threadIdx.x;
    a[i] = std::sqrtf(1.0 - double(i) * double(i) / double(N) / double(N));
}

int main()
{
    double a[N] = { 0 };
    double* p_a;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&p_a, N * sizeof(double));

   
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    integral <<<1, N >>> (p_a);

    cudaStatus = cudaMemcpy(a, p_a, N * sizeof(double), cudaMemcpyDeviceToHost);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
        return 1;
    }
    
    double q = 0;
    for (int i = 0; i < N; ++i) {
        q += a[i];
    }
    printf("Pi is %f\n", q*4/N);

    cudaFree(p_a);
    return 0;
}
