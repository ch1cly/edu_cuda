
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 1000


__device__ bool isInCircle(double *x, double *y)
{
    return (*x) * (*x) + (*y) * (*y) <= 1;
}

__global__ void piCalc(double* p_a) {

    double x = double(blockIdx.x) / N;
    double y = double(threadIdx.x) / N;
    isInCircle(&x, &y) ? p_a[threadIdx.x * N + blockIdx.x] = 1 : p_a[threadIdx.x * N + blockIdx.x] = 0;
}

int main()
{
    double *a = new double[N*N];
    double* p_a;


    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&p_a, N * N * sizeof(double));
    
   
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    piCalc <<< N, N >>> (p_a);

    cudaStatus = cudaMemcpy(a, p_a, N * N * sizeof(double), cudaMemcpyDeviceToHost);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
        return 1;
    }
    
    double q = 0;
    for (int i = 0; i < N*N; ++i) {
        q += a[i];
    }
    printf("Pi is %f\n", q*4/N/N);
    
    delete a;
    cudaFree(p_a);
    return 0;
}
