
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 1000


__device__ bool isInCircle(double *x, double *y)
{
    return (*x) * (*x) + (*y) * (*y) <= 1;
}

__global__ void piCalc(unsigned int* p_a) {

    double x = double(blockIdx.x) / N;
    double y = double(threadIdx.x) / N;
    isInCircle(&x, &y) ? atomicAdd(p_a, 1) : 0;
}

int main()
{
    unsigned int a = 0;
    unsigned int* p_a;


    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&p_a, sizeof(unsigned int));
    
    cudaStatus = cudaMemcpy(p_a, &a, sizeof(unsigned int), cudaMemcpyHostToDevice);
   
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    piCalc <<< N, N >>> (p_a);

    cudaStatus = cudaMemcpy(&a, p_a, sizeof(unsigned int), cudaMemcpyDeviceToHost);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
        return 1;
    }
    
    double q = 0;
    for (int i = 0; i < N*N; ++i) {
        q  = double(a);
    }
    printf("Pi is %f\n", q*4/N/N);
    
    cudaFree(p_a);
    return 0;
}