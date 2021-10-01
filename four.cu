
%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 1000


__global__ void integral(double **a)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    if (x * x + y * y > 1){
        a[x][y] = 0;
    } 
    else{
        a[x][y] = 1;
    }
}

int main()
{
    double a[N][N] = { 0 };
    double** p_a;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&p_a, N * sizeof(*double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc p_a failed!");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        cudaStatus = cudaMalloc((void**)&(*p_a+i), N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc inside p_a failed!");
        return 1;
      }
    }
   
    

    integral <<<N, N >>> (p_a);

    cudaStatus = cudaMemcpy(a, p_a, N * sizeof(*double), cudaMemcpyDeviceToHost);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
        return 1;
    }
 
  for (int i = 0; i < N; ++i) {
      cudaStatus = cudaMemcpy((*a+i, *p_a+i, N * sizeof(double), cudaMemcpyDeviceToHost);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost inside failed!");
        return 1;
    }
  }
    
    double q = 0;
    for(int j = 0; j < N; ++j){
    for (int i = 0; i < N; ++i) {
        q += a[j][i];
    }
    }
    printf("Pi is %f\n", q*4);

     for (int i = 0; i < N; ++i) {
       cudaFree(*p_a+i);
     }

    cudaFree(p_a);
    return 0;
}