#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>


#define N 100


__device__ double f(double* x) {
    return *x * *x;
}


__global__ void integralByMiddleQuad(double* c, const double* h, const double* a)
{
    __shared__ double csh[32];

    double x = *a + (double)(blockIdx.x * 32 + threadIdx.x) * *h + *h/2;


    csh[threadIdx.x] = f(&x);

    __syncthreads();

    if (blockIdx.x * 32 + threadIdx.x < N ) {
        
        c[blockIdx.x * 32 + threadIdx.x] = csh[threadIdx.x];

    }

}



__global__ void integralByTrapez(double* c, const double* h, const double* a)
{
    __shared__ double csh[32];

    double x = *a + (double)(blockIdx.x * 32 + threadIdx.x) * *h;

    //printf("x = %f\n", x);

    csh[threadIdx.x] = f(&x);

    __syncthreads();

    if (blockIdx.x * 32 + threadIdx.x < N + 1) {
        //printf("NUM %d = threadash[%d] = %f  \n", blockIdx.x * 32 + threadIdx.x, threadIdx.x, csh[threadIdx.x]);
        c[blockIdx.x * 32 + threadIdx.x] = csh[threadIdx.x];
        // printf("c = %f\n", c[blockIdx.x * 32 + threadIdx.x]);
    }

}


int main()
{

    double a = 3;
    double b = 6;
    double h = (b - a) / N;
    double c_midd[N];
    double c_trap[N+1];
    double ansv = 63;



    double* dev_a = 0;
    double* dev_h = 0;
    double* dev_c_midd = 0;
    double* dev_c_trap = 0;

    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c_midd, (N) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&dev_c_trap, (N+1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 111;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 2;
    }

    cudaStatus = cudaMalloc((void**)&dev_h, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 3;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, &a, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 4;
    }

    cudaStatus = cudaMemcpy(dev_h, &h, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 5;
    }
    cudaStatus = cudaMemcpy(dev_c_midd, c_midd, (N) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 551;
    }

    cudaStatus = cudaMemcpy(dev_c_trap, c_trap, (N+1) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 552;
    }

    
    {
        // Launch a kernel on the GPU with one thread for each element.
        int blockSize;

        blockSize = N / 32 + 1;
        // Launch a kernel on the GPU with one thread for each element.
        integralByMiddleQuad << <blockSize, 32 >> > (dev_c_midd, dev_h, dev_a);




        cudaStatus = cudaMemcpy(c_midd, dev_c_midd, (N) * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return 6;
        }

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return 7;
        }

        double sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (c_midd[i]);
        }
        sum /= N / (b - a);
        printf("integralByMiddleQuad = %f\n error = %e\n", sum, abs(ansv-sum));

    }


    {
        // Launch a kernel on the GPU with one thread for each element.
        int blockSize;

        blockSize = N / 32 + 1;
        // Launch a kernel on the GPU with one thread for each element.
        integralByTrapez << <blockSize, 32 >> > (dev_c_trap, dev_h, dev_a);




        cudaStatus = cudaMemcpy(c_trap, dev_c_trap, (N+1) * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return 6;
        }

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return 7;
        }

        double sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (c_trap[i]+ c_trap[i+1])/2;
        }
        sum /= N / (b - a);
        printf("integralByTrap = %f\n error = %e\n", sum, abs(ansv - sum));
    }


    
        double sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += (c_trap[i] + 4*c_midd[i] + c_trap[i + 1]) / 6;
        }
        sum /= N / (b - a);
        printf("integralBySimpson = %f\n error = %e\n", sum, abs(ansv - sum));


    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 8;
    }



    cudaFree(dev_c_trap);
    cudaFree(dev_c_midd);
    cudaFree(dev_a);
    cudaFree(dev_h);



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
