#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void add(int* a, int* b, int* c) {
    *c = *a + *b;
}

int main() {
    int a, b, c;
    
    // host copies of variables a, b & c
    int* d_a, * d_b, * d_c;
    // device copies of variables a, b & c
    int size = sizeof(int);
    
    // Allocate space for device copies of a, b, c
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // Setup input values  
    c = 0;
    //a = 3;
    //b = 5;
    std::cout << "Input first figure" << std::endl;
    std::cin >> a;
    std::cout << "Input second figure" << std::endl;
    std::cin >> b;
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    
    // Launch add() kernel on GPU
    add<<<1,1>>> (d_a, d_b, d_c);
    
    // Copy result back to host
    cudaError err = cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }
    printf("\nresult is %d\n", c);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
