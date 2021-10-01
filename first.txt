%%cu
#include <stdio.h>
#include <stdlib.h>
__global__ void add(int *a, int *b, int *c) {
*c = *a + *b;
}
int main() {
int a, b, c;
// host copies of variables a, b & c
int *d_a, *d_b, *d_c;
// device copies of variables a, b & c
int size = sizeof(int);
// Allocate space for device copies of a, b, c
cudaMalloc((void **)&d_a, size);
cudaMalloc((void **)&d_b, size);
cudaMalloc((void **)&d_c, size);
// Setup input values  
c = 0;
a = 3;
b = 5;
// Copy inputs to device
cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
// Launch add() kernel on GPU
add<<<1,1>>>(d_a, d_b, d_c);
// Copy result back to host
cudaError err = cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
  if(err!=cudaSuccess) {
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
  }
printf("result is %d\n",c);
// Cleanup
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
return 0;
}




%%cu
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() {
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, 0);
printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
printf("Total global memory : %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024); 
printf("Clock rate: %d\n", deviceProp.clockRate); 
printf("Peak memory clock frequency in kilohertz: %d\n", deviceProp.memoryClockRate); 


printf("Global memory bus width in bits: %d\n", deviceProp.memoryBusWidth);

printf("Conclusion: %s is pretty good for me obv.\n", deviceProp.name); 
return 0;
}
