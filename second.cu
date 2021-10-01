%%cu
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() {
cudaDeviceProp deviceProp; cudaGetDeviceProperties(&deviceProp, 0);
printf("Device name : %s\n", deviceProp.name); 
printf("Total global memory : %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024); 
printf("Shared memory per block : %d\n", deviceProp.sharedMemPerBlock);
printf("Registers per block : %d\n", deviceProp.regsPerBlock);
printf("Warp size : %d\n", deviceProp.warpSize); 
printf("Memory pitch : %d\n", deviceProp.memPitch); 
printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
printf("Clock rate: %d\n", deviceProp.clockRate);
printf("Size of L2 cache in bytes  %d\n", deviceProp.l2CacheSize );
printf("Global memory bus width in bits: %d\n", deviceProp.memoryBusWidth);
printf("Max threads per block : %d\n", deviceProp.maxThreadsPerBlock);
printf("Max threads dimensions : x = %d, y = %d, z = %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
printf("description is same as in https://www.techpowerup.com/gpu-specs/tesla-k80.c2616")
}

