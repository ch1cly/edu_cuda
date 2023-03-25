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
