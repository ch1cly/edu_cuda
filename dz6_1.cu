#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE double
// функция перемножения матриц
__global__ void matrixMult(const BASE_TYPE* A, const
	BASE_TYPE* B, BASE_TYPE* C, int Acols, int Bcols)
{
	int i0 = Acols * (blockDim.y * blockIdx.y +
		threadIdx.y);
	int j0 = blockDim.x * blockIdx.x + threadIdx.x;
	BASE_TYPE sum = 0;

		for (int k = 0; k < Acols; k++)
			sum += A[i0 + k] * B[k * Bcols + j0];

	int ind = Bcols * (blockDim.y * blockIdx.y +
		threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	C[ind] = sum;
}
int toMultiple(int a, int b) {
	int mod = a % b;
	if (mod != 0) {
		mod = b - mod;
		return a + mod;
	}
	return a;
}
int main()
{
	//start, stop - for Kernel time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// количество строк и столбцов матрицы
	int Arows = 100;
	int Acols = 200;
	int Brows = Acols;
	int Bcols = 150;
	Arows = toMultiple(Arows, BLOCK_SIZE);
	printf("Arows = %d\n", Arows);
	Acols = toMultiple(Acols, BLOCK_SIZE);
	printf("Acols = %d\n", Acols);
	Brows = toMultiple(Brows, BLOCK_SIZE);
	printf("Brows = %d\n", Brows);
	Bcols = toMultiple(Bcols, BLOCK_SIZE);
	printf("Bcols = %d\n", Bcols);
	size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
	size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
	size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);
	BASE_TYPE* h_A = (BASE_TYPE*)malloc(Asize);
	BASE_TYPE* h_B = (BASE_TYPE*)malloc(Bsize);
	BASE_TYPE* h_AB = (BASE_TYPE*)malloc(Csize);
	BASE_TYPE* h_BA = (BASE_TYPE*)malloc(Csize);

	for (int i = 0; i < Arows * Acols; ++i) {
		h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	for (int i = 0; i < Brows * Bcols; ++i) {
		h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	BASE_TYPE* d_A = NULL;
	cudaMalloc((void**)&d_A, Asize);
	BASE_TYPE* d_B = NULL;
	cudaMalloc((void**)&d_B, Bsize);
	BASE_TYPE* d_AB = NULL;
	cudaMalloc((void**)&d_AB, Csize);

	BASE_TYPE* d_BA = NULL;
	cudaMalloc((void**)&d_BA, Csize);

	cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(Bcols / BLOCK_SIZE, Arows /
		BLOCK_SIZE);


	cudaEventRecord(start, 0);
	matrixMult << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_AB, Acols, Bcols);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %.2f milliseconds\n",
		KernelTime);
	cudaMemcpy(h_AB, d_AB, Csize, cudaMemcpyDeviceToHost);


	cudaEventRecord(start, 0);
	matrixMult << <blocksPerGrid, threadsPerBlock >> > (d_B, d_A, d_BA, Acols, Bcols);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime1;
	cudaEventElapsedTime(&KernelTime1, start, stop);
	printf("KernelTime: %.2f milliseconds\n",
		KernelTime1);
	cudaMemcpy(h_BA, d_BA, Csize, cudaMemcpyDeviceToHost);

	bool b = true;
	for (int i = 0; i < Csize; ++i) {
		if (abs(h_AB[i] - h_BA[i]) > 1e-9) {
			b = false;
			break;
		}
	}
	if (!b) {
		printf("not comm\n");
	}
	else {
		printf("comm\n");
	}
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_AB);
	cudaFree(d_BA);
	free(h_A);
	free(h_B);
	free(h_AB);
	free(h_BA);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}