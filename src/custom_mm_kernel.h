#ifndef __CUSTOM_MM_KERNEL_H__
#define __CUSTOM_MM_KERNEL_H__

/*
MM reproduced from:
https://gist.github.com/peterwittek/6303527
*/

#include <iostream>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <curand.h>

// Print matrix A
void print_matrix(const float *A, int A_rows, int A_cols)
{
	for (int i = 0; i < A_rows; ++i)
	{
		for (int j = 0; j < A_cols; ++j)
		{
			std::cout << A[j * A_rows + i] << " ";
		}
		std::cout << std::endl;
	}
}

void mmul_wrapper(const float *A, const float *B, float *C,
				  const int A_rows, const int A_cols,
				  const int B_rows, const int B_cols,
				  const int C_rows, const int C_cols)
{

	if (A_cols != B_rows || C_rows != A_rows || C_cols != B_cols)
	{
		std::cerr << "Invalid dimensions.";
	}

	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, A_rows * A_cols * sizeof(float));
	cudaMalloc(&d_B, B_rows * B_cols * sizeof(float));
	cudaMalloc(&d_C, C_rows * C_cols * sizeof(float));

	cudaMemcpy(d_A, A, A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(float), cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "cuBLAS initialization error.";
	}

	float alpha = 1.0;
	float beta = 0.0;
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
						 B_cols, A_rows, A_cols,
						 &alpha, d_B, B_cols,
						 d_A, A_cols,
						 &beta, d_C, B_cols);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "Kernel execution error.";
	}

	cudaMemcpy(C, d_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost);

	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "Shutdown error!";
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

#endif // __CUSTOM_MM_KERNEL_H__