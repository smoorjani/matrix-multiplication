#ifndef __CUBLAS_MM_KERNEL_H__
#define __CUBLAS_MM_KERNEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

void cublas_mm_wrapper(double *h_A, int h_A_rows, int h_A_cols, double *h_B, int h_B_rows, int h_B_cols) {
    const int m = h_A_rows;
    const int k = h_A_cols;
    const int n = h_B_cols;

    int h_C_rows = h_A_rows;
    int h_C_cols = h_B_cols;
    double *h_C = (double *)malloc(h_C_rows * h_C_cols * sizeof(*h_C));

    if (h_A_cols != h_B_rows) {
		std::cerr << "Invalid dimensions.";
    }
    
    cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cuBLAS initialization error.";
    }
    
    double *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, h_A_rows * h_A_cols * sizeof(double));
	cudaMalloc(&d_B, h_B_rows * h_B_cols * sizeof(double));
	cudaMalloc(&d_C, h_C_rows * h_C_cols * sizeof(double));

	cudaMemcpy(d_A, h_A, h_A_rows * h_A_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, h_B_rows * h_B_cols * sizeof(double), cudaMemcpyHostToDevice);

    float alpha = 1.0;
	float beta = 0.0;
	status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, alpha,
        d_A, m, 
        d_B, n, beta,
        d_C, m
    );

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Kernel execution error.";
	}

	cudaMemcpy(h_C, d_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost);

	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Shutdown error!";
    }
    
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
            printf("%f \t", h_C[i * m + j]);
        printf("\n");
    }

    cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main()
{
    // (3 x 4) x (4 x 2) = 3 x 2
    int h_A_rows = 3;
    int h_A_cols = 4;
    double *h_A = (double *)malloc(h_A_rows * h_A_cols * sizeof(*h_A));

    // Column-major ordering
    h_A[0] = 1.0f;
    h_A[1] = 0.0f;
    h_A[2] = 5.0f;
    h_A[3] = 0.0f;
    h_A[4] = 4.0f;
    h_A[5] = 2.0f;
    h_A[6] = 5.0f;
    h_A[7] = 3.0f;
    h_A[8] = 7.0f;
    h_A[9] = 1.0f;
    h_A[10] = 2.0f;
    h_A[11] = 9.0f;


    int h_B_rows = 4;
    int h_B_cols = 2;
    double *h_B = (double *)malloc(h_B_rows * h_B_cols * sizeof(*h_B));

    // Column-major ordering
    h_B[0] = 1.0f;
    h_B[1] = 0.0f;
    h_B[2] = 5.0f;
    h_B[3] = 0.0f;
    h_B[4] = 4.0f;
    h_B[5] = 2.0f;
    h_B[6] = 0.0f;
    h_B[7] = 0.0f;

    cublas_mm_wrapper(h_A, h_A_rows, h_A_cols, h_B, h_B_rows, h_B_cols);

    free(h_A);
    free(h_B);

    // expected
    // 26  4
    // 15  8
    // 40 24
}

#endif // __CUSPARSE_MM_KERNEL_H__