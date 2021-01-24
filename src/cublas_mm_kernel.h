#ifndef __CUBLAS_MM_KERNEL_H__
#define __CUBLAS_MM_KERNEL_H__

#include <iostream>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

void cublas_mm_wrapper(float *h_A, int h_A_rows, int h_A_cols,
                       float *h_B, int h_B_rows, int h_B_cols,
                       float *h_C)
{
    const int m = h_A_rows;
    const int k = h_A_cols;
    const int n = h_B_cols;

    int h_C_rows = h_A_rows;
    int h_C_cols = h_B_cols;

    if (h_A_cols != h_B_rows)
    {
        std::cerr << "Invalid dimensions.";
    }

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS initialization error.";
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A_rows * h_A_cols * sizeof(float));
    cudaMalloc(&d_B, h_B_rows * h_B_cols * sizeof(float));
    cudaMalloc(&d_C, h_C_rows * h_C_cols * sizeof(float));

    cudaMemcpy(d_A, h_A, h_A_rows * h_A_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, h_B_rows * h_B_cols * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0;
    float beta = 0.0;
    status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha,
        d_A, m,
        d_B, k, &beta,
        d_C, m);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }

    cudaMemcpy(h_C, d_C, h_C_rows * h_C_cols * sizeof(float), cudaMemcpyDeviceToHost);

    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Shutdown error!";
    }

    /*
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
            printf("%f \t", h_C[i * m + j]);
        printf("\n");
    }
    */

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#endif // __CUBLAS_MM_KERNEL_H__
