#ifndef __CUBLAS_BMM_KERNEL_H__
#define __CUBLAS_BMM_KERNEL_H__

#include <iostream>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

void cublas_bmm_wrapper(cublasHandle_t handle,
                       float *h_A, int h_A_rows, int h_A_cols,
                       float *h_B, int h_B_rows, int h_B_cols,
                       float *h_C, int batchCount)
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

    // cublasHandle_t handle;
    // cublasStatus_t status = cublasCreate(&handle);
    // if (status != CUBLAS_STATUS_SUCCESS)
    // {
    //     std::cerr << "cuBLAS initialization error.";
    // }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A_rows * h_A_cols * sizeof(float));
    cudaMalloc(&d_B, h_B_rows * h_B_cols * sizeof(float));
    cudaMalloc(&d_C, h_C_rows * h_C_cols * sizeof(float));

    cudaMemcpy(d_A, h_A, h_A_rows * h_A_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, h_B_rows * h_B_cols * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t status = cublasSgemmBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha,
        d_A, m,
        d_B, k, &beta,
        d_C, m, batchCount);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }

    cudaMemcpy(h_C, d_C, h_C_rows * h_C_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // status = cublasDestroy(handle);
    // if (status != CUBLAS_STATUS_SUCCESS)
    // {
    //     std::cerr << "Shutdown error!";
    // }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#endif // __CUBLAS_BMM_KERNEL_H__
