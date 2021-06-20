#ifndef __CUBLAS_MM_KERNEL_H__
#define __CUBLAS_MM_KERNEL_H__

#include <iostream>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

void cublas_mm_wrapper(cublasHandle_t handle,
                       float *d_A, float *d_B, float *d_C,
                       int m, int k, int n) {
    cudaMemset(d_C, 0, m * n * sizeof(float));
    
    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, &alpha,
        d_B, n,
        d_A, k, &beta,
        d_C, n);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }

}

#endif // __CUBLAS_MM_KERNEL_H__
