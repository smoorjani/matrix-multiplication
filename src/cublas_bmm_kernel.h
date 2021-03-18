#ifndef __CUBLAS_BMM_KERNEL_H__
#define __CUBLAS_BMM_KERNEL_H__

// https://stackoverflow.com/questions/23743384/how-performing-multiple-matrix-multiplications-in-cuda/23743838#23743838

#include <iostream>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static inline void sgemmbatched(
        const cublasHandle_t handle,
        const cublasOperation_t &trans_a,
        const cublasOperation_t &trans_b,
        const int batch_size,

        const float **a, const int a_rows, const int a_cols, const int lda,
        const float **b, const int b_rows, const int b_cols, const int ldb,
        float **c, const int c_rows, const int c_cols, const int ldc,
        const float *alpha,
        const float *beta ) {
        int m, n, k;
        if (trans_a == cublasOperation_t::CUBLAS_OP_N)
        {
                m = a_rows;
                k = a_cols;
        }
        else
        {
                m = a_cols;
                k = a_rows;
        }
        if (trans_b == cublasOperation_t::CUBLAS_OP_N)
        {
                assert(k == b_rows);
                k = b_rows;
                n = b_cols;
        }
        else
        {
                assert(k == b_cols);
                k = b_cols;
                n = b_rows;
        }
        assert(m == c_rows);
        assert(n = c_cols);


        cublasStatus_t status = cublasSgemmBatched(handle,
                trans_a, trans_b,
                m, n, k,
                alpha,
                a, lda,
                b, ldb,
                beta,
                c, ldc,
                batch_size
        );

        assert(status == CUBLAS_STATUS_SUCCESS);
}

void cublas_bmm_wrapper(cublasHandle_t handle,
    float **A, float **B, float **C,
    size_t a_rows, size_t b_rows, size_t b_cols,
    size_t batch_size) {

    cudaError_t cudaStatus;
    cublasStatus_t status;

    float **d_A = (float**)malloc(batch_size * sizeof(float*));
    float **d_B = (float**)malloc(batch_size * sizeof(float*));
    float **d_C = (float**)malloc(batch_size * sizeof(float*));

    size_t size_A = a_rows * b_rows * sizeof(float);
    size_t size_B = b_rows * b_cols * sizeof(float);
    size_t size_C = a_rows * b_cols * sizeof(float);

    for (int i = 0 ; i < batch_size; i ++) {
        gpuErrchk(cudaMalloc(&d_A[i], size_A));
        gpuErrchk(cudaMemcpy(d_A[i], A[i], size_A, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc(&d_B[i], size_B));
        gpuErrchk(cudaMemcpy(d_B[i], B[i], size_B, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc(&d_C[i], size_C));
        gpuErrchk(cudaMemcpy(d_C[i], C[i], size_C, cudaMemcpyHostToDevice));
    }
    cudaCheckErrors("inner cudaMalloc/cudaMemcpy fail");

    const float **d_A_arr = 0, **d_B_arr = 0;
    float **d_C_arr = 0;
    gpuErrchk(cudaMalloc(&d_A_arr, batch_size * sizeof(float*)));
    gpuErrchk(cudaMalloc(&d_B_arr, batch_size * sizeof(float*)));
    gpuErrchk(cudaMalloc(&d_C_arr, batch_size * sizeof(float*)));
    cudaCheckErrors("outer cudaMalloc fail");

    gpuErrchk(cudaMemcpy(d_A_arr, d_A, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B_arr, d_B, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C_arr, d_C, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheckErrors("outer cudaMemcpy H2D fail");

    const float alpha = 1.0f, beta = 0.0f;

    sgemmbatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        batch_size,
        d_A_arr, a_rows, b_rows, a_rows,
        d_B_arr, b_rows, b_cols, b_rows,
        d_C_arr, a_rows, b_cols, a_rows,
        &alpha, &beta
    );

    for (int i = 0; i < batch_size; i++) {
        gpuErrchk(cudaMemcpy(C[i], d_C[i], size_C, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_A[i]));
        gpuErrchk(cudaFree(d_B[i]));
        gpuErrchk(cudaFree(d_C[i]));
    }
    cudaCheckErrors("cudaMemcpy D2H fail");

    gpuErrchk(cudaFree(d_A_arr));
    gpuErrchk(cudaFree(d_B_arr));
    gpuErrchk(cudaFree(d_C_arr));

    free(d_A);
    free(d_B);
    free(d_C);
    cudaCheckErrors("free cuda memory fail");
 }

#endif // __CUBLAS_BMM_KERNEL_H__
