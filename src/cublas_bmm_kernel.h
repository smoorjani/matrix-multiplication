#ifndef __CUBLAS_BMM_KERNEL_H__
#define __CUBLAS_BMM_KERNEL_H__

// https://stackoverflow.com/questions/23743384/how-performing-multiple-matrix-multiplications-in-cuda/23743838#23743838

#include <iostream>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printArrayS(float *ptr, int rows, int cols, char mode, char *name)
{
    printf("%s\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (mode == 'B') /* Normal mode */
            {
                if (ptr[i * cols + j] >= 0)
                    printf(" %3.6f ", ptr[i * cols + j]);
                else
                    printf("%3.6f ", ptr[i * cols + j]);
            }
            else /* Transpose mode */
            {
                if (ptr[j * rows + i] >= 0)
                    printf("%3.6f ", ptr[j * rows + i]);
                else
                    printf("%3.6f ", ptr[j * rows + i]);
            }
        }
        printf("\n");
    }
}


void cublas_bmm_wrapper(cublasHandle_t handle,
               float **A, float **B, float **C,
               size_t a_rows, size_t b_cols, size_t b_rows,
               size_t batch_dim) {

    float *d_A[batch_dim];
    float *d_B[batch_dim];
    float *d_C[batch_dim];

    size_t c_size = sizeof(float) * a_rows * b_cols;
    size_t a_size = sizeof(float) * a_rows * b_rows;
    size_t b_size = sizeof(float) * b_rows * b_cols;

    const float **d_A_arr, **d_B_arr;
    float **d_C_arr;

    for (int i = 0; i < batch_dim; i++) {
        gpuErrchk(cudaMalloc((void **)&d_A[i], a_size));
        gpuErrchk(cudaMalloc((void **)&d_B[i], b_size));
        gpuErrchk(cudaMalloc((void **)&devP[i], c_size));
    }

    gpuErrchk(cudaMalloc((void **)&d_A_arr, batch_dim * sizeof(float *)));
    gpuErrchk(cudaMalloc((void **)&d_B_arr, batch_dim * sizeof(float *)));
    gpuErrchk(cudaMalloc((void **)&d_C_arr, batch_dim * sizeof(float *)));

    for (int i = 0; i < batch_dim; i++) {
        gpuErrchk(cudaMemcpy(d_A[i], A[i], a_size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_B[i], B[i], b_size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(devP[i], C[i], c_size, cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaMemcpy(d_A_arr, d_A, batch_dim * sizeof(float *), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B_arr, d_B, batch_dim * sizeof(float *), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C_arr, devP, batch_dim * sizeof(float *), cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cublas_result = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N
                                       , b_cols, a_rows, b_rows
                                       , &alpha, d_B_arr, b_cols, d_A_arr, b_rows
                                       , &beta, d_C_arr, b_cols
                                       , batch_dim);
    assert(cublas_result == CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i < batch_dim; i++)
    {
        gpuErrchk(cudaMemcpy(C[i], devP[i], c_size, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_A[i]));
        gpuErrchk(cudaFree(d_B[i]));
        gpuErrchk(cudaFree(devP[i]));
    }
    gpuErrchk(cudaFree(d_A_arr));
    gpuErrchk(cudaFree(d_B_arr));
    gpuErrchk(cudaFree(d_C_arr));
}

#endif // __CUBLAS_BMM_KERNEL_H__
