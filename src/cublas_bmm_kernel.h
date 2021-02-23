#ifndef __CUBLAS_BMM_KERNEL_H__
#define __CUBLAS_BMM_KERNEL_H__

#include <iostream>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define ROWM 4
#define COLM 3
#define COLN 5

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

void cublas_bmm_wrapper(cublasHandle_t handle, 
    float **A, float **B, float **C,
    size_t a_rows, size_t b_rows, size_t b_cols, 
    size_t batch_size) {

    float *d_A[batch_size];
    float *d_B[batch_size];
    float *d_C[batch_size];
    size_t numC = sizeof(float) *a_rows*b_cols;
    size_t numA = sizeof(float) *a_rows*b_rows;
    size_t numB = sizeof(float) *b_rows*b_cols;
    const float **d_A_array, **d_B_array;
    float **d_C_array;

    for (int i = 0 ; i < batch_size; i ++) {
        cudaMalloc((void**)&d_A[ i ], numA );
        cudaMalloc((void**)&d_B[ i ], numB );
        cudaMalloc((void**)&d_C[ i ], numC );
    }

    cudaMalloc((void**)&d_A_array, batch_size*sizeof(float *));
    cudaMalloc((void**)&d_B_array, batch_size*sizeof(float *));
    cudaMalloc((void**)&d_C_array, batch_size*sizeof(float *));

    cudaCheckErrors("cudaMalloc fail");
    for (int i = 0 ; i < batch_size; i ++ ) {
        cudaMemcpy(d_A[i], A[i], numA , cudaMemcpyHostToDevice);
        cudaMemcpy(d_B[i], B[i], numB , cudaMemcpyHostToDevice);
        cudaMemcpy(d_C[i], C[i], numC , cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_A_array, d_A, batch_size*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, d_B, batch_size*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array, d_C, batch_size*sizeof(float *), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    // change to cublasDgemmBatched for double
    cublasStatus_t status = cublasSgemmBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        a_rows, b_cols, b_rows, &alpha,
        d_A_array, a_rows,
        d_B_array, b_rows, &beta,
        d_C_array, a_rows, batch_size);

    assert(status == CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i < batch_size; i ++ ) {
        cudaMemcpy(C[i], d_C[i], numC, cudaMemcpyDeviceToHost);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }

    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    cudaCheckErrors("cudaMemcpy D2H fail");
}

int main(){

  float h_M1[ROWM][COLM], h_M2[ROWM][COLM];
  float h_N1[COLM][COLN], h_N2[COLM][COLN];
  float h_P1[ROWM][COLN], h_P2[ROWM][COLN];
  float *h_Marray[2], *h_Narray[2], *h_Parray[2];
  for (int i = 0; i < ROWM; i++)
    for (int j = 0; j < COLM; j++){
      h_M1[i][j] = 1.0f; h_M2[i][j] = 2.0f;}
  for (int i = 0; i < COLM; i++)
    for (int j = 0; j < COLN; j++){
      h_N1[i][j] = 1.0f; h_N2[i][j] = 1.0f;}
  for (int i = 0; i < ROWM; i++)
    for (int j = 0; j < COLN; j++){
      h_P1[i][j] = 0.0f; h_P2[i][j] = 0.0f;}

  h_Marray[0] = &(h_M1[0][0]);
  h_Marray[1] = &(h_M2[0][0]);
  h_Narray[0] = &(h_N1[0][0]);
  h_Narray[1] = &(h_N2[0][0]);
  h_Parray[0] = &(h_P1[0][0]);
  h_Parray[1] = &(h_P2[0][0]);

  GPU_Multi(h_Marray, h_Narray, h_Parray, ROWM, COLN, COLM, 2, 1.0f, 0.0f);
  for (int i = 0; i < ROWM; i++)
    for (int j = 0; j < COLN; j++){
      if (h_P1[i][j] != COLM*1.0f)
      {
        printf("h_P1 mismatch at %d,%d was: %f should be: %f\n"
          , i, j, h_P1[i][j], COLM*1.0f); return 1;
      }
      if (h_P2[i][j] != COLM*2.0f)
      {
        printf("h_P2 mismatch at %d,%d was: %f should be: %f\n"
          , i, j, h_P2[i][j], COLM*2.0f); return 1;
      }
    }
  printf("Success!\n");
  return 0;
}

// void cublas_bmm_wrapper(cublasHandle_t handle,
//                        float *h_A, int h_A_rows, int h_A_cols,
//                        float *h_B, int h_B_rows, int h_B_cols,
//                        float *h_C, int batchCount)
// {
//     const int m = h_A_rows;
//     const int k = h_A_cols;
//     const int n = h_B_cols;

//     int h_C_rows = h_A_rows;
//     int h_C_cols = h_B_cols;

//     if (h_A_cols != h_B_rows)
//     {
//         std::cerr << "Invalid dimensions.";
//     }

//     // cublasHandle_t handle;
//     // cublasStatus_t status = cublasCreate(&handle);
//     // if (status != CUBLAS_STATUS_SUCCESS)
//     // {
//     //     std::cerr << "cuBLAS initialization error.";
//     // }

//     const float *d_A, *d_B;
//     float *d_C;
//     cudaMalloc(&d_A, h_A_rows * h_A_cols * sizeof(float));
//     cudaMalloc(&d_B, h_B_rows * h_B_cols * sizeof(float));
//     cudaMalloc(&d_C, h_C_rows * h_C_cols * sizeof(float));

//     cudaMemcpy(d_A, h_A, h_A_rows * h_A_cols * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, h_B_rows * h_B_cols * sizeof(float), cudaMemcpyHostToDevice);

//     float alpha = 1.0;
//     float beta = 0.0;
//     cublasStatus_t status = cublasSgemmBatched(
//         handle, CUBLAS_OP_N, CUBLAS_OP_N,
//         m, n, k, &alpha,
//         d_A, m,
//         d_B, k, &beta,
//         d_C, m, batchCount);

//     if (status != CUBLAS_STATUS_SUCCESS)
//     {
//         std::cerr << "Kernel execution error.";
//     }

//     cudaMemcpy(h_C, d_C, h_C_rows * h_C_cols * sizeof(float), cudaMemcpyDeviceToHost);

//     // status = cublasDestroy(handle);
//     // if (status != CUBLAS_STATUS_SUCCESS)
//     // {
//     //     std::cerr << "Shutdown error!";
//     // }

//     cudaFree((float*) d_A);
//     cudaFree((float*) d_B);
//     cudaFree(d_C);
// }

#endif // __CUBLAS_BMM_KERNEL_H__
