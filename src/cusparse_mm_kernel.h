#ifndef __CUSPARSE_MM_KERNEL_H__
#define __CUSPARSE_MM_KERNEL_H__

#include <iostream>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cusparse_v2.h>

// https://stackoverflow.com/questions/29688627/sparse-matrix-matrix-multiplication-in-cuda-using-cusparse

// Sparse (CSR) * Dense matmul
// A * B = C

// (m x k) * (k * n) = (m x n)
// note: row_ind.len = lda + 1

void cusparse_mm_wrapper(double *h_A, int *h_A_ColIndices, int *h_A_RowIndices,
                         int nnzA, int h_A_rowptr_size,
                         double *h_B_dense, int h_B_rows, int h_B_cols,
                         double *h_C_dense)
{
    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseSafeCall(cusparseCreate(&handle));
    const int m = h_A_rowptr_size - 1;
    const int k = h_B_rows;
    const int n = h_B_cols;

    // Host side dense matrices
    //double *h_C_dense = (double *)malloc(m * n * sizeof(*h_C_dense));
    // Create device arrays and copy host arrays to them
    double *d_B_dense;
    gpuErrchk(cudaMalloc(&d_B_dense, k * n * sizeof(*d_B_dense)));
    double *d_C_dense;
    gpuErrchk(cudaMalloc(&d_C_dense, m * n * sizeof(*d_C_dense)));

    // copy B from host to device
    gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense, k * n * sizeof(*d_B_dense), cudaMemcpyHostToDevice));

    // Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    // Descriptor for sparse matrix B
    cusparseMatDescr_t descrB;
    cusparseSafeCall(cusparseCreateMatDescr(&descrB));
    cusparseSafeCall(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE));
    // Descriptor for sparse matrix C
    cusparseMatDescr_t descrC;
    cusparseSafeCall(cusparseCreateMatDescr(&descrC));
    cusparseSafeCall(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE));

    int nnzB = 0; //   Number of nonzero elements in dense matrix B
    // Device side number of nonzero elements per row of matrix B
    int *d_nnzPerVectorB;
    gpuErrchk(cudaMalloc(&d_nnzPerVectorB, k * sizeof(*d_nnzPerVectorB)));
    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, k, n, descrB, d_B_dense, k, d_nnzPerVectorB, &nnzB));
    // Host side number of nonzero elements per row of matrix B
    int *h_nnzPerVectorB = (int *)malloc(k * sizeof(*h_nnzPerVectorB));
    gpuErrchk(cudaMemcpy(h_nnzPerVectorB, d_nnzPerVectorB, k * sizeof(*h_nnzPerVectorB), cudaMemcpyDeviceToHost));

    // Device side sparse matrix A
    double *d_A;
    gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
    int *d_A_RowIndices;
    gpuErrchk(cudaMalloc(&d_A_RowIndices, (m + 1) * sizeof(*d_A_RowIndices)));
    int *d_A_ColIndices;
    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
    // Copy A from host to device
    cudaMemcpy(d_A, h_A, nnzA * sizeof(*d_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_RowIndices, h_A_RowIndices, (m + 1) * sizeof(*d_A_RowIndices), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_ColIndices, h_A_ColIndices, nnzA * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice);

    // Device side sparse matrix B
    double *d_B;
    gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));
    int *d_B_RowIndices;
    gpuErrchk(cudaMalloc(&d_B_RowIndices, (k + 1) * sizeof(*d_B_RowIndices)));
    int *d_B_ColIndices;
    gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));
    // Dense B to Sparse B
    cusparseSafeCall(cusparseDdense2csr(handle, k, n, descrB, d_B_dense, k, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

    // Move sparse B from device to host
    double *h_B = (double *)malloc(nnzB * sizeof(*h_B));
    int *h_B_ColIndices = (int *)malloc(nnzB * sizeof(*h_B_ColIndices));
    int *h_B_RowIndices = (int *)malloc((k + 1) * sizeof(*h_B_RowIndices));

    gpuErrchk(cudaMemcpy(h_B, d_B, nnzB * sizeof(*h_B), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_B_RowIndices, d_B_RowIndices, (k + 1) * sizeof(*h_B_RowIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_B_ColIndices, d_B_ColIndices, nnzB * sizeof(*h_B_ColIndices), cudaMemcpyDeviceToHost));

    // Device side sparse matrix C
    int *d_C_RowIndices;
    gpuErrchk(cudaMalloc(&d_C_RowIndices, (m + 1) * sizeof(*d_C_RowIndices)));

    // Performing the matrix - matrix multiplication
    int baseC, nnzC = 0;
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzC;

    cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA,
                                         d_A_RowIndices, d_A_ColIndices, descrB, nnzB, d_B_RowIndices, d_B_ColIndices, descrC, d_C_RowIndices,
                                         nnzTotalDevHostPtr));
    if (nnzTotalDevHostPtr != NULL)
        nnzC = *nnzTotalDevHostPtr;
    else
    {
        cudaMemcpy(&nnzC, d_C_RowIndices + m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }

    // device side sparse matrix C
    double *d_C;
    gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(double)));
    int *d_C_ColIndices;
    gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
    // host side sparse matrix c
    double *h_C = (double *)malloc(nnzC * sizeof(*h_C));
    int *h_C_ColIndices = (int *)malloc(nnzC * sizeof(*h_C_ColIndices));
    int *h_C_RowIndices = (int *)malloc((m + 1) * sizeof(*h_C_RowIndices));

    cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA,
                                      d_A, d_A_RowIndices, d_A_ColIndices, descrB, nnzB, d_B, d_B_RowIndices, d_B_ColIndices, descrC,
                                      d_C, d_C_RowIndices, d_C_ColIndices));

    cusparseSafeCall(cusparseDcsr2dense(handle, m, n, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, m));

    gpuErrchk(cudaMemcpy(h_C, d_C, nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (m + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices, nnzC * sizeof(*h_C_ColIndices), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    /*
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
            printf("%f \t", h_C_dense[i * m + j]);
        printf("\n");
    }
    */

    // free(h_C_dense);
    cudaFree(h_B_dense);
    cudaFree(h_C_dense);

    cudaFree(d_nnzPerVectorB);
    free(h_nnzPerVectorB);

    cudaFree(d_A);
    cudaFree(d_A_RowIndices);
    cudaFree(d_A_ColIndices);

    cudaFree(d_B);
    cudaFree(d_B_RowIndices);
    cudaFree(d_B_ColIndices);

    cudaFree(d_C);
    cudaFree(d_C_RowIndices);
    cudaFree(d_C_ColIndices);

    free(h_C);
    free(h_C_RowIndices);
    free(h_C_ColIndices);

    return;
}

void dense_to_csr(double *h_A_dense, const int Nrows, const int Ncols, double **h_A_val, int **h_A_colind, int **h_A_rowptr, int *nnzA)
{
    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseSafeCall(cusparseCreate(&handle));

    //create device array and copy host to it
    double *d_A_dense;
    gpuErrchk(cudaMalloc(&d_A_dense, Nrows * Ncols * sizeof(*d_A_dense)));
    gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

    // Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int nnz = 0;           //   Number of nonzero elements in dense matrix
    const int lda = Nrows; //   Leading dimension of dense matrix
    // Device side number of nonzero elements per row
    int *d_nnzPerVector;
    gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
    // Host side number of nonzero elements per row
    int *h_nnzPerVector = (int *)malloc(Nrows * sizeof(*h_nnzPerVector));
    gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost));

    // Device side dense matrix
    double *d_A;
    gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
    int *d_A_RowIndices;
    gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
    int *d_A_ColIndices;
    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

    cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

    // Host side dense matrix
    double *h_A = (double *)malloc(nnz * sizeof(*h_A));
    int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
    int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
    gpuErrchk(cudaMemcpy(h_A, d_A, nnz * sizeof(*h_A), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

    *h_A_val = h_A;
    *h_A_rowptr = h_A_RowIndices;
    *h_A_colind = h_A_ColIndices;
    *nnzA = nnz;

    gpuErrchk(cudaFree(d_nnzPerVector));
    free(h_nnzPerVector);

    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_A_RowIndices));
    gpuErrchk(cudaFree(d_A_ColIndices));

    return;
}

#endif // __CUSPARSE_MM_KERNEL_H__
