#ifndef __BASELINE_MM_H_
#define __BASELINE_MM_H_

#include <iostream>
#include <array>
#include <torch/extension.h>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <sys/time.h>
typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp () {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

#define NUM_THREADS (64)

__global__ void dummyKernel()
{
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    printf("%d\n", tid);
}

void dummy_kernel_launch() {
    dim3 threads_per_block(NUM_THREADS);
    dim3 blocks_per_grid(NUM_THREADS);
    dummyKernel<<<blocks_per_grid, threads_per_block>>>();
    checkCudaStatus(cudaDeviceSynchronize());
}

__global__ void check_equal(float *d_Arr, float *h_Arr, size_t rows, size_t cols)
{
    // checks two values are equal between two arrays
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= rows * cols) {
        return;
    }

    if (d_Arr[tid] == h_Arr[tid]) {
        printf("Equal %d\n", tid);
    } else {
        printf("Not Equal %d\n", tid);
    }
}

void cublas_mm_wrapper(cublasHandle_t handle,
                       torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
                       int a_rows, int a_cols, int b_rows, int b_cols,
                       bool transa, bool transb) {

    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();

    cublasOperation_t trans_a = (!transb) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_b = (!transa) ? CUBLAS_OP_N : CUBLAS_OP_T;

    int m = b_cols;
    int n = a_rows;
    int k = b_rows;
    int ldb = b_cols;
    int lda = b_rows;
    int ldc = b_cols;

    if (transb && transa) {
        m = b_rows;
        n = a_cols;
        k = b_cols;
        lda = n;
        ldb = k;
	    ldc = m;
    } else if (transa) {
        m = b_cols;
        n = a_cols;
        k = b_rows;
        lda = n;
        ldb = m;
	    ldc = m;
    } else if (transb) {
        m = b_rows;
        n = a_rows;
        k = b_cols;
        lda = k;
        ldb = k;
	    ldc = m;
    }

    float alpha = 1.0;
    float beta = 0.0;
    checkCublasStatus(cublasSgemm(
        handle, trans_a, trans_b,
        m, n, k, &alpha,
        d_B_arr, ldb,
        d_A_arr, lda, &beta,
        d_C_arr, ldc));
}


void cublas_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
               size_t a_rows, size_t a_cols, size_t b_cols, size_t b_rows,
               size_t batch_dim, bool transa, bool transb) {
    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();

    const float alpha = 1.0f, beta = 0.0f;

    cublasOperation_t trans_a = (!transb) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_b = (!transa) ? CUBLAS_OP_N : CUBLAS_OP_T;

    int m = b_cols;
    int n = a_rows;
    int k = b_rows;

    int ldb = b_cols;
    int lda = b_rows;
    int ldc = b_cols;

    if (transb && transa) {
        m = b_rows;
        n = a_cols;
        k = b_cols;
        lda = n;
        ldb = k;
        ldc = m;
    } else if (transa) {
        m = b_cols;
        n = a_cols;
        k = b_rows;
        lda = n;
        ldb = m;
	    ldc = m;
    } else if (transb) {
        m = b_rows;
        n = a_rows;
        k = b_cols;
        lda = k;
        ldb = k;
	    ldc = m;
    }

    checkCublasStatus(cublasSgemmStridedBatched(handle, trans_a, trans_b
                                       , m, n, k
                                       , &alpha, d_B_arr, ldb, m * k
                                       , d_A_arr, lda, n * k
                                       , &beta, d_C_arr, ldc, m * n
                                       , batch_dim));
}

/*
cuSPARSE Kernels
https://stackoverflow.com/questions/29688627/sparse-matrix-matrix-multiplication-in-cuda-using-cusparse

Sparse (CSR) * Dense matmul
A * B = C

(m x k) * (k * n) = (m x n)
note: row_ind.len = lda + 1
*/
void cusparse_mm_wrapper(cusparseHandle_t handle,
                         float *dA_values, int *dA_columns, int *dA_csrOffsets,
                         int A_nnz, int A_rows, int A_cols,
                         torch::Tensor B, int B_rows, int B_cols,
                         torch::Tensor C)
{
    float *dB = B.data_ptr<float>();
    float *dC = C.data_ptr<float>();

    int ldb = B_cols;
    int ldc = B_cols;

    // CUSPARSE APIs
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Create sparse matrix A in CSR format
    cusparseSafeCall(cusparseCreateCsr(&matA, A_rows, A_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense matrix B
    cusparseSafeCall(cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create dense matrix C
    cusparseSafeCall(cusparseCreateDnMat(&matC, A_rows, B_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseSafeCall(cusparseSpMM_bufferSize(handle, transA, transB,
                                           &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufferSize));
    checkCudaStatus(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMM
    cusparseSafeCall(cusparseSpMM(handle, transA, transB,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, dBuffer));

    // destroy matrix/vector descriptors
    cusparseSafeCall(cusparseDestroySpMat(matA));
    cusparseSafeCall(cusparseDestroyDnMat(matB));
    cusparseSafeCall(cusparseDestroyDnMat(matC));

    return;
}

void dense_to_csr(cusparseHandle_t handle, 
                  torch::Tensor dense, const int num_rows, const int num_cols,
                  float **d_csr_values, int **d_csr_columns, int **d_csr_offsets, int *nnz)
{

    int ld = num_cols;
    float *d_dense = dense.data_ptr<float>();
    // CUSPARSE APIs
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Allocate memory for offsets
    checkCudaStatus(cudaMalloc((void**) &(*d_csr_offsets), (num_rows + 1) * sizeof(int)));

    // Create dense matrix A
    cusparseSafeCall(cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create sparse matrix B in CSR format
    cusparseSafeCall(cusparseCreateCsr(&matB, num_rows, num_cols, 0, *d_csr_offsets, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // allocate an external buffer if needed
    cusparseSafeCall(cusparseDenseToSparse_bufferSize(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
    checkCudaStatus(cudaMalloc(&dBuffer, bufferSize));

    // execute Sparse to Dense conversion
    cusparseSafeCall(cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz_tmp;
    cusparseSafeCall(cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz_tmp));
    *nnz = nnz_tmp;

    // allocate CSR column indices and values
    checkCudaStatus(cudaMalloc((void**) &(*d_csr_columns), nnz_tmp * sizeof(int)));
    checkCudaStatus(cudaMalloc((void**) &(*d_csr_values),  nnz_tmp * sizeof(float)));
    // reset offsets, column indices, and values pointers
    cusparseSafeCall(cusparseCsrSetPointers(matB, *d_csr_offsets, *d_csr_columns, *d_csr_values));
    // execute Sparse to Dense conversion
    cusparseSafeCall(cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));
    // destroy matrix/vector descriptors
    cusparseSafeCall(cusparseDestroyDnMat(matA));
    cusparseSafeCall(cusparseDestroySpMat(matB));
    // device memory deallocation
    checkCudaStatus(cudaFree(dBuffer));
}

/*
    cusparseOrder_t order[] = {CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL};
    cusparseOperation_t transop[] = {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE}; 

    int i = 0;
    int m, n, k;
    for (auto b_order : order) {
            for (auto transA : transop) {
                for (auto transB: transop) {
			printf("%d\n", i);
			
			if (i !=5 ) {
			    i++;
			    continue;
			}
			
			if (transA == CUSPARSE_OPERATION_TRANSPOSE && transB == CUSPARSE_OPERATION_TRANSPOSE) {
			   m = A_cols;
			   n = B_rows;
			   k = B_cols;
			   ldb = B_cols;
			   ldc = A_cols;
			} else if (transB == CUSPARSE_OPERATION_TRANSPOSE) {
			    m = A_rows;
			    n = B_rows;
			    k = A_cols;
			    ldb = B_cols;
			    ldc = A_rows;
			} else if (transA == CUSPARSE_OPERATION_TRANSPOSE) {
			    m = A_cols;
			    n = B_cols;
			    k = A_rows;
			    ldb = B_rows;
			    ldc = A_cols;
			} else {
			    m = A_rows;
			    n = B_cols;
			    k = B_rows;
			    ldb = B_rows;
			    ldc = A_rows;
			}

			printf("m: %d, n: %d, k: %d, ldb: %d, ldc: %d\n", m, n, k, ldb, ldc);
			
			try {
                        // Create sparse matrix A in CSR format
                        cusparseSafeCall(cusparseCreateCsr(&matA, m, k, A_nnz,
                                                        dA_csrOffsets, dA_columns, dA_values,
                                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
                        // Create dense matrix B
                        cusparseSafeCall(cusparseCreateDnMat(&matB, k, n, ldb, dB,
                                                            CUDA_R_32F, b_order));
                        // Create dense matrix C
                        cusparseSafeCall(cusparseCreateDnMat(&matC, m, n, ldc, dC,
                                                            CUDA_R_32F, b_order));
                        // allocate an external buffer if needed
                        float alpha = 1.0f;
                        float beta = 0.0f;

                        //cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
                        //cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

                        cusparseSafeCall(cusparseSpMM_bufferSize(handle, transA, transB,
                                                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
                        checkCudaStatus(cudaMalloc(&dBuffer, bufferSize));

                        // execute SpMM
                        cusparseSafeCall(cusparseSpMM(handle, transA, transB,
                                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

                        // destroy matrix/vector descriptors
                        cusparseSafeCall(cusparseDestroySpMat(matA));
                        cusparseSafeCall(cusparseDestroyDnMat(matB));
                        cusparseSafeCall(cusparseDestroyDnMat(matC));

                        cuda_print_arr<float>(dC, m, n);
                        i++;
			} catch (const std::exception& e) {
			    printf("Didn't work\n");
			}

                }
            }
    }
    */

    // device memory deallocation
    // checkCudaStatus(cudaFree(dBuffer))
    // checkCudaStatus(cudaFree(dA_csrOffsets))
    // checkCudaStatus(cudaFree(dA_columns))
    // checkCudaStatus(cudaFree(dA_values))
    // checkCudaStatus(cudaFree(dB))
    //checkCudaStatus( cudaFree(dC) );

void free_csr(float *d_csr_values, int *d_csr_columns, int *d_csr_offsets) {
    checkCudaStatus(cudaFree(d_csr_values));
    checkCudaStatus(cudaFree(d_csr_columns));
    checkCudaStatus(cudaFree(d_csr_offsets));
}

#endif // __BASELINE_MM_H_
