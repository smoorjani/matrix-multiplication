#ifndef __MM_KERNEL_H__
#define __MM_KERNEL_H__

#include <iostream>
#include <array>
#include <torch/extension.h>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cublasLt.h>

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

    //cuda_print_arr<float>(dA_values, 1, A_nnz);
    //cuda_print_arr<int>(dA_columns, 1, A_nnz);
    //cuda_print_arr<int>(dA_csrOffsets, 1, A_rows + 1);
    //cuda_print_arr<float>(dB, B_rows, B_cols);
    printf("A_rows: %d, A_cols: %d, B_rows: %d, B_cols: %d, nnz: %d\n", A_rows, A_cols, B_rows, B_cols, A_nnz);
    int ldb = B_rows;
    int ldc = A_rows;

    // CUSPARSE APIs
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseOrder_t order[] = {CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL};
    cusparseOrder_t transop[] = {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE} 

    for (auto b_order : order) {
        for (auto c_order : order) {
            for (auto transA : transop) {
                for (auto transB: transop) {
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
                                                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
                        checkCudaStatus(cudaMalloc(&dBuffer, bufferSize));

                        // execute SpMM
                        cusparseSafeCall(cusparseSpMM(handle, transA, transB,
                                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

                        // destroy matrix/vector descriptors
                        cusparseSafeCall(cusparseDestroySpMat(matA));
                        cusparseSafeCall(cusparseDestroyDnMat(matB));
                        cusparseSafeCall(cusparseDestroyDnMat(matC));

                        cuda_print_arr<float>(dC, A_rows, B_cols);

                }
            }
        }
    }
    // // Create sparse matrix A in CSR format
    // cusparseSafeCall(cusparseCreateCsr(&matA, A_rows, A_cols, A_nnz,
    //                                   dA_csrOffsets, dA_columns, dA_values,
    //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // // Create dense matrix B
    // cusparseSafeCall(cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
    //                                     CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // // Create dense matrix C
    // cusparseSafeCall(cusparseCreateDnMat(&matC, A_rows, B_cols, ldc, dC,
    //                                     CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // // allocate an external buffer if needed
    // float alpha = 1.0f;
    // float beta = 0.0f;

    // cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // cusparseSafeCall(cusparseSpMM_bufferSize(handle, transA, transB,
    //                                        &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    // checkCudaStatus(cudaMalloc(&dBuffer, bufferSize));

    // // execute SpMM
    // cusparseSafeCall(cusparseSpMM(handle, transA, transB,
    //                             &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // // destroy matrix/vector descriptors
    // cusparseSafeCall(cusparseDestroySpMat(matA));
    // cusparseSafeCall(cusparseDestroyDnMat(matB));
    // cusparseSafeCall(cusparseDestroyDnMat(matC));

    // device memory deallocation
    // checkCudaStatus(cudaFree(dBuffer))
    // checkCudaStatus(cudaFree(dA_csrOffsets))
    // checkCudaStatus(cudaFree(dA_columns))
    // checkCudaStatus(cudaFree(dA_values))
    // checkCudaStatus(cudaFree(dB))
    //checkCudaStatus( cudaFree(dC) );
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

void free_csr(float **d_csr_values, int **d_csr_columns, int **d_csr_offsets) {
    checkCudaStatus(cudaFree(d_csr_values));
    checkCudaStatus(cudaFree(d_csr_columns));
    checkCudaStatus(cudaFree(d_csr_offsets));
}



int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const float *A,
                   int lda,
                   const float *B,
                   int ldb,
                   float *C,
                   int ldc) {return;}
    /*
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    float *Atransform = NULL, *Btransform = NULL;
    float *Ctransform                   = NULL;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(float) * roundoff(k, 32) / 32 * ldatransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(float) * roundoff(k, 32) / 32 * ldbtransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(float) * roundoff(n, 32) / 32 * ldctransform));

    checkCublasStatus(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    //checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUDA_R_32F));
    // tensor op igemm kernels only support NT gemm
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for original matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_32F, m, k, ldatransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // data memory order is set to CUBLASLT_ORDER_COL4_4R2_8C in order to achieve best performance on Turing devices.
    // for best performance on Ampere, consider setting the memory order to CUBLASLT_ORDER_COL32_2R_4R4.
    checkCublasStatus(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_32F, n, k, ldbtransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32F, m, n, ldctransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // ---------------------------------------------------------------------------------------------
    // transforms and computation

    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0));

    // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0));

    // no need to transform C matrix as beta is assumed to be 0
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     Atransform,
                                     AtransformDesc,
                                     Btransform,
                                     BtransformDesc,
                                     &beta,
                                     Ctransform,
                                     CtransformDesc,
                                     Ctransform,
                                     CtransformDesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    opTranspose = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    // transform outputs to COL order
    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CtransformDesc, &transformBeta, NULL, NULL, C, Cdesc, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (CtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(CtransformDesc));
    if (BtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(BtransformDesc));
    if (AtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(AtransformDesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if (transformDesc) checkCublasStatus(cublasLtMatrixTransformDescDestroy(transformDesc));

    // wait until device is done before freeing transformed buffers
    checkCudaStatus(cudaDeviceSynchronize());
    if (Ctransform) checkCudaStatus(cudaFree(Ctransform));
    if (Btransform) checkCudaStatus(cudaFree(Btransform));
    if (Atransform) checkCudaStatus(cudaFree(Atransform));
}
*/
#endif // __MM_KERNEL_H__
