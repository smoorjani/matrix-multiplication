#ifndef __MM_KERNEL_H__
#define __MM_KERNEL_H__

#include <iostream>
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
    gpuErrchk(cudaDeviceSynchronize());
}

void cublas_mm_wrapper(cublasHandle_t handle,
                       torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
                       int a_rows, int a_cols, int b_rows, int b_cols,
                       bool transa, bool transb) {

    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();

    printf("transa: %d, transb: %d\n", transa, transb);
    printf("a_rows: %d, b_rows: %d, b_cols: %d\n", a_rows, b_rows, b_cols);
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
	k = b_cols;
	ldb = b_rows;
        lda = b_cols;
	ldc = lda;
    } else if (transa) {
        n = a_cols;
	m = b_cols;
	k = b_rows;
	lda = k;
        ldb = m;
	ldc = m;
    } else if (transb) {
        m = b_rows;
        n = a_cols;
	k = a_rows;
	ldb = n;
        lda = k;
	ldc = lda;
    }


    printf("m: %d, n: %d, k: %d\n", m, n, k);
    printf("lda: %d, ldb: %d, ldc: %d\n", lda, ldb, ldc);

    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t status = cublasSgemm(
        handle, trans_a, trans_b,
        m, n, k, &alpha,
        d_B_arr, ldb,
        d_A_arr, lda, &beta,
        d_C_arr, ldc);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }
}

__global__ void check_equal(float *d_Arr, float *h_Arr, size_t rows, size_t cols)
{
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
// https://stackoverflow.com/questions/23743384/how-performing-multiple-matrix-multiplications-in-cuda/23743838#23743838

void cublas_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
               size_t a_rows, size_t a_cols, size_t b_cols, size_t b_rows,
               size_t batch_dim, bool transa, bool transb) {
    
    // ==============
    timestamp_t t0 = get_timestamp();
    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();
    /*
    printf("chkpt1\n");
    float *C;
    gpuErrchk(cudaMalloc(&C, a_rows * b_cols * sizeof(float)));
    gpuErrchk(cudaMemcpy(C, d_C_arr, a_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice));
    printf("chkpt2\n");
    dim3 threads_per_block(NUM_THREADS);
    dim3 C_blocks_per_grid((a_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);
    check_equal<<<threads_per_block, C_blocks_per_grid>>>(C, d_C_arr, a_rows, b_cols);
    printf("chkpt3\n");
    gpuErrchk(cudaDeviceSynchronize());
    */
    timestamp_t t1 = get_timestamp();
    double secs = (t1 - t0) / 1000000.0L;
    printf("Preprocessing: %f\n", secs);

    t0 = get_timestamp();
    const float alpha = 1.0f, beta = 0.0f;

    cublasOperation_t trans_a = (!transb) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_b = (!transa) ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasStatus_t status = cublasSgemmStridedBatched(handle, trans_a, trans_b
                                       , b_cols, a_rows, b_rows
                                       , &alpha, d_B_arr, b_cols, b_rows * b_cols
                                       , d_A_arr, b_rows, a_rows * b_rows
                                       , &beta, d_C_arr, b_cols, a_rows * b_cols
                                       , batch_dim);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }
    /*
    gpuErrchk(cudaDeviceSynchronize());
    printf("chkpt4\n");
    //gpuErrchk(cudaMemcpy(C, d_C_arr, a_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice));
    //check_equal<<<threads_per_block, C_blocks_per_grid>>>(C, d_C_arr, a_rows, b_cols);
    //gpuErrchk(cudaDeviceSynchronize());
    printf("chkpt5\n");
    gpuErrchk(cudaMemcpy(C, d_C_arr, a_rows * b_cols * sizeof(float), cudaMemcpyDeviceToDevice));
    check_equal<<<threads_per_block, C_blocks_per_grid>>>(C, d_C_arr, a_rows, b_cols);
    gpuErrchk(cudaDeviceSynchronize());
    printf("chkpt6\n");
    */

    gpuErrchk(cudaStreamSynchronize(0));
    t1 = get_timestamp();
    secs = (t1 - t0) / 1000000.0L;
    printf("Batch GEMM: %f\n", secs);
}

void cublas_4d_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
               size_t a_rows, size_t a_cols, size_t b_cols, size_t b_rows,
               size_t batch_dim1, size_t batch_dim2,
               bool transa, bool transb) {

    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();
    const float alpha = 1.0f, beta = 0.0f;

    cublasOperation_t trans_a = (!transa) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_b = (!transb) ? CUBLAS_OP_N : CUBLAS_OP_T;
    printf("transa: %d\n", (int) transa);
    printf("transb: %d\n", (int) transb);
    // (64, 512) (64, 256) transa
    // m = 512, n = 256, k = 64
    // m = a_cols -> a_cols
    // n = b_cols -> b_cols
    // k = b_rows -> b_rows 
    // (512, 64) (256, 64) transb
    // m = 512, n = 256, k = 64
    // m = a_rows -> a_rows
    // n = b_rows -> b_rows
    // k = b_cols -> b_cols
    // (512, 64) (64, 256)
    // m = 512, n = 256, k = 64
    // m = a_rows -> b_cols
    // n = b_cols -> a_rows
    // k = b_rows -> b_rows

    size_t m = 0;
    size_t n = 0;
    size_t k = 0;

    if (trans_a && trans_b) {
        m = a_rows;
        n = b_cols;
        k = b_rows;
    } else if (trans_a) {
        m = a_cols;
        n = b_cols;
        k = b_rows;
    } else if (trans_b) {
        m = a_rows;
        n = b_rows;
        k = b_cols;
    } else {
        m = b_cols;
        n = a_rows;
        k = b_rows;
    }

    if (trans_b) {
        m = a_rows;
        n = b_rows;
        k = b_cols;
        cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T
                                       , m, n, k
                                       , &alpha, d_B_arr, n, b_rows * b_cols
                                       , d_A_arr, m, a_rows * a_cols
                                       , &beta, d_C_arr, a_rows, a_rows * b_rows
                                       , batch_dim1 * batch_dim2);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "Kernel execution error.";
        }
    } else {
        cublasStatus_t status = cublasSgemmStridedBatched(handle, trans_b, trans_a
                                       , m, n, k
                                       , &alpha, d_B_arr, m, b_rows * b_cols
                                       , d_A_arr, k, a_rows * b_rows
                                       , &beta, d_C_arr, m, a_rows * b_cols
                                       , batch_dim1 * batch_dim2);
    
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "Kernel execution error.";
        }
    }

    
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
                         double *d_A, int *d_A_ColIndices, int *d_A_RowIndices,
                         int nnzA, int A_rowptr_size,
                         double *d_B_dense, int B_rows, int B_cols,
                         double *d_C_dense)
{
    // Initialize cuSPARSE
    // cusparseHandle_t handle;
    // cusparseSafeCall(cusparseCreate(&handle));
    const int m = A_rowptr_size - 1;
    const int k = B_rows;
    const int n = B_cols;

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
    //cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, k, n, descrB, d_B_dense, k, d_nnzPerVectorB, &nnzB));

    // Device side sparse matrix B
    double *d_B;
    gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));
    int *d_B_RowIndices;
    gpuErrchk(cudaMalloc(&d_B_RowIndices, (k + 1) * sizeof(*d_B_RowIndices)));
    int *d_B_ColIndices;
    gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));
    // Dense B to Sparse B
    //cusparseSafeCall(cusparseDdense2csr(handle, k, n, descrB, d_B_dense, k, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

    // Device side sparse matrix C
    int *d_C_RowIndices;
    gpuErrchk(cudaMalloc(&d_C_RowIndices, (m + 1) * sizeof(*d_C_RowIndices)));

    // Performing the matrix - matrix multiplication
    int baseC, nnzC = 0;
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzC;

    //cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    //cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA,
    //                                     d_A_RowIndices, d_A_ColIndices, descrB, nnzB, d_B_RowIndices, d_B_ColIndices, descrC, d_C_RowIndices,
    //                                     nnzTotalDevHostPtr));
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

    //cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA,
    //                                  d_A, d_A_RowIndices, d_A_ColIndices, descrB, nnzB, d_B, d_B_RowIndices, d_B_ColIndices, descrC,
    //                                  d_C, d_C_RowIndices, d_C_ColIndices));

    //cusparseSafeCall(cusparseDcsr2dense(handle, m, n, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, m));

    cudaFree(d_nnzPerVectorB);

    cudaFree(d_B);
    cudaFree(d_B_RowIndices);
    cudaFree(d_B_ColIndices);

    cudaFree(d_C);
    cudaFree(d_C_RowIndices);
    cudaFree(d_C_ColIndices);

    return;
}

void dense_to_csr(cusparseHandle_t handle, 
                  double *d_A_dense, const int Nrows, const int Ncols,
                  double **d_A_val, int **d_A_colind, int **d_A_rowptr, int *nnzA)
{
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
    //cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));

    // Device side sparse matrix
    double *d_A;
    gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
    int *d_A_RowIndices;
    gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
    int *d_A_ColIndices;
    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

    //cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

    *d_A_val = d_A;
    *d_A_rowptr = d_A_RowIndices;
    *d_A_colind = d_A_ColIndices;
    *nnzA = nnz;

    gpuErrchk(cudaFree(d_nnzPerVector));
    return;
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
