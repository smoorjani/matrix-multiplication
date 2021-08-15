#ifndef __MM_KERNEL_H__
#define __MM_KERNEL_H__

#include <iostream>
#include <torch/extension.h>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

//#define NUM_THREADS 64;
int NUM_THREADS = 64;

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

__global__ void packed_accessor_kernel(
    // figure out if this is coalesced
    torch::PackedTensorAccessor32<float, 3> accessor,
    float** trace
) {
  int NUM_THREADS = 64;
  int batch_size = accessor.size(0);
  int n_rows = accessor.size(1);
  int n_cols = accessor.size(2);
    
  int idx_per_row = (int) ((float) (n_cols + NUM_THREADS - 1) / NUM_THREADS);
  // find thread id, row = threadid/64, col = threadid%64
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int batch = 0, row = 0, col = 0;

  if (n_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;

    batch = idx / (idx_per_row * n_rows);
    row = (idx / idx_per_row) % n_rows;
    col = (idx % idx_per_row) * NUM_THREADS + off;
  } else {
    batch = threadId / (n_rows * n_cols);
    row = (threadId % (n_rows * n_cols))/ n_cols;
    col = threadId % n_cols;
  }

  if (batch >= batch_size || row >= n_rows || col >= n_cols) {
    return;
  }
  
  trace[batch][row * n_cols + col] = accessor[batch][row][col];
}


__global__ void packed_setter_kernel(
    torch::PackedTensorAccessor32<float, 3> accessor,
    float** trace
) {
  int NUM_THREADS = 64;
  int batch_size = accessor.size(0);
  int n_rows = accessor.size(1);
  int n_cols = accessor.size(2);
  int idx_per_row = (int) ((float) (n_cols + NUM_THREADS - 1) / NUM_THREADS);
  // find thread id, row = threadid/64, col = threadid%64
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int batch= 0, row = 0, col = 0;

  if (n_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;

    batch = idx / (idx_per_row * n_rows);
    row = (idx / idx_per_row) % n_rows;
    col = (idx % idx_per_row) * NUM_THREADS + off;
  } else {
    batch = threadId / (n_rows * n_cols);
    row = (threadId % (n_rows * n_cols))/ n_cols;
    col = threadId % n_cols;
  }

  if (batch >= batch_size || row >= n_rows || col >= n_cols) {
    return;
  }

  accessor[batch][row][col] = trace[batch][row * n_cols + col];
}

// https://stackoverflow.com/questions/23743384/how-performing-multiple-matrix-multiplications-in-cuda/23743838#23743838

void cublas_bmm_wrapper_accessor(cublasHandle_t handle,
               torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
               size_t a_rows, size_t b_cols, size_t b_rows,
               size_t batch_dim) {

    //auto A_accessory = d_A.packed_accessor32<float,3>();
    //float **d_A_arry = NULL;
    //test_kernel<<<4,2>>>();
    //packed_accessor_kernel<<<(a_rows * b_rows + NUM_THREADS)/NUM_THREADS, NUM_THREADS>>>(A_accessory, d_A_arry);
    //cudaDeviceSynchronize();
    //return;

    size_t a_size = sizeof(float) * a_rows * b_rows;
    size_t b_size = sizeof(float) * b_rows * b_cols;
    size_t c_size = sizeof(float) * a_rows * b_cols;

    float **d_A_arr, **d_B_arr, **d_C_arr;
    float **h_A_arr = (float **) malloc(batch_dim * sizeof(float*));
    float **h_B_arr = (float **) malloc(batch_dim * sizeof(float*));
    float **h_C_arr = (float **) malloc(batch_dim * sizeof(float*));

    gpuErrchk(cudaMalloc((void **)&d_A_arr, batch_dim * sizeof(float *)));
    gpuErrchk(cudaMalloc((void **)&d_B_arr, batch_dim * sizeof(float *)));
    gpuErrchk(cudaMalloc((void **)&d_C_arr, batch_dim * sizeof(float *)));

    for (int i = 0; i < batch_dim; i++) {
        gpuErrchk(cudaMalloc(&h_A_arr[i], a_size));
        gpuErrchk(cudaMalloc(&h_B_arr[i], b_size));
        gpuErrchk(cudaMalloc(&h_C_arr[i], c_size));
    }

    gpuErrchk(cudaMemcpy(d_A_arr, h_A_arr, batch_dim * sizeof(float*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B_arr, h_B_arr, batch_dim * sizeof(float*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C_arr, h_C_arr, batch_dim * sizeof(float*), cudaMemcpyHostToDevice));

    auto A_accessor = d_A.packed_accessor32<float,3>();
    auto B_accessor = d_B.packed_accessor32<float,3>();
    auto C_accessor = d_C.packed_accessor32<float,3>();

    // execute 1 time with a_rows threads
    dim3 thread_per_block(NUM_THREADS);
    dim3 A_blocks_per_grid((batch_dim * a_rows * b_rows + NUM_THREADS - 1)/NUM_THREADS);
    dim3 B_blocks_per_grid((batch_dim * b_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);
    dim3 C_blocks_per_grid((batch_dim * a_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);

    // <<<total threads/64 ,64>>>>
    packed_accessor_kernel<<<A_blocks_per_grid, thread_per_block>>>(A_accessor, d_A_arr);
    cudaDeviceSynchronize();
    packed_accessor_kernel<<<B_blocks_per_grid, thread_per_block>>>(B_accessor, d_B_arr);
    cudaDeviceSynchronize();
  
    const float alpha = 1.0f, beta = 0.0f;
       
    cublasStatus_t status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N
                                       , b_cols, a_rows, b_rows
                                       , &alpha, d_B_arr, b_cols, d_A_arr, b_rows
                                       , &beta, d_C_arr, b_cols
                                       , batch_dim);
    
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }

    packed_setter_kernel<<<C_blocks_per_grid, thread_per_block>>>(C_accessor, d_C_arr);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < batch_dim; i++) {
        gpuErrchk(cudaFree(h_A_arr[i]));
        gpuErrchk(cudaFree(h_B_arr[i]));
        gpuErrchk(cudaFree(h_C_arr[i]));
    }
    gpuErrchk(cudaFree(d_A_arr));
    gpuErrchk(cudaFree(d_B_arr));
    gpuErrchk(cudaFree(d_C_arr));
    
    free(h_A_arr);
    free(h_B_arr);
    free(h_C_arr);
}


void cublas_bmm_wrapper(cublasHandle_t handle,
               float *d_A, float *d_B, float *d_C,
               size_t a_rows, size_t b_cols, size_t b_rows,
               size_t batch_dim) {


    size_t c_size = sizeof(float) * a_rows * b_cols;
    size_t a_size = sizeof(float) * a_rows * b_rows;
    size_t b_size = sizeof(float) * b_rows * b_cols;

    float **d_A_arr, **d_B_arr;
    float **d_C_arr;

    gpuErrchk(cudaMalloc((void **)&d_A_arr, batch_dim * sizeof(float *)));
    gpuErrchk(cudaMalloc((void **)&d_B_arr, batch_dim * sizeof(float *)));
    gpuErrchk(cudaMalloc((void **)&d_C_arr, batch_dim * sizeof(float *)));

    // assumes contiguous memory
    for (int i = 0; i < batch_dim; i++) {
        gpuErrchk(cudaMemcpy(d_A_arr[i], d_A + (batch_dim * a_size * i), (batch_dim * a_size), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_B_arr[i], d_B + (batch_dim * b_size * i), (batch_dim * b_size), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_C_arr[i], d_C + (batch_dim * c_size * i), (batch_dim * c_size), cudaMemcpyDeviceToDevice));
    }

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cublas_result = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N
                                       , b_cols, a_rows, b_rows
                                       , &alpha, d_B_arr, b_cols, d_A_arr, b_rows
                                       , &beta, d_C_arr, b_cols
                                       , batch_dim);
    assert(cublas_result == CUBLAS_STATUS_SUCCESS);

    gpuErrchk(cudaFree(d_A_arr));
    gpuErrchk(cudaFree(d_B_arr));
    gpuErrchk(cudaFree(d_C_arr));
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
    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, k, n, descrB, d_B_dense, k, d_nnzPerVectorB, &nnzB));

    // Device side sparse matrix B
    double *d_B;
    gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));
    int *d_B_RowIndices;
    gpuErrchk(cudaMalloc(&d_B_RowIndices, (k + 1) * sizeof(*d_B_RowIndices)));
    int *d_B_ColIndices;
    gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));
    // Dense B to Sparse B
    cusparseSafeCall(cusparseDdense2csr(handle, k, n, descrB, d_B_dense, k, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

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

    cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA,
                                      d_A, d_A_RowIndices, d_A_ColIndices, descrB, nnzB, d_B, d_B_RowIndices, d_B_ColIndices, descrC,
                                      d_C, d_C_RowIndices, d_C_ColIndices));

    cusparseSafeCall(cusparseDcsr2dense(handle, m, n, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, m));

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
    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));

    // Device side sparse matrix
    double *d_A;
    gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
    int *d_A_RowIndices;
    gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
    int *d_A_ColIndices;
    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

    cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

    *d_A_val = d_A;
    *d_A_rowptr = d_A_RowIndices;
    *d_A_colind = d_A_ColIndices;
    *nnzA = nnz;

    gpuErrchk(cudaFree(d_nnzPerVector));
    return;
}

#endif // __MM_KERNEL_H__
