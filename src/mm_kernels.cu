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

__global__ void packed_1d_accessor_kernel(
    // figure out if this is coalesced
    torch::PackedTensorAccessor32<float, 2> accessor,
    float* trace
) {
  int n_rows = accessor.size(0);
  int n_cols = accessor.size(1);
    
  int idx_per_row = (int) ((float) (n_cols + NUM_THREADS - 1) / NUM_THREADS);
  // find thread id, row = threadid/64, col = threadid%64
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int row = 0, col = 0;

  if (n_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;

    row = idx;
    col = row * (idx_per_row - 1) + off;
  } else {
    row = threadId / n_cols;
    col = threadId % n_cols;
  }

  if (row >= n_rows || col >= n_cols) {
    return;
  }
  trace[row * n_cols + col] = accessor[row][col];
}

__global__ void packed_1d_setter_kernel(
    // figure out if this is coalesced
    torch::PackedTensorAccessor32<float, 2> accessor,
    float* trace
) {
  int n_rows = accessor.size(0);
  int n_cols = accessor.size(1);
    
  int idx_per_row = (int) ((float) (n_cols + NUM_THREADS - 1) / NUM_THREADS);
  // find thread id, row = threadid/64, col = threadid%64
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int row = 0, col = 0;

  if (n_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;

    row = idx;
    col = row * (idx_per_row - 1) + off;
  } else {
    row = threadId / n_cols;
    col = threadId % n_cols;
  }

  if (row >= n_rows || col >= n_cols) {
    return;
  }
  
  accessor[row][col] = trace[row * n_cols + col];
}

void cublas_mm_wrapper(cublasHandle_t handle,
                       torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
                       int a_rows, int b_rows, int b_cols) {

    /*
    size_t a_size = sizeof(float) * a_rows * b_rows;
    size_t b_size = sizeof(float) * b_rows * b_cols;
    size_t c_size = sizeof(float) * a_rows * b_cols;

    float *d_A_arr, *d_B_arr, *d_C_arr;
    gpuErrchk(cudaMalloc((void **)&d_A_arr, a_size));
    gpuErrchk(cudaMalloc((void **)&d_B_arr, b_size));
    gpuErrchk(cudaMalloc((void **)&d_C_arr, c_size));

    auto A_accessor = d_A.packed_accessor32<float,2>();
    auto B_accessor = d_B.packed_accessor32<float,2>();
    auto C_accessor = d_C.packed_accessor32<float,2>();
 
    dim3 thread_per_block(NUM_THREADS);
    dim3 A_blocks_per_grid((a_rows * b_rows + NUM_THREADS - 1)/NUM_THREADS);
    dim3 B_blocks_per_grid((b_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);
    dim3 C_blocks_per_grid((a_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);

    packed_1d_accessor_kernel<<<A_blocks_per_grid, thread_per_block>>>(A_accessor, d_A_arr);   
    packed_1d_accessor_kernel<<<B_blocks_per_grid, thread_per_block>>>(B_accessor, d_B_arr);
    */
    
    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();

    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        b_cols, a_rows, b_rows, &alpha,
        d_B_arr, b_cols,
        d_A_arr, b_rows, &beta,
        d_C_arr, b_cols);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }
    
    /*
    packed_1d_setter_kernel<<<C_blocks_per_grid, thread_per_block>>>(C_accessor, d_C_arr);
    
    gpuErrchk(cudaFree(d_A_arr));
    gpuErrchk(cudaFree(d_B_arr));
    gpuErrchk(cudaFree(d_C_arr));
    */
}

__global__ void packed_2d_accessor_kernel(
    // figure out if this is coalesced
    torch::PackedTensorAccessor32<float, 3> accessor,
    float* trace
) {
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
  
  trace[batch * n_rows * n_cols + row * n_cols + col] = accessor[batch][row][col];
}

__global__ void packed_2d_accessor_kernel_combined(
    // figure out if this is coalesced
    torch::PackedTensorAccessor32<float, 3> a_accessor,
    torch::PackedTensorAccessor32<float, 3> b_accessor,
    float* a_trace,
    float* b_trace
) {
  int batch_size = a_accessor.size(0);
  int a_rows = a_accessor.size(1);
  int a_cols = a_accessor.size(2);
  int b_rows = b_accessor.size(1);
  int b_cols = b_accessor.size(2);
    
  int a_idx_per_row = (int) ((float) (a_cols + NUM_THREADS - 1) / NUM_THREADS);
  int b_idx_per_row = (int) ((float) (b_cols + NUM_THREADS - 1) / NUM_THREADS);
  // find thread id, row = threadid/64, col = threadid%64
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int a_batch = 0, a_row = 0, a_col = 0;
  int b_batch = 0, b_row = 0, b_col = 0;

  if (a_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;

    a_batch = idx / (a_idx_per_row * a_rows);
    a_row = (idx / a_idx_per_row) % a_rows;
    a_col = (idx % a_idx_per_row) * NUM_THREADS + off;
  } else {
    a_batch = threadId / (a_rows * a_cols);
    a_row = (threadId % (a_rows * a_cols))/ a_cols;
    a_col = threadId % a_cols;
  }

  if (b_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;

    b_batch = idx / (b_idx_per_row * b_rows);
    b_row = (idx / b_idx_per_row) % b_rows;
    b_col = (idx % b_idx_per_row) * NUM_THREADS + off;
  } else {
    b_batch = threadId / (b_rows * b_cols);
    b_row = (threadId % (b_rows * b_cols))/ b_cols;
    b_col = threadId % b_cols;
  }

  if (a_batch < batch_size && a_row < a_rows && a_col < a_cols) {
    a_trace[a_batch * a_rows * a_cols + a_row * a_cols + a_col] = a_accessor[a_batch][a_row][a_col];
  }

  if (b_batch < batch_size && b_row < b_rows && b_col < b_cols) {
    b_trace[b_batch * b_rows * b_cols + b_row * b_cols + b_col] = b_accessor[b_batch][b_row][b_col];
  }
 
}


__global__ void packed_2d_setter_kernel(
    torch::PackedTensorAccessor32<float, 3> accessor,
    float* trace
) {
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

  accessor[batch][row][col] = trace[batch * n_rows * n_cols + row * n_cols + col];
}

// https://stackoverflow.com/questions/23743384/how-performing-multiple-matrix-multiplications-in-cuda/23743838#23743838

void cublas_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
               size_t a_rows, size_t b_cols, size_t b_rows,
               size_t batch_dim) {
    
    // ==============
    timestamp_t t0 = get_timestamp();
    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();
    /*
    size_t a_size = sizeof(float) * a_rows * b_rows;
    size_t b_size = sizeof(float) * b_rows * b_cols;
    size_t c_size = sizeof(float) * a_rows * b_cols;
    
    // setting up device arrays for bmm
    float *d_A_arr, *d_B_arr, *d_C_arr;
    gpuErrchk(cudaMalloc(&d_A_arr, batch_dim * a_size));
    gpuErrchk(cudaMalloc(&d_B_arr, batch_dim * b_size));
    gpuErrchk(cudaMalloc(&d_C_arr, batch_dim * c_size));
    
    // creating accessors for tensors
    auto A_accessor = d_A.packed_accessor32<float,3>();
    auto B_accessor = d_B.packed_accessor32<float,3>();
    auto C_accessor = d_C.packed_accessor32<float,3>();

    // const int num_streams = 16;
    // cudaStream_t streams[num_streams];
    // execute 1 time with a_rows threads
    // << blocks_per_grid/NUM_THREADS, NUM_THREADS, 0, streams[i]>>
    dim3 thread_per_block(NUM_THREADS);
    dim3 C_blocks_per_grid((batch_dim * a_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);
    dim3 AB_blocks_per_grid((batch_dim * a_rows * b_rows + batch_dim * a_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);
    */
    timestamp_t t1 = get_timestamp();
    double secs = (t1 - t0) / 1000000.0L;
    printf("Preprocessing: %f\n", secs);
    /*
    // ==============
    t0 = get_timestamp();
    packed_2d_accessor_kernel_combined<<<AB_blocks_per_grid, thread_per_block>>>(A_accessor, B_accessor, d_A_arr, d_B_arr);
    gpuErrchk(cudaDeviceSynchronize());
    t1 = get_timestamp();
    secs = (t1 - t0) / 1000000.0L;
    printf("Accessor Kernel: %f\n", secs);
    // ==============
    */
    t0 = get_timestamp();
    const float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N
                                       , b_cols, a_rows, b_rows
                                       , &alpha, d_B_arr, b_cols, b_rows * b_cols
                                       , d_A_arr, b_rows, a_rows * b_rows
                                       , &beta, d_C_arr, b_cols, a_rows * b_cols
                                       , batch_dim);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }

    gpuErrchk(cudaDeviceSynchronize());
    t1 = get_timestamp();
    secs = (t1 - t0) / 1000000.0L;
    printf("Batch GEMM: %f\n", secs);
    /*
    // ==============
    t0 = get_timestamp();
    packed_2d_setter_kernel<<<C_blocks_per_grid, thread_per_block>>>(C_accessor, d_C_arr);
    gpuErrchk(cudaDeviceSynchronize());
    t1 = get_timestamp();
    secs = (t1 - t0) / 1000000.0L;
    printf("Setter: %f\n", secs);
    // ==============
    t0 = get_timestamp();
    gpuErrchk(cudaFree(d_A_arr));
    gpuErrchk(cudaFree(d_B_arr));
    gpuErrchk(cudaFree(d_C_arr));
    t1 = get_timestamp();
    secs = (t1 - t0) / 1000000.0L;
    printf("Freeing Memory: %f\n", secs);
    // ==============
    */
}

__global__ void packed_2d_accessor_kernel_4d(
    // figure out if this is coalesced
    torch::PackedTensorAccessor32<float, 4> accessor,
    float* trace
) {
  // turns a 4d matrix into a 2d array of [batch_size][matrix_idx]
  // batch size spans the first two dimensions
  // matrix idx spans the last two (rows/cols)
  int batch_dim1 = accessor.size(0);
  int batch_dim2 = accessor.size(1);
  int n_rows = accessor.size(2);
  int n_cols = accessor.size(3);
    
  int idx_per_row = (int) ((float) (n_cols + NUM_THREADS - 1) / NUM_THREADS);
  // find thread id, row = threadid/64, col = threadid%64
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int d_1 = 0, d_2 = 0, row = 0, col = 0;

  if (n_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;

    d_1 = idx / (idx_per_row * n_rows * batch_dim2);
    d_2 = (idx / (idx_per_row * n_rows)) % batch_dim2;
    row = (idx / idx_per_row) % n_rows;
    col = (idx % idx_per_row) * NUM_THREADS + off;
  } else {
    d_1 = threadId / (batch_dim2 * n_rows * n_cols);
    d_2 = (threadId % (batch_dim2 * n_rows * n_cols))/ (n_rows * n_cols);
    row = (threadId % (n_rows * n_cols))/ n_cols;
    col = threadId % n_cols;
  }

  if (d_1 >= batch_dim1 || d_2 >= batch_dim2 || row >= n_rows || col >= n_cols) {
    return;
  }

  trace[threadId] = accessor[d_1][d_2][row][col];
}

__global__ void packed_2d_setter_kernel_4d(
    // figure out if this is coalesced
    torch::PackedTensorAccessor32<float, 4> accessor,
    float* trace
) {
  // turns a 4d matrix into a 2d array of [batch_size][matrix_idx]
  // batch size spans the first two dimensions
  // matrix idx spans the last two (rows/cols)
  int batch_dim1 = accessor.size(0);
  int batch_dim2 = accessor.size(1);
  int n_rows = accessor.size(2);
  int n_cols = accessor.size(3);
    
  int idx_per_row = (int) ((float) (n_cols + NUM_THREADS - 1) / NUM_THREADS);
  // find thread id, row = threadid/64, col = threadid%64
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int d_1 = 0, d_2 = 0, row = 0, col = 0;

  if (n_cols >= NUM_THREADS) {
    int idx = threadId / NUM_THREADS;
    int off = threadId % NUM_THREADS;


    d_1 = idx / (idx_per_row * n_rows * batch_dim2);
    d_2 = (idx / (idx_per_row * n_rows)) % batch_dim2;
    row = (idx / idx_per_row) % n_rows;
    col = (idx % idx_per_row) * NUM_THREADS + off;
  } else {
    d_1 = threadId / (batch_dim2 * n_rows * n_cols);
    d_2 = (threadId % (batch_dim2 * n_rows * n_cols))/ (n_rows * n_cols);
    row = (threadId % (n_rows * n_cols))/ n_cols;
    col = threadId % n_cols;
  }

  if (d_1 >= batch_dim1 || d_2 >= batch_dim2 || row >= n_rows || col >= n_cols) {
    return;
  }
  
  accessor[d_1][d_2][row][col] = trace[threadId];
}

void cublas_4d_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
               size_t a_rows, size_t b_cols, size_t b_rows,
               size_t batch_dim1, size_t batch_dim2) {

    /*
    size_t a_size = sizeof(float) * a_rows * b_rows;
    size_t b_size = sizeof(float) * b_rows * b_cols;
    size_t c_size = sizeof(float) * a_rows * b_cols;

    // setting up device arrays for bmm
    float *d_A_arr, *d_B_arr, *d_C_arr;
    gpuErrchk(cudaMalloc(&d_A_arr, batch_dim1 * batch_dim2 * a_size));
    gpuErrchk(cudaMalloc(&d_B_arr, batch_dim1 * batch_dim2 * b_size));
    gpuErrchk(cudaMalloc(&d_C_arr, batch_dim1 * batch_dim2 * c_size));

    auto A_accessor = d_A.packed_accessor32<float,4>();
    auto B_accessor = d_B.packed_accessor32<float,4>();
    auto C_accessor = d_C.packed_accessor32<float,4>();

    // execute 1 time with a_rows threads
    dim3 thread_per_block(NUM_THREADS);
    dim3 A_blocks_per_grid((batch_dim1 * batch_dim2 * a_rows * b_rows + NUM_THREADS - 1)/NUM_THREADS);
    dim3 B_blocks_per_grid((batch_dim1 * batch_dim2 * b_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);
    dim3 C_blocks_per_grid((batch_dim1 * batch_dim2 * a_rows * b_cols + NUM_THREADS - 1)/NUM_THREADS);

    // <<<total threads/64 ,64>>>>
    packed_2d_accessor_kernel_4d<<<A_blocks_per_grid, thread_per_block>>>(A_accessor, d_A_arr);
    packed_2d_accessor_kernel_4d<<<B_blocks_per_grid, thread_per_block>>>(B_accessor, d_B_arr);
 
    gpuErrchk(cudaDeviceSynchronize());
    */

    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();

    const float alpha = 1.0f, beta = 0.0f;
       
    cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N
                                       , b_cols, a_rows, b_rows
                                       , &alpha, d_B_arr, b_cols, b_rows * b_cols
                                       , d_A_arr, b_rows, a_rows * b_rows
                                       , &beta, d_C_arr, b_cols, a_rows * b_cols
                                       , batch_dim1 * batch_dim2);
    
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Kernel execution error.";
    }

    /*
    gpuErrchk(cudaDeviceSynchronize());

    packed_2d_setter_kernel_4d<<<C_blocks_per_grid, thread_per_block>>>(C_accessor, d_C_arr);
    
    gpuErrchk(cudaFree(d_A_arr));
    gpuErrchk(cudaFree(d_B_arr));
    gpuErrchk(cudaFree(d_C_arr));
    */
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
                   int ldc) {
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

#endif // __MM_KERNEL_H__
