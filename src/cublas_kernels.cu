#ifndef __CUBLAS_BMM_KERNEL_H__
#define __CUBLAS_BMM_KERNEL_H__

// https://stackoverflow.com/questions/23743384/how-performing-multiple-matrix-multiplications-in-cuda/23743838#23743838

#include <iostream>
#include <torch/extension.h>
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
*/


void cublas_mm_wrapper(cublasHandle_t handle,
                       float *d_A, float *d_B, float *d_C,
                       int m, int k, int n) {
    cudaMemset(d_C, 0, m * n * sizeof(float));
 

    /* 
    printf("m: %d, k: %d, n: %d\n", m, k, n);
 
    float *h_A = (float *) malloc(sizeof(float) * m * k);
    float *h_B = (float *) malloc(sizeof(float) * k * n);
    gpuErrchk(cudaMemcpy(h_A, d_A, sizeof(float) * m * k, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_B, d_B, sizeof(float) * k * n, cudaMemcpyDeviceToHost));

    print_arr(h_A, m, k);
    print_arr(h_B, k, n);
    */

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


void printArrayS(float *ptr, int rows, int cols, char mode, char *name) {
    printf("%s\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mode == 'B') /* Normal mode */ {
                if (ptr[i * cols + j] >= 0)
                    printf(" %3.6f ", ptr[i * cols + j]);
                else
                    printf("%3.6f ", ptr[i * cols + j]);
            } else /* Transpose mode */ {
                if (ptr[j * rows + i] >= 0)
                    printf("%3.6f ", ptr[j * rows + i]);
                else
                    printf("%3.6f ", ptr[j * rows + i]);
            }
        }
        printf("\n");
    }
}

/*
__global__ void packed_accessor_kernel(
    torch::PackedTensorAccessor32<float, 3> accessor,
    float** trace) {
  int i=threadIdx.x;
  // should access row i
  cudaMemcpy(trace[i], accessor[i]);
}
*/
__global__ void packed_accessor_kernel(
    torch::PackedTensorAccessor32<float, 3> accessor,
    float** trace, int size) {
  int i=threadIdx.x;
  // should access row i
  for (int j = 0; j < size; j++) {
    //cudaMemcpy(&trace[i][j], accessor[i][j].data(), sizeof(float), cudaMemcpyDeviceToDevice);
    trace[i][j] = *(accessor[i][j].data());
    
  }
}

void cublas_bmm_wrapper_accessor(cublasHandle_t handle,
               torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
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

    auto A_accessor = d_A.packed_accessor32<float,3>();
    float **trace = d_A_arr;
    // execute 1 time with a_rows threads
    packed_accessor_kernel<<<1, a_rows>>>(A_accessor, (float**) (d_A_arr), a_size);

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

// https://stackoverflow.com/questions/29688627/sparse-matrix-matrix-multiplication-in-cuda-using-cusparse

// Sparse (CSR) * Dense matmul
// A * B = C

// (m x k) * (k * n) = (m x n)
// note: row_ind.len = lda + 1

void cusparse_mm_wrapper(cusparseHandle_t handle,
                         double *h_A, int *h_A_ColIndices, int *h_A_RowIndices,
                         int nnzA, int h_A_rowptr_size,
                         double *h_B_dense, int h_B_rows, int h_B_cols,
                         double *h_C_dense)
{
    // Initialize cuSPARSE
    // cusparseHandle_t handle;
    // cusparseSafeCall(cusparseCreate(&handle));
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

void dense_to_csr(cusparseHandle_t handle, 
                  double *h_A_dense, const int Nrows, const int Ncols,
                  double **h_A_val, int **h_A_colind, int **h_A_rowptr, int *nnzA)
{
    // Initialize cuSPARSE
    // cusparseHandle_t handle;
    // cusparseSafeCall(cusparseCreate(&handle));

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






// void cublas_bmm_wrapper(cublasHandle_t handle,
//                float **A, float **B, float **C,
//                size_t a_rows, size_t b_cols, size_t b_rows,
//                size_t batch_dim) {

//     float *d_A[batch_dim];
//     float *d_B[batch_dim];
//     float *d_C[batch_dim];

//     size_t c_size = sizeof(float) * a_rows * b_cols;
//     size_t a_size = sizeof(float) * a_rows * b_rows;
//     size_t b_size = sizeof(float) * b_rows * b_cols;

//     const float **d_A_arr, **d_B_arr;
//     float **d_C_arr;

//     for (int i = 0; i < batch_dim; i++) {
//         gpuErrchk(cudaMalloc((void **)&d_A[i], a_size));
//         gpuErrchk(cudaMalloc((void **)&d_B[i], b_size));
//         gpuErrchk(cudaMalloc((void **)&d_C[i], c_size));
//     }

//     gpuErrchk(cudaMalloc((void **)&d_A_arr, batch_dim * sizeof(float *)));
//     gpuErrchk(cudaMalloc((void **)&d_B_arr, batch_dim * sizeof(float *)));
//     gpuErrchk(cudaMalloc((void **)&d_C_arr, batch_dim * sizeof(float *)));

//     for (int i = 0; i < batch_dim; i++) {
//         gpuErrchk(cudaMemcpy(d_A[i], A[i], a_size, cudaMemcpyHostToDevice));
//         gpuErrchk(cudaMemcpy(d_B[i], B[i], b_size, cudaMemcpyHostToDevice));
//         gpuErrchk(cudaMemcpy(d_C[i], C[i], c_size, cudaMemcpyHostToDevice));
//     }

//     gpuErrchk(cudaMemcpy(d_A_arr, d_A, batch_dim * sizeof(float *), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_B_arr, d_B, batch_dim * sizeof(float *), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_C_arr, d_C, batch_dim * sizeof(float *), cudaMemcpyHostToDevice));

//     const float alpha = 1.0f, beta = 0.0f;
//     cublasStatus_t cublas_result = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N
//                                        , b_cols, a_rows, b_rows
//                                        , &alpha, d_B_arr, b_cols, d_A_arr, b_rows
//                                        , &beta, d_C_arr, b_cols
//                                        , batch_dim);
//     assert(cublas_result == CUBLAS_STATUS_SUCCESS);

//     for (int i = 0; i < batch_dim; i++)
//     {
//         gpuErrchk(cudaMemcpy(C[i], d_C[i], c_size, cudaMemcpyDeviceToHost));
//         gpuErrchk(cudaFree(d_A[i]));
//         gpuErrchk(cudaFree(d_B[i]));
//         gpuErrchk(cudaFree(d_C[i]));
//     }
//     gpuErrchk(cudaFree(d_A_arr));
//     gpuErrchk(cudaFree(d_B_arr));
//     gpuErrchk(cudaFree(d_C_arr));
// }

#endif // __CUBLAS_BMM_KERNEL_H__
