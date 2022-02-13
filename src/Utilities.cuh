#ifndef __UTILITIES_KERNEL_H__
#define __UTILITIES_KERNEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>


/********************/
/* CUDA ERROR CHECK */
/********************/
inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

/******************/
/* ARRAY PRINTING */
/******************/

void print_arr(float *arr, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", arr[i * n + j]);
        }
        printf("\n");
    }
}

void print_arr_ptr(float **arr, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", arr[i][j]);
        }
        printf("\n");
    }
}

template <typename T>
void cuda_print_arr(T *d_arr, int m, int n) {
    T *h_arr = (T *) malloc(m * n * sizeof(T));
    checkCudaStatus(cudaMemcpy(h_arr, d_arr, m * n * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            //printf("%.4f ", h_arr[i * n + j]);
	    std::cout << h_arr[i * n + j] << " ";
        }
        printf("\n");
    }
}

void cuda_print_batched_arr(float *d_arr, int batch_size, int m, int n) {
    float *h_arr = (float *) malloc(batch_size * m * n * sizeof(float));
    checkCudaStatus(cudaMemcpy(h_arr, d_arr, batch_size * m * n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int k = 0; k < batch_size; k++) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", h_arr[k * m * n + i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    }
}

/**********************/
/* CUBLAS ERROR CHECK */
/**********************/

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED"; 
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR"; 
    }
    return "unknown error";
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        printf("%s\n", cublasGetErrorString(status));
        throw std::logic_error("cuBLAS API failed");
    }
}

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {

        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

        case CUSPARSE_STATUS_ZERO_PIVOT:
            return "CUSPARSE_STATUS_ZERO_PIVOT";
    }

    return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    if(CUSPARSE_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUSPARSE error in file '%s', line %Ndims\nobjs %d\nerror %Ndims: %d, %s\nterminating!\n",__FILE__, __LINE__,err, \
                                _cusparseGetErrorEnum(err)); \
    //    cudaDeviceReset(); assert(0); 
    }
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

#endif // __UTILITIES_KERNEL_H__
