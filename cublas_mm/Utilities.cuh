#ifndef __UTILITIES_KERNEL_H__
#define __UTILITIES_KERNEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) { exit(code); }
   }
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

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
        fprintf(stderr, "CUSPARSE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs",__FILE__, __LINE__,err, \
                                _cusparseGetErrorEnum(err)); \
        cudaDeviceReset(); assert(0); \
    }
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

#endif // __UTILITIES_KERNEL_H__