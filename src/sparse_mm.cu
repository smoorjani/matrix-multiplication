#ifndef __SPARSE_MM_H_
#define __SPARSE_MM_H_

#include <iostream>
#include <array>
#include <torch/extension.h>
//#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <sys/time.h>

void csr_preproc(torch::Tensor row_ind, torch::Tensor col_ind, torch::Tensor val) {
    
}

#endif // __SPARSE_MM_H_