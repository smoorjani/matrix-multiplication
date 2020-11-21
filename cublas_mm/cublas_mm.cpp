/*
 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/

#include <torch/extension.h>

// Remove this for python setup.py install
#include "cublas_mm_kernel.cu"


void print_matrix(const float *A, int A_rows, int A_cols);
void fill_rand(float *A, int A_rows, int A_cols);
void mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
void random_mmul();


void random_mmul_wrapper() {
    random_mmul();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mmul", &random_mmul_wrapper, "Random Matrix Multiplication");
}
