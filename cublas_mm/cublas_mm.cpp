/*
 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/

#include <torch/extension.h>

// Remove this for python setup.py install
#include "cublas_mm_kernel.h"


void print_matrix(const float *A, int A_rows, int A_cols);
// void fill_rand(float *A, int A_rows, int A_cols);
void mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
void mmul_wrapper(const float *A, const float *B, float *C);


torch::Tensor torch_mmul(torch::Tensor A, torch::Tensor B) {
    float* A_arr = A.data_ptr<float>();
    float* B_arr = B.data_ptr<float>();

    int A_rows = A.size(0);
    int A_cols = A.size(1);
    int B_cols = B.size(1);

    torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
    float* C_arr = C.data_ptr<float>();

    mmul(A_arr, B_arr, C_arr, A_rows, A_cols, B_cols);
    print_matrix(C_arr. C.size(0), C.size(1));
    // C = torch::from_blob(C_arr.ptr<float>(), {A_rows, B_cols});
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mmul", &torch_mmul, "Random Matrix Multiplication");
}
