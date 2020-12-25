/*
 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/

#include <torch/extension.h>
#include "cublas_mm_kernel.h"

torch::Tensor row_major_mmul(torch::Tensor A, torch::Tensor B) {

    float* A_arr = A.data_ptr<float>();
    float* B_arr = B.data_ptr<float>();

    int A_rows = A.size(0);
    int A_cols = A.size(1);
    int B_rows = B.size(0);
    int B_cols = B.size(1);

    torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
    int C_rows = C.size(0);
    int C_cols = C.size(1);

    float* C_arr = C.data_ptr<float>();

    mmul_wrapper(A_arr, B_arr, C_arr, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols);
    return C;
}

torch::Tensor torch_mmul(torch::Tensor A, torch::Tensor B) {
    // torch passes in with column major
    // the current 

    auto A_tensor = torch::transpose(A, 0, 1);
    auto B_tensor = torch::transpose(B, 0, 1);

    float* A_arr = B_tensor.data_ptr<float>();
    float* B_arr = A_tensor.data_ptr<float>();

    int A_rows = B_tensor.size(0);
    int A_cols = B_tensor.size(1);
    int B_rows = A_tensor.size(0);
    int B_cols = A_tensor.size(1);

    torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
    int C_rows = C.size(0);
    int C_cols = C.size(1);

    float* C_arr = C.data_ptr<float>();

    mmul_wrapper(A_arr, B_arr, C_arr, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols);
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mmul", &torch_mmul, "Random Matrix Multiplication");
}
