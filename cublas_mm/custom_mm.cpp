/*
 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/

#include <torch/extension.h>
#include "custom_mm_kernel.h"
#include "cusparse_mm_kernel.h"
#include "cublas_mm_kernel.h"

torch::Tensor row_major_mmul(torch::Tensor A, torch::Tensor B)
{

  float *A_arr = A.data_ptr<float>();
  float *B_arr = B.data_ptr<float>();

  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
  int C_rows = C.size(0);
  int C_cols = C.size(1);

  float *C_arr = C.data_ptr<float>();

  mmul_wrapper(A_arr, B_arr, C_arr, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols);
  return C;
}

torch::Tensor torch_mmul(torch::Tensor A, torch::Tensor B)
{
  // torch passes in with column major
  // the current

  auto A_tensor = torch::transpose(A, 0, 1);
  auto B_tensor = torch::transpose(B, 0, 1);

  float *A_arr = B_tensor.data_ptr<float>();
  float *B_arr = A_tensor.data_ptr<float>();

  int A_rows = B_tensor.size(0);
  int A_cols = B_tensor.size(1);
  int B_rows = A_tensor.size(0);
  int B_cols = A_tensor.size(1);

  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
  int C_rows = C.size(0);
  int C_cols = C.size(1);

  float *C_arr = C.data_ptr<float>();

  mmul_wrapper(A_arr, B_arr, C_arr, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols);
  return C;
}

torch::Tensor cublas_mmul(torch::Tensor B, torch::Tensor A)
{
  // torch passes in with column major
  // the current

  // row major?
  auto A_tensor = torch::transpose(A, 0, 1);
  auto B_tensor = torch::transpose(B, 0, 1);

  float *A_arr = A_tensor.data_ptr<float>();
  float *B_arr = B_tensor.data_ptr<float>();

  int A_rows = A_tensor.size(0);
  int A_cols = A_tensor.size(1);
  int B_rows = B_tensor.size(0);
  int B_cols = B_tensor.size(1);

  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
  int C_rows = C.size(0);
  int C_cols = C.size(1);

  float *C_arr = C.data_ptr<float>();

  cublas_mm_wrapper(A_arr, A_rows, A_cols, B_arr, B_rows, B_cols, C_arr);
  auto accessor = C.accessor<float,2>();

  for (int i = 0; i < C_rows; i++) {
    for (int j = 0; j < C_cols; j++) {
      accessor[i][j] = C_arr[i * C_rows + j];
    }
  }

  /*
  for (int i = 0; i < C_rows; i++)
  {
    for (int j = 0; j < C_cols; j++)
      printf("%f \t", C_arr[i * C_rows + j]);
    printf("\n");
  }
  */

  return C;
}

torch::Tensor cusparse_mmul(torch::Tensor A, torch::Tensor B)
{
  // torch passes in with column major
  // the current

  // convert to double
  double *A_arr = A.data_ptr<double>();
  double *B_arr = B.data_ptr<double>();

  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
  // int C_rows = C.size(0);
  // int C_cols = C.size(1);

  // double *C_arr = C.data_ptr<double>();

  double *h_A_val = nullptr;
  int *h_A_colind = nullptr;
  int *h_A_rowptr = nullptr;
  int nnzA = 0;
  int h_A_rowptr_size = A_rows + 1;

  dense_to_csr(A_arr, A_rows, A_cols, &h_A_val, &h_A_colind, &h_A_rowptr, &nnzA);

  cusparse_mm_wrapper(h_A_val, h_A_colind, h_A_rowptr,
                      nnzA, h_A_rowptr_size, B_arr, B_rows, B_cols);

  // TODO: fill values of C in
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("mmul", &torch_mmul, "Torch Matrix Multiplication");
  m.def("cublas_mmul", &cublas_mmul, "cuBLAS Torch Matrix Multiplication");
  m.def("cusparse_mmul", &cusparse_mmul, "cuSPARSE Torch Matrix Multiplication");
}
