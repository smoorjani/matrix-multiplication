/*
 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/

#include <assert.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "Utilities.cuh"
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "cusparse_mm_kernel.h"
#include "cublas_mm_kernel.h"

// TODO: add handle initialization for cusparse
cublasHandle_t g_cublas_handle = nullptr;
cusparseHandle_t g_cusparse_handle = nullptr;

void init_cublas_handle() {
  cublasStatus_t status = cublasCreate(&g_cublas_handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "cuBLAS initialization error.";
  }
}

void destroy_cublas_handle() {
  cublasStatus_t status = cublasDestroy(g_cublas_handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "Shutdown error!";
  }
}

void init_cusparse_handle() {
  cusparseSafeCall(cusparseCreate(&g_cusparse_handle));
}

void destroy_cusparse_handle() {
  cusparseSafeCall(cusparseDestroy(g_cusparse_handle));
}

torch::Tensor cublas_mmul(torch::Tensor B, torch::Tensor A)
{
  // torch passes in with column major
  // the current

  auto A_tensor = torch::transpose(A, 0, 1);
  auto B_tensor = torch::transpose(B, 0, 1);

  float *A_arr = A_tensor.data_ptr<float>();
  float *B_arr = B_tensor.data_ptr<float>();

  int A_rows = A_tensor.size(0);
  int A_cols = A_tensor.size(1);
  int B_rows = B_tensor.size(0);
  int B_cols = B_tensor.size(1);

  torch::Tensor C = torch::zeros({B_cols, A_rows}, torch::kFloat32);
  int C_rows = C.size(0);
  int C_cols = C.size(1);

  float *C_arr = C.data_ptr<float>();

  cublas_mm_wrapper(g_cublas_handle, A_arr, A_rows, A_cols, B_arr, B_rows, B_cols, C_arr);
  auto accessor = C.accessor<float, 2>();

  for (int i = 0; i < C_rows; i++)
  {
    for (int j = 0; j < C_cols; j++)
    {
      accessor[i][j] = C_arr[i * C_cols + j];
    }
  }

  return C;
}

torch::Tensor cublas_bmm(torch::Tensor B, torch::Tensor A, int dim)
{
  // torch passes in with column major
  // the current
  torch::Tensor A_tensor;
  torch::Tensor B_tensor;
  torch::Tensor C;

  int A_rows;
  int A_cols;
  int B_rows;
  int B_cols;
  int C_rows;
  int C_cols;
  int batch_dim;

  if (dim == 3) {
    A_tensor = torch::transpose(A, 1, 2);
    B_tensor = torch::transpose(B, 1, 2);

    A_rows = A_tensor.size(1);
    A_cols = A_tensor.size(2);
    B_rows = B_tensor.size(1);
    B_cols = B_tensor.size(2);

    batch_dim = A_tensor.size(0);
    assert(batch_dim == B_tensor.size(0));

    torch::Tensor C = torch::zeros({batch_dim, B_cols, A_rows}, torch::kFloat32);
    int C_rows = C.size(0);
    int C_cols = C.size(1);

    float *A_arr = A_tensor.data_ptr<float>();
    float *B_arr = B_tensor.data_ptr<float>();
    float *C_arr = C.data_ptr<float>();

    cublas_bmm_wrapper(g_cublas_handle, A_arr, A_rows, A_cols, B_arr, B_rows, B_cols, C_arr, batch_dim);
    auto accessor = C.accessor<float, 3>();

    for (int b = 0; b < batch_dim; b++) {
      for (int i = 0; i < C_rows; i++)
      {
        for (int j = 0; j < C_cols; j++)
        {
          accessor[b][i][j] = C_arr[i * C_cols + j];
        }
      }
    }
    
    return C;
  } 
}

torch::Tensor cusparse_mmul(torch::Tensor B, torch::Tensor A)
{
  // torch passes in with column major
  // the current

  auto A_tensor = torch::transpose(A, 0, 1);
  auto B_tensor = torch::transpose(B, 0, 1);
  
  double *A_arr = A_tensor.data_ptr<double>();
  double *B_arr = B_tensor.data_ptr<double>();

  int A_rows = A_tensor.size(0);
  int A_cols = A_tensor.size(1);
  int B_rows = B_tensor.size(0);
  int B_cols = B_tensor.size(1);

  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kDouble);
  int C_rows = C.size(0);
  int C_cols = C.size(1);

  double *C_arr = C.data_ptr<double>();

  double *h_A_val = nullptr;
  int *h_A_colind = nullptr;
  int *h_A_rowptr = nullptr;
  int nnzA = 0;
  int h_A_rowptr_size = A_rows + 1;

  dense_to_csr(g_cusparse_handle, A_arr, A_rows, A_cols, &h_A_val, &h_A_colind, &h_A_rowptr, &nnzA);

  for (int i = 0; i < nnzA; i++)
  {
    h_A_colind[i] += 1;
  }

  for (int i = 0; i < h_A_rowptr_size; i++)
  {
    h_A_rowptr[i] += 1;
  }

  cusparse_mm_wrapper(g_cusparse_handle, h_A_val, h_A_colind, h_A_rowptr, nnzA,
                      h_A_rowptr_size, B_arr, B_rows, B_cols, C_arr);

  auto accessor = C.accessor<double, 2>();

  for (int i = 0; i < C_rows; i++)
  {
    for (int j = 0; j < C_cols; j++)
    {
      accessor[i][j] = C_arr[i * C_rows + j];
    }
  }

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("init_cublas", &init_cublas_handle, "Create cuBLAS handle.");
  m.def("destroy_cublas", &destroy_cublas_handle, "Destroy cuBLAS handle.");
  m.def("init_cusparse", &init_cusparse_handle, "Create cuSPARSE handle.");
  m.def("destroy_cusparse", &destroy_cusparse_handle, "Destroy cuSPARSE handle.");
  m.def("cublas_mmul", &cublas_mmul, "cuBLAS Torch Matrix Multiplication");
  m.def("cublas_bmm", &cublas_bmm, "cuBLAS Batched Torch Matrix Multiplication");
  m.def("cusparse_mmul", &cusparse_mmul, "cuSPARSE Torch Matrix Multiplication");
}
