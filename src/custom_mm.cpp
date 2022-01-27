/*
 Intermediate connection between python front-end and CUDA back-end
 Use this file mostly for forward declaration ands calling CUDA functions.
 The pybind11 linker is at the bottom of the file.

 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/
#include <iostream>
#include <assert.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDABlas.h>

// dummy kernel forward declaration
void dummy_kernel_launch();

// cublas mm forward declaration   
void cublas_mm_wrapper(cublasHandle_t handle,
                       torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
                       int a_rows, int a_cols, int b_rows, int b_cols, bool transa, bool transb);

// cublas bmm forward declaration
void cublas_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor A, torch::Tensor B, torch::Tensor C,
               size_t a_rows, size_t a_cols, size_t b_cols, size_t b_rows,
               size_t batch_dim, bool transa, bool transb);

// cusparse mm forward declaration
void cusparse_mm_wrapper(cusparseHandle_t handle,
                         float *dA_values, int *dA_columns, int *dA_csrOffsets,
                         int A_nnz, int A_num_rows, int A_num_cols,
                         torch::Tensor B, int B_num_rows, int B_num_cols,
                         torch::Tensor C);
void dense_to_csr(cusparseHandle_t handle, 
                  torch::Tensor dense, const int num_rows, const int num_cols,
                  float **d_csr_values, int **d_csr_columns, int **d_csr_offsets, int *nnz);

// cublasLT forward declaration
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m, int n, int k,
                   const float *A, int lda,
                   const float *B, int ldb,
                   float *C, int ldc);

// naive mm forward declaration
void naive_batched_matmul(
              torch::Tensor A, torch::Tensor B, torch::Tensor C,
              int a_rows, int a_cols, int b_rows, int b_cols,
              int batch_dim);

// manual handles in case none exist.
// initialization/destruction functions are near the bottom
cublasHandle_t g_cublas_handle = nullptr;
cusparseHandle_t g_cusparse_handle = nullptr;
cublasLtHandle_t g_cublaslt_handle = nullptr;
torch::Device device(torch::kCUDA);

torch::Tensor cublas_mmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, bool transa, bool transb)
{
  // handles 2d matrix multiplications with cuBLAS
  // remember that cuBLAS is column-major while torch passes in with row-major
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  // if (handle == nullptr && g_cublas_handle != nullptr) {
  //   handle = g_cublas_handle;
  // } else {
  //   throw std::invalid_argument("No initialized cuBLAS handle");
  // }

  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  cublas_mm_wrapper(handle, A, B, C, A_rows, A_cols, B_rows, B_cols, transa, transb);

  return C;
}

torch::Tensor cublas_bmm(torch::Tensor A, torch::Tensor B, torch::Tensor C, int dim, bool transa, bool transb)
{
  // handles batched matrix multiplications with cuBLAS
  // remember that cuBLAS is column-major while torch passes in with row-major
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  // if (handle == nullptr && g_cublas_handle != nullptr) {
  //   handle = g_cublas_handle;
  // } else {
  //   throw std::invalid_argument("No initialized cuBLAS handle");
  // }

  if (dim == 3) {
    int A_rows = A.size(1);
    int A_cols = A.size(2);
    int B_rows = B.size(1);
    int B_cols = B.size(2);

    int batch_dim = A.size(0);
    assert(batch_dim == B.size(0));
    cublas_bmm_wrapper(handle, A, B, C, A_rows, A_cols, B_cols, B_rows, batch_dim, transa, transb);
    return C;
  } else if (dim ==4) {
    int A_rows = A.size(2);
    int A_cols = A.size(3);
    int B_rows = B.size(2);
    int B_cols = B.size(3);

    int batch_dim1 = A.size(0);
    int batch_dim2 = A.size(1);
    assert(batch_dim1 == B.size(0));
    assert(batch_dim2 == B.size(1));

    cublas_bmm_wrapper(handle, A, B, C, A_rows, A_cols, B_cols, B_rows, batch_dim1 *batch_dim2, transa, transb);
    return C;
  } else if (dim == 2) {
    return cublas_mmul(A, B, C, transa, transb);
  } else {
    throw std::invalid_argument("Invalid dim argument.");
  }
}

torch::Tensor naive_bmm(torch::Tensor A, torch::Tensor B, torch::Tensor C, int dim)
{
  // handles matrix multiplications (2-d and batched) with a naive kernel
  if (dim == 3) {
    int A_rows = A.size(1);
    int A_cols = A.size(2);
    int B_rows = B.size(1);
    int B_cols = B.size(2);

    int batch_dim = A.size(0);
    assert(batch_dim == B.size(0));
    naive_batched_matmul(A, B, C, A_rows, A_cols, B_rows, B_cols, batch_dim);
    return C;
  } else if (dim == 4) {
    int A_rows = A.size(2);
    int A_cols = A.size(3);
    int B_rows = B.size(2);
    int B_cols = B.size(3);

    int batch_dim1 = A.size(0);
    int batch_dim2 = A.size(1);
    assert(batch_dim1 == B.size(0));
    assert(batch_dim2 == B.size(1));

    naive_batched_matmul(A, B, C, A_rows, A_cols, B_rows, B_cols, batch_dim1 * batch_dim2);
    return C;
  } else if (dim == 2) {
    int A_rows = A.size(0);
    int A_cols = A.size(1);
    int B_rows = B.size(0);
    int B_cols = B.size(1);

    naive_batched_matmul(A, B, C, A_rows, A_cols, B_rows, B_cols, 1);
  } else {
    throw std::invalid_argument("Invalid dim argument.");
  }
}


torch::Tensor cusparse_mmul(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes two dense matrices, and sparsifies A.
  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);;

  float *d_A_values = nullptr;
  int *d_A_columns = nullptr;
  int *d_A_offsets = nullptr;
  int nnzA = 0;

  dense_to_csr(g_cusparse_handle, A, A_rows, A_cols, &d_A_values, &d_A_columns, &d_A_offsets, &nnzA);

  cusparse_mm_wrapper(g_cusparse_handle, d_A_values, d_A_columns, d_A_offsets, nnzA,
                      A_rows, A_cols, B, B_rows, B_cols, C);

  return C;
}

torch::Tensor cublaslt_mmul(torch::Tensor A, torch::Tensor B)
{
  // uses cuBLASLt to do a 2-d matrix multiplications
  // currently is not working, do not use.

  throw std::logic_error("Function not yet implemented");

  float *A_arr = A.data_ptr<float>();
  float *B_arr = B.data_ptr<float>();

  int A_dim = A.dim();
  int B_dim = B.dim();
  int lda = A.size(0);
  int ldb = B.size(0);
  int ldc = B.size(0);
  int m = A.size(A_dim - 2);
  int k = A.size(A_dim - 1);
  int n = B.size(B_dim - 1);   

  torch::Tensor C = torch::zeros({m,n}, torch::kFloat32);
  float *C_arr = C.data_ptr<float>();

  LtIgemmTensor(g_cublaslt_handle, m, n, k, A_arr, lda, B_arr, ldb, C_arr, ldc);

  return C;
}

// handle initialization/destruction functions

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
  cusparseStatus_t status = cusparseCreate(&g_cusparse_handle);
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    std::cerr << "cuSPARSE initialization error.";
  }
}

void destroy_cusparse_handle() {
  cusparseStatus_t status = cusparseDestroy(g_cusparse_handle);
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    std::cerr << "Shutdown error!";
  }
}

void init_cublaslt_handle() {
  cublasStatus_t status = cublasLtCreate(&g_cublaslt_handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "cuBLASLt initialization error.";
  }
}

void destroy_cublaslt_handle() {
  cublasStatus_t status = cublasLtDestroy(g_cublaslt_handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "Shutdown error!";
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("init_cublas", &init_cublas_handle, "Create cuBLAS handle.");
  m.def("destroy_cublas", &destroy_cublas_handle, "Destroy cuBLAS handle.");
  m.def("init_cublaslt", &init_cublaslt_handle, "Create cuBLASLt handle.");
  m.def("destroy_cublaslt", &destroy_cublaslt_handle, "Destroy cuBLASLt handle.");
  m.def("init_cusparse", &init_cusparse_handle, "Create cuSPARSE handle.");
  m.def("destroy_cusparse", &destroy_cusparse_handle, "Destroy cuSPARSE handle.");
  m.def("cublas_mmul", &cublas_mmul, "cuBLAS Torch Matrix Multiplication");
  m.def("cublas_bmm", &cublas_bmm, "cuBLAS Batched Torch Matrix Multiplication");
  m.def("cusparse_mmul", &cusparse_mmul, "cuSPARSE Torch Matrix Multiplication");
  m.def("cublaslt_mmul", &cublaslt_mmul, "cuBLASLt Torch Matrix Multiplication");
  m.def("dummy_kernel", &dummy_kernel_launch, "Launch dummy kernel.");
  m.def("naive_bmm", &naive_bmm, "A naive implementation of Batched Matrix Multiplication");
}
