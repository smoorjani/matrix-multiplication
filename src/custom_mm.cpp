/*
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

// dummy kernel forward declaration
void dummy_kernel_launch();

// cublas mm forward declaration   
void cublas_mm_wrapper(cublasHandle_t handle,
                       torch::Tensor d_A, torch::Tensor d_B, torch::Tensor d_C,
                       int m, int k, int n);

// cublas bmm forward declaration
void cublas_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor A, torch::Tensor B, torch::Tensor C,
               size_t a_rows, size_t b_cols, size_t b_rows,
               size_t batch_dim);
void cublas_4d_bmm_wrapper(cublasHandle_t handle,
               torch::Tensor A, torch::Tensor B, torch::Tensor C,
               size_t a_rows, size_t b_cols, size_t b_rows,
               size_t batch_dim1, size_t batch_dim2);

// cusparse mm forward declaration
void cusparse_mm_wrapper(cusparseHandle_t handle,
                         double *h_A, int *h_A_ColIndices, int *h_A_RowIndices,
                         int nnzA, int h_A_rowptr_size,
                         double *h_B_dense, int h_B_rows, int h_B_cols,
                         double *h_C_dense);
void dense_to_csr(cusparseHandle_t handle, 
                  double *h_A_dense, const int Nrows, const int Ncols,
                  double **h_A_val, int **h_A_colind, int **h_A_rowptr, int *nnzA);

// cublasLT forward declaration
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m, int n, int k,
                   const float *A, int lda,
                   const float *B, int ldb,
                   float *C, int ldc);

cublasHandle_t g_cublas_handle = nullptr;
cusparseHandle_t g_cusparse_handle = nullptr;
cublasLtHandle_t g_cublaslt_handle = nullptr;
torch::Device device(torch::kCUDA);

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

torch::Tensor cublas_mmul(torch::Tensor A, torch::Tensor B)
{
  // torch passes in with column major
  // the current

  float *A_arr = A.data_ptr<float>();
  float *B_arr = B.data_ptr<float>();

  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_cols = B.size(1);

  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kFloat32);
  float *C_arr = C.data_ptr<float>();

  cublas_mm_wrapper(g_cublas_handle, A, B, C, A_rows, A_cols, B_cols);

  return C;
}

torch::Tensor cublas_bmm(torch::Tensor A, torch::Tensor B, int dim)
{
  if (dim == 3) {
    int A_rows = A.size(1);
    int B_rows = B.size(1);
    int B_cols = B.size(2);

    int batch_dim = A.size(0);
    assert(batch_dim == B.size(0));
    torch::Tensor C = torch::zeros({batch_dim, A_rows, B_cols}, torch::kFloat32).contiguous();
    //C = C.to(device);
    std::cout << "A C++ Before: " << A.device() << std::endl;
    std::cout << "B C++ Before: " << B.device() << std::endl;
    std::cout << "C C++ Before: " << C.device() << std::endl;
    cublas_bmm_wrapper(g_cublas_handle, A, B, C, A_rows, B_cols, B_rows, batch_dim);
    std::cout << "A C++ After: " << A.device() << std::endl;
    std::cout << "B C++ After: " << B.device() << std::endl;
    std::cout << "C C++ After: " << C.device() << std::endl;
    return C; //return C;
  } else if (dim ==4) {
    int A_rows = A.size(2);
    int B_rows = B.size(2);
    int B_cols = B.size(3);

    int batch_dim1 = A.size(0);
    int batch_dim2 = A.size(1);
    assert(batch_dim1 == B.size(0));
    assert(batch_dim2 == B.size(1));

    torch::Tensor C = torch::zeros({batch_dim1, batch_dim2, A_rows, B_cols}, torch::kFloat32).contiguous();
    //C = C.to(device);
    std::cout << "A C++ Before: " << A.device() << std::endl;
    std::cout << "B C++ Before: " << B.device() << std::endl;
    std::cout << "C C++ Before: " << C.device() << std::endl;
    cublas_4d_bmm_wrapper(g_cublas_handle, A, B, C, A_rows, B_cols, B_rows, batch_dim1, batch_dim2);
    std::cout << "A C++ After: " << A.device() << std::endl;
    std::cout << "B C++ After: " << B.device() << std::endl;
    std::cout << "C C++ After: " << C.device() << std::endl;
    return C; //return C;
  } else if (dim == 2) {
    return cublas_mmul(A, B);
  } else {
    throw std::invalid_argument("Invalid dim argument.");
  }
}

torch::Tensor cusparse_mmul(torch::Tensor A, torch::Tensor B)
{
  double *A_arr = A.data_ptr<double>();
  double *B_arr = B.data_ptr<double>();

  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);;

  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::kDouble);
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


  return C;
}

torch::Tensor cublaslt_mmul(torch::Tensor A, torch::Tensor B)
{
  // torch passes in with column major
  // the current

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
}
