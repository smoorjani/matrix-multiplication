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
void free_csr(float *d_csr_values, int *d_csr_columns, int *d_csr_offsets);

// naive mm forward declaration
void naive_batched_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                          int a_rows, int a_cols, int b_rows, int b_cols,
                          int batch_dim);

void naive_spmm_wrapper(float *dA_values, int *dA_columns, int *dA_csrOffsets,
				int nnzA, int A_rows, int A_cols,
				torch::Tensor B, int B_rows, int B_cols,
				torch::Tensor C);

// tiledspmm naive mm forward declaration
void TiledSpMM_inspect(int M, int N, int K, int *rowdispl, int *colindex, T *nnzvalue, TiledSpMM_handle *handle);
void TiledSpMM_multiply(T *B, T *C, TiledSpMM_handle handle);
void TiledSpMM_free(TiledSpMM_handle handle);

// manual handles in case none exist.
// initialization/destruction functions are near the bottom
cublasHandle_t g_cublas_handle = nullptr;
cusparseHandle_t g_cusparse_handle = nullptr;
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

torch::Tensor naive_spmm(torch::Tensor A_values, torch::Tensor A_columns, torch::Tensor A_offsets, int nnzA, int A_rows, int A_cols, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes two dense matrices, and sparsifies A.
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  float *d_A_values = A_values.data_ptr<float>();
  int *d_A_columns = A_columns.data_ptr<int>();
  int *d_A_offsets = A_offsets.data_ptr<int>();

  naive_spmm_wrapper(d_A_values, d_A_columns, d_A_offsets, nnzA, A_rows, A_cols, B, B_rows, B_cols, C);
  return C;
}

torch::Tensor naive_spmm_csr_conversion(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes two dense matrices, and sparsifies A.
  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  float *d_A_values = nullptr;
  int *d_A_columns = nullptr;
  int *d_A_offsets = nullptr;
  int nnzA = 0;

  dense_to_csr(g_cusparse_handle, A, A_rows, A_cols, &d_A_values, &d_A_columns, &d_A_offsets, &nnzA);

  naive_spmm_wrapper(d_A_values, d_A_columns, d_A_offsets, nnzA, A_rows, A_cols, B, B_rows, B_cols, C);

  free_csr(d_A_values, d_A_columns, d_A_offsets);
  return C;
}

torch::Tensor cusparse_mmul(torch::Tensor A_values, torch::Tensor A_columns, torch::Tensor A_offsets, int nnzA, int A_rows, int A_cols, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes a sparse A and a dense B.
  int B_rows = B.size(0);
  int B_cols = B.size(1);
  
  float *d_A_values = A_values.data_ptr<float>();
  int *d_A_columns = A_columns.data_ptr<int>();
  int *d_A_offsets = A_offsets.data_ptr<int>();

  cusparse_mm_wrapper(g_cusparse_handle, d_A_values, d_A_columns, d_A_offsets, nnzA,
                      A_rows, A_cols, B, B_rows, B_cols, C);
  return C;
}

torch::Tensor cusparse_mmul_csr_conversion(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with cuSPARSE
  // this function takes two dense matrices, and sparsifies A.
  int A_rows = A.size(0);
  int A_cols = A.size(1);
  int B_rows = B.size(0);
  int B_cols = B.size(1);

  float *d_A_values = nullptr;
  int *d_A_columns = nullptr;
  int *d_A_offsets = nullptr;
  int nnzA = 0;

  dense_to_csr(g_cusparse_handle, A, A_rows, A_cols, &d_A_values, &d_A_columns, &d_A_offsets, &nnzA);

  cusparse_mm_wrapper(g_cusparse_handle, d_A_values, d_A_columns, d_A_offsets, nnzA,
                      A_rows, A_cols, B, B_rows, B_cols, C);
   
  free_csr(d_A_values, d_A_columns, d_A_offsets);
  return C;
}

/*
The following methods use the TiledSpMM interface for matrix multiplication
You must do an inspection step where you create the handle for each unique matrix
Note that it uses the following convention:
C = M x K
A = M x N
B = N x K
*/

struct TiledSpMM_handle* spmm_handle = nullptr;

void tiledspmm_inspect_naive(torch::Tensor A_values, torch::Tensor A_columns, atorch::Tensor A_offsets, int M, int N, int K)
{
  // handles 2d sparse-dense matrix multiplications with the TiledSpMM interface
  // THIS IS A NAIVE METHOD, NOT THE FULL TILEDSPMM
  // this function takes a sparse A and dimensions to create a handle.

  // M = A_rows
  // N = A_cols
  // K = B_cols
  
  float *d_A_values = A_values.data_ptr<float>();
  int *d_A_columns = A_columns.data_ptr<int>();
  int *d_A_offsets = A_offsets.data_ptr<int>();

  spmm_handle = new TiledSpMM_handle;  

  TiledSpMM_inspect(M, N, K, d_A_offsets, d_A_columns, d_A_values, spmm_handle);
}

torch::Tensor tiledspmm_mm_naive(torch::Tensor B, torch::Tensor C)
{
  // handles 2d sparse-dense matrix multiplications with the TiledSpMM interface
  // THIS IS A NAIVE METHOD, NOT THE FULL TILEDSPMM
  // this function takes a dense B and uses stored handle for A.
  float *d_B = B.data_ptr<float>();
  float *d_C = C.data_ptr<float>();

  TiledSpMM_multiply(d_B, d_C, spmm_handle);
  return C;
}

void clean_tiledspmm_handle()
{
  // handles 2d sparse-dense matrix multiplications with the TiledSpMM interface
  // THIS IS A NAIVE METHOD, NOT THE FULL TILEDSPMM
  // this function frees the handle for A

  TiledSpMM_free(spmm_handle);
  delete spmm_handle;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("init_cublas", &init_cublas_handle, "Create cuBLAS handle.");
  m.def("destroy_cublas", &destroy_cublas_handle, "Destroy cuBLAS handle.");

  m.def("init_cusparse", &init_cusparse_handle, "Create cuSPARSE handle.");
  m.def("destroy_cusparse", &destroy_cusparse_handle, "Destroy cuSPARSE handle.");

  m.def("cublas_mmul", &cublas_mmul, "cuBLAS Torch Matrix Multiplication");
  m.def("cublas_bmm", &cublas_bmm, "cuBLAS Batched Torch Matrix Multiplication");
  m.def("cusparse_mmul", &cusparse_mmul, "cuSPARSE Torch Matrix Multiplication");

  m.def("dummy_kernel", &dummy_kernel_launch, "Launch dummy kernel.");
  m.def("naive_spmm", &naive_spmm, "A naive implementation of Sparse Matrix Multiplication");

  m.def("tiledspmm_naive_inspect", &tiledspmm_inspect_naive, "Inspect function for Naive CSR MM");
  m.def("tiledspmm_naive_mm", &tiledspmm_mm_naive, "MM function for Naive CSR MM");
  m.def("tiledspmm_naive_clean", &clean_tiledspmm_handle, "Cleanup function for Naive CSR MM");
}
