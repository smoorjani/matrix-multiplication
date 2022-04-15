/*
 Intermediate connection between python front-end and CUDA back-end
 Use this file mostly for forward declaration ands calling CUDA functions.
 The pybind11 linker is at the bottom of the file.

 For full documentation:
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
*/
#include <iostream>
#include <assert.h>
#include <unordered_map>
#include <string>
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

void torch_cusparse_mm_wrapper(cusparseHandle_t handle,
                         float *dA_values, int *dA_columns, int *dA_csrOffsets,
                         int A_nnz, int A_num_rows, int A_num_cols,
                         torch::Tensor B, int B_num_rows, int B_num_cols,
                         torch::Tensor C);

// naive mm forward declaration
void naive_batched_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                          int a_rows, int a_cols, int b_rows, int b_cols,
                          int batch_dim);

void naive_spmm_wrapper(float *dA_values, int *dA_columns, int *dA_csrOffsets,
				int nnzA, int A_rows, int A_cols,
				torch::Tensor B, int B_rows, int B_cols,
				torch::Tensor C);

#define T float
 //TiledSpMM HANDLE STRUCTURE
struct TiledSpMM_handle {
  int M;
  int N;
  int K;

  int numblock;
  int tilesize;
  int *tiledispl;
  int *mapdispl;
  int *mapindex;
  int *elldispl;
  unsigned short *ellindex;
  T *ellvalue;

  dim3 grid;
  dim3 block;
};

int handle_len = 0;
TiledSpMM_handle *handle = nullptr;
std::unordered_map<std::string, int> layer_lookup;

// tiledspmm naive mm forward declaration
void TiledSpMM_coo2csr(int M, int N, long nnz,
                       int *coo_rowidx, int *coo_colidx, T *coo_value,
                       long **csr_displ, long **csr_index, T **csr_value);

void TiledSpMM_inspect(int M, int N, int K,
                       long *csr_displ, long *csr_index,
                       T *csr_value, TiledSpMM_handle *handle);

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

 // CuSPARSE HANDLE STRUCTURE
struct CuSPARSE_handle {
  int M;
  int N;
  int K;

  long *displ;
  long *colind;
  float *values;

  int nnz;
};

int cusparse_handle_len = 0;
CuSPARSE_handle *cu_handle = nullptr;
std::unordered_map<std::string, int> cusparse_layer_lookup;

void cusparse_inspect(torch::Tensor displ, torch::Tensor colindex, torch::Tensor value, int nnz, int M, int N, int K, std::string layer)
{
  long *_displ = displ.data_ptr<long>();
  long *_colindex = colindex.data_ptr<long>();
  float *_value = value.data_ptr<float>();

  CuSPARSE_handle *cur_handle = new CuSPARSE_handle;

  cur_handle->M = M;
  cur_handle->N = N;
  cur_handle->K = K;
  cur_handle->nnz = nnz;

  cur_handle->displ = _displ;
  cur_handle->colind = _colindex;
  cur_handle->values = _value;

  cusparse_handle_len++;
  cu_handle = (CuSPARSE_handle*) realloc(cu_handle, cusparse_handle_len * sizeof(CuSPARSE_handle));
  cu_handle[cusparse_handle_len - 1] = *cur_handle;
  cusparse_layer_lookup[layer] = cusparse_handle_len - 1;
}

torch::Tensor cusparse_mmul_opt(torch::Tensor B, torch::Tensor C, std::string layer) {
  int handle_id = cusparse_layer_lookup[layer];
  if (handle_id > cusparse_handle_len) {
    throw std::runtime_error("Invalid handle_id!");
  }

  CuSPARSE_handle cur_handle = cu_handle[handle_id];

  torch_cusparse_mm_wrapper(g_cusparse_handle, cur_handle.values, cur_handle.colind, cur_handle.displ, cur_handle.nnz,
                      cur_handle.M, cur_handle.K, B, cur_handle.K, cur_handle.N, C);
  return C;
}
    
void cusparse_free() {
    for (int i = 0; i < cusparse_handle_len; i++) {
      cudaFree(cu_handle[i].displ);
      cudaFree(cu_handle[i].colind);
      cudaFree(cu_handle[i].values);
    }
    free(cu_handle);
    cusparse_handle_len = 0;
    cu_handle = nullptr;
}    

/*
The following methods use the TiledSpMM interface for matrix multiplication
You must do an inspection step where you create the handle for each unique matrix
Note that it uses the following convention:
C = M x K
A = M x N
B = N x K
*/

// interfacing with torch
void torch_TiledSpMM_inspect_coo(int M, int N, int K, long nnz,
                             torch::Tensor rowptr, torch::Tensor colptr,
                             torch::Tensor value, std::string layer) {
    int *_rowptr = rowptr.data_ptr<int>();
    int *_colptr = colptr.data_ptr<int>();
    T *_value = value.data_ptr<T>();

    //CSR DATA STRUCTURES
    long *csr_displ;
    long *csr_index;
    T *csr_value;

    //CONVERT COO->CSR
    TiledSpMM_coo2csr(M, N, nnz, _rowptr, _colptr, _value, &csr_displ, &csr_index, &csr_value);
    
    TiledSpMM_handle *cur_handle = new TiledSpMM_handle;
    TiledSpMM_inspect(M, N, K, csr_displ, csr_index, csr_value, cur_handle);

    delete[] csr_displ;
    delete[] csr_index;
    delete[] csr_value;

    handle_len++;
    handle = (TiledSpMM_handle*) realloc(handle, handle_len * sizeof(TiledSpMM_handle));
    handle[handle_len - 1] = *cur_handle;
    layer_lookup[layer] = handle_len - 1;
}

void torch_TiledSpMM_inspect_csr(int M, int N, int K,
                             torch::Tensor displ, torch::Tensor colindex,
                             torch::Tensor value, std::string layer) {
    long *_displ = displ.data_ptr<long>();
    long *_colindex = colindex.data_ptr<long>();
    T *_value = value.data_ptr<T>();
    
    TiledSpMM_handle *cur_handle = new TiledSpMM_handle;
    TiledSpMM_inspect(M, N, K, _displ, _colindex, _value, cur_handle);

    handle_len++;
    handle = (TiledSpMM_handle*) realloc(handle, handle_len * sizeof(TiledSpMM_handle));
    handle[handle_len - 1] = *cur_handle;
    layer_lookup[layer] = handle_len - 1;
}

void torch_TiledSpMM_multiply(torch::Tensor B, torch::Tensor C, std::string layer) {
    int handle_id = layer_lookup[layer];
    if (handle_id > handle_len) {
    	throw std::runtime_error("Invalid handle_id!");
    }

    TiledSpMM_handle cur_handle = handle[handle_id];
    T *_B = B.data_ptr<T>();
    T *_C = C.data_ptr<T>();

    TiledSpMM_multiply(_B, _C, cur_handle);
}

void torch_TiledSpMM_free() {
    for (int i = 0; i < handle_len; i++) {
      TiledSpMM_free(handle[i]);
    }
    free(handle);
    handle_len = 0;
    handle = nullptr;
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

  m.def("tiledspmm_inspect_csr", &torch_TiledSpMM_inspect_csr, "Inspect function for TiledSpMM with CSR input");
  m.def("tiledspmm_inspect_coo", &torch_TiledSpMM_inspect_coo, "Inspect function for TiledSpMM with COO input");
  m.def("tiledspmm_mm", &torch_TiledSpMM_multiply, "MM function for TiledSpMM");
  m.def("tiledspmm_clean", &torch_TiledSpMM_free, "Cleanup function for TiledSpMM");

  m.def("cusparse_inspect", &cusparse_inspect, "Inspect function for CuSPARSE with CSR input");
  m.def("cusparse_mmul_opt", &cusparse_mmul_opt, "MM function for CuSPARSE");
  m.def("cusparse_clean", &cusparse_free, "Cleanup function for CuSPARSE");
}
