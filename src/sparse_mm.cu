#ifndef __SPARSE_MM_
#define __SPARSE_MM_

#include <torch/extension.h>

#define T double

__global__ void kernel_CSR(T *B, T *C, int *rowdispl, int *colindex, T *nnzvalue, int M, int K) {
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < M){
    B += blockIdx.y*K;
    C += blockIdx.y*M;
    T acc = 0.0;
    for(int k = rowdispl[tid]; k < rowdispl[tid+1]; k++)
      acc += B[colindex[k]]*nnzvalue[k];
    C[tid] = acc;
  }
};

struct TiledSpMM_handle {
  int M;
  int N;
  int K;

  int *rowdispl;
  int *colindex;
  T *nnzvalue;
};

void TiledSpMM_inspect(int M, int N, int K, int *rowdispl, int *colindex, T *nnzvalue, TiledSpMM_handle *handle) {
  handle->M = M;
  handle->N = N;
  handle->K = K;

  int *rowdispl_d;
  int *colindex_d;
  T *nnzvalue_d;

  cudaMalloc(&rowdispl_d,(M+1)*sizeof(int));
  cudaMalloc(&colindex_d,rowdispl[M]*sizeof(int));
  cudaMalloc(&nnzvalue_d,rowdispl[M]*sizeof(T));

  cudaMemcpy(rowdispl_d,rowdispl,(M+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(colindex_d,colindex,rowdispl[M]*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(nnzvalue_d,nnzvalue,rowdispl[M]*sizeof(T),cudaMemcpyHostToDevice);

  handle->rowdispl = rowdispl_d;
  handle->colindex = colindex_d;
  handle->nnzvalue = nnzvalue_d;  
};

void TiledSpMM_multiply(T *B, T *C, TiledSpMM_handle handle) {
  int M = handle.M;
  int N = handle.N;
  int K = handle.K;
  int *rowdispl = handle.rowdispl;
  int *colindex = handle.colindex;
  T *nnzvalue = handle.nnzvalue;

  int blocksize = 32;
  dim3 grid((M+blocksize-1)/blocksize,N);
  dim3 block(blocksize);
  kernel_CSR<<<grid,block>>>(B, C, rowdispl, colindex, nnzvalue, M, K);
};

void TiledSpMM_free(TiledSpMM_handle handle) {
  cudaFree(handle.rowdispl);
  cudaFree(handle.colindex);
  cudaFree(handle.nnzvalue);
};

#endif // __SPARSE_MM_