// Taken from https://github.com/salehjg/batch-matmul-cuda
#ifndef __NAIVE_SPARSE_MM_H_
#define __NAIVE_SPARSE_MM_H_

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <torch/extension.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#include "naive_reducer.cuh"

#define THREADS 256
#define FULL_MASK 0xffffffff


// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, ReductionType REDUCE, bool HAS_VALUE>
__global__ void spmm_kernel(const int *rowptr_data, const int *col_data,
                            const scalar_t *value_data,
                            const scalar_t *mat_data, scalar_t *out_data,
                            int64_t *arg_out_data, int B, int M, int N, int K) {

  // We ignore blockIdx.y here, because threads
  // across `blockIdx.y` are treated equally.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int row = thread_idx >> 5;            // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1); // thread_idx % 32
  int batch_idx = row / M;

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + mat_col_idx;

  // Helper arrays for warp communication.
  int mat_row, mat_rows[32];
  scalar_t val, vals[HAS_VALUE ? 32 : 1];

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (batch_idx < B) {
    int row_start = __ldg(rowptr_data + (row % M));
    int row_end = __ldg(rowptr_data + (row % M) + 1);
    int col_idx = row_start + lane_idx;

    scalar_t result = Reducer<scalar_t, REDUCE>::init();
    int64_t arg;

    // Iterate over all `col` indices in parallel within a warp.
    for (int c = row_start; c < row_end; c += 32) {

      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = __ldg(col_data + col_idx) * K;
        if (HAS_VALUE)
          val = __ldg(value_data + col_idx);
      } else {
        mat_row = -1;
        if (HAS_VALUE)
          val = (scalar_t)0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i++) {
        // Communication between all threads in a warp.
        mat_rows[i] = __shfl_sync(FULL_MASK, mat_row, i);
        if (HAS_VALUE)
          vals[i] = __shfl_sync(FULL_MASK, val, i);
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if (lane_idx < leftover && mat_rows[i] != -1) {
          // Coalesced memory access into `mat`.
          val = __ldg(mat_data + batch_idx * N * K + mat_rows[i] + mat_col_idx);
          if (HAS_VALUE)
            val = vals[i] * val;
          Reducer<scalar_t, REDUCE>::update(&result, val, &arg, c + i);
        }
      }
    }

    if (lane_idx < leftover) {
      // Coalesced write into `out`.
      Reducer<scalar_t, REDUCE>::write(out_data + out_idx, result,
                                       arg_out_data + out_idx, arg,
                                       row_end - row_start);
    }
  }
}


void naive_spmm_wrapper(float *dA_values, int *dA_columns, int *dA_csrOffsets,
				int nnzA, int A_rows, int A_cols,
				torch::Tensor B, int B_rows, int B_cols,
				torch::Tensor C) {
	float *dB = B.data_ptr<float>();
    float *dC = C.data_ptr<float>();

	auto m = A_rows;
	auto n = B_rows;
	auto k = B_cols;
	int batch_size = B.numel() / (n * k);

	auto BLOCKS = dim3((32 * batch_size * m + THREADS - 1) / THREADS, (k + 31) / 32);
	// auto stream = aten::cuda::getCurrentCUDAStream();

	std::string reduce = "sum";

	auto sizes = B.sizes().vec();
	sizes[B.dim() - 2] = m;
	auto out = torch::empty(sizes, B.options());

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	int64_t *arg_out_data = nullptr;
	if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
		arg_out = torch::full_like(out, nnzA, B.options());
		arg_out_data = arg_out.value().data_ptr<int64_t>();
	}
  
	AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      spmm_kernel<float, REDUCE, true><<<BLOCKS, THREADS, 0>>>(dA_csrOffsets, dA_columns, dA_values, dB, dC, arg_out_data, batch_size, m, n, k);
    });

}

void naive_batched_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, int a_rows, int a_cols, int b_rows, int b_cols, int batch_dim) {return;}

#endif // __NAIVE_SPARSE_MM_H_