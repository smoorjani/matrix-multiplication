// Taken from https://github.com/salehjg/batch-matmul-cuda

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <torch/extension.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#include "reducer.cuh"
#include "utils.cuh"

#define THREADS 256
#define FULL_MASK 0xffffffff


#define CUDA_ERROR_CHECK

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


// C = AB
template <int BLOCK_SIZE>
__global__ void kernel_batched_matmul(
		const float * matA,
		const float * matB,
		float * matC,
		int dim0,
		int dim1A, int dim2A,
		int dim1B, int dim2B,
		int dim1C, int dim2C){
	extern __shared__ float smem[];

	const unsigned int len_subA = BLOCK_SIZE * dim2A,len_subB = BLOCK_SIZE * dim1B; //len of sub matrices of A and B.
	const unsigned long
		len_A = dim0*dim1A*dim2A,
		len_B = dim0*dim1B*dim2B,
		len_C = dim0*dim1C*dim2C;
	const unsigned long
		len_A_signleBatch = dim1A*dim2A,
		len_B_signleBatch = dim1B*dim2B,
		len_C_signleBatch = dim1C*dim2C;
	const unsigned int BLOCKSIZE_P2 = BLOCK_SIZE*BLOCK_SIZE;

    // Block index
    unsigned int bx = blockIdx.x; // mapped to the sub-matrices of output
    unsigned int by = blockIdx.y; // mapped to the sub-matrices of output
    unsigned int bz = blockIdx.z; // batch index

    // Thread index
    unsigned int  tx = threadIdx.x;
    unsigned int  ty = threadIdx.y;

    unsigned int  c_pos_x, c_pos_y;
    c_pos_x = bx*BLOCK_SIZE + tx;
    c_pos_y = by*BLOCK_SIZE + ty;

    unsigned long gidx1,gidx2;
    unsigned int _d1,_d2;


	unsigned long offsetA = (by * BLOCK_SIZE) * dim2A;
	unsigned long offsetB = (bx * BLOCK_SIZE); //first row (d1=0)

	// Load sub matrices from global memory into shared memory

	unsigned long idxA, idxB;
	idxA = ty* BLOCK_SIZE + tx;
	idxB = ty* BLOCK_SIZE + tx;

	while(idxA < len_subA){//Block-stride loop
		gidx1 = offsetA + idxA;
		if(idxA < len_subA && gidx1 < len_A) {
			smem[idxA] = matA[bz * len_A_signleBatch + gidx1];
		}else{
			smem[idxA] = 0;
		}
		idxA += BLOCKSIZE_P2;
	}

	///TODO: It might be better to store transposed subMatB in shared memory to avoid shared memory read conflict.
	///      But then we might get shared memory write conflict. (?)
	while(idxB < len_subB ){//Block-stride loop
		//gidx2 = offsetB + (bx*BLOCK_SIZE)*dim2B + (idxB % dim2B);
		_d2 = idxB%BLOCK_SIZE;
		_d1 = (idxB/BLOCK_SIZE);
		gidx2 = offsetB + _d1*dim2B + _d2;
		if(idxB < len_subB && _d1<dim1B && _d2<dim2B){
			smem[len_subA+idxB] = matB[bz * len_B_signleBatch +gidx2];
		}else{
			smem[len_subA+idxB] = 0;
		}
		idxB += BLOCKSIZE_P2;
	}

	__syncthreads();

    	// Multiply and add each result to produce output element of current thread in the thread block.
    if(c_pos_x<dim2C && c_pos_y<dim1C){
    	unsigned long idx = ty* BLOCK_SIZE + tx;
    	float output_element = 0.0f;

    	//dim2A=dim1B is common equal dimension of 2 matrices  --- block-stride loop
    	for (int k = 0; k < dim2A; k++) {
    		output_element += smem[ty*dim2A+k] * smem[len_subA+ k*BLOCK_SIZE+tx];
    	}

    	///TODO: Check matC index to not to exceed the len of matC!
    	matC[bz * len_C_signleBatch + c_pos_y*dim2C + c_pos_x] = output_element;

    }
}

void naive_batched_matmul(torch::Tensor d_A, torch::Tensor d_B,
            torch::Tensor d_C, int a_rows, int a_cols, int b_rows,
            int b_cols, int batch_dim) {

    float *d_A_arr = d_A.data_ptr<float>();
    float *d_B_arr = d_B.data_ptr<float>();
    float *d_C_arr = d_C.data_ptr<float>();

	const int BLOCK_DIM = 6;
	dim3 blocksize(BLOCK_DIM,BLOCK_DIM,1);
	dim3 gridsize(0,0,0);
	gridsize.x = (b_cols + BLOCK_DIM-1)/BLOCK_DIM;
	gridsize.y = (a_rows + BLOCK_DIM-1)/BLOCK_DIM;
	gridsize.z = (batch_dim);
	unsigned long sharedmemsize = (BLOCK_DIM*a_cols + BLOCK_DIM* b_rows)*sizeof(float);
	printf("@batched_matmul:\n");
	printf("\tBLOCK:(%d, %d)\n",blocksize.x,blocksize.y);
	printf("\t GRID:(%d, %d, %d)\n",gridsize.x,gridsize.y,gridsize.z);
	printf("\t SHARED: %d Bytes\n",sharedmemsize);

	if(BLOCK_DIM==6){
		kernel_batched_matmul<6> <<<gridsize, blocksize, sharedmemsize>>>(
				d_A_arr,
				d_B_arr,
				d_C_arr,
				batch_dim,

				a_rows, //hA
				a_cols, //wA

				b_rows, //hA
				b_cols, //wA

				a_rows,
				b_cols);
		CudaCheckError();
	}else{
		printf("ERR@batched_matmul: UNDEFINED BLOCK_DIM.\n"); return;
	}

}


// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, ReductionType REDUCE, bool HAS_VALUE>
__global__ void spmm_kernel(const int64_t *rowptr_data, const int64_t *col_data,
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


void naive_spmm(float *dA_values, int *dA_columns, int *dA_csrOffsets,
				int A_rows, int A_cols,
				torch::Tensor B, int B_rows, int B_cols,
				torch::Tensor C) {
	float *dB = B.data_ptr<float>();
    float *dC = C.data_ptr<float>();

	auto M = A_rows;
	auto N = B_rows;
	auto K = B_cols;
	auto B = B.numel() / (N * K);

	int64_t *arg_out_data = nullptr;
	if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
		arg_out = torch::full_like(out, col.numel(), rowptr.options());
		arg_out_data = arg_out.value().data_ptr<int64_t>();
	}
  
	spmm_kernel<float, REDUCE, true><<<BLOCKS, THREADS, 0, stream>>>(
		dA_csrOffsets, dA_columns, dA_values, dB, dC, arg_out_data,
		B, M, N, K);

}