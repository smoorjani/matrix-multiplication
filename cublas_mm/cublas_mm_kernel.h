#ifndef __CUBLAS_MM_KERNEL_H__
#define __CUBLAS_MM_KERNEL_H__

'''
MM reproduced from:
https://gist.github.com/peterwittek/6303527
'''

#include <iostream>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <curand.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int ci(int row, int column, int nColumns) {
  return row*nColumns+column;
}

// Print matrix A
void print_matrix(const float *A, int A_rows, int A_cols) {
    for(int i = 0; i < A_rows; ++i){
        for(int j = 0; j < A_cols; ++j){
            std::cout << A[j * A_rows + i] << " ";
        }
        std::cout << std::endl;
    }
}

void fill_rand(float *A, int A_rows, int A_cols) {
	curandGenerator_t random;
	curandCreateGenerator(&random, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(random, (unsigned long long) clock());
	curandGenerateUniform(random, A, A_rows * A_cols);
}

void mmul_wrapper(const float *A, const float *B, float *C, 
				  const int A_rows, const int A_cols,
				  const int B_rows, const int B_cols,
				  const int C_rows, const int C_cols) {

	float *gpu_A, *gpu_B, *gpu_C;
	cudaMalloc(&gpu_A, A_rows * A_cols * sizeof(float));
	cudaMalloc(&gpu_B, B_rows * B_cols * sizeof(float));
	cudaMalloc(&gpu_C, C_rows * C_cols * sizeof(float));

	cudaMemcpy(gpu_A, A, A_rows * A_cols * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B, B, B_rows * B_cols * sizeof(float),cudaMemcpyHostToDevice);

	cublasHandle_t handle;

	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "!!!! CUBLAS initialization error\n";
	}
	
	float alpha = 1.0f;
	float beta = 0.0f;
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
										B_cols, A_rows, A_cols, 
										&alpha, gpu_B, B_cols, 
												gpu_A, A_cols, 
										&beta,  gpu_C, B_cols);

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Kernel execution error.";
	}

	cudaMemcpy(C, gpu_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost);

	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Shutdown error!";
	}

	cudaFree(gpu_A);
	cudaFree(gpu_B);
	cudaFree(gpu_C);	
}

#endif // __CUBLAS_MM_KERNEL_H__