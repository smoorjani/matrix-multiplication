#ifndef __CUBLAS_MM_KERNEL_H__
#define __CUBLAS_MM_KERNEL_H__

#include <iostream>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <curand.h>

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

// A * B on GPU and store in C
// C(m,n) = A(m,k) * B(k,n)
void mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m;
    int ldb = k;
    int ldc = m;

	const float a = 1;
	const float b = 0;
	const float *alpha = &a;
	const float *beta = &b;

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cublasDestroy(handle);
}

void mmul_wrapper(const float *A, const float *B, float *C, 
				  const int A_rows, const int A_cols,
				  const int B_rows, const int B_cols,
				  const int C_rows, const int C_cols) {
	
	// Allocate 3 arrays on GPU
	float *gpu_A, *gpu_B, *gpu_C;
	cudaMalloc(&gpu_A, A_rows * A_cols * sizeof(float));
	cudaMalloc(&gpu_B, B_rows * B_cols * sizeof(float));
	cudaMalloc(&gpu_C, C_rows * C_cols * sizeof(float));

	print_matrix(A, A_rows, A_cols);
	print_matrix(B, B_rows, B_cols);
	// Fill the arrays A and B on GPU with random numbers
	// fill_rand(gpu_A, A_rows, A_cols);
	// fill_rand(gpu_B, B_rows, B_cols);
	for (int row = 0; row < A_rows; row++) {
		for (int col = 0; col < A_cols; col++) {
			cudaMemcpy(&gpu_A[row * A_cols + col], &A[row * A_cols + col], sizeof(float), cudaMemcpyHostToDevice);
		}
	}

	for (int row = 0; row < A_rows; row++) {
		for (int col = 0; col < A_cols; col++) {
			cudaMemcpy(&gpu_B[rows * B_cols + col], &B[row * B_cols + col], sizeof(float), cudaMemcpyHostToDevice);
		}
	}

	float *cpu_A = (float *)malloc(A_rows * A_cols * sizeof(float));
	float *cpu_B = (float *)malloc(B_rows * B_cols * sizeof(float));
	cudaMemcpy(cpu_A, gpu_A, A_rows * A_cols * sizeof(float), cudaMemcpyDeviceToHost);
	print_matrix(cpu_A, A_rows, A_cols);
	cudaMemcpy(cpu_B, gpu_B, B_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost);
	print_matrix(cpu_B, B_rows, B_cols);

	mmul(gpu_A, gpu_B, gpu_C, A_rows, A_cols, B_cols);

	cudaMemcpy(C, gpu_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost);

	//Free GPU memory
	cudaFree(gpu_A);
	cudaFree(gpu_B);
	cudaFree(gpu_C);

}

void random_mmul() {
	// Allocate 3 arrays on CPU
	int A_rows, A_cols, B_rows, B_cols, C_rows, C_cols;
	A_rows = A_cols = B_rows = B_cols = C_rows = C_cols = 4;
	
	float *cpu_A = (float *)malloc(A_rows * A_cols * sizeof(float));
	float *cpu_B = (float *)malloc(B_rows * B_cols * sizeof(float));
	float *cpu_C = (float *)malloc(C_rows * C_cols * sizeof(float));

	// Allocate 3 arrays on GPU
	float *gpu_A, *gpu_B, *gpu_C;
	cudaMalloc(&gpu_A, A_rows * A_cols * sizeof(float));
	cudaMalloc(&gpu_B, B_rows * B_cols * sizeof(float));
	cudaMalloc(&gpu_C, C_rows * C_cols * sizeof(float));

	// Fill the arrays A and B on GPU with random numbers
	fill_rand(gpu_A, A_rows, A_cols);
	fill_rand(gpu_B, B_rows, B_cols);

	// copy to CPU and print
	cudaMemcpy(cpu_A, gpu_A, A_rows * A_cols * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(cpu_A, A_rows, A_cols);
	cudaMemcpy(cpu_B, gpu_B, B_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost);
	print_matrix(cpu_B, B_rows, B_cols);

	mmul(gpu_A, gpu_B, gpu_C, A_rows, A_cols, B_cols);

	cudaMemcpy(cpu_C, gpu_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost);
	print_matrix(cpu_C, C_rows, C_cols);

	//Free GPU memory
	cudaFree(gpu_A);
	cudaFree(gpu_B);
	cudaFree(gpu_C);	

	// Free CPU memory
	free(cpu_A);
	free(cpu_B);
	free(cpu_C);
}

// int main() {
// 	random_mmul();
// }

#endif // __CUBLAS_MM_KERNEL_H__
