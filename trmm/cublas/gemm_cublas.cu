#include <cublas_v2.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "utils.h"

int find_max(std::vector<int> v) {
  int max = -10000;
  for (auto e: v) {
    if (e > max) max = e;
  }
  return max;
}

float testCuBLAS(int M, int iters, int warmup) {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  float* A;
  float* B;
  float* C;

  auto op_a = CUBLAS_OP_T;

  int lda = M;
  int ldb = M;
  int ldc = M;

  CUDA_CHECK(cudaMalloc((void**)&A, M * M * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&B, M * M * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&C, M * M * sizeof(float)));

  auto runner = [&]() {
    float time = 0;
    for (int i = 0; i < iters; ++i) {
      cudaEvent_t start, end;
      float elapsed = 0;

      // Timing info
      cudaEventCreate(&start);
      cudaEventCreate(&end);
      cudaEventRecord(start);

      const float alpha = 1.0;

      cublasStatus_t cublas_result = cublasStrmm(cublas_handle,
						 CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
						 op_a, CUBLAS_DIAG_NON_UNIT,
						 M, M,
						 &alpha,
						 A, lda,
						 B, ldb,
						 C, ldc);

      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed, start, end);
      time += elapsed;
      assert(cublas_result == CUBLAS_STATUS_SUCCESS);
    }
    return (time / iters);
  };

  if (warmup) { runner(); }
  float time = runner();

  CUDA_CHECK(cudaFree((void*)A));
  CUDA_CHECK(cudaFree((void*)B));
  CUDA_CHECK(cudaFree((void*)C));

  return time;
}

int main(int argc, char *argv[]) {
  int M = std::stoi(argv[1]);
  int iters = std::stoi(argv[2]);
  int warmup = std::stoi(argv[3]);

  float time = testCuBLAS(M, iters, warmup);
  std::cout << "RESULTS," << time << std::endl;
}
