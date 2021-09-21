#include "kernels.hpp"
#include<iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__device__ __host__ int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}

__global__ void taco_binarySearchBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, idx * values_per_block);
}

__host__ int * taco_binarySearchBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int block_size, int num_blocks) {
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, values_per_block, num_blocks);
  return results;
}

__device__ __host__ int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}

__global__ void computeCSRLB(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B,
			     taco_tensor_t * __restrict__ C, int32_t* i_blockStarts, int32_t m) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  __shared__ float precomputedA_ALL[256];
  float * precomputedA = precomputedA_ALL + warp * 32;
  for (int32_t pprecomputedA = 0; pprecomputedA < 8; pprecomputedA++) {
    precomputedA[pprecomputedA] = 0.0;
  }
  for (int32_t nnz = 0; nnz < 8; nnz++) {
    int32_t fpos1 = warp * 8 + nnz;
    int32_t fposA = block * 64 + fpos1;
    if (fposA >= A2_pos[A1_dimension])
      break;

    int32_t f = A2_crd[fposA];
    precomputedA[nnz] = precomputedA[nnz] + A_vals[fposA];
  }
  #pragma unroll 4
  for (int32_t dense_val = 0; dense_val < m / 32; dense_val++) {
    int32_t k = dense_val * 32 + thread;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 8;
    int32_t fposA = block * 64 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 8; nnz++) {
      int32_t fpos1 = warp * 8 + nnz;
      int32_t fposA = block * 64 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      int32_t kB = f * B2_dimension + k;
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = k * C1_dimension + i;
      atomicAdd(&C_vals[iC], B_vals[kB] * precomputedA[nnz]);
    }
  }
}

__global__ void computeCSRNoLB(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B,
			       taco_tensor_t * __restrict__ C, int32_t* i_blockStarts, int32_t m) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C2_dimension = (int)(C->dimensions[1]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t f1 = blockIdx.x;
  int32_t i0 = f1 / ((B2_dimension + 63) / 64);
  int32_t j0 = f1 % ((B2_dimension + 63) / 64);
  int32_t f2 = threadIdx.x;
  int32_t i10 = f2 / 8;
  int32_t j10 = f2 % 8;
  for (int32_t i11 = 0; i11 < 8; i11++) {
    int32_t i1 = i10 * 8 + i11;
    int32_t i = i0 * 64 + i1;
    if (i >= A1_dimension)
      continue;

    for (int32_t kA = A2_pos[i]; kA < A2_pos[(i + 1)]; kA++) {
      int32_t k = A2_crd[kA];
      for (int32_t j11 = 0; j11 < 8; j11++) {
	int32_t j1 = j10 * 8 + j11;
	int32_t j = j0 * 64 + j1;
	int32_t jB = k * B2_dimension + j;
	int32_t jC = i * C2_dimension + j;
	if (j >= B2_dimension)
	  continue;

	C_vals[jC] = C_vals[jC] + A_vals[kA] * B_vals[jB];
      }
    }
  }
}

float compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B, int32_t m, int32_t mode, int32_t iters) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  float* __restrict__ C_vals = (float*)(C->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);

  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 63) / 64 + 1)));
  gpuErrchk(cudaMallocManaged((void**)&(C->vals), sizeof(float) * m * m));

  cudaEvent_t start, end;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  if (mode == 0) {
    for (int i = 0; i < iters; ++i) {
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, (int32_t) 0, A1_dimension,
							 (int32_t) 64, (int32_t) 256, ((A2_pos[A1_dimension] + 63) / 64));
      computeCSRLB<<<(A2_pos[A1_dimension] + 63) / 64, (32 * 8)>>>(A, B, C, i_blockStarts, m);
    }
  } else {
    int num_blocks = (((A1_dimension + 63) / 64) * ((B2_dimension + 63) / 64));
    for (int i = 0; i < iters; ++i) {
      computeCSRNoLB<<<num_blocks, (32 * 8)>>>(A, B, C, i_blockStarts, m);
    }
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);

  cudaFree(i_blockStarts);
  return elapsed;
}
