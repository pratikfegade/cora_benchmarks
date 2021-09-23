#include <iostream>
#include "taco.h"
#include "utils.cuh"

using namespace taco;
using namespace std::chrono;

const IndexVar io("io"), jo("jo"), ko("ko"), ii("ii"), ji("ji"), ki("ki");
int WARP_SIZE = 32;

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<float> A, int m, int bs, IndexExpr precomputedAExpr,
			  int NNZ_PER_WARP=1, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), nnz_pre("nnz_pre");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  TensorVar precomputedA("precomputedA", Type(Float32, {Dimension(nnz)}), taco::dense);
  return stmt.reorder({io, jo, ko, ii, ji, ki})
  .fuse(io, jo, f)
    .pos(f, fpos, A(io, jo, ii, ji))
    .split(fpos, block, fpos1, NNZ_PER_TB)
    .split(fpos1, warp, nnz, NNZ_PER_WARP)
    .split(ko, dense_val_unbounded, thread, WARP_SIZE)
    .reorder({block, warp, thread, dense_val_unbounded, nnz})
    .bound(dense_val_unbounded, dense_val, (m / bs) / WARP_SIZE, BoundType::MaxExact)
    .unroll(dense_val, 4)
    .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
    .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
    .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

__global__
void computeDeviceKernel0(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* io_blockStarts, int mb){
  int A1_dimension = (int)(A->dimensions[0]);
  int A3_dimension = (int)(A->dimensions[2]);
  int A4_dimension = (int)(A->dimensions[3]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  int B3_dimension = (int)(B->dimensions[2]);
  int B4_dimension = (int)(B->dimensions[3]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C2_dimension = (int)(C->dimensions[1]);
  int C3_dimension = (int)(C->dimensions[2]);
  int C4_dimension = (int)(C->dimensions[3]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  float Cl[32];
  for (int i = 0; i < 32; ++i) {
    Cl[i] = 0;
  }

  #pragma unroll 4
  for (int32_t dense_val = 0; dense_val < mb/32; dense_val++) {
    int32_t ko = dense_val * 32 + thread;
    int32_t pA2_begin = io_blockStarts[block];
    int32_t pA2_end = io_blockStarts[(block + 1)];
    int32_t fposA = block * 8 + warp;
    int32_t io_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t io = io_pos;
    for (int32_t nnz = 0; nnz < 1; nnz++) {
      int32_t fposA = block * 8 + warp;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      int32_t koB = f * B2_dimension + ko;
      while (fposA == A2_pos[(io_pos + 1)]) {
        io_pos = io_pos + 1;
        io = io_pos;
      }
      int32_t koC = io * C2_dimension + ko;
      for (int32_t ii = 0; ii < A3_dimension; ii++) {
        int32_t iiC = koC * C3_dimension + ii;
        int32_t iiA = fposA * A3_dimension + ii;
        for (int32_t ji = 0; ji < A4_dimension; ji++) {
          int32_t jiB = koB * B3_dimension + ji;
          int32_t jiA = iiA * A4_dimension + ji;
          for (int32_t ki = 0; ki < B4_dimension; ki++) {
            int32_t kiC = iiC * C4_dimension + ki;
            int32_t kiB = jiB * B4_dimension + ki;
	    Cl[ji] += B_vals[kiB] * A_vals[jiA];
          }
        }
	for (int32_t ki = 0; ki < B4_dimension; ki++) {
	  int32_t kiC = iiC * C4_dimension + ki;
	  C_vals[kiC] = Cl[ki];
	}
      }
    }
  }

}

float compute(taco_tensor_t *C, taco_tensor_t *B, taco_tensor_t *A, int m, int bs, int iters) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);

  int32_t* io_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&io_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 7) / 8 + 1)));
  gpuErrchk(cudaMallocManaged((void**)&(C->vals), sizeof(float) * m * m));
  io_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, io_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 8, (int32_t) 256, ((A2_pos[A1_dimension] + 7) / 8));

  int num_blocks = (A2_pos[A1_dimension] + 7) / 8;
  std::cout << "NB " << num_blocks << " " << A2_pos[A1_dimension] << std::endl;
  cudaEvent_t start, end;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  for (int i = 0; i < iters; ++i) {
    computeDeviceKernel0<<<num_blocks, (32 * 8)>>>(A, B, C, io_blockStarts, m/bs);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(io_blockStarts);
  return elapsed;
}



int main(int argc, char* argv[]) {
  int m = std::atoi(argv[1]);
  int bs = std::atoi(argv[2]);
  int mb = m/bs;
  Tensor<float> A("A", {mb, mb, bs, bs}, {Dense, Compressed, Dense, Dense});
  Tensor<float> B("B", {mb, mb, bs, bs}, {Dense, Dense, Dense, Dense});
  Tensor<float> C("C", {mb, mb, bs, bs}, {Dense, Dense, Dense, Dense});

  for (int i = 0; i < mb; ++i) {
    for (int j = 0; j < i + 1; ++j) {
      for (int ii = 0; ii < bs; ++ii) {
  	for (int ji = 0; ji < bs; ++ji) {
  	  float rand_float = (float)rand()/(float)(RAND_MAX);
  	  A.insert({i, j, ii, ji}, rand_float);
  	}
      }
    }
    for (int j = i + 1; j < mb; j++) {
      for (int ii = 0; ii < bs; ++ii) {
  	for (int ji = 0; ji < bs; ++ji) {
  	  float rand_float = (float)rand()/(float)(RAND_MAX);
  	  B.insert({i, j, ii, ji}, rand_float);
  	  C.insert({i, j, ii, ji}, rand_float);
  	}
      }
    }
  }
  A.pack();
  B.pack();

  auto At = A.getTacoTensorT();
  auto Bt = B.getTacoTensorT();
  auto Ct = C.getTacoTensorT();

  int witers = 100;
  int iters = 100;
  // Warm up
  compute(Ct, Bt, At, m, bs, witers);

  float time = compute(Ct, Bt, At, m, bs, iters);
  time /= iters;
  std::cout << "RESULTS," << time << std::endl;



  // IndexExpr precomputedA = A(io, jo, ii, ji);
  // IndexExpr precomputedB = B(jo, ko, ji, ki);
  // C(io, ko, ii, ki) += precomputedB * precomputedA;

  // IndexStmt stmt = C.getAssignment().concretize();
  // stmt = scheduleSpMMGPU(stmt, A, m, bs, precomputedA);

  // C.compile(stmt);
}
