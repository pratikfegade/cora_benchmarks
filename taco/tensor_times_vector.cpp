#include <iostream>
#include "taco.h"
#include "kernels.hpp"

using namespace taco;
using namespace std::chrono;

const IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
int WARP_SIZE = 32;

float measure_time(std::function<float()> runner) {
  int w_iters = 1000;
  int a_iters = 1000;
  for (int i = 0; i < w_iters; ++i) {
    runner();
  }

  float exe_time = 0.0;
  for (int i = 0; i < a_iters; ++i) {
    exe_time += runner();
  }

  return exe_time / a_iters;
}

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<float> A, int m, IndexExpr precomputedExprA,
			  IndexExpr precomputedExprB, int NNZ_PER_WARP=8, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), nnz_pre("nnz_pre");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  TensorVar precomputedA("precomputedA", Type(Float32, {Dimension(nnz)}), taco::dense);
  TensorVar precomputedB("precomputedB", Type(Float32, {Dimension(nnz)}), taco::dense);
  return stmt.reorder({i, j, k})
          .fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(k, dense_val_unbounded, thread, WARP_SIZE)
          .reorder({block, warp, thread, dense_val_unbounded, nnz})
          .precompute(precomputedExprA, nnz, nnz, precomputedA)
          .bound(dense_val_unbounded, dense_val, -1, BoundType::MaxExact)
          .unroll(dense_val, 4)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

int main(int argc, char* argv[]) {
  int m = std::atoi(argv[1]);
  int mode = std::atoi(argv[2]);
  int NUM_I = m;
  int NUM_J = m;
  int NUM_K = m;
  float SPARSITY = .3;
  Tensor<float> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<float> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));

  srand(434321);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < i + 1; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({i, j}, rand_float);
      C.insert({i, j}, rand_float);
      A.insert({i, j}, rand_float);
    }
    for (int j = i + 1; j < m; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({i, j}, rand_float);
      C.insert({i, j}, rand_float);
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
  compute(Ct, At, Bt, m, mode, witers);

  float time = compute(Ct, At, Bt, m, mode, iters);
  time /= iters;

  std::cout << "RESULTS," << time << std::endl;
}
