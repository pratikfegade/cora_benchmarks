#include <chrono>
#include <functional>
#include <assert.h>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <mkl.h>

float measure_time(std::function<float()> runner, int w_iters, int a_iters) {
  for (int i = 0; i < w_iters; ++i) {
    runner();
  }

  float exe_time = 0.0;
  for (int i = 0; i < a_iters; ++i) {
    exe_time += runner();
  }

  return exe_time / a_iters;
}

float testTRMM(int M, int iters, int warmup) {
  const float* a = static_cast<const float*>(malloc(M * M * sizeof(float)));
  float* b = static_cast<float*>(malloc(M * M * sizeof(float)));

  auto runner = [&] {
    using namespace std::chrono;
    time_point<system_clock> start = system_clock::now();
    cblas_strmm(CblasRowMajor, CblasLeft,
		CblasLower, CblasNoTrans,
		CblasNonUnit,
		M, M,
		1.0,
		a, M,
		b, M);
    time_point<system_clock> end = system_clock::now();
    std::chrono::duration<float> exe_time = (end - start);
    return duration_cast<microseconds>(exe_time).count();
  };

  return measure_time(runner, warmup ? iters : 0, iters);
}

int main(int argc, char *argv[]) {
  mkl_set_threading_layer(MKL_THREADING_GNU);
  int M = std::stoi(argv[1]);
  int iters = std::stoi(argv[2]);
  int warmup = std::stoi(argv[3]);

  auto time = testTRMM(M, iters, warmup);
  time /= 1000;

  std::cout << "RESULTS," << time << std::endl;
}
