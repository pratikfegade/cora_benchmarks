#include <fstream>
#include <chrono>
#include <vector>
#include <iostream>
#include <functional>

using namespace std::chrono;

double measure_time(std::function<void()> runner) {
  int w_iters = 1000;
  int a_iters = 1000;
  for (int i = 0; i < w_iters; ++i) {
    runner();
  }

  time_point<system_clock> start = system_clock::now();
  for (int i = 0; i < a_iters; ++i) {
    runner();
  }
  time_point<system_clock> end = system_clock::now();
  duration<double> exe_time = (end - start);
  return (duration_cast<microseconds>(exe_time).count() * 1.0) / a_iters;
}

int num_heads = 8;

std::pair<long long int, double> gen_data_csf(int batch_size, std::vector<int> lens) {
  int len_sum = 0;
  for (int i = 0; i < batch_size; ++i) {
    len_sum += lens[i];
  }

  int n_idx3_ints = num_heads * batch_size + 1;
  int n_idx4_ints = num_heads * len_sum + 1;
  int* idx3 = new int[n_idx3_ints];
  int* idx4 = new int[n_idx4_ints];

  auto runner = [&] {
    int ctr3 = 0;
    int ctr4 = 0;
    int pos2 = 0;
    int pos3 = 0;

    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_heads; ++j) {
	idx3[pos2] = ctr3;

	for (int k = 0; k < lens[i]; ++k) {
	  idx4[pos3] = ctr4;
	  ctr4 += lens[i];
	  pos3++;
	}

	ctr3 += lens[i];
	pos2++;
      }
    }

    idx3[pos2] = ctr3;
    idx4[pos3] = ctr4;
  };

  int nbytes = (n_idx3_ints + n_idx4_ints) * sizeof(int);
  float time = measure_time(runner);
  return std::make_pair(nbytes, time);
}

std::pair<long long int, double> gen_data_cora(int batch_size, std::vector<int> lens) {
  int n_af1_ints = batch_size + 1;
  int* af1 = new int[n_af1_ints];

  auto runner = [&] {
    int ctr1 = 0;

    int i = 0;
    for (;i < batch_size; ++i) {
      af1[i] = ctr1;
      ctr1 += lens[i] * lens[i];
    }
    af1[i] = ctr1;
  };

  int nbytes = n_af1_ints * sizeof(int);
  float time = measure_time(runner);
  return std::make_pair(nbytes, time);
}

int main(int argc, char** argv) {
  int batch_size = std::stoi(argv[1]);
  int num_batches = std::stoi(argv[2]);
  std::string data_file = argv[3];
  std::string mode = argv[4];

  std::fstream fs;
  fs.open(data_file);
  if (!fs.is_open()){
    printf("Error opening input\n");
    exit(EXIT_FAILURE);
  }

  double total_time = 0;
  double total_mem = 0;
  for (int i = 0; i < num_batches; ++i) {
    std::vector<int> lens(batch_size, -1);

    for (int j = 0; j < batch_size; ++j){
      fs >> lens[j];
    }

    std::pair<long long int, double>  p;
    if (mode == "csf") {
      p = gen_data_csf(batch_size, lens);
    } else {
      p = gen_data_cora(batch_size, lens);
    }
    total_mem += p.first;
    total_time += p.second;
  }

  total_time /= num_batches;
  total_mem /= num_batches;
  total_time *= 0.001;

  std::cout << "RESULTS," << total_time << std::endl;
  std::cout << "MEM," << total_mem << std::endl;
  return 0;
}
