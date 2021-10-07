#include <cuda_runtime_api.h>
#include <cuda.h>

#include <fstream>
#include <chrono>
#include <vector>
#include <iostream>
#include <functional>

using namespace std::chrono;

#define ceil(a, b) ((a + b - 1) / b)

template<class T>
class Triple {
public:
  T fusion_time;
  T tensor_time;
  T copy_time;
};

Triple<double> measure_time(std::function<Triple<double>()> runner) {
  int w_iters = 250;
  int a_iters = 250;
  for (int i = 0; i < w_iters; ++i) {
    runner();
  }

  Triple<double> times;
  times.fusion_time = 0;
  times.tensor_time = 0;
  times.copy_time = 0;
  for (int i = 0; i < a_iters; ++i) {
    auto p = runner();
    times.fusion_time += p.fusion_time;
    times.tensor_time += p.tensor_time;
    times.copy_time += p.copy_time;
  }
  times.fusion_time /= a_iters;
  times.tensor_time /= a_iters;
  times.copy_time /= a_iters;
  return times;
}

int sum(std::vector<int> vec) {
  int sum = 0;
  for (auto s: vec) sum += s;
  return sum;
}

int pad_amount(std::vector<int> batch, int factor) {
  int s = sum(batch);
  if (s % factor == 0) return factor;
  else return factor - (s % factor);
}

int num_heads = 8;
int qkv = 3;

class Stats {
public:
  float fusion_time = 0;
  float fusion_mem = 0;
  float tensor_time = 0;
  float tensor_mem = 0;
  float copy_time = 0;
};

class Allocator {
public:
  virtual std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) {
    std::cout << "Unimplemented get_needed_allocs!" << std::endl;
    return {};
  }

  virtual void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) {
    std::cout << "Unimplemented construct!" << std::endl;
  }

  virtual bool is_fusion_alloc() {
    return false;
  }
};

/**************************** CoRa *****************************/
///// Templates
template<int PAD>
class BatchLenPadStorageTemplate: public Allocator {
public:
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    return {batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* idx = allocs[0];

    int ctr = 0;
    for (int i = 0; i < batch_size + 1; ++i) {
      idx[i] = ctr;
      ctr += PAD*ceil(lens[i], PAD);
    }
    idx[batch_size] = ctr;
  }
};

template<int PAD>
class BatchLen2PadStorageTemplate: public Allocator {
public:
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    return {batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* idx = allocs[0];

    int ctr = 0;
    for (int i = 0; i < batch_size + 1; ++i) {
      idx[i] = ctr;
      ctr += PAD*ceil(lens[i], PAD) * PAD*ceil(lens[i], PAD);
    }
    idx[batch_size] = ctr;
  }
};

class FusionAllocator: public Allocator {
  bool is_fusion_alloc() override {
    return true;
  }
};

class CoRaBatchSizeSeqLenFusionAllocatorTemplate: public FusionAllocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int ctr = 0;
    for (int o = 0; o < batch_size; ++o) {
      ctr += lens[o];
    }

    return {ctr, ctr, batch_size};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* fo = allocs[0];
    int* fi = allocs[1];
    int* oif = allocs[2];

    int ctr = 0;
    for (int o = 0; o < batch_size; ++o) {
      oif[o] = ctr;
      for (int i = 0; i < lens[o]; ++i) {
	fo[ctr] = o;
	fi[ctr] = i;
	ctr++;
      }
    }
  }
};

template<int PAD>
class CSFAttentionMatrixAllocatorTemplate: public Allocator {
public:
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int len_sum = 0;
    for (int i = 0; i < batch_size; ++i) {
      len_sum += PAD*ceil(lens[i], PAD);
    }

    int n_idx3_ints = batch_size + 1;
    int n_idx4_ints = num_heads * len_sum + 1;
    return {n_idx3_ints, n_idx4_ints};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* idx3 = allocs[0];
    int* idx4 = allocs[1];

    int ctr3 = 0;
    int ctr4 = 0;
    int pos2 = 0;
    int pos3 = 0;

    for (int i = 0; i < batch_size; ++i) {
      idx3[pos2] = ctr3;
      for (int k = 0; k < PAD*ceil(lens[i], PAD); ++k) {
	for (int j = 0; j < num_heads; ++j) {
	  idx4[pos3] = ctr4;
	  ctr4 += PAD*ceil(lens[i], PAD);
	  pos3++;
	}

	ctr3 += lens[i];
      }
      pos2++;
    }

    idx3[pos2] = ctr3;
    idx4[pos3] = ctr4;
  }
};

///// Oeprations
// Input
class CoRaInputAllocator: public BatchLenPadStorageTemplate<1> {};

class CSFInputAllocator: public BatchLenPadStorageTemplate<1> {};

// PreLinear
class CoRaPreLinearOutAllocator: public BatchLenPadStorageTemplate<64> {};

class CSFPreLinearOutAllocator: public Allocator {
public:
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    return {qkv * batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* idx = allocs[0];

    int pos = 0;
    for (int i = 0; i < qkv; ++i) {
      for (int j = 0; j < batch_size; ++j) {
	idx[i * batch_size + j] = pos;
	pos += 64 * ceil(lens[i], 64);
      }
    }
    idx[qkv * batch_size + 1] = pos;
  }
};

class CoRaPreLinearFusionAllocator: public CoRaBatchSizeSeqLenFusionAllocatorTemplate {};

// QKt
class CoRaQKtOutAllocator: public BatchLen2PadStorageTemplate<64> {};

class CSFQKtOutAllocator: public CSFAttentionMatrixAllocatorTemplate<64> {};

class CoRaQKtFusionAllocator: public FusionAllocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int ctr = 0;
    for (int o = 0; o < batch_size; ++o) {
      ctr += ceil(lens[o], 64) * ceil(lens[o], 64);
    }

    return {ctr, ctr, batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* fo = allocs[0];
    int* fi = allocs[1];
    int* oif = allocs[2];

    int ctr = 0;
    for (int o = 0; o < batch_size; ++o) {
      oif[o] = ctr;
      for (int i = 0; i < ceil(lens[o], 64) * ceil(lens[o], 64); ++i) {
	fo[ctr] = o;
	fi[ctr] = i;
	ctr++;
      }
    }
  }
};

// Softmax
class CoRaSoftmaxOutAllocator: public BatchLen2PadStorageTemplate<64> {};

class CSFSoftmaxOutAllocator: public CSFAttentionMatrixAllocatorTemplate<64> {};

class CoRaSoftmaxFusionAllocator: public CoRaBatchSizeSeqLenFusionAllocatorTemplate {};

// AttnV
class CoRaAttnVOutAllocator: public BatchLenPadStorageTemplate<64> {};

class CSFAttnVOutAllocator: public BatchLenPadStorageTemplate<64> {};

class CoRaAttnVFusionAllocator: public FusionAllocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int ctr = 0;
    for (int o = 0; o < batch_size; ++o) {
      ctr += ceil(lens[o], 64);
    }

    return {ctr, ctr, batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* fo = allocs[0];
    int* fi = allocs[1];
    int* oif = allocs[2];

    int ctr = 0;
    for (int o = 0; o < batch_size; ++o) {
      oif[o] = ctr;
      for (int i = 0; i < ceil(lens[o], 64); ++i) {
	fo[ctr] = o;
	fi[ctr] = i;
	ctr++;
      }
    }
  }
};

// PostLinear
class CoRaPostLinearOutAllocator: public BatchLenPadStorageTemplate<1> {};

class CSFPostLinearOutAllocator: public BatchLenPadStorageTemplate<1> {};

class CoRaPostLinearFusionAllocator: public CoRaBatchSizeSeqLenFusionAllocatorTemplate {};

// LayerNorm1
class CoRaLayerNormOutAllocator: public BatchLenPadStorageTemplate<1> {};

class CSFLayerNormOutAllocator: public BatchLenPadStorageTemplate<1> {};

class CoRaLayerNorm1FusionAllocator: public CoRaBatchSizeSeqLenFusionAllocatorTemplate {};

// FF1
class CoRaFF1OutAllocator: public BatchLenPadStorageTemplate<1> {};

class CSFFF1OutAllocator: public BatchLenPadStorageTemplate<1> {};

class CoRaFF1FusionAllocator: public CoRaBatchSizeSeqLenFusionAllocatorTemplate {};

// FF2
class CoRaFF2OutAllocator: public BatchLenPadStorageTemplate<1> {};

class CSFFF2OutAllocator: public BatchLenPadStorageTemplate<1> {};

class CoRaFF2FusionAllocator: public CoRaBatchSizeSeqLenFusionAllocatorTemplate {};

// LayerNorm2
class CoRaLayerNorm2FusionAllocator: public CoRaBatchSizeSeqLenFusionAllocatorTemplate {};

std::vector<int*> alloc_memory(std::vector<int> sizes) {
  std::vector<int*> ret;
  for (auto size: sizes) {
    ret.push_back(new int[size]);
  }
  return ret;
}

void free_allocs(std::vector<int*> allocs) {
  for (auto alloc: allocs) {
    delete[] alloc;
  }
}

void free_device_allocs(std::vector<int*> allocs) {
  for (auto alloc: allocs) {
    cudaFree(alloc);
  }
}

Stats run(std::vector<int> lens, std::vector<Allocator*> allocators) {
  float fusion_mem = 0;
  float tensor_mem = 0;
  int batch_size = lens.size();

  std::vector<std::vector<int*>> allocated_mems;
  std::vector<int*> allocated_raw_mems;
  std::vector<int*> allocated_raw_device_mems;
  std::vector<int> allocated_mem_sizes;

  for (Allocator* allocator: allocators) {
    std::vector<int> allocs_needed = allocator->get_needed_allocs(batch_size, lens);
    int total_mem_needed = sum(allocs_needed);
    int* raw_mem = new int[total_mem_needed];
    std::vector<int*> allocated_mem;
    int pos = 0;
    for (int i = 0; i < allocs_needed.size(); ++i) {
      allocated_mem.push_back(raw_mem + pos);
      pos += allocs_needed[i];
    }
    allocated_raw_mems.push_back(raw_mem);
    allocated_mems.push_back(allocated_mem);
    allocated_mem_sizes.push_back(total_mem_needed);
    int* device_raw_mem;
    cudaMalloc((void **) &device_raw_mem, total_mem_needed);
    allocated_raw_device_mems.push_back(device_raw_mem);
    if (allocator->is_fusion_alloc()) {
      fusion_mem += sum(allocs_needed);
    } else {
      tensor_mem += sum(allocs_needed);;
    }
  }

  auto runner = [&]() {
    double fusion_time = 0;
    double tensor_time = 0;
    float copy_time = 0;
    for (size_t i = 0; i < allocators.size(); ++i) {
      time_point<system_clock> startt = system_clock::now();
      allocators[i]->construct(allocated_mems[i], batch_size, lens);
      time_point<system_clock> endt = system_clock::now();
      if (allocators[i]->is_fusion_alloc()) {
	fusion_time += duration_cast<nanoseconds>(endt - startt).count();
      } else {
	tensor_time += duration_cast<nanoseconds>(endt - startt).count();
      }

      cudaEvent_t start, end;
      cudaEventCreate(&start);
      cudaEventCreate(&end);
      cudaEventRecord(start);
      cudaMemcpy(allocated_raw_device_mems[i], allocated_raw_mems[i], allocated_mem_sizes[i] * sizeof(int), cudaMemcpyHostToDevice);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      float this_copy_time = 0;
      cudaEventElapsedTime(&copy_time, start, end);
      copy_time += this_copy_time;
    }
    return Triple<double>({fusion_time, tensor_time, static_cast<double>(copy_time) * 1000 * 1000});
  };

  auto time_triple = measure_time(runner);
  auto fusion_time = static_cast<float>(time_triple.fusion_time);
  auto tensor_time = static_cast<float>(time_triple.tensor_time);
  auto copy_time = static_cast<float>(time_triple.copy_time);

  free_allocs(allocated_raw_mems);
  free_device_allocs(allocated_raw_device_mems);

  fusion_mem *= sizeof(int);
  tensor_mem *= sizeof(int);
  return {fusion_time, fusion_mem, tensor_time, tensor_mem, copy_time};
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

  std::vector<Allocator*> allocators;
  if (mode == "csf_vanilla") {
    // Add all storage allocators repeating as needed per op
    // PreLinear
    allocators.push_back(new CSFInputAllocator());        // Input
    allocators.push_back(new CSFPreLinearOutAllocator());

    // QKt
    allocators.push_back(new CSFPreLinearOutAllocator()); // Q
    allocators.push_back(new CSFPreLinearOutAllocator()); // K
    allocators.push_back(new CSFQKtOutAllocator());       // Output

    // Softmax
    allocators.push_back(new CSFQKtOutAllocator());       // Attn
    allocators.push_back(new CSFSoftmaxOutAllocator());   // Output

    // Attn
    allocators.push_back(new CSFSoftmaxOutAllocator());   // Attn
    allocators.push_back(new CSFPreLinearOutAllocator()); // V
    allocators.push_back(new CSFAttnVOutAllocator());     // Output

    // PostLinear
    allocators.push_back(new CSFAttnVOutAllocator());     // A
    allocators.push_back(new CSFInputAllocator());        // A2
    allocators.push_back(new CSFPostLinearOutAllocator());// Output

    // LayerNorm1
    allocators.push_back(new CSFPostLinearOutAllocator());// A
    allocators.push_back(new CSFLayerNormOutAllocator()); // Output

    // FF1
    allocators.push_back(new CSFLayerNormOutAllocator()); // A
    allocators.push_back(new CSFFF1OutAllocator());       // Output

    // FF2
    allocators.push_back(new CSFFF1OutAllocator());       // A
    allocators.push_back(new CSFLayerNormOutAllocator()); // A2
    allocators.push_back(new CSFFF2OutAllocator());       // Output

    // LayerNorm2
    allocators.push_back(new CSFFF2OutAllocator());       // A
    allocators.push_back(new CSFLayerNormOutAllocator()); // Output
  } else if (mode == "csf_opt") {
    // Add all storage allocators repeating as needed per op
    // PreLinear
    allocators.push_back(new CSFInputAllocator());        // Input
    allocators.push_back(new CSFPreLinearOutAllocator());

    // QKt
    allocators.push_back(new CSFQKtOutAllocator());       // Output

    // Softmax
    allocators.push_back(new CSFSoftmaxOutAllocator());   // Output

    // Attn
    allocators.push_back(new CSFAttnVOutAllocator());     // Output

    // PostLinear
    allocators.push_back(new CSFPostLinearOutAllocator());// Output

    // LayerNorm1
    allocators.push_back(new CSFLayerNormOutAllocator()); // Output

    // FF1
    allocators.push_back(new CSFFF1OutAllocator());       // Output

    // FF2
    allocators.push_back(new CSFFF2OutAllocator());       // Output

    // LayerNorm2
  } else if (mode == "cora_vanilla") {
    // Add all fusion allocators
    allocators.push_back(new CoRaPreLinearFusionAllocator());
    allocators.push_back(new CoRaQKtFusionAllocator());
    allocators.push_back(new CoRaSoftmaxFusionAllocator());
    allocators.push_back(new CoRaAttnVFusionAllocator());
    allocators.push_back(new CoRaPostLinearFusionAllocator());
    allocators.push_back(new CoRaLayerNorm1FusionAllocator());
    allocators.push_back(new CoRaFF1FusionAllocator());
    allocators.push_back(new CoRaFF2FusionAllocator());
    allocators.push_back(new CoRaLayerNorm2FusionAllocator());

    // Add all storage allocators repeating as needed per op
    // PreLinear
    allocators.push_back(new CoRaInputAllocator());        // Input
    allocators.push_back(new CoRaPreLinearOutAllocator());

    // QKt
    allocators.push_back(new CoRaPreLinearOutAllocator()); // Q
    allocators.push_back(new CoRaPreLinearOutAllocator()); // K
    allocators.push_back(new CoRaQKtOutAllocator());       // Output

    // Softmax
    allocators.push_back(new CoRaQKtOutAllocator());       // Attn
    allocators.push_back(new CoRaSoftmaxOutAllocator());   // Output

    // Attn
    allocators.push_back(new CoRaSoftmaxOutAllocator());   // Attn
    allocators.push_back(new CoRaPreLinearOutAllocator()); // V
    allocators.push_back(new CoRaAttnVOutAllocator());     // Output

    // PostLinear
    allocators.push_back(new CoRaAttnVOutAllocator());     // A
    allocators.push_back(new CoRaInputAllocator());        // A2
    allocators.push_back(new CoRaPostLinearOutAllocator());// Output

    // LayerNorm1
    allocators.push_back(new CoRaPostLinearOutAllocator());// A
    allocators.push_back(new CoRaLayerNormOutAllocator()); // Output

    // FF1
    allocators.push_back(new CoRaLayerNormOutAllocator()); // A
    allocators.push_back(new CoRaFF1OutAllocator());       // Output

    // FF2
    allocators.push_back(new CoRaFF1OutAllocator());       // A
    allocators.push_back(new CoRaLayerNormOutAllocator()); // A2
    allocators.push_back(new CoRaFF2OutAllocator());       // Output

    // LayerNorm2
    allocators.push_back(new CoRaFF2OutAllocator());       // A
    allocators.push_back(new CoRaLayerNormOutAllocator()); // Output
  } else if (mode == "cora_opt") {
    // Add all fusion allocators
    allocators.push_back(new CoRaPreLinearFusionAllocator());
    allocators.push_back(new CoRaQKtFusionAllocator());
    allocators.push_back(new CoRaAttnVFusionAllocator());

    // Add all storage allocators repeating as needed per op
    // PreLinear
    allocators.push_back(new CoRaInputAllocator());        // Input
    allocators.push_back(new CoRaPreLinearOutAllocator());

    // QKt
    allocators.push_back(new CoRaQKtOutAllocator());       // Output

    // Softmax
    allocators.push_back(new CoRaSoftmaxOutAllocator());   // Output

    // Attn
    allocators.push_back(new CoRaAttnVOutAllocator());     // Output

    // PostLinear
    allocators.push_back(new CoRaPostLinearOutAllocator());// Output

    // LayerNorm1
    allocators.push_back(new CoRaLayerNormOutAllocator()); // Output

    // FF1
    allocators.push_back(new CoRaFF1OutAllocator());       // Output

    // FF2
    allocators.push_back(new CoRaFF2OutAllocator());       // Output

    // LayerNorm2
  }

  Stats aggregate;
  for (int i = 0; i < num_batches; ++i) {
    std::vector<int> lens(batch_size, -1);

    for (int j = 0; j < batch_size; ++j){
      fs >> lens[j];
    }

    lens.push_back(pad_amount(lens, 64));

    Stats stats = run(lens, allocators);

    aggregate.fusion_time += stats.fusion_time;
    aggregate.fusion_mem += stats.fusion_mem;
    aggregate.tensor_time += stats.tensor_time;
    aggregate.tensor_mem += stats.tensor_mem;
    aggregate.copy_time += stats.copy_time;
  }

  aggregate.fusion_time /= num_batches;
  aggregate.fusion_mem /= num_batches;
  aggregate.tensor_time /= num_batches;
  aggregate.tensor_mem /= num_batches;
  aggregate.copy_time /= num_batches;

  aggregate.fusion_mem /= 1024;
  aggregate.tensor_mem /= 1024;

  aggregate.fusion_time /= (1000 * 1000);
  aggregate.tensor_time /= (1000 * 1000);
  aggregate.copy_time /= (1000 * 1000);

  std::cout << "RESULTS," << aggregate.fusion_time << "," << aggregate.tensor_time << "," << aggregate.copy_time << std::endl;
  std::cout << "MEM," << aggregate.fusion_mem << "," << aggregate.tensor_mem << std::endl;
  return 0;
}
