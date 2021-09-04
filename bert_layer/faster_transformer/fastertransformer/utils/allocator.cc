#include <unordered_map>

namespace fastertransformer {
long long int total_allocated_memory = 0;
long long int max_allocated_memory = -1;
std::unordered_map<void*, size_t> alloc_sizes;
}
