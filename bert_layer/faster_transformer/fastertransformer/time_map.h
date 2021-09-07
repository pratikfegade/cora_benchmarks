#pragma once

#define OP_TIMES 1

#define START_OPTIME_MEASUREMENT  \
  {cudaEventCreate(&start);	  \
   cudaEventCreate(&end);            \
   cudaEventRecord(start);}	     \

#define END_OPTIME_MEASUREMENT(op)	      	   \
  {cudaEventRecord(end);			   \
   cudaEventSynchronize(end);			   \
   cudaEventElapsedTime(&elapsed, start, end);	     \
   TimeMap::AddTime(op, elapsed);}		     \


#include <unordered_map>
#include <iostream>

namespace fastertransformer {
  class TimeMap {
  public:
    static void AddTime(std::string op, float time) {
      auto it = times.find(op);
      if (it == times.end()) {
	times[op] = time;
      } else {
	it->second += time;
      }
    }

    static void Print() {
      for (auto it: times) {
	std::cout << "RESULTS," << it.first << "," << it.second << std::endl;
      }
    }

  private:
    static std::unordered_map<std::string, float> times;
  };
}
