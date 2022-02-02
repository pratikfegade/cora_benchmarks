/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastertransformer/bert_encoder_transformer.h"
#include "fastertransformer/time_map.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <fstream>
#include <vector>

using namespace fastertransformer;

std::unordered_map<std::string, float> fastertransformer::TimeMap::times;
bool fastertransformer::TimeMap::do_profile;
fastertransformer::MemoryTracker fastertransformer::MemoryTracker::instance;

template <typename T>
void device_malloc(T **ptr, int size);

template <typename T>
void device_malloc_one(T **ptr, int batchSize, int seqLen, float rate);

template <typename T>
float encoder_sample(std::vector<int> batch,
		     int num_layers,
		     int seq_len,
		     int head_num,
		     int size_per_head,
		     bool is_remove_padding,
		     int int8_mode,
		     int ite,
		     bool allow_gemm_test=false);

int main(int argc, char* argv[])
{
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  if(argc != 11 && argc != 12)
  {
    printf("[ERROR] encoder_sample data_file num_batches batch_size num_layers seq_len head_num size_per_head is_fp16 is_remove_padding int8_mode\n");
    printf("e.g., ./bin/encoder_sample <file> 10 1 12 128 12 64 0 0 0\n");
    return 0;
  }
  printf("Device %s\n", prop.name);
  bool allow_gemm_test = false;
  if (argc >= 10)
    allow_gemm_test = (atoi(argv[9]) == 1) ? true : false;

  printf("Device %s\n", prop.name);
  std::string data_file = argv[1];
  int num_batches = atoi(argv[2]);
  int batch_size = atoi(argv[3]);
  int num_layers = atoi(argv[4]);
  int seq_len = atoi(argv[5]);
  int head_num = atoi(argv[6]);
  int size_per_head = atoi(argv[7]);
  int is_fp16 = atoi(argv[8]);
  bool is_remove_padding = (bool)atoi(argv[9]);
  int int8_mode = atoi(argv[10]);

  std::cout << "File " << data_file << std::endl;
  std::cout << "NumB " << num_batches << std::endl;
  std::cout << "BatS " << batch_size << std::endl;
  std::cout << "NumL " << num_layers << std::endl;
  std::cout << "SeqL " << seq_len << std::endl;
  std::cout << "HedN " << head_num << std::endl;
  std::cout << "HedS " << size_per_head << std::endl;
  std::cout << "FP16 " << is_fp16 << std::endl;
  std::cout << "RemP " << is_remove_padding << std::endl;
  std::cout << "Int8 " << int8_mode << std::endl;
#ifdef OP_TIMES
  std::cout << "Profiling Op times" << std::endl;
#endif


  int ite = 10;

  if(is_fp16 != 0 && is_fp16 != 1) {
    printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
    return -1;
  }

  std::vector<int> all_lengths;
  std::ifstream l_file(data_file);
  int l;
  while (l_file >> l) {
    all_lengths.push_back(l);
  }
  num_batches = std::min(num_batches, static_cast<int>(all_lengths.size()) / batch_size);
  std::cout << "Possible NumB " << num_batches << std::endl;

  int counter = 0;
  float total_time = 0.0;
  for (int i = 0; i < num_batches; ++i) {
    std::vector<int> batch;
    for (int j = 0; j < batch_size; ++j) {
      int l = all_lengths[counter++];;
      batch.push_back(l);
    }

    if(is_fp16 == 0)
      total_time += encoder_sample<float>(batch, num_layers, seq_len, head_num, size_per_head,
					  is_remove_padding, int8_mode, ite, allow_gemm_test);
    else if(is_fp16 == 1)
      total_time += encoder_sample<half>(batch, num_layers, seq_len, head_num, size_per_head,
					 is_remove_padding, int8_mode, ite, allow_gemm_test);
  }

  total_time /= (num_batches * num_layers * ite);
#ifdef OP_TIMES
  fastertransformer::TimeMap::MeanBy(num_batches * num_layers * ite);
  fastertransformer::TimeMap::Print();
#endif

#ifdef MEM_PROF
  fastertransformer::MemoryTracker::print();
#endif

#ifdef OP_TIMES
  printf("RESULTS,Total,%.5f\n", total_time);
#else
  printf("RESULTS,%.5f\n", total_time);
#endif

  return 0;
}

template <typename T>
void device_malloc(T **ptr, int size)
{
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  T *tmp = new T[size];
  for(int i = 0; i < size; i++)
  {
    tmp[i] = (T)((rand() % 100) / 50.0f) - 1.0f;
  }
  cudaMemcpy(*ptr, tmp, sizeof(T) * size, cudaMemcpyHostToDevice);
  delete tmp;

}

template <typename T>
void device_malloc_one(T **ptr, int batchSize, int seqLen, int avg_len)
{
  int size = batchSize * seqLen * seqLen;
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  T *tmp = new T[size];
  int j = 0;
  for (int b = 0 ; b < batchSize ; b++)
  {
    for (int l1 = 0 ; l1 < seqLen ; l1++)
    {
      for (int l2 = 0 ; l2 < avg_len ; l2++)
        tmp[j++] = (T)(1.0f);
      for (int l2 = avg_len ; l2 < seqLen ; l2++)
        tmp[j++] = T(0.0f);
    }
  }
  cudaMemcpy(*ptr, tmp, sizeof(T) * size, cudaMemcpyHostToDevice);
  delete tmp;
}

template <typename T>
float encoder_sample(std::vector<int> batch,
		     int num_layers,
		     int seq_len,
		     int head_num,
		     int size_per_head,
		     bool is_remove_padding,
		     int int8_mode,
		     int ite,
		     bool allow_gemm_test)
{
  int batch_size = batch.size();
  int hidden_dim = head_num * size_per_head;

  T *d_from_tensor = NULL, *d_transformer_out = NULL;
  T *d_attr_kernel_Q = NULL, *d_attr_kernel_K = NULL, *d_attr_kernel_V = NULL;
  T *d_attr_bias_Q = NULL, *d_attr_bias_K = NULL, *d_attr_bias_V = NULL;
  T *d_attr_mask = NULL, *d_attr_output_kernel = NULL, *d_attr_output_bias = NULL;
  T *d_attr_output_layernorm_beta = NULL;
  T *d_attr_output_layernorm_gamma = NULL;
  T *d_inter_kernel = NULL, *d_inter_bias = NULL;
  T *d_output_kernel = NULL, *d_output_bias = NULL, *d_output_layernorm_beta = NULL, *d_output_layernorm_gamma = NULL;
  float *amaxList = NULL;
  int* d_trt_seqlen_offset = NULL;

  // pre_process buffer
  T *d_from_tensor_with_padding = NULL;
  T *d_transformer_out_with_padding = NULL;

  int* d_sequence_length;
  int *d_sequence_id_offset;
  int *d_tmp_sequence_id_offset;
  int *d_valid_word_num;

  int max_seq_len = -1000;
  for (int i = 0; i < batch_size; ++i) {
    if (batch[i] > max_seq_len) max_seq_len = batch[i];
  }

  seq_len = max_seq_len;

  int from_seq_len = max_seq_len;
  int to_seq_len = max_seq_len;

  int* h_sequence_length = new int[batch_size];
  for(int i = 0; i < batch_size; i++)
  {
    if (is_remove_padding) {
      h_sequence_length[i] = batch[i];
    } else {
      h_sequence_length[i] = max_seq_len;
    }
  }

  int h_trt_seqlen_size = 0;
  if(is_remove_padding)
  {
    h_trt_seqlen_size = batch_size + 1;

    int* h_trt_seqlen_offset = new int[h_trt_seqlen_size];
    h_trt_seqlen_offset[0] = 0;
    for(int i = 1; i < h_trt_seqlen_size; i++)
    {
      h_trt_seqlen_offset[i] = h_trt_seqlen_offset[i-1] + h_sequence_length[i - 1];
    }
    cudaMalloc(&d_trt_seqlen_offset, sizeof(int) * (h_trt_seqlen_size));
    cudaMemcpy(d_trt_seqlen_offset, h_trt_seqlen_offset, sizeof(int) * (h_trt_seqlen_size), cudaMemcpyHostToDevice);
    device_malloc_one(&d_attr_mask, batch_size, seq_len, seq_len/2);
    delete [] h_trt_seqlen_offset;
  }
  else
  {
    h_trt_seqlen_size = batch_size * 2 + 1;

    int* h_trt_seqlen_offset = new int[h_trt_seqlen_size];
    h_trt_seqlen_offset[0] = 0;
    for(int i = 1; i < h_trt_seqlen_size; i++)
    {
      if(i % 2 == 1)
        h_trt_seqlen_offset[i] = h_trt_seqlen_offset[i-1] + h_sequence_length[(i-1) / 2];
      else
        h_trt_seqlen_offset[i] = seq_len * (i / 2);
    }
    cudaMalloc(&d_trt_seqlen_offset, sizeof(int) * (h_trt_seqlen_size));
    cudaMemcpy(d_trt_seqlen_offset, h_trt_seqlen_offset, sizeof(int) * (h_trt_seqlen_size), cudaMemcpyHostToDevice);
    device_malloc_one(&d_attr_mask, batch_size, seq_len, seq_len);
    delete [] h_trt_seqlen_offset;
  }

  size_t free_bytes, total_bytes;
  check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
  float free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
  float total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  // printf("before allocate free %.2f GB total %.2f GB\n", free, total);

  cudaMalloc((void**)&d_sequence_length, sizeof(int) * (ceil(batch_size/4.) * 4));

  device_malloc(&d_from_tensor, batch_size * seq_len * hidden_dim);
  device_malloc(&d_transformer_out, batch_size * seq_len * hidden_dim);
  device_malloc(&d_attr_kernel_Q, hidden_dim * hidden_dim * 3);
  d_attr_kernel_K = d_attr_kernel_Q + hidden_dim * hidden_dim;
  d_attr_kernel_V = d_attr_kernel_K + hidden_dim * hidden_dim;
  //device_malloc(&d_attr_kernel_K, hidden_dim * hidden_dim);
  //device_malloc(&d_attr_kernel_V, hidden_dim * hidden_dim);
  device_malloc(&d_attr_bias_Q, hidden_dim);
  device_malloc(&d_attr_bias_K, hidden_dim);
  device_malloc(&d_attr_bias_V, hidden_dim);
  device_malloc(&d_attr_output_kernel, hidden_dim * hidden_dim);
  device_malloc(&d_attr_output_bias, hidden_dim);
  device_malloc(&d_attr_output_layernorm_beta, hidden_dim);
  device_malloc(&d_attr_output_layernorm_gamma, hidden_dim);
  device_malloc(&d_inter_kernel, hidden_dim * hidden_dim * 4);
  device_malloc(&d_inter_bias, hidden_dim * 4);
  device_malloc(&d_output_kernel, hidden_dim * hidden_dim * 4);
  device_malloc(&d_output_bias, hidden_dim);
  device_malloc(&d_output_layernorm_beta, hidden_dim);
  device_malloc(&d_output_layernorm_gamma, hidden_dim);
  device_malloc(&amaxList, ACTIVATION_AMAX_NUM + 9*hidden_dim + INT8O_GEMM_NUM + TRT_FUSED_MHA_AMAX_NUM);

  if(is_remove_padding == true)
  {
    const int pre_process_buf_size = ceil((batch_size * from_seq_len + 1) * sizeof(int) / 4.) * 4;
    cudaMalloc((void**)&d_sequence_id_offset, sizeof(int) * batch_size * from_seq_len);
    cudaMalloc((void**)&d_tmp_sequence_id_offset, pre_process_buf_size);
    d_valid_word_num = (int*)d_tmp_sequence_id_offset + batch_size * from_seq_len;
    device_malloc(&d_from_tensor_with_padding, batch_size * from_seq_len * hidden_dim);
    device_malloc(&d_transformer_out_with_padding, batch_size * from_seq_len * hidden_dim);
  }

  check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
  free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
  total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  // printf("After allocate free %.2f GB used %.2f GB total %.2f GB\n", free, total - free, total);

  cublasHandle_t cublasHandle;
  cublasLtHandle_t cublaslt_handle;
  check_cuda_error(cublasCreate(&cublasHandle));
  check_cuda_error(cublasLtCreate(&cublaslt_handle));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  const fastertransformer::OperationType type = sizeof(T) == sizeof(float) ? OperationType::FP32 : OperationType::FP16;
  typedef BertEncoderTransformerTraits<type, cuda::OpenMultiHeadAttention> EncoderTraits_;
  fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
  BertInitParam<T> encoder_param; //init param here

  encoder_param.from_tensor = d_from_tensor;
  encoder_param.to_tensor = d_from_tensor;
  encoder_param.self_attention.query_weight.kernel = d_attr_kernel_Q;
  encoder_param.self_attention.key_weight.kernel = d_attr_kernel_K;
  encoder_param.self_attention.value_weight.kernel = d_attr_kernel_V;
  encoder_param.self_attention.query_weight.bias = d_attr_bias_Q;
  encoder_param.self_attention.key_weight.bias = d_attr_bias_K;
  encoder_param.self_attention.value_weight.bias = d_attr_bias_V;
  encoder_param.attr_mask = d_attr_mask;
  encoder_param.self_attention.attention_output_weight.kernel = d_attr_output_kernel;
  encoder_param.self_attention.attention_output_weight.bias = d_attr_output_bias;
  encoder_param.self_layernorm.beta = d_attr_output_layernorm_beta;
  encoder_param.self_layernorm.gamma = d_attr_output_layernorm_gamma;
  encoder_param.ffn.intermediate_weight.kernel = d_inter_kernel;
  encoder_param.ffn.intermediate_weight.bias = d_inter_bias;
  encoder_param.ffn.output_weight.kernel = d_output_kernel;
  encoder_param.ffn.output_weight.bias = d_output_bias;
  encoder_param.ffn_layernorm.beta = d_output_layernorm_beta;
  encoder_param.ffn_layernorm.gamma = d_output_layernorm_gamma;
  encoder_param.transformer_out = d_transformer_out;
  encoder_param.cublas_handle = cublasHandle;
  encoder_param.cublaslt_handle = cublaslt_handle;
  encoder_param.stream = stream;
  encoder_param.amaxList = amaxList;
  encoder_param.layer_idx = 0;
  encoder_param.layer_num = num_layers;
  encoder_param.trt_seqlen_offset = d_trt_seqlen_offset;
  encoder_param.trt_seqlen_size = h_trt_seqlen_size;

  BertEncoderTransformer<EncoderTraits_> *encoder_transformer_ =
          new BertEncoderTransformer<EncoderTraits_>(int8_mode, allow_gemm_test);

  encoder_transformer_->allocateBuffer(&allocator, batch_size, from_seq_len, to_seq_len, head_num, size_per_head);

#ifdef OP_TIMES
  fastertransformer::TimeMap::StopProfiling();
#endif

  //warm up
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  for (int i = 0; i < ite; ++i)
  {
    if(is_remove_padding == true)
    {
      cudaMemcpyAsync(d_sequence_length, h_sequence_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice, stream);
      int* h_valid_word_num = new int[1];
      build_sequence_length_padding_offset_kernelLauncher(d_sequence_length,
            batch_size, seq_len, d_valid_word_num, d_tmp_sequence_id_offset, stream);
      cudaMemcpyAsync(h_valid_word_num, d_valid_word_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
      const int valid_word_num = h_valid_word_num[0];
      delete h_valid_word_num;

      remove_sequence_length_padding_kernelLauncher(d_from_tensor_with_padding,
                                                    d_from_tensor,
                                                    d_tmp_sequence_id_offset,
                                                    d_sequence_id_offset,
                                                    valid_word_num, hidden_dim,
                                                    stream);

      encoder_param.sequence_id_offset = d_sequence_id_offset;
      encoder_param.valid_word_num = valid_word_num;
    }
    else
    {
      encoder_param.valid_word_num = batch_size * seq_len;
    }


    encoder_transformer_->initialize(encoder_param);
    for(int i = 0; i < num_layers; i++)
      encoder_transformer_->forward();

    if(is_remove_padding == true)
    {
      rebuild_sequence_length_padding_kernelLauncher(d_transformer_out, d_transformer_out_with_padding,
                                                      d_sequence_id_offset,
                                                      encoder_param.valid_word_num, hidden_dim,
                                                      encoder_param.stream);
    }
  }

#ifdef OP_TIMES
  fastertransformer::TimeMap::StartProfiling();
#endif

  float total_time = 0.0;
  for (int i = 0; i < ite; ++i)
  {
    if(is_remove_padding == true)
    {
      cudaMemcpyAsync(d_sequence_length, h_sequence_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice, stream);
      int* h_valid_word_num = new int[1];
      build_sequence_length_padding_offset_kernelLauncher(d_sequence_length,
            batch_size, seq_len, d_valid_word_num, d_tmp_sequence_id_offset, stream);
      cudaMemcpyAsync(h_valid_word_num, d_valid_word_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
      const int valid_word_num = h_valid_word_num[0];
      delete h_valid_word_num;

      remove_sequence_length_padding_kernelLauncher(d_from_tensor_with_padding,
                                                    d_from_tensor,
                                                    d_tmp_sequence_id_offset,
                                                    d_sequence_id_offset,
                                                    valid_word_num, hidden_dim,
                                                    stream);

      encoder_param.sequence_id_offset = d_sequence_id_offset;
      encoder_param.valid_word_num = valid_word_num;
    }
    else
    {
      encoder_param.valid_word_num = batch_size * seq_len;
    }

    struct timeval start, end;
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);

    for(int i = 0; i < num_layers; i++){
      if (int8_mode != 0)
        encoder_param.layer_idx = i;
      encoder_transformer_->initialize(encoder_param);
      encoder_transformer_->forward();
    }

    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();

    total_time += ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001);

    if(is_remove_padding == true)
    {
      rebuild_sequence_length_padding_kernelLauncher(d_transformer_out, d_transformer_out_with_padding,
                                                      d_sequence_id_offset,
                                                      encoder_param.valid_word_num, hidden_dim,
                                                      encoder_param.stream);
    }
  }

#ifdef OP_TIMES
  fastertransformer::TimeMap::StopProfiling();
#endif

  cudaFree(d_sequence_length);
  cudaFree(d_from_tensor);
  cudaFree(d_transformer_out);
  cudaFree(d_attr_kernel_Q);
  cudaFree(d_attr_bias_Q);
  cudaFree(d_attr_bias_K);
  cudaFree(d_attr_bias_V);
  cudaFree(d_attr_output_kernel);
  cudaFree(d_attr_output_bias);
  cudaFree(d_attr_output_layernorm_beta);
  cudaFree(d_attr_output_layernorm_gamma);
  cudaFree(d_inter_kernel);
  cudaFree(d_inter_bias);
  cudaFree(d_output_kernel);
  cudaFree(d_output_bias);
  cudaFree(d_output_layernorm_beta);
  cudaFree(d_output_layernorm_gamma);
  cudaFree(amaxList);

  if(is_remove_padding == true)
  {
    cudaFree(d_sequence_id_offset);
    cudaFree(d_tmp_sequence_id_offset);
    cudaFree(d_from_tensor_with_padding);
    cudaFree(d_transformer_out_with_padding);
  }


  // printf("[INFO] batch_size %d seq_len %d layer %d FT-CPP-time %.2f ms ( %d iterations) \n", batch_size, seq_len, num_layers,
  // ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite, ite);
  delete encoder_transformer_;
  return total_time;
}
