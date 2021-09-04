import os
import sys
import numpy as np
import time
import tvm
import argparse
import ast
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import run_utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--out-file', dest='out_file', type=str)
args = parser.parse_args()

def flops_for_dataset_batch(dataset, batch_size, max_batches):
    batches = run_utils.get_nlp_batches(batch_size, max_batches, dataset, run_utils.DATA_DIR)

    MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(dataset), 32)
    NUM_HEADS = 8
    HEAD_SIZE = 64
    MODEL_DIM = NUM_HEADS * HEAD_SIZE
    FF_DIM = 2048
    QKV_NUM = 3

    giga = 1e9
    total_dense_gflops = 0.0
    total_ragged_gflops = 0.0
    for batch in batches:
        if len(batch) != batch_size:
            continue

        batch_max_len = np.amax(batch)
        sum1 = run_utils.prefix_sum(batch_size, lambda i: batch[i])
        sum2 = run_utils.prefix_sum(batch_size, lambda i: batch[i] * batch[i])

        def get_pre_linear_flops():
            dense_flops = (QKV_NUM * batch_size * batch_max_len * NUM_HEADS * HEAD_SIZE * MODEL_DIM + # MM
                           QKV_NUM * batch_size * batch_max_len * NUM_HEADS * HEAD_SIZE)  # Bias add
            ragged_flops = QKV_NUM * sum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM
            return dense_flops, ragged_flops

        def get_qkt_flops():
            dense_flops = batch_size * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE
            ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE
            return dense_flops, ragged_flops

        def get_softmax_flops():
            dense_flops = batch_size * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE * 5 # (max + max_sub + exp + sum + exp_div)
            ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE * 5
            return dense_flops, ragged_flops

        def get_attn_v_flops():
            dense_flops = batch_size * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE
            ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE
            return dense_flops, ragged_flops

        def get_post_linear_flops():
            dense_flops = (batch_size * batch_max_len * NUM_HEADS * HEAD_SIZE * MODEL_DIM + # MM
                           batch_size * batch_max_len * MODEL_DIM)                          # Bias add
            ragged_flops = sum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM
            return dense_flops, ragged_flops

        def get_norm_add1_flops():
            dense_flops = batch_size * batch_max_len * MODEL_DIM * 4 # (residual_add + mean + std + div)
            ragged_flops = sum1 * MODEL_DIM * 4
            return dense_flops, ragged_flops

        def get_ff1_flops():
            dense_flops = (batch_size * batch_max_len * FF_DIM * MODEL_DIM + # MM
                           batch_size * batch_max_len * FF_DIM +             # Bias add
                           batch_size * batch_max_len * FF_DIM)              # Relu
            ragged_flops = (sum1 * FF_DIM * MODEL_DIM + # MM
                            sum1 * FF_DIM +             # Bias add
                            sum1 * FF_DIM)              # Relu
            return dense_flops, ragged_flops

        def get_ff2_flops():
            dense_flops = batch_size * batch_max_len * FF_DIM * MODEL_DIM              # Relu
            ragged_flops = sum1 * FF_DIM * MODEL_DIM
            return dense_flops, ragged_flops

        def get_norm_add2_flops():
            dense_flops = batch_size * batch_max_len * MODEL_DIM * 4 # (residual_add + mean + std + div)
            ragged_flops = sum1 * MODEL_DIM * 4
            return dense_flops, ragged_flops

        flops_ops = [
            get_qkt_flops(),
            get_softmax_flops(),
            get_attn_v_flops(),
            get_post_linear_flops(),
            get_norm_add1_flops(),
            get_ff1_flops(),
            get_ff2_flops(),
            get_norm_add2_flops()
        ]

        for op in flops_ops:
            total_dense_gflops += op[0] / giga
            total_ragged_gflops += op[1] / giga

    total_dense_gflops /= len(batches)
    total_ragged_gflops /= len(batches)
    return total_dense_gflops, total_ragged_gflops


with open(args.out_file, 'w') as out_file:
    print('Dataset', 'Batch Size', 'Dense Flops', 'Ragged Flops', sep=',', file=out_file)
    for dataset in run_utils.DATASETS:
        for i in range(8):
            batch_size = 1 << i
            dense, ragged = flops_for_dataset_batch(dataset, batch_size, args.max_batches)
            print(dataset, i, dense, ragged, sep=',', file=out_file)
