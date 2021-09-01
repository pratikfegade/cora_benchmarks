import sys
import numpy as np
import time
import tvm
import argparse
import ast
sys.path.append("../")
import run_utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--dataset', nargs='?', default='random_384_512')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
NUM_HEADS = 8
HEAD_SIZE = 64
MODEL_DIM = NUM_HEADS * HEAD_SIZE
FF_DIM = 2048
QKV_NUM = 3

batches = run_utils.get_nlp_batches(args.batch_size, args.max_batches, args.dataset, args.datadir)

giga = 1e9
total_dense_gflops = 0.0
total_ragged_gflops = 0.0
for batch in batches:
    batch_max_len = np.amax(batch)
    sum1 = run_utils.prefix_sum(BATCH_SIZE, lambda i: batch[i])
    sum2 = run_utils.prefix_sum(BATCH_SIZE, lambda i: batch[i] * batch[i])

    def get_pre_linear_flops():
        dense_flops = (QKV_NUM * BATCH_SIZE * batch_max_len * NUM_HEADS * HEAD_SIZE * MODEL_DIM + # MM
                       QKV_NUM * BATCH_SIZE * batch_max_len * NUM_HEADS * HEAD_SIZE)  # Bias add
        ragged_flops = QKV_NUM * sum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM
        return dense_flops, ragged_flops

    def get_qkt_flops():
        dense_flops = BATCH_SIZE * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE
        ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE
        return dense_flops, ragged_flops

    def get_softmax_flops():
        dense_flops = BATCH_SIZE * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE * 5 # (max + max_sub + exp + sum + exp_div)
        ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE * 5
        return dense_flops, ragged_flops

    def get_attn_v_flops():
        dense_flops = BATCH_SIZE * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE
        ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE
        return dense_flops, ragged_flops

    def get_post_linear_flops():
        dense_flops = (BATCH_SIZE * batch_max_len * NUM_HEADS * HEAD_SIZE * MODEL_DIM + # MM
                       BATCH_SIZE * batch_max_len * MODEL_DIM)                          # Bias add
        ragged_flops = sum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM
        return dense_flops, ragged_flops

    def get_norm_add1_flops():
        dense_flops = BATCH_SIZE * batch_max_len * MODEL_DIM * 4 # (residual_add + mean + std + div)
        ragged_flops = sum1 * MODEL_DIM * 4
        return dense_flops, ragged_flops

    def get_ff1_flops():
        dense_flops = (BATCH_SIZE * batch_max_len * FF_DIM * MODEL_DIM + # MM
                       BATCH_SIZE * batch_max_len * FF_DIM +             # Bias add
                       BATCH_SIZE * batch_max_len * FF_DIM)              # Relu
        ragged_flops = (sum1 * FF_DIM * MODEL_DIM + # MM
                        sum1 * FF_DIM +             # Bias add
                        sum1 * FF_DIM)              # Relu
        return dense_flops, ragged_flops

    def get_ff2_flops():
        dense_flops = BATCH_SIZE * batch_max_len * FF_DIM * MODEL_DIM              # Relu
        ragged_flops = sum1 * FF_DIM * MODEL_DIM
        return dense_flops, ragged_flops

    def get_norm_add2_flops():
        dense_flops = BATCH_SIZE * batch_max_len * MODEL_DIM * 4 # (residual_add + mean + std + div)
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
print(total_dense_gflops, total_ragged_gflops)
