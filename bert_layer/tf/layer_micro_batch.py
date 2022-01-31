import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import run_utils
import utils

tf.config.run_functions_eagerly(False)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
# parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=75)
# parser.add_argument("--discard-iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=75)
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=40)
parser.add_argument("--discard-iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=20)
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--dataset', nargs='?', default='random_384_512')
parser.add_argument("--no-ub", action="store_true")
args = parser.parse_args()

batch_size = args.batch_size
head_size = 64
num_heads = 8
ff_dim = 2048
model_dim = 512

pre_w = tf.constant(np.random.random_sample((3,num_heads,model_dim,head_size)).astype(np.float32))
pre_b = tf.constant(np.random.random_sample((3,num_heads,1,head_size,)).astype(np.float32))
pst_w = tf.constant(np.random.random_sample((model_dim,model_dim)).astype(np.float32))
pst_b = tf.constant(np.random.random_sample((model_dim,)).astype(np.float32))

@tf.function(experimental_compile=args.xla)
def mha(input,max_len,mask,micro_batch_size):
    pre_mm_out = tf.linalg.matmul(input, pre_w) + pre_b
    pre_out = tf.reshape(pre_mm_out, (3,num_heads,micro_batch_size,max_len,head_size))
    q, k, v = tf.split(pre_out, 3, axis=0)
    q = tf.reshape(q, (num_heads,micro_batch_size,max_len,head_size))
    k = tf.reshape(k, (num_heads,micro_batch_size,max_len,head_size))
    v = tf.reshape(v, (num_heads,micro_batch_size,max_len,head_size))
    k = tf.transpose(k, perm=[0,1,3,2])
    qkt_out = tf.matmul(q, k)
    attn_scores_out = tf.nn.softmax(qkt_out+mask, axis=3)
    attn_out = tf.matmul(attn_scores_out, v)
    attn_out = tf.transpose(attn_out, perm=[1,2,0,3])
    attn_out = tf.reshape(attn_out, (micro_batch_size,max_len,model_dim))
    pst_out = tf.matmul(attn_out, pst_w) + pst_b
    return pst_out

def run_batches(micro_batch_size, discard_iter, iterations):
    batches = run_utils.get_nlp_batches(batch_size, args.max_batches, args.dataset)
    batches = [i for i in batches if len(i) == batch_size]
    times = []
    for batch in batches:
        batch = np.sort(batch)

        bs = batch_size
        ubs = micro_batch_size
        micro_batches = np.split(batch, bs // ubs)

        data_dict = {}
        idx = 0
        for micro_batch in micro_batches:
            max_len = int(np.amax(micro_batch))
            mask = np.full((1,ubs,max_len,max_len), 0.0, dtype='float32')
            # for i in range(ubs):
            #     for j in range(max_len):
            #         if j >= micro_batch[i]:
            #             mask[0][i][j] = np.full((max_len,), -float('inf'), dtype='float32')
            #         else:
            #             mask[0][i][j][j+1:] = np.full((max_len - j - 1,), -float('inf'), dtype='float32')

            inputs = tf.constant(np.random.random_sample((ubs*max_len,model_dim)).astype(np.float32))
            mask = tf.constant(np.random.random_sample((1,ubs,max_len,max_len)).astype(np.float32))
            data_dict[idx] = (max_len, inputs, mask)
            idx += 1

        for i in range(discard_iter + iterations):
            batch_time = 0
            idx = 0
            for micro_batch in micro_batches:
                max_len, inputs, mask = data_dict[idx]

                t0 = time.perf_counter()
                mha(inputs, max_len, mask, micro_batch_size)
                t1 = time.perf_counter()
                batch_time += t1 - t0
                idx += 1
            times.append(batch_time)

    total = sum(times[discard_iter*len(batches):])
    avg = total / (iterations*len(batches)) * 1000.0
    return avg

if args.no_ub:
    min_time = run_batches(batch_size, args.discard_iter, args.iterations)
    print('RESULTS',str(min_time),str(batch_size),sep=',')
else:
    min_time = float('inf')
    min_ubs = -1
    for ubs in [2, 4, 8, 16, 32, 64, 128]:
        if ubs > batch_size:
            break
        ubs_time = run_batches(ubs, 2, 5)
        print(ubs, ubs_time)
        if ubs_time < min_time:
            min_time = ubs_time
            min_ubs = ubs

    min_time = run_batches(min_ubs, args.discard_iter, args.iterations)
    print('RESULTS',str(min_time),str(min_ubs),sep=',')
