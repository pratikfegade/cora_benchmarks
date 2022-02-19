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
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=20)
parser.add_argument("--discard-iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=40)
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

min_ubs = {
    ('race', 128): 32, ('wiki_128', 32): 32,
    ('cola', 32): 32, ('mnli', 128): 64
}

bs = batch_size
if args.no_ub: ubs = bs
else: ubs = min_ubs[(args.dataset, batch_size)]
micro_batch_size = ubs

pre_w = tf.constant(np.random.random_sample((3,num_heads,model_dim,head_size)).astype(np.float32))
pre_b = tf.constant(np.random.random_sample((3,num_heads,1,head_size,)).astype(np.float32))
pst_w = tf.constant(np.random.random_sample((model_dim,model_dim)).astype(np.float32))
pst_b = tf.constant(np.random.random_sample((model_dim,)).astype(np.float32))

@tf.function(experimental_compile=args.xla)
def pre_linear(input):
    pre_mm_out = tf.linalg.matmul(input, pre_w) + pre_b
    pre_out = tf.reshape(pre_mm_out, (3,num_heads,micro_batch_size,max_len,head_size))
    q, k, v = tf.split(pre_out, 3, axis=0)
    return q, k, v

@tf.function(experimental_compile=args.xla)
def qkt(q, k, mask):
    q = tf.reshape(q, (num_heads,micro_batch_size,max_len,head_size))
    k = tf.reshape(k, (num_heads,micro_batch_size,max_len,head_size))
    k = tf.transpose(k, perm=[0,1,3,2])
    return tf.matmul(q, k) + mask

@tf.function(experimental_compile=args.xla)
def softmax(attn):
    return tf.nn.softmax(attn, axis=3)

@tf.function(experimental_compile=args.xla)
def attn_v(attn, v):
    v = tf.reshape(v, (num_heads,micro_batch_size,max_len,head_size))
    attn_out = tf.matmul(attn, v)
    attn_out = tf.transpose(attn_out, perm=[1,2,0,3])
    return tf.reshape(attn_out, (micro_batch_size,max_len,model_dim))

@tf.function(experimental_compile=args.xla)
def post_linear(attn_out):
    return tf.matmul(attn_out, pst_w) + pst_b

batches = run_utils.get_nlp_batches(batch_size, args.max_batches, args.dataset)
op_order = [
    'pre_linear',
    'qkt',
    'softmax',
    'attn_v',
    'post_linear',
]

op_times = {}
for op in op_order:
    op_times[op] = []

for batch in batches:
    batch = np.sort(batch)
    op_ubs_times = {}
    for op in op_order:
        op_ubs_times[op] = 0.0

    micro_batches = np.split(batch, bs // ubs)

    for micro_batch in micro_batches:
        max_len = int(np.amax(micro_batch))
        mask = np.full((1,micro_batch_size,max_len,max_len), 0.0, dtype='float32')
        # for i in range(micro_batch_size):
        #     for j in range(max_len):
        #         if j >= micro_batch[i]:
        #             mask[0][i][j] = np.full((max_len,), -float('inf'), dtype='float32')
        #         else:
        #             mask[0][i][j][j+1:] = np.full((max_len - j - 1,), -float('inf'), dtype='float32')
        input = tf.constant(np.random.random_sample((micro_batch_size*max_len,model_dim)).astype(np.float32))
        mask = tf.constant(mask.astype(np.float32))
        q = tf.constant(np.random.random_sample((1,num_heads,micro_batch_size,max_len,head_size)).astype(np.float32))
        k = tf.constant(np.random.random_sample((1,num_heads,micro_batch_size,max_len,head_size)).astype(np.float32))
        v = tf.constant(np.random.random_sample((1,num_heads,micro_batch_size,max_len,head_size)).astype(np.float32))
        attn = tf.constant(np.random.random_sample((num_heads,micro_batch_size,max_len,max_len)).astype(np.float32))
        attn_out = tf.constant(np.random.random_sample((micro_batch_size,max_len,model_dim)).astype(np.float32))

        ops = {
            'pre_linear': (pre_linear, [input]),
            'qkt': (qkt, [q, k, mask]),
            'softmax': (softmax, [attn]),
            'attn_v': (attn_v, [attn, v]),
            'post_linear': (post_linear, [attn_out])
        }

        for op in op_order:
            func = ops[op][0]
            inputs = ops[op][1]

            for i in range(args.discard_iter + args.iterations):
                t0 = time.perf_counter()
                func(*inputs)
                t1 = time.perf_counter()
                if i >= args.discard_iter:
                    op_ubs_times[op] += (t1 - t0)

    for op in op_ubs_times:
        op_times[op].append(op_ubs_times[op])

for op, times in op_times.items():
    # total = sum(times[args.discard_iter*args.max_batches:])
    # avg = total / (args.iterations*args.max_batches) * 1000.0
    total = sum(times)
    avg = total / (args.iterations*args.max_batches) * 1000.0
    print('RESULTS',op,str(avg),sep=',')