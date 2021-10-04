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
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=50)
parser.add_argument("--discard-iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=50)
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--dataset', nargs='?', default='random_384_512')
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

# @tf.function(experimental_compile=args.xla)
def mha(input,max_len,mask):
    pre_mm_out = tf.linalg.matmul(input, pre_w) + pre_b
    pre_out = tf.reshape(pre_mm_out, (3,num_heads,batch_size,max_len,head_size))
    q, k, v = tf.split(pre_out, 3, axis=0)
    q = tf.reshape(q, (num_heads,batch_size,max_len,head_size))
    k = tf.reshape(k, (num_heads,batch_size,max_len,head_size))
    v = tf.reshape(v, (num_heads,batch_size,max_len,head_size))
    k = tf.transpose(k, perm=[0,1,3,2])
    qkt_out = tf.matmul(q, k)
    attn_scores_out = tf.nn.softmax(qkt_out+mask, axis=3)
    attn_out = tf.matmul(attn_scores_out, v)
    attn_out = tf.transpose(attn_out, perm=[1,2,0,3])
    attn_out = tf.reshape(attn_out, (batch_size,max_len,model_dim))
    pst_out = tf.matmul(attn_out, pst_w) + pst_b
    return pst_out

batches = run_utils.get_nlp_batches(batch_size, args.max_batches, args.dataset)
times = []
for batch in batches:
    max_len = int(np.amax(batch))

    # tf.profiler.experimental.start('log')
    inputs = tf.constant(np.random.random_sample((batch_size*max_len,model_dim)).astype(np.float32))
    mask = tf.constant(np.random.random_sample((1,batch_size,max_len,max_len)).astype(np.float32))
    for i in range(args.discard_iter + args.iterations):
        t0 = time.perf_counter()
        mha(inputs,max_len,mask)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    # tf.profiler.experimental.stop()

total = sum(times[args.discard_iter*args.max_batches:])
avg = total / (args.iterations*args.max_batches) * 1000.0
print('RESULTS',str(avg),sep=',')
