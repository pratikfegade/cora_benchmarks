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
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=200, type=int)
parser.add_argument('--witers', dest='witers', default=50, type=int)
parser.add_argument('--iters', dest='iters', default=200, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random_384_512')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
NUM_HEADS = 8
HEAD_SIZE = 64
MODEL_DIM = NUM_HEADS * HEAD_SIZE
FF_DIM = 2048

def load_module(op_name):
    return tvm.runtime.module.load_module(run_utils.MODULE_DIR + '/' + op_name + '.so')

def load_ibuf_info(op_name):
    bufs = [[], []]
    with open(run_utils.MODULE_DIR + '/' + op_name + '_bufs.txt') as topo_file:
        for line in topo_file:
            arr = line.strip().split(' ')
            data = (ast.literal_eval(arr[1]), arr[2])
            if arr[0] == 'h':
                bufs[0].append(data)
            else:
                bufs[1].append(data)
    return bufs

def create_ibufs(ibuf_info, cpu_ctx, dev_ctx):
    host_bufs = [tvm.nd.array(run_utils.create_numpy_array(i[0], i[1]), cpu_ctx) for i in ibuf_info[0]]
    dev_bufs = [tvm.nd.array(run_utils.create_numpy_array(i[0], i[1]), dev_ctx) for i in ibuf_info[1]]
    return host_bufs, dev_bufs

class Op:
    def __init__(self, name, module_name, tensor_inputs, cpu_ctx, dev_ctx):
        self.name = name
        self.module_name = module_name
        self.tensor_inputs = tensor_inputs
        self.module = load_module(module_name)
        ibuf_info = load_ibuf_info(module_name)
        self.host_ibufs, self.dev_ibufs = create_ibufs(ibuf_info, cpu_ctx, dev_ctx)

    def execute(self, l_inputs):
        inputs = self.tensor_inputs + l_inputs + self.host_ibufs + self.dev_ibufs
        self.module(*inputs)

dev_ctx = run_utils.get_ctx(args.target)
cpu_ctx = run_utils.get_ctx("llvm")

ops = {
    'pre_linear2': Op('pre_linear2', 'pre_linear2', [], cpu_ctx, dev_ctx),
    'qkt2': Op('qkt2', 'qkt2', [], cpu_ctx, dev_ctx),
    'softmax2': Op('softmax2', 'softmax2', [], cpu_ctx, dev_ctx),
    'attn_v2': Op('attn_v2', 'attn_v2', [], cpu_ctx, dev_ctx),
    'post_linear2': Op('post_linear2', 'post_linear2', [], cpu_ctx, dev_ctx),
    'norm_add1': Op('norm_add1', 'norm_add', [], cpu_ctx, dev_ctx),
    'ff1': Op('ff1', 'ff1', [], cpu_ctx, dev_ctx),
    'ff2': Op('ff2', 'ff2', [], cpu_ctx, dev_ctx),
    'norm_add2': Op('norm_add2', 'norm_add', [], cpu_ctx, dev_ctx),
}

ops_order = [
    ops['pre_linear2'],
    ops['qkt2'],
    ops['softmax2'],
    ops['attn_v2'],
    ops['post_linear2'],
    ops['norm_add1'],
    ops['ff1'],
    ops['ff2'],
    ops['norm_add2'],
]

# l_inputs: Allocate tensors
batches = run_utils.get_nlp_batches(args.dataset, args.datadir)
batches = run_utils.add_padded_sum(batches, 128)

pre_linear_in_w = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
pre_linear_in_b = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE,), "float32", dev_ctx, lw_args={})
post_linear_in_w = run_utils.create_tvm_array((NUM_HEADS * HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
post_linear_in_b = run_utils.create_tvm_array((MODEL_DIMm), "float32", dev_ctx, lw_args={})
ff1_in_w = run_utils.create_tvm_array((MODEL_DIM, FF_DIM), "float32", dev_ctx, lw_args={})
ff1_in_b = run_utils.create_tvm_array((FF_DIM,), "float32", dev_ctx, lw_args={})
ff2_in_w = run_utils.create_tvm_array((FF_DIM, MODEL_DIM), "float32", dev_ctx, lw_args={})

times = []
ctr = 0
for batch in batches:
    ctr += 1
    print('BATCH', ctr)
    batch = np.sort(batch).astype('int32')
    batch_size_ = BATCH_SIZE + 1

    sum1 = run_utils.prefix_sum(batch_size_, lambda i: batch[i])
    sum16 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 16))
    sum32 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 32))
    sum64 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64))
    sum264 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64) * utils.ceilmult(batch[i], 64))

    # t_inputs: Allocate tensors
    pre_linear_in_qkv = run_utils.create_ragged_array((batch_size_ * MAX_LEN, MODEL_DIM), sum1*MODEL_DIM, "float32", dev_ctx)
    pre_linear_out = run_utils.create_ragged_array((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                   3*sum64*NUM_HEADS*HEAD_SIZE, "float32", dev_ctx)

    qkt_in_q = pre_linear_out
    qkt_in_k = pre_linear_out
    qkt_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum264, "float32", dev_ctx)

    softmax_in = qkt_out
    softmax_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum264, "float32", dev_ctx)

    attn_v_in_attn = softmax_out
    attn_v_in_v = pre_linear_out
    attn_v_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                               NUM_HEADS*HEAD_SIZE*sum64, "float32", dev_ctx)

    post_linear_in_a = attn_v_out
    # post_linear_out = run_utils.create_ragged_array((batch_size_ * MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)
    post_linear_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)

    norm_add1_in_a1 = pre_linear_in_qkv.create_view((batch_size_, MAX_LEN, MODEL_DIM))
    # norm_add1_in_a2 = post_linear_out.create_view((batch_size_, MAX_LEN, MODEL_DIM))
    norm_add1_in_a2 = post_linear_out
    norm_add1_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)

    ff1_in_a = norm_add1_out
    ff1_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, FF_DIM), FF_DIM*sum1, "float32", dev_ctx)

    ff2_in_a = ff1_out
    ff2_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)

    norm_add2_in_a1 = norm_add1_out
    norm_add2_in_a2 = ff2_out
    norm_add2_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)


    ops['pre_linear2'].tensor_inputs = [pre_linear_in_qkv, pre_linear_in_w, pre_linear_in_b, pre_linear_out]
    ops['qkt2'].tensor_inputs = [qkt_in_q, qkt_in_k, qkt_out]
    ops['softmax2'].tensor_inputs = [softmax_in, softmax_out]
    ops['attn_v2'].tensor_inputs = [attn_v_in_v, attn_v_in_attn, attn_v_out]
    ops['post_linear2'].tensor_inputs = [post_linear_in_a, post_linear_in_w, post_linear_in_b, post_linear_out]
    ops['norm_add1'].tensor_inputs = [norm_add1_in_a1, norm_add1_in_a2, norm_add1_out]
    ops['ff1'].tensor_inputs = [ff1_in_a, ff1_in_w, ff1_in_b, ff1_out]
    ops['ff2'].tensor_inputs = [ff2_in_a, ff2_in_w, ff2_out]
    ops['norm_add2'].tensor_inputs = [norm_add2_in_a1, norm_add2_in_a2, norm_add2_out]

    l_inputs = [tvm.nd.array(batch, cpu_ctx)]

    start = time.perf_counter()

    this_times = []
    for i in range(args.witers + args.iters):
        start = time.perf_counter()
        for op in ops_order: op.execute(l_inputs)
        dev_ctx.sync()
        end = time.perf_counter()
        this_times.append(end - start)

    this_times = this_times[args.witers:]
    times.append(sum(this_times) / args.iters)

total_time = sum(times)*1000.0
print(total_time / (len(batches)))