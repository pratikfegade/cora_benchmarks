import gc
import os
import sys
import numpy as np
import time
import tvm
import argparse
import ast
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils
from common import Op

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=200, type=int)
parser.add_argument('--witers', dest='witers', default=20, type=int)
parser.add_argument('--iters', dest='iters', default=40, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--average', dest='average', default=False, action='store_true')
parser.add_argument('--bin-packed', dest='bin_packed', default=False, action='store_true')
parser.add_argument('--masked-mha', dest='masked_mha', default=False, action='store_true')
parser.add_argument('--plain-mha', dest='plain_mha', default=False, action='store_true')
parser.add_argument('--per-op', dest='per_op', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random_384_512')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MAX_LEN = max(64, utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32))
NUM_HEADS = 8
HEAD_SIZE = 64
MODEL_DIM = NUM_HEADS * HEAD_SIZE
FF_DIM = 2048

dev_ctx = run_utils.get_ctx(args.target)
cpu_ctx = run_utils.get_ctx("llvm")

only_mha = args.plain_mha or args.masked_mha

qkt_module = 'qkt_cpu'
attn_v_module = 'attn_v_cpu'
softmax_module = 'softmax_cpu'
assert not (args.bin_packed and only_mha)
assert not (args.masked_mha and args.plain_mha)
if args.bin_packed:
    qkt_module = 'qkt_bin_packed'
    attn_v_module = 'attn_v_bin_packed'

ops = {
    'pre_linear': Op('pre_linear', 'pre_linear_blasized', BATCH_SIZE, [], cpu_ctx, dev_ctx),
    'add_pad': Op('add_pad', 'add_padding_cpu', BATCH_SIZE, [], cpu_ctx, dev_ctx),
    'qkt': Op('qkt', qkt_module, BATCH_SIZE, [], cpu_ctx, dev_ctx, variants=[1,2]),
    'softmax': Op('softmax', softmax_module, BATCH_SIZE, [], cpu_ctx, dev_ctx),
    'attn_v': Op('attn_v', attn_v_module, BATCH_SIZE, [], cpu_ctx, dev_ctx),
    'rem_pad': Op('rem_pad', 'remove_padding_cpu', BATCH_SIZE, [], cpu_ctx, dev_ctx),
    'post_linear': Op('post_linear', 'post_linear_blasized', BATCH_SIZE, [], cpu_ctx, dev_ctx),
}

ops_order = [
    ops['pre_linear'],
    ops['add_pad'],
    ops['qkt'],
    ops['softmax'],
    ops['attn_v'],
    ops['rem_pad'],
    ops['post_linear'],
]


# l_inputs: Allocate tensors
batches = run_utils.get_nlp_batches(args.batch_size, args.max_batches, args.dataset)
if args.dataset not in ['race', 'squadv2']:
    batches = run_utils.reverse_sort_batches(batches)
if args.average:
    for i in range(len(batches)):
        avg = np.mean(batches[i])
        batches[i].fill(avg)
        batches[i] = batches[i].astype('int32')
batches = run_utils.append_padded_sum(batches, 64)

pre_linear_in_w = run_utils.create_tvm_array((3, NUM_HEADS * HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
pre_linear_in_b = run_utils.create_tvm_array((3, NUM_HEADS * HEAD_SIZE,), "float32", dev_ctx, lw_args={})
post_linear_in_w = run_utils.create_tvm_array((MODEL_DIM, NUM_HEADS * HEAD_SIZE), "float32", dev_ctx, lw_args={})
post_linear_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
if not only_mha:
    norm_add1_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
    norm_add1_in_g = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
    norm_add2_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
    norm_add2_in_g = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
    ff1_in_w = run_utils.create_tvm_array((MODEL_DIM, FF_DIM), "float32", dev_ctx, lw_args={})
    ff1_in_b = run_utils.create_tvm_array((FF_DIM,), "float32", dev_ctx, lw_args={})
    ff2_in_w = run_utils.create_tvm_array((FF_DIM, MODEL_DIM), "float32", dev_ctx, lw_args={})
    ff2_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})

times = []
time_dict = {}
if args.per_op:
    for op in ops_order:
        time_dict[op.name] = []
batch_size_ = BATCH_SIZE + 1
optimal_variants = None
if len(batches[-1]) != batch_size_:
    batches.pop()
print([len(i) for i in batches])


if True:
    # t_inputs: Allocate tensors
    pre_linear_in_qkv = run_utils.create_tvm_array((batch_size_ * MAX_LEN, MODEL_DIM), "float32", dev_ctx)
    pre_linear_out = run_utils.create_tvm_array((3, batch_size_ * MAX_LEN, NUM_HEADS * HEAD_SIZE), "float32", dev_ctx)

    add_pad_a = pre_linear_out.create_view((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE))
    add_pad_o = run_utils.create_tvm_array((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE), "float32", dev_ctx)

    qkt_in_q = add_pad_o
    qkt_in_k = add_pad_o
    qkt_out = run_utils.create_tvm_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), "float32", dev_ctx)

    softmax_in = qkt_out
    softmax_out = run_utils.create_tvm_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), "float32", dev_ctx)

    attn_v_in_attn = softmax_out
    attn_v_in_v = add_pad_o
    attn_v_out = run_utils.create_tvm_array((batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE), "float32", dev_ctx)

    rem_pad_a = attn_v_out
    rem_pad_o = run_utils.create_tvm_array((batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE), "float32", dev_ctx)

    post_linear_in_a = rem_pad_o.create_view((batch_size_ * MAX_LEN, NUM_HEADS * HEAD_SIZE))
    post_linear_in_a2 = pre_linear_in_qkv.create_view((batch_size_, MAX_LEN, MODEL_DIM))
    post_linear_out = run_utils.create_tvm_array((batch_size_ * MAX_LEN, MODEL_DIM), "float32", dev_ctx)
    # post_linear_out = pre_linear_out.create_view((batch_size_ * MAX_LEN, MODEL_DIM))

    gc.collect()

    ops['pre_linear'].tensor_inputs = [pre_linear_in_qkv, pre_linear_in_w, pre_linear_in_b, pre_linear_out]
    ops['add_pad'].tensor_inputs = [add_pad_a, add_pad_o]
    ops['qkt'].tensor_inputs = [qkt_in_q, qkt_in_k, qkt_out]
    ops['softmax'].tensor_inputs = [softmax_in, softmax_out]
    ops['attn_v'].tensor_inputs = [attn_v_in_v, attn_v_in_attn, attn_v_out]
    ops['rem_pad'].tensor_inputs = [rem_pad_a, rem_pad_o]
    ops['post_linear'].tensor_inputs = [post_linear_in_a, post_linear_in_w, post_linear_in_b, post_linear_out]

for batch in batches:
    sum1 = run_utils.prefix_sum(batch_size_, lambda i: batch[i])
    sum16 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 16))
    sum32 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 32))
    sum64 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64))
    sum264 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64) *
                                  utils.ceilmult(batch[i], 64))

    l_inputs = [tvm.nd.array(batch, cpu_ctx)]

    if not optimal_variants:
        optimal_variants = {}
        for op in ops_order:
            optimal_variants[op.name] = op.profile_variants(l_inputs, dev_ctx)
        print(optimal_variants)

    if args.per_op:
        ops['pre_linear'].set_inputs_and_variant(l_inputs, 0, int(sum(batch))//64)
        ops['post_linear'].set_inputs_and_variant(l_inputs, 0, int(sum(batch))//64)
        this_time = 0
        for op in ops_order:
            print('Executing', op.name)
            if op.name == 'pre_linear' or op.name == 'post_linear':
                op_time = op.execute_multiple(l_inputs, dev_ctx, f_sum=int(sum(batch))//64)
            else:
                op_time = op.execute_multiple(l_inputs, dev_ctx)
            print('  Time', op_time)
            time_dict[op.name].append(op_time)
            this_time += op_time
        times.append(this_time)
    else:
        for op in ops_order: op.set_inputs_and_variant(l_inputs, 0)
        ops['pre_linear'].set_inputs_and_variant(l_inputs, 0, int(sum(batch))//64)
        ops['post_linear'].set_inputs_and_variant(l_inputs, 0, int(sum(batch))//64)
        for i in range(args.witers):
            for op in ops_order: op.execute()
        dev_ctx.sync()
        start = time.perf_counter()
        for i in range(args.iters):
            for op in ops_order: op.execute()
        dev_ctx.sync()
        end = time.perf_counter()
        times.append((end - start) / args.iters)

        for op in ops_order: op.reset()

    gc.collect()

if args.per_op:
    for op in ops_order:
        op_times = time_dict[op.name]
        time = (sum(op_times)*1000.0) / len(op_times)
        print('RESULTS', op.name, time, sep=',')

total_time = sum(times)*1000.0
if args.per_op:
    pass
    # print('RESULTS,Sum', total_time / (len(batches)), sep=',')
else:
    print('RESULTS', total_time / (len(batches)), sep=',')
