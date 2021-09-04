import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils

parser = run_utils.get_cmd_parser()
args = parser.parse_args()

BATCH_SIZE = te.var('bs')
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
NUM_HEADS = 8
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')

TILE1=64
TILE2=16
def len1_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
def len2_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [s1], [], lambda: lambda s: utils.ceilmult(s + 1, pad))
l1ufw = len1_ufw('s1', TILE1)
l2ufw = len2_ufw('s2', TILE2)

luf1 = l1ufw.get_uf()
luf2 = l2ufw.get_uf()
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: luf1,
    3: luf2,
}

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
width_ufs=[ls[0], ls[2], ls[1], luf1]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[2], ls[1]]
Amax = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS), [bd, s1, md], loop_ufs,
                         lambda ds, rds: tvm.max(A[ds[bd], ds[s1], ds[md], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Amax', reduce_axis_ufs = [('k', luf2)])

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
Aexp = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                         lambda ds: tvm.exp((A[ds[bd], ds[s1], ds[md], ds[s2]] -
                                             Amax[ds[bd], ds[s1], ds[md]]) * scale), name = 'Aexp')

loop_ufs=[ls[0], ls[2], ls[1]]
Asum = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS), [bd, s1, md], loop_ufs,
                         lambda ds, rds: tvm.sum(Aexp[ds[bd], ds[s1], ds[md], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Asum', reduce_axis_ufs = [('k', luf2)])

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
width_ufs=[[ls[0], ls[2], ls[1], luf1]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: Aexp[ds[bd], ds[s1], ds[md], ds[s2]] / Asum[ds[bd], ds[s1], ds[md]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = tvm.thread_axis("threadIdx.x")
thread_y = tvm.thread_axis("threadIdx.y")
block_x = tvm.thread_axis("blockIdx.x")
block_y = tvm.thread_axis("blockIdx.y")


ntx = 16
ko, ki = s[Amax].split(s[Amax].op.reduce_axis[0], factor = ntx)
Amax_rf = s.rfactor(Amax, ki, 1)

ko, ki = s[Asum].split(s[Asum].op.reduce_axis[0], factor = ntx)
Asum_rf = s.rfactor(Asum, ki, 1)

b, s1, h, s2 = s[O].leaf_iter_vars
f = s[O].fuse(b, s1)
s[O].bind(f, block_x)
# s[O].bind(h, thread_y)

xo, xi = s[O].split(s2, factor = ntx)
s[O].bind(xi, thread_x)
s[Amax].bind(s[Amax].op.reduce_axis[0], thread_x)
s[Asum].bind(s[Asum].op.reduce_axis[0], thread_x)

s[Amax].compute_at(s[O], h)
s[Amax_rf].compute_at(s[Amax], s[Amax].leaf_iter_vars[3])
s[Asum].compute_at(s[O], h)
s[Asum_rf].compute_at(s[Asum], s[Asum].leaf_iter_vars[3])
s[Aexp].compute_inline()

s[Amax].set_scope('local')
s[Amax_rf].set_scope('local')
s[Asum].set_scope('local')
s[Asum_rf].set_scope('local')
s[Aexp].set_scope('local')

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (l1ufw.get_fn(lens)(b) *
                                                       l1ufw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (l1ufw.get_fn(lens)(b) *
                                                       l1ufw.get_fn(lens)(b)))
    }

inputs = [[lens], [BATCH_SIZE, A, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE))

# out = out[2]
# ctr = 0
# out = out.flatten()
# for i in range(args.batch_size):
#     rounded = utils.ceilmult(batches[0][i], 32)
#     this_extent = utils.ceilmult(batches[0][i], 32)
#     this_storage_extent = utils.ceilmult(batches[0][i], 64) * utils.ceilmult(batches[0][i], 64) * NUM_HEADS
#     print(batches[0][i], rounded, 1 / rounded, np.mean(out[ctr:ctr + this_extent]))
#     ctr += this_storage_extent
