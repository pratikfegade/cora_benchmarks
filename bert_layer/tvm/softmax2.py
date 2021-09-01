import os
import numpy as np
import run_utils
import argparse
import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw

parser = run_utils.get_cmd_parser()
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
NUM_HEADS = 8
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1_1', 1)
lufw32 = len_ufw('s2_32', 32)
lufw64 = len_ufw('s64', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufw1.get_uf(),
    3: lufw32.get_uf(),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
width_ufs=[ls[0], lufw64.get_uf(), ls[1], lufw64.get_uf()]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)


loop_ufs=[ls[0], ls[2], ls[1]]
Amax = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS), [bd, s1, md], loop_ufs,
                         lambda ds, rds: tvm.max(A[ds[bd], ds[s1], ds[md], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Amax', reduce_axis_ufs = [('k', lufw32.get_uf())])

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
Aexp = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                         lambda ds: tvm.exp((A[ds[bd], ds[s1], ds[md], ds[s2]] -
                                             Amax[ds[bd], ds[s1], ds[md]]) * scale), name = 'Aexp')

loop_ufs=[ls[0], ls[2], ls[1]]
Asum = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS), [bd, s1, md], loop_ufs,
                         lambda ds, rds: tvm.sum(Aexp[ds[bd], ds[s1], ds[md], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Asum', reduce_axis_ufs = [('k', lufw32.get_uf())])

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: Aexp[ds[bd], ds[s1], ds[md], ds[s2]] / Asum[ds[bd], ds[s1], ds[md]],
                      name = 'O', width_uf_lists=None if args.dense_storage else [width_ufs])

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")

    ko, ki = s[Amax].split(s[Amax].op.reduce_axis[0], factor = 32)
    Amax_rf = s.rfactor(Amax, ki, 1)

    ko, ki = s[Asum].split(s[Asum].op.reduce_axis[0], factor = 32)
    Asum_rf = s.rfactor(Asum, ki, 1)

    b, s1, h, s2 = s[O].leaf_iter_vars
    f = s[O].fuse(b, s1)
    s[O].bind(f, block_x)
    # s[O].bind(h, thread_y)

    xo, xi = s[O].split(s2, factor = 32)
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
    inputs = [[lens], [A, O]]
else:
    inputs = [[lens], [A, Amax, Aexp, Asum, O]]

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (lufw64.get_fn(lens)(b) *
                                                       lufw64.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (lufw64.get_fn(lens)(b) *
                                                       lufw64.get_fn(lens)(b)))
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn)
# out = out[1].asnumpy()
# for i in range(args.batch_size):
#     rounded = utils.ceilmult(batches[0][i], 32)
#     print(1 / rounded, np.mean(out[i, 0, 0, 0:rounded]))