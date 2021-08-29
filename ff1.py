import numpy as np
import math
import os
import utils
import run_utils
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw

parser = run_utils.get_cmd_parser()
args = parser.parse_args()

BATCH_SIZE = args.batch_size + 1
IN_SIZE = 512
OUT_SIZE = 2048
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 64)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_ufw(name): return Ufw(name, "l", (1, MAX_LEN), [bd], [lens], lambda lens: lambda b: lens[b])
lufw = len_ufw('s1')

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: lufw.get_uf(),
    2: Uf.from_constant('id', IN_SIZE, "l"),
    3: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((IN_SIZE, OUT_SIZE), name='W')
B = te.placeholder((OUT_SIZE, ), name='B')

loop_ufs=[ls[0], ls[1], ls[3]]
width_ufs=None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
M = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(W[k, ds[od]] * A[ds[bd], ds[s1], k], axis = k, dimensions = [id]),
                      name = 'M', width_uf_lists=width_ufs)

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.max(M[ds[bd], ds[s1], ds[od]], 0) + B[ds[od]],
                      name = 'O', width_uf_lists=None if args.dense_storage else width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    b, l, o, k = tuple(M.op.axis) + tuple(M.op.reduce_axis)
    l = s[M].fuse(b, l, padding = 2)
    loi, li = s[M].split(l, factor=2)

    ooi, oi = s[M].split(o, factor=2)

    koi, ki = s[M].split(k, factor=4)
    koo, koi = s[M].split(koi, factor=2)

    s[M].reorder(koo, koi, loi, ooi, ki, li, oi)

    if not args.debug_code:
        s[M].unroll(koi)
        s[M].unroll(loi)
        s[M].unroll(ooi)
        s[M].unroll(ki)
        s[M].unroll(li)
        s[M].unroll(oi)

    O_b, O_l, O_o = tuple(O.op.axis)
    O_l = s[O].fuse(O_b, O_l, padding = 32)

    O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
    O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=2)
    O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

    O_o_o_i, O_o_i = s[O].split(O_o, factor=4)
    O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
    O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
    s[O].reorder(O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)

    A_shared = s.cache_read(A, "shared", [M])
    A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
    A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
    s[A_shared].compute_at(s[M], koo)

    W_shared = s.cache_read(W, "shared", [M], vanilla=True)
    W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
    s[W_shared].compute_at(s[M], koo)

    B_shared = s.cache_read(B, "shared", [O], vanilla=True)
    B_shared_ax0, = tuple(B_shared.op.axis)

    s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.y"))
    s[O].bind(O_o_o_o_o, te.thread_axis("blockIdx.x"))
    O_l_o_o_i_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
    s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
    O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
    s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
    s[M].compute_at(s[O], O_l_o_i_o_o_i_fused)
    s[B_shared].compute_at(s[O], O_l_o_i_o_o_i_fused)

    A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
    A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
    s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
    A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
    s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
    s[A_shared].mark_no_bounds_check()

    W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
    W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=4)
    s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
    W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=32)
    s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

    B_shared_ax0_o, B_shared_ax0_i = s[B_shared].split(B_shared_ax0, factor=2)
    s[B_shared].vectorize(B_shared_ax0_i)
    B_shared_ax0_o_o, B_shared_ax0_o_i = s[B_shared].split(B_shared_ax0_o, factor=32)
    s[B_shared].bind(B_shared_ax0_o_i, te.thread_axis("threadIdx.x"))

    s.fuse_tensor_dimensions(M, 0, 1)
    s.fuse_tensor_dimensions(A_shared, 0, 1)

    s[M].set_scope('local')

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: IN_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        O: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                       run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)))
    }

inputs = [[lens], [A, W, B, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=32)

# A, W, O  = out
# for i in range(BATCH_SIZE):
    # length = batches[0][i]
    # print(batches[0][i], np.mean(O[i,0:length,:]))
