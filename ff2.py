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

IN_SIZE = 2048
OUT_SIZE = 512
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 64)

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw32 = len_ufw('s1', 32)

ls =  {
    0: Uf.from_constant('bd', args.batch_size, "l"),
    1: lufw1.get_uf(),
    2: Uf.from_constant('id', IN_SIZE, "l"),
    3: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((args.batch_size, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((IN_SIZE, OUT_SIZE), name='W')

loop_ufs=[ls[0], ls[1], ls[3]]
width_ufs=None if args.dense_storage else [[ls[0], lufw32.get_uf(), ls[3]]]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
O = te.ragged_compute((args.batch_size, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(W[k, ds[od]] * A[ds[bd], ds[s1], k], axis = k, dimensions = [id]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    # O_local, = s.cache_write([O], "local", pass_storage_layouts=True)
    O_local, = s.cache_write([O], "local", storage_layout_mode='loop_layout')

    b, l, o, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    l = s[O_local].fuse(b, l, padding = 2)
    loi, li = s[O_local].split(l, factor=2)

    ooi, oi = s[O_local].split(o, factor=2)

    koi, ki = s[O_local].split(k, factor=4)
    koo, koi = s[O_local].split(koi, factor=2)

    s[O_local].reorder(koo, koi, loi, ooi, ki, li, oi)

    if not args.debug_code:
        s[O_local].unroll(koi)
        s[O_local].unroll(loi)
        s[O_local].unroll(ooi)
        s[O_local].unroll(ki)
        s[O_local].unroll(li)
        s[O_local].unroll(oi)

    O_b, O_l, O_o, O_k = tuple(O.op.axis) + tuple(O.op.reduce_axis)
    O_l = s[O].fuse(O_b, O_l, padding = 32)

    O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
    O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=2)
    O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

    O_o_o_i, O_o_i = s[O].split(O_o, factor=4)
    O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
    O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
    s[O].reorder(O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)

    A_shared = s.cache_read(A, "shared", [O_local])
    A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
    A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
    s[A_shared].compute_at(s[O_local], koo)

    W_shared = s.cache_read(W, "shared", [O_local], vanilla=True)
    W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
    s[W_shared].compute_at(s[O_local], koo)

    s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.y"))
    s[O].bind(O_o_o_o_o, te.thread_axis("blockIdx.x"))
    O_l_o_o_i_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
    s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
    O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
    s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
    s[O_local].compute_at(s[O], O_l_o_i_o_o_i_fused)

    A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
    A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
    if not args.debug_functions: s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
    A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
    s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
    s[A_shared].mark_no_bounds_check()

    W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
    W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=4)
    if not args.debug_functions: s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
    W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=32)
    s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))


    s.fuse_tensor_dimensions(O_local, 0, 1)
    s.fuse_tensor_dimensions(A_shared, 0, 1)


    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        # A: IN_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)),
        # O: OUT_SIZE * (args.batch_size * MAX_LEN if args.dense_storage else
                       # run_utils.prefix_sum(len(lens), lambda b: lufw32.get_fn(lens)(b)))
        A: IN_SIZE * args.batch_size * MAX_LEN,
        O: OUT_SIZE * args.batch_size * MAX_LEN
    }

inputs = [[lens], [A, W, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn)

# A, W, O  = out
# for i in range(args.batch_size):
    # length = batches[0][i]
    # print(batches[0][i], np.mean(O[i,0:length,:]))
