import numpy as np
import math
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

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
NUM_HEADS = 8
IN_SIZE = 512
OUT_SIZE = 64
QKV_NUM = 3
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
qkv = Dim('qkv')
md = Dim('md')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('qkv', QKV_NUM, "l"),
    1: Uf.from_constant('bd', BATCH_SIZE, "l"),
    2: Uf.from_constant('md', NUM_HEADS, "l"),
    3: lufw1.get_uf(),
    4: Uf.from_constant('id', IN_SIZE, "l"),
    5: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[1], ls[3], ls[4]]
width_ufs=[ls[1], ls[3], ls[4]]
QKV = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                            name='QKV', width_ufs=width_ufs)

W = te.placeholder((QKV_NUM, NUM_HEADS, OUT_SIZE, IN_SIZE), name='W')
B = te.placeholder((QKV_NUM, NUM_HEADS, OUT_SIZE), name='B')

loop_ufs=[ls[0], ls[1], ls[2], ls[5], ls[3]]
width_ufs=None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
S = te.ragged_compute((QKV_NUM, BATCH_SIZE, NUM_HEADS, OUT_SIZE, MAX_LEN), [qkv, bd, md, od, s1], loop_ufs,
                      lambda ds: tvm.sum(W[ds[qkv], ds[md], ds[od], k] * QKV[ds[bd], ds[s1], k],
                                         axis = k, dimensions = [id]),
                      name = 'S', width_uf_lists=width_ufs)

if args.layout_unfused:
    width_ufs=None if args.dense_storage else [[ls[0], ls[1], ls[2], ls[5], lufw1.get_uf()]]
else:
    width_ufs=None if args.dense_storage else [[ls[0], ls[1], ls[2], ls[5], lufw64.get_uf()]]
O = te.ragged_compute((QKV_NUM, BATCH_SIZE, NUM_HEADS, OUT_SIZE, MAX_LEN), [qkv, bd, md, od, s1], loop_ufs,
                      lambda ds: S[ds[qkv], ds[bd], ds[md], ds[od], ds[s1]] + B[ds[qkv], ds[md], ds[od]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    s.fuse_tensor_dimensions(QKV, 0, 1)

    s.reorder_tensor_dimensions(S, 3, 4)
    s.reorder_tensor_dimensions(S, 2, 3)
    s.fuse_tensor_dimensions(S, 1, 2)
    S_q, S_b, S_n, S_h, S_l, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)
    s[S].reorder(S_q, S_b, S_l, S_n, S_h)
    S_l = s[S].fuse(S_b, S_l)
    S_o = s[S].fuse(S_n, S_h)
    s[S].reorder(S_q, S_o, S_l)

    S_o_o_i, S_o_i = s[S].split(S_o, factor=4)

    S_k_o_o, S_k_o_i = s[S].split(S_k, factor=4)
    s[S].reorder(S_k_o_o, S_k_o_i, S_o_o_i, S_o_i, S_l)

    s[S].unroll(S_l)
    s[S].unroll(S_o_i)

    O_q, O_b, O_n, O_h, O_l = tuple(O.op.axis) + tuple(O.op.reduce_axis)
    s[O].reorder(O_q, O_b, O_l, O_n, O_h)
    O_o = s[O].fuse(O_n, O_h)
    O_l = s[O].fuse(O_b, O_l, padding=128)
    s[O].reorder(O_q, O_o, O_l)

    O_o_o_i, O_o_i = s[O].split(O_o, factor=32)
    O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=2)

    O_l_o_i, O_l_i = s[O].split(O_l, factor=2)
    O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=32)
    O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

    s[O].reorder(O_o_o_o_i, O_l_o_o_o, O_l_o_o_i, O_o_o_i, O_l_o_i, O_o_i, O_l_i)

    A_shared = s.cache_read(QKV, "shared", [S])
    A_shared_ax0, A_shared_ax1, A_shared_ax2 = tuple(A_shared.op.axis)
    s[A_shared].compute_at(s[S], S_k_o_o)

    W_shared = s.cache_read(W, "shared", [S], vanilla=True)
    W_shared_axm1, W_shared_ax0, W_shared_ax1, W_shared_ax2 = tuple(W_shared.op.axis)
    W_shared_ax1 = s[W_shared].fuse(W_shared_ax1, W_shared_ax2)
    s[W_shared].compute_at(s[S], S_k_o_o)

    s[O].bind(O_q, te.thread_axis("blockIdx.z"))
    s[O].bind(O_o_o_o_i, te.thread_axis("blockIdx.y"))
    s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.x"))
    O_b_o_o_i_o_o_o_i_fused_l_o_o_i_fused = O_l_o_o_i
    s[O].bind(O_b_o_o_i_o_o_o_i_fused_l_o_o_i_fused, te.thread_axis("vthread"))
    O_b_o_i_o_o_i_fused_l_o_i_fused = s[O].fuse(O_o_o_i, O_l_o_i)
    s[O].bind(O_b_o_i_o_o_i_fused_l_o_i_fused, te.thread_axis("threadIdx.x"))
    s[S].compute_at(s[O], O_b_o_i_o_o_i_fused_l_o_i_fused)

    A_shared_ax0_ax1_fused_ax2_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1, A_shared_ax2)
    A_shared_ax0_ax1_fused_ax2_fused_o, A_shared_ax0_ax1_fused_ax2_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused_ax2_fused, factor=1)
    s[A_shared].vectorize(A_shared_ax0_ax1_fused_ax2_fused_i)
    A_shared_ax0_ax1_fused_ax2_fused_o_o, A_shared_ax0_ax1_fused_ax2_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_ax2_fused_o, factor=64)
    s[A_shared].bind(A_shared_ax0_ax1_fused_ax2_fused_o_i, te.thread_axis("threadIdx.x"))

    W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
    W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=1)
    s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
    W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=64)
    s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

    s[S].set_scope('local')
    s[S].mark_no_bounds_check()
    s[O].mark_no_bounds_check()
    s[A_shared].mark_no_bounds_check()
    s.fuse_tensor_dimensions(A_shared, 0, 1)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    s.fuse_tensor_dimensions(QKV, 0, 1)
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    if args.layout_unfused: out_fn = lufw1.get_fn(lens)
    else: out_fn = lufw64.get_fn(lens)

    return {
        QKV: IN_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)),
        O: QKV_NUM * NUM_HEADS * OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                                             run_utils.prefix_sum(len(lens), lambda b: out_fn(b)))
    }

bQKV = tvm.decl_buffer([BATCH_SIZE*MAX_LEN, IN_SIZE], name = "bQKV")
binds = {QKV: bQKV}
if args.target == "cuda":
    inputs = [[lens], [BS_VAR, bQKV, W, B, O]]
else:
    inputs = [[lens], [BS_VAR, bQKV, W, B, S, O]]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

q_size = 0
for length in batches[0]:
    q_size += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE

# ctr = 0
# O  = out[-1]
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * NUM_HEADS * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]), np.mean(O[ctr+q_size:ctr+q_size + this_extent]),
#           np.mean(O[ctr+2*q_size:ctr+2*q_size + this_extent]))
#     ctr += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE
