import numpy as np
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
parser.add_argument('--kt', dest='kt', default=8, type=int)
# parser.add_argument('--nt', dest='nt', default=16, type=int)
args = parser.parse_args()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
HEAD_SIZE = 64

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

if args.no_raggedness:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [], [], lambda : lambda : utils.ceilmult(MAX_LEN, pad))
else:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s', 1)

if args.dataset in ['mprc', 'cola']: lufwp = len_ufw('s', 32)
else: lufwp = len_ufw('s', 64)
sufwp = len_ufw('s', 64)

lbduf = Uf.from_constant('bd', BS_VAR, "l")
ls = {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufwp.get_uf(),
    3: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    4: Uf.from_constant('qk', 3, "l"),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[2]]
width_ufs=[ls[0], sufwp.get_uf(), ls[1], sufwp.get_uf()]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s2, md, s1], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[4], ls[0], ls[2], ls[1], ls[3]]
width_ufs=[ls[4], ls[0], sufwp.get_uf(), ls[1], ls[3]]
V = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[lbduf, ls[2], ls[1], ls[3]]
width_ufs=None if args.dense_storage else [[ls[0], sufwp.get_uf(), ls[1], ls[3]]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s2, md, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[s2], ds[md], rds['k']] *
                                              V[2, ds[bd], rds['k'], ds[md], ds[hd]],
                                              axis=rds['k'], dimensions=[s1]),
                      name = 'O', reduce_axis_ufs = [('k', lufw1.get_uf())],
                      width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    S = s.cache_write(O, 'local')

    S_b, S_l, S_h, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)

    S_l_o_i, S_l_i = s[S].split(S_l, factor=2)

    S_k_o_o, S_k_o_i = s[S].split(S_k, factor=16)
    s[S].reorder(S_b, S_o, S_k_o_o, S_k_o_i, S_l_o_i, S_l_i)

    O_b, O_l, O_h, O_o, O_k = tuple(O.op.axis) + tuple(O.op.reduce_axis)

    xo, xi = s[O].split(O_l, factor = 64)
    yo, yi = s[O].split(O_o, factor = 64)
    s[O].reorder(O_h, O_b, xo, yo, xi, yi)
    f1 = s[O].fuse(xo, yo)
    O_b = s[O].fuse(O_b, f1)
    O_l = xi
    O_o = yi

    O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
    O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=4)
    O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

    O_o_o_o_i, O_o_o_i = s[O].split(O_o, factor=32)
    O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=2)
    s[O].reorder(O_b, O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i)

    A_shared = s.cache_read(A, "shared", [S])
    A_shared_axm1, A_shared_ax0, A_shared_ax1, A_shared_ax2 = tuple(A_shared.op.axis)
    s[A_shared].compute_at(s[S], S_k_o_o)

    V_shared = s.cache_read(V, "shared", [S])
    V_shared_axm2, V_shared_axm1, V_shared_ax0, V_shared_ax1, V_shared_ax2 = tuple(V_shared.op.axis)
    s[V_shared].compute_at(s[S], S_k_o_o)

    O_b_l_o_o_o_fused_o_o_o_o_fused = s[O].fuse(O_b, O_l_o_o_o, O_o_o_o_o)
    s[O].bind(O_b_l_o_o_o_fused_o_o_o_o_fused, te.thread_axis("blockIdx.x"))
    s[O].bind(O_h, te.thread_axis("blockIdx.y"))
    O_l_o_o_i_fused_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
    s[O].bind(O_l_o_o_i_fused_o_o_o_i_fused, te.thread_axis("vthread"), no_unroll_vthread=True)
    O_l_o_i_fused_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
    s[O].bind(O_l_o_i_fused_o_o_i_fused, te.thread_axis("threadIdx.x"))
    s[S].compute_at(s[O], O_l_o_i_fused_o_o_i_fused)

    A_shared_ax0_ax1_f_ax2_f = s[A_shared].fuse(A_shared_ax0, A_shared_ax1, A_shared_ax2)
    A_shared_ax0_ax1_f_ax2_f_o, A_shared_ax0_ax1_f_ax2_f_i = s[A_shared].split(A_shared_ax0_ax1_f_ax2_f, factor=4)
    s[A_shared].vectorize(A_shared_ax0_ax1_f_ax2_f_i)
    A_shared_ax0_ax1_f_ax2_f_o_o, A_shared_ax0_ax1_f_ax2_f_o_i = s[A_shared].split(A_shared_ax0_ax1_f_ax2_f_o, factor=128)
    s[A_shared].bind(A_shared_ax0_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

    V_shared_ax0_ax1_f_ax2_f = s[V_shared].fuse(V_shared_ax0, V_shared_ax1, V_shared_ax2)
    V_shared_ax0_ax1_f_ax2_f_o, V_shared_ax0_ax1_f_ax2_f_i = s[V_shared].split(V_shared_ax0_ax1_f_ax2_f, factor=4)
    s[V_shared].vectorize(V_shared_ax0_ax1_f_ax2_f_i)
    V_shared_ax0_ax1_f_ax2_f_o_o, V_shared_ax0_ax1_f_ax2_f_o_i = s[V_shared].split(V_shared_ax0_ax1_f_ax2_f_o, factor=128)
    s[V_shared].bind(V_shared_ax0_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))


def size_fn(l_inputs):
    if args.no_raggedness: return {}
    else:
        lens = l_inputs[0]
        return {
            V: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (sufwp.get_fn(lens)(b))),
            A: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (sufwp.get_fn(lens)(b) *
                                                                      sufwp.get_fn(lens)(b))),
            O: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: sufwp.get_fn(lens)(b))
        }

prep_code_mode = 'no_prep_code' if args.no_raggedness else 'with_prep_code'
inputs = [[lens], [BS_VAR, V, A, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR),
                                        prep_code_mode=prep_code_mode)

# _, V, A, O  = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     rounded64 = utils.ceilmult(length, 64)
#     this_extent = rounded64 * NUM_HEADS * HEAD_SIZE
#     print(length, run_utils.stats(O[ctr:ctr + this_extent]))
#     ctr += this_extent
