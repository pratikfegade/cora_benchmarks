import numpy as np
import math
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append("../../")
import utils
import run_utils

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--tile-size', dest='tile_size', default=128, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--data-file', nargs='?', default='random')
args = parser.parse_args()

BATCH_SIZE = args.batch_size

ms = te.placeholder((BATCH_SIZE,), name = 'ms', dtype = 'int32')
ns = te.placeholder((BATCH_SIZE,), name = 'ns', dtype = 'int32')
ks = te.placeholder((BATCH_SIZE,), name = 'ks', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')
MIN_DIM, MAX_DIM = 4*args.tile_size, 12*args.tile_size

def f_mufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ms], lambda b: lambda b: args.tile_size * ms[b])
def f_nufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ns], lambda b: lambda b: args.tile_size * ns[b])
def f_kufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ks], lambda b: lambda b: args.tile_size * ks[b])

mufw = f_mufw('m')
nufw = f_nufw('m')
kufw = f_kufw('m')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: mufw.get_uf(),
    2: nufw.get_uf(),
    3: kufw.get_uf(),
}

loop_ufs=[ls[0], ls[1], ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, kd], loop_ufs, name='A', width_ufs=None, dtype='float32')
loop_ufs=[ls[0], ls[2], ls[3]]
B = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, nd, kd], loop_ufs, name='B', width_ufs=None, dtype='float32')

loop_ufs=[ls[0], ls[1], ls[2]]
O = te.ragged_compute((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[md], rds['k']] * B[ds[bd], ds[nd], rds['k']],
                                              axis=rds['k'], dimensions=[kd]),
                      name = 'O', reduce_axis_ufs = [('k', kufw.get_uf())], width_uf_lists=None)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    O_local, = s.cache_write([O], "local", storage_layout_mode='loop_layout')

    b, l, o, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
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

    O_b, O_l, O_o = tuple(O.op.axis)

    O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
    O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=2)
    O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

    O_o_o_i, O_o_i = s[O].split(O_o, factor=4)
    O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
    O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
    s[O].reorder(O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)
    if not args.debug_functions: s[O].vectorize(O_o_i)

    A_shared = s.cache_read(A, "shared", [O_local])
    A_shared_ax01, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
    s[A_shared].compute_at(s[O_local], koo)

    B_shared = s.cache_read(B, "shared", [O_local])
    B_shared_ax01, B_shared_ax0, B_shared_ax1 = tuple(B_shared.op.axis)
    s[B_shared].compute_at(s[O_local], koo)

    O_fused = s[O].fuse(O_l_o_o_o, O_o_o_o_o)
    O_fused = s[O].fuse(O_b, O_fused)
    s[O].bind(O_fused, te.thread_axis("blockIdx.x"))
    O_l_o_o_i_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
    s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
    O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
    s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
    s[O_local].compute_at(s[O], O_l_o_i_o_o_i_fused)

    A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
    A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=4)
    if not args.debug_functions: s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
    A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
    s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

    B_shared_ax0_ax1_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1)
    B_shared_ax0_ax1_fused_o, B_shared_ax0_ax1_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused, factor=4)
    if not args.debug_functions: s[B_shared].vectorize(B_shared_ax0_ax1_fused_i)
    B_shared_ax0_ax1_fused_o_o, B_shared_ax0_ax1_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_o, factor=32)
    s[B_shared].bind(B_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: BATCH_SIZE * MAX_DIM * MAX_DIM,
        B: BATCH_SIZE * MAX_DIM * MAX_DIM,
        O: BATCH_SIZE * MAX_DIM * MAX_DIM,
    }

inputs = [[ms, ns, ks], [A, B, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, run_function=run_utils.run_vbatch_gemm)

# A, W, O  = out
# for i in range(BATCH_SIZE):
    # length = batches[0][i]
    # print(batches[0][i], np.mean(O[i,0:length,:]))
