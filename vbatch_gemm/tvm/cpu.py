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

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=2, type=int)
parser.add_argument('--tile-size', dest='tile_size', default=128, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
parser.add_argument('--data-file', nargs='?', default='random')

parser.add_argument('--m1', dest='m1', default=2, type=int)
parser.add_argument('--m2', dest='m2', default=1, type=int)
parser.add_argument('--n1', dest='n1', default=32, type=int)
parser.add_argument('--n2', dest='n2', default=4, type=int)
parser.add_argument('--k1', dest='k1', default=8, type=int)
parser.add_argument('--k2', dest='k2', default=8, type=int)
parser.add_argument('--fs', dest='fs', default=3, type=int)
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
loop_ufs=[ls[0], ls[3], ls[2]]
B = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, kd, nd], loop_ufs, name='B', width_ufs=None, dtype='float32')

loop_ufs=[ls[0], ls[1], ls[2]]
Op = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs, name='Op', width_ufs=None, dtype='float32')

loop_ufs=[ls[0], ls[1], ls[2]]
S = te.ragged_compute((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[md], rds['k']] * B[ds[bd], rds['k'], ds[nd]],
                                              axis=rds['k'], dimensions=[kd]),
                      name = 'S', reduce_axis_ufs = [('k', kufw.get_uf())], width_uf_lists=None)

loop_ufs=[ls[0], ls[1], ls[2]]
O = te.ragged_compute((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs,
                      lambda ds: S[ds[bd], ds[md], ds[nd]] + Op[ds[bd], ds[md], ds[nd]],
                      name = 'O', width_uf_lists=None)

s = tvm.create_schedule([O.op])

prep_code_mode='with_prep_code'
if True:
    O_local = S
    lls = [Uf.from_constant('m', MAX_DIM, 'l'), Uf.from_constant('n', MAX_DIM, 'l'), Uf.from_constant('k', MAX_DIM, 'l')]
    Bl = s.cache_read(B, 'local', [O_local], layouts='dense', loop_layout=[lls[2], lls[1]])
    Al = s.cache_read(A, 'local', [O_local], layouts='dense', loop_layout=[lls[0], lls[2]])

    # All = s.cache_read(Al, 'local', [O_local], layouts='dense')

    b, m, n, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)

    m1, m2 = args.m1, args.m2
    n1, n2 = args.n1, args.n2
    k1, k2 = args.k1, args.k2

    if not (m1 * m2 <= 128 and n1 * n2 <= 128 and k1 * k2 <= 128):
        exit(0)

    mo, mii = s[O_local].split(m, factor=m1)
    mo, mi = s[O_local].split(mo, factor=m2)

    no, nii = s[O_local].split(n, factor=n1)
    no, ni = s[O_local].split(no, factor=n2)

    ko, kii = s[O_local].split(k, factor=k1)
    ko, ki = s[O_local].split(ko, factor=k2)
    s[O_local].reorder(ko, mo, no, ki, mi, ni, kii, mii, nii)
    s[Bl].compute_at(s[O_local], ko)
    s[Al].compute_at(s[O_local], ko)

    # s[All].compute_at(s[O_local], kii)

    if not args.debug_code:
        # s[All].unroll(s[All].leaf_iter_vars[1])
        s[O_local].vectorize(nii)
        s[O_local].unroll(kii)
        s[O_local].unroll(mii)

    O_b, O_m, O_n = tuple(O.op.axis) + tuple(O.op.reduce_axis)
    O_m_o_o, O_m_i = s[O].split(O_m, factor=128)
    O_m_i_o, O_m_i_i = s[O].split(O_m_i, factor=128)
    O_n_o_o, O_n_i = s[O].split(O_n, factor=128)
    O_n_i_o, O_n_i_i = s[O].split(O_n_i, factor=128)

    s[O].reorder(O_b, O_m_o_o, O_n_o_o, O_m_i_o, O_n_i_o, O_m_i_i, O_n_i_i)

    if args.fs == 1:
        s[O].parallel(O_b)
        prep_code_mode='no_prep_code'
    elif args.fs == 2:
        fused = s[O].fuse(O_b, O_m_o_o)
        s[O].parallel(fused)
    else:
        fused = s[O].fuse(O_b, O_m_o_o)
        fused = s[O].fuse(fused, O_n_o_o)
        s[O].parallel(fused)
    s[O_local].compute_at(s[O], O_n_i_o)

    s[Al].mark_no_bounds_check()
    s[Bl].mark_no_bounds_check()
    s[S].mark_no_bounds_check()

    s.split_tensor_dimension(Al, 0, m1)
    s.reorder_tensor_dimensions(Al, 1, 2)

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: BATCH_SIZE * MAX_DIM * MAX_DIM,
        B: BATCH_SIZE * MAX_DIM * MAX_DIM,
        O: BATCH_SIZE * MAX_DIM * MAX_DIM,
    }

bO = tvm.tir.decl_buffer((BATCH_SIZE, MAX_DIM, MAX_DIM), name="bO")
binds = {Op: bO, O: bO}
if args.only_prep_code: prep_code_mode = 'only_prep_code'
inputs = [[ms, ns, ks], [A, B, bO]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                               run_function=run_utils.run_vbatch_gemm,
                               prep_code_mode=prep_code_mode, binds=binds)

# A, W, O  = out
# for i in range(BATCH_SIZE):
    # length = batches[0][i]
    # print(batches[0][i], np.mean(O[i,0:length,:]))
