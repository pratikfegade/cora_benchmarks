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
parser.add_argument('--nt', dest='nt', default=8, type=int)
parser.add_argument('--kt', dest='kt', default=4, type=int)
parser.add_argument('--masked-mha', dest='masked_mha', default=False, action='store_true')
parser.add_argument('--hfuse', dest='hfuse', default=False, action='store_true')
args = parser.parse_args()

args.target = run_utils.get_arm_target()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def lbw(name): return Ufw(name, "l", (0, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.floormult(lens[b], 64))
def ubw(name): return Ufw(name, "l", (32, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], 32))
def nbw(name): return Ufw(name, "l", (64, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], 64))
lbufw = lbw('lb')
ubufw = ubw('ub')
nbufw = nbw('nb')
lb_uf = lbufw.get_uf()
ub_uf = ubufw.get_uf()
nb_uf = nbufw.get_uf()

lbduf = Uf.from_constant('bd', BS_VAR, "l")
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: Uf.from_constant('md', NUM_HEADS, "l"),
    # 2: lufw.get_uf(),
    # 3: lufw.get_uf(),
    4: Uf.from_constant('hd', HEAD_SIZE, "l"),
    5: Uf.from_constant('qk', 3, "l"),
}

loop_ufs=[ls[5], ls[0], nb_uf, ls[1], ls[4]]
width_ufs = None if args.dense_storage else [ls[5], ls[0], nb_uf, ls[1], ls[4]]
Q = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[5], ls[0], nb_uf, ls[1], ls[4]]
width_ufs = None if args.dense_storage else [ls[5], ls[0], nb_uf, ls[1], ls[4]]
K = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[lbduf, ub_uf, ls[1], nb_uf]
width_ufs = None if args.dense_storage else [[ls[0], nb_uf, ls[1], nb_uf]]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[0, ds[bd], ds[s1], ds[md], k] * K[1, ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions=[hd]),
                      name = 'S', width_uf_lists=width_ufs)

def get_threshold(ds):
    if args.masked_mha:
        return ds[s1] + 1
    else:
        return lens[ds[bd]]

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.if_then_else(ds[s2] >= get_threshold(ds), -float('inf'), S[ds[bd], ds[s1], ds[md], ds[s2]]),
                      name = 'O', width_uf_lists=width_ufs)

output_layout = O.op.output_layout(0)
s = tvm.create_schedule([O.op])

def schedule_op(S, O, tile_x, tile_y, suffix):
    Ks = s.cache_read(K, "shared", [S], layouts='dense', suffix=suffix)

    s[S].set_scope('local')
    S_b_c, S_m_c, S_h_c, S_n_c, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)

    S_m_c_o_i, S_m_c_i = s[S].split(S_m_c, factor=2)
    S_m_c_o_o_i, S_m_c_o_i = s[S].split(S_m_c_o_i, factor=1)
    S_m_c_o_o_o, S_m_c_o_o_i = s[S].split(S_m_c_o_o_i, factor=8)

    S_n_c_o_i, S_n_c_i = s[S].split(S_n_c, factor=8)
    S_n_c_o_o_i, S_n_c_o_i = s[S].split(S_n_c_o_i, factor=1)

    S_k_o, S_k_i = s[S].split(S_k, factor=64)
    S_k_i_o, S_k_i_i = s[S].split(S_k_i, factor=16)
    s[S].reorder(S_b_c, S_m_c_o_o_o, S_n_c_o_o_i, S_m_c_o_o_i, S_k_o, S_m_c_o_i, S_n_c_o_i, S_k_i_o, S_k_i_i, S_m_c_i, S_n_c_i)

    s[S].unroll(S_k_i_i)
    s[S].unroll(S_n_c_i)
    s[S].unroll(S_m_c_i)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = tile_x)
    yo, yi = s[O].split(y, factor = tile_y)
    s[O].reorder(b, xo, yo, h, xi, yi)
    f1 = s[O].fuse(xo, yo)
    f1 = s[O].fuse(b, f1)
    s[O].parallel(f1)

    O_m, O_n = xi, yi

    O_m_o_i, O_m_i = s[O].split(O_m, factor=2)
    O_m_o_o, O_m_o_i = s[O].split(O_m_o_i, factor=8)

    O_n_o_i, O_n_i = s[O].split(O_n, factor=8)
    s[O].reorder(O_m_o_o, O_n_o_i, O_m_o_i, O_m_i, O_n_i)
    s[S].compute_at(s[O], O_n_o_i)
    s[Ks].compute_at(s[O], O_n_o_i)

    s[S].vectorize(S_n_c_i)
    s[O].vectorize(O_n_i)

    s.reorder_tensor_dimensions(Ks, 2, 3)
    s.reorder_tensor_dimensions(Ks, 3, 4)

G1, G2 = s.split_for_bin_packing([S], O, {O.op.axis[1]: lb_uf}, include_inputs=True)
S1, O1 = G1
S2, O2 = G2
schedule_op(S1, O1, 64, 64, '1')
schedule_op(S2, O2, 32, 64, '2')

if args.hfuse:
    s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]),
             (s[O2].op, s[O2].leaf_iter_vars[0])])

bO = tvm.tir.decl_buffer(output_layout, name="bO")
inputs = [[lens], [BS_VAR, Q, K, bO]]
binds = {O1:bO, O2:bO}

def size_fn(l_inputs):
    lens = l_inputs[0]
    if args.dense_storage: return {}
    return {
        Q: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (nbufw.get_fn(lens)(b))),
        K: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (nbufw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (nbufw.get_fn(lens)(b) *
                                                       nbufw.get_fn(lens)(b)))
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64, binds=binds,
                                        hoist_lets_above_parallel_loop=False, hoist_loads=True,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))


# _, Q, K, O = out[0:4]
# O = O.flatten()
# ctr = 0
# for length in batches[0]:
#     rounded = utils.ceilmult(length, TILE)
#     this_extent = rounded
#     this_storage_extent = rounded * rounded * NUM_HEADS
#     # print(rounded, np.mean(O[ctr:ctr+this_storage_extent]))
#     print(rounded, np.mean(O[ctr:ctr+length]))
#     ctr += this_storage_extent
