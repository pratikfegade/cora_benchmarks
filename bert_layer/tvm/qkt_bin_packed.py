import numpy as np
import os
import argparse
import utils
import run_utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw

parser = run_utils.get_cmd_parser()
args = parser.parse_args()

NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
RTILE=4
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')
qk = Dim('qk')

def lbw(name): return Ufw(name, "l", (0, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.floormult(lens[b], 64))
def ubw(name): return Ufw(name, "l", (32, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], 32))
lbufw = lbw('lb')
ubufw = ubw('ub')

qk_uf = Uf.from_constant('qk', 3, "l")
bd_uf = Uf.from_constant('bd', args.batch_size, "l")
md_uf = Uf.from_constant('md', NUM_HEADS, "l")
lb_uf = lbufw.get_uf()
ub_uf = ubufw.get_uf()
hd_uf = Uf.from_constant('hd', HEAD_SIZE, "l")

loop_ufs=[qk_uf, bd_uf, ub_uf, md_uf, hd_uf]
width_ufs = None if args.dense_storage else loop_ufs
Q = te.ragged_placeholder((3, args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[qk_uf, bd_uf, ub_uf, md_uf, hd_uf]
width_ufs = None if args.dense_storage else loop_ufs
K = te.ragged_placeholder((3, args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[bd_uf, ub_uf, md_uf, ub_uf]
width_ufs = None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
S = te.ragged_compute((args.batch_size, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[0, ds[bd], ds[s1], ds[md], k] * K[1, ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions=[hd]),
                      name = 'S', width_uf_lists=width_ufs)

O = te.ragged_compute((args.batch_size, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.if_then_else(ds[s1] >= lens[ds[bd]], -float('inf'), S[ds[bd], ds[s1], ds[md], ds[s2]]),
                      name = 'O', width_uf_lists=width_ufs)

output_layout = O.op.output_layout(0)
s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")
ntx = 16
nty = 16

def schedule_op(S, O, tile_x, tile_y, suffix):
    Qs = s.cache_read(Q, "shared", [S], layouts='dense', suffix=suffix)
    Ks = s.cache_read(K, "shared", [S], layouts='dense', suffix=suffix)

    Ql = s.cache_read(Qs, "local", [S], layouts='dense', suffix=suffix)
    Kl = s.cache_read(Ks, "local", [S], layouts='dense', suffix=suffix)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor=tile_x)
    yo, yi = s[O].split(y, factor=tile_x)

    s[O].reorder(b, xo, yo, h, xi, yi)
    f1 = s[O].fuse(xo, yo)
    f2 = s[O].fuse(b, f1)
    s[O].bind(f2, block_x())
    s[O].bind(h, block_y())
    s[Qs].compute_at(s[O], h)
    s[Ks].compute_at(s[O], h)

    xio, xii = s[O].split(xi, factor = nty)
    yio, yii = s[O].split(yi, factor = ntx)
    s[O].bind(xii, thread_y())
    s[O].bind(yii, thread_x())
    s[O].bind(yio, tvm.thread_axis("vthread", name='vth1'), no_unroll_vthread=True)
    s[O].bind(xio, tvm.thread_axis("vthread", name='vth2'), no_unroll_vthread=True)
    s[O].reorder(xio, yii, yio, xii)
    s[S].compute_at(s[O], xii)

    x, h, y, k = s[S].leaf_iter_vars[1:5]
    s[S].reorder(h, k, x, y)
    s[Ql].compute_at(s[S], k)
    s[Kl].compute_at(s[S], k)

    x, h, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3], s[Ks].leaf_iter_vars[4]
    s[Ks].reorder(h, y, x)
    f = s[Ks].fuse(x, y)
    fo, fi = s[Ks].split(f, factor = ntx * nty * 4)
    fio, fii = s[Ks].split(fi, factor = ntx * 4)
    fiio, fiii = s[Ks].split(fii, factor = 4)
    s[Ks].bind(fio, thread_y())
    s[Ks].bind(fiio, thread_x())
    if not args.debug_functions: s[Ks].vectorize(fiii)

    x, h, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3], s[Qs].leaf_iter_vars[4]
    s[Qs].reorder(h, y, x)
    f = s[Qs].fuse(x, y)
    fo, fi = s[Qs].split(f, factor = ntx * nty * 4)
    fio, fii = s[Qs].split(fi, factor = ntx * 4)
    fiio, fiii = s[Qs].split(fii, factor = 4)
    s[Qs].bind(fio, thread_y())
    s[Qs].bind(fiio, thread_x())
    if not args.debug_functions: s[Qs].vectorize(fiii)

    s.reorder_tensor_dimensions(Ks, 2, 3)
    s.reorder_tensor_dimensions(Ks, 3, 4)
    s.reorder_tensor_dimensions(Qs, 2, 3)
    s.reorder_tensor_dimensions(Qs, 3, 4)

    s[S].set_scope('local')

G1, G2, G3, G4 = s.split_for_bin_packing([S], O, {O.op.axis[1]: lb_uf, O.op.axis[3]: lb_uf}, include_inputs=True)
S1, O1 = G1
S2, O2 = G2
S3, O3 = G3
S4, O4 = G4
schedule_op(S1, O1, 64, 64, '1')
schedule_op(S2, O2, 32, 64, '2')
schedule_op(S3, O3, 64, 32, '3')
schedule_op(S4, O4, 32, 32, '4')

s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0]),
         (s[O3].op, s[O3].leaf_iter_vars[0]), (s[O4].op, s[O4].leaf_iter_vars[0])])

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        Q: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (ubufw.get_fn(lens)(b))),
        K: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (ubufw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (ubufw.get_fn(lens)(b) *
                                                       ubufw.get_fn(lens)(b)))
    }

bO = tvm.tir.decl_buffer(output_layout, name="bO")
inputs = [[lens], [Q, K, bO]]
binds = {O1:bO, O2:bO, O3:bO, O4:bO}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds)
Q, K, O = out
for i in range(args.batch_size):
    length = batches[0][i]
    rounded = utils.ceilmult(length, TILE)
    print(rounded, np.mean(O[i,0:rounded,:,0:rounded]))
