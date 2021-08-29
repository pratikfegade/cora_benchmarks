import numpy as np
import math
import os
import argparse
import run_utils
import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw

parser = run_utils.get_cmd_parser()
args = parser.parse_args()

BATCH_SIZE = args.batch_size + 1
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 64)
NUM_HEADS = 8
HEAD_SIZE = 64
OUT_SIZE = 512

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
hd = Dim('hd')
od = Dim('od')
mdhd = Dim('mdhd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufw1.get_uf(),
    3: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    4: Uf.from_constant('od', OUT_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[ls[0], ls[1], lufw64.get_uf(), ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((NUM_HEADS * HEAD_SIZE, OUT_SIZE), name='W')

loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, NUM_HEADS * HEAD_SIZE), name = 'k')
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(A[ds[bd], tvm.floordiv(k, HEAD_SIZE), ds[s1], tvm.floormod(k, HEAD_SIZE)] *
                                         W[k, ds[od]], axis=k, dimensions = [mdhd]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    tile = 128
    rtile = 8
    nt = tile // rtile
    ks = utils.next_power_of_2((NUM_HEADS * HEAD_SIZE) / (6144 // tile))

    thread_x = lambda: tvm.thread_axis("threadIdx.x")
    thread_y = lambda: tvm.thread_axis("threadIdx.y")
    block_x = lambda: tvm.thread_axis("blockIdx.x")
    block_y = lambda: tvm.thread_axis("blockIdx.y")
    vthread = lambda: tvm.thread_axis("vthread")

    Ol = s.cache_write(O, "local")
    As = s.cache_read(A, "shared", [Ol], loop_layout=[ls[0], ls[1], ls[2], ls[3]], layouts=[ls[0], ls[1], ls[2], ls[3]])
    Ws = s.cache_read(W, "shared", [Ol], vanilla=True)

    Al = s.cache_read(As, "local", [Ol])
    Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)

    b, l, h = s[O].leaf_iter_vars
    y = s[O].fuse(b, l, padding = tile)
    x = h
    yo, yi = s[O].split(y, factor = tile)
    xo, xi = s[O].split(x, factor = tile)
    s[O].bind(yo, block_y())
    s[O].bind(xo, block_x())

    yio, yii = s[O].split(yi, factor = nt)
    xio, xii = s[O].split(xi, factor = nt)
    s[O].bind(xii, thread_x())
    s[O].bind(yii, thread_y())
    s[O].bind(xio, tvm.thread_axis("vthread", name='vth1'), no_unroll_vthread = True)
    s[O].bind(yio, tvm.thread_axis("vthread", name='vth2'), no_unroll_vthread = True)
    s[Ol].compute_at(s[O], xii)

    b, x, y, k = s[Ol].leaf_iter_vars
    s[Ol].reorder(k, x, y)
    ko, ki = s[Ol].split(k, nparts = ks)
    s[As].compute_at(s[Ol], ko)
    s[Ws].compute_at(s[Ol], ko)
    s[Al].compute_at(s[Ol], ki)
    s[Wl].compute_at(s[Ol], ki)

    b, h, l, i = s[As].leaf_iter_vars
    s[As].reorder(h, b)
    f = s[As].fuse(b, l)
    f = s[As].fuse(f, i)
    fo, fi = s[As].split(f, factor = nt * nt * 4)
    fio, fii = s[As].split(fi, factor = nt * 4)
    fiio, fiii = s[As].split(fii, factor = 4)
    s[As].bind(fio, thread_y())
    s[As].bind(fiio, thread_x())
    if not args.debug_functions: s[As].vectorize(fiii)

    s.reorder_tensor_dimensions(As, 0, 1)
    s.fuse_tensor_dimensions(As, 1, 2)

    # s.fuse_tensor_dimensions(O, 0, 1)

    x, y = s[Ws].leaf_iter_vars
    f = s[Ws].fuse(x, y)
    fo, fi = s[Ws].split(f, factor = nt * nt * 4)
    fio, fii = s[Ws].split(fi, factor = nt * 4)
    fiio, fiii = s[Ws].split(fii, factor = 4)
    s[Ws].bind(fio, thread_y())
    s[Ws].bind(fiio, thread_x())
    if not args.debug_functions: s[Ws].vectorize(fiii)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw64.get_fn(lens)(b)),
        O: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                       run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)))
    }

# bO = tvm.decl_buffer([BATCH_SIZE * MAX_LEN, OUT_SIZE], name = "bO")
# inputs = [[lens], [A, W, bO]]
# binds = {O: bO}
inputs = [[lens], [A, W, O]]
binds = {}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, pad_sum=128)

# A, W, O = out
# for i in range(BATCH_SIZE):
    # length = batches[0][i]
    # print(batches[0][i], np.mean(O[i,0:length,:]))
