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

def len_ufw(name): return Ufw(name, "l", (TILE, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], TILE))
lufw = len_ufw('s')

ls =  {
    0: Uf.from_constant('bd', args.batch_size, "l"),
    1: Uf.from_constant('md', NUM_HEADS, "l"),
    2: lufw.get_uf(),
    3: lufw.get_uf(),
    4: Uf.from_constant('hd', HEAD_SIZE, "l"),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[4]]
width_ufs = None if args.dense_storage else loop_ufs
Q = te.ragged_placeholder((args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[3], ls[1], ls[4]]
width_ufs = None if args.dense_storage else loop_ufs
K = te.ragged_placeholder((args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
width_ufs = None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
S = te.ragged_compute((args.batch_size, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[ds[bd], ds[s1], ds[md], k] * K[ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions=[hd]),
                      name = 'S', width_uf_lists=width_ufs)

O = te.ragged_compute((args.batch_size, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.if_then_else(ds[s1] >= lens[ds[bd]], -float('inf'), S[ds[bd], ds[s1], ds[md], ds[s2]]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    thread_x = lambda: tvm.thread_axis("threadIdx.x")
    thread_y = lambda: tvm.thread_axis("threadIdx.y")
    block_x = lambda: tvm.thread_axis("blockIdx.x")
    block_y = lambda: tvm.thread_axis("blockIdx.y")

    ntx = 16
    nty = 16

    Qs = s.cache_read(Q, "shared", [S], layouts='dense')
    Ks = s.cache_read(K, "shared", [S], layouts='dense')

    Ql = s.cache_read(Qs, "local", [S], layouts='dense')
    Kl = s.cache_read(Ks, "local", [S], layouts='dense')

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)

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
    s[O].bind(yio, tvm.thread_axis("vthread"))
    s[O].bind(xio, tvm.thread_axis("vthread"))
    s[O].reorder(xio, yii, yio, xii)
    s[S].compute_at(s[O], xii)

    x, h, y, k = s[S].leaf_iter_vars[1:5]
    s[S].reorder(h, k, x, y)
    s[Ql].compute_at(s[S], k)
    s[Kl].compute_at(s[S], k)

    x, h, y = s[Ks].leaf_iter_vars[1], s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3]
    s[Ks].reorder(h, y, x)
    f = s[Ks].fuse(x, y)
    fo, fi = s[Ks].split(f, factor = ntx * nty * 4)
    fio, fii = s[Ks].split(fi, factor = ntx * 4)
    fiio, fiii = s[Ks].split(fii, factor = 4)
    s[Ks].bind(fio, thread_y())
    s[Ks].bind(fiio, thread_x())
    s[Ks].vectorize(fiii)

    x, h, y = s[Qs].leaf_iter_vars[1], s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3]
    s[Qs].reorder(h, y, x)
    f = s[Qs].fuse(x, y)
    fo, fi = s[Qs].split(f, factor = ntx * nty * 4)
    fio, fii = s[Qs].split(fi, factor = ntx * 4)
    fiio, fiii = s[Qs].split(fii, factor = 4)
    s[Qs].bind(fio, thread_y())
    s[Qs].bind(fiio, thread_x())
    s[Qs].vectorize(fiii)

    s.reorder_tensor_dimensions(Ks, 1, 2)
    s.reorder_tensor_dimensions(Ks, 2, 3)
    s.reorder_tensor_dimensions(Qs, 1, 2)
    s.reorder_tensor_dimensions(Qs, 2, 3)

    s[S].set_scope('local')

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        Q: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (lufw.get_fn(lens)(b))),
        K: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (lufw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (lufw.get_fn(lens)(b) *
                                                       lufw.get_fn(lens)(b)))
    }

inputs = [[lens], [Q, K, O]]
with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=True):
    if args.debug_code:
        lowered = tvm.lower(s, inputs, args.target, simple_mode = True)
        print(lowered)
        # fadd, _ = tvm.build(s, inputs, args.target)
        # if args.target == 'cuda':
            # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        # else:
            # print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd, i_bufs = tvm.build(s, inputs, args.target)
        # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
        out, batches = run_utils.run2(fadd, i_bufs, inputs[1], size_fn, args)
        # Q, K, O = out
        # for i in range(args.batch_size):
        #     length = batches[0][i]
        #     rounded = utils.ceilmult(length, TILE)
        #     print(rounded, np.mean(O[i,0:rounded,:,0:rounded]))
