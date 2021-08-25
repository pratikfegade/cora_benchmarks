import os
import argparse
import utils
import run_utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--peel-loops', dest='peel_loops', default=False, action='store_true')
parser.add_argument('--unroll-loops', dest='unroll_loops', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=False, action='store_true')
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

NUM_HEADS = 8
HEAD_SIZE = 64
TILE=32
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def lb(name): return Uf(name, "l", (0, MAX_LEN), [bd], lambda b: utils.floormult(lens[b], 64))
def ub(name): return Uf(name, "l", (32, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], 32))

bd_uf = Uf.from_constant('bd', args.batch_size, "l")
md_uf = Uf.from_constant('md', NUM_HEADS, "l")
lb_uf = lb('lb')
ub_uf = ub('ub')
hd_uf = Uf.from_constant('hd', HEAD_SIZE, "l")

loop_ufs=[bd_uf, ub_uf, md_uf, hd_uf]
width_ufs = None if args.dense_storage else loop_ufs
Q = te.ragged_placeholder((args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[bd_uf, ub_uf, md_uf, hd_uf]
width_ufs = None if args.dense_storage else loop_ufs
K = te.ragged_placeholder((args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[bd_uf, ub_uf, md_uf, ub_uf]
width_ufs = None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
O = te.ragged_compute((args.batch_size, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[ds[bd], ds[s1], ds[md], k] * K[ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions=[hd]),
                      name = 'O', width_uf_lists=width_ufs)

output_layout = O.op.output_layout(0)
s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

ntx = 16
nty = 16

def schedule_op(O, tile_x, tile_y, suffix):
    Ol = s.cache_write(O, 'local')

    Qs = s.cache_read(Q, "shared", [Ol], layouts='dense', suffix=suffix)
    Ks = s.cache_read(K, "shared", [Ol], layouts='dense', suffix=suffix)

    Ql = s.cache_read(Qs, "local", [Ol], layouts='dense', suffix=suffix)
    Kl = s.cache_read(Ks, "local", [Ol], layouts='dense', suffix=suffix)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = tile_x)
    yo, yi = s[O].split(y, factor = tile_y)

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
    s[Ol].compute_at(s[O], xii)

    x, h, y, k = s[Ol].leaf_iter_vars[1:5]
    s[Ol].reorder(h, k, x, y)
    s[Ql].compute_at(s[Ol], k)
    s[Kl].compute_at(s[Ol], k)

    x, h, y = s[Ks].leaf_iter_vars[1], s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3]
    s[Ks].reorder(h, y, x)
    f = s[Ks].fuse(x, y)
    fo, fi = s[Ks].split(f, factor = ntx * nty * 4)
    fio, fii = s[Ks].split(fi, factor = ntx * 4)
    fiio, fiii = s[Ks].split(fii, factor = 4)
    s[Ks].bind(fio, thread_y())
    s[Ks].bind(fiio, thread_x())
    if not args.debug_functions: s[Ks].vectorize(fiii)

    x, h, y = s[Qs].leaf_iter_vars[1], s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3]
    s[Qs].reorder(h, y, x)
    f = s[Qs].fuse(x, y)
    fo, fi = s[Qs].split(f, factor = ntx * nty * 4)
    fio, fii = s[Qs].split(fi, factor = ntx * 4)
    fiio, fiii = s[Qs].split(fii, factor = 4)
    s[Qs].bind(fio, thread_y())
    s[Qs].bind(fiio, thread_x())
    if not args.debug_functions: s[Qs].vectorize(fiii)

    s.reorder_tensor_dimensions(Ks, 1, 2)
    s.reorder_tensor_dimensions(Ks, 2, 3)
    s.reorder_tensor_dimensions(Qs, 1, 2)
    s.reorder_tensor_dimensions(Qs, 2, 3)

O1, O2, O3, O4 = s.split_for_bin_packing(O, {O.op.axis[1]: lb_uf, O.op.axis[3]: lb_uf})
schedule_op(O1, 64, 64, '1')
schedule_op(O2, 32, 64, '2')
schedule_op(O3, 64, 32, '3')
schedule_op(O4, 32, 32, '4')

s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0]),
         (s[O3].op, s[O3].leaf_iter_vars[0]), (s[O4].op, s[O4].leaf_iter_vars[0])])

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

bO = tvm.tir.decl_buffer(output_layout, name="bO")
inputs = [[lens], [Q, K, bO]]
binds = {O1:bO, O2:bO, O3:bO, O4:bO}
with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=not args.debug_functions):
    if args.debug_code:
        # lowered = tvm.lower(s, inputs, args.target, simple_mode=True, binds=binds)
        # print(lowered)
        fadd, _ = tvm.build(s, inputs, args.target, binds=binds)
        if args.target == 'cuda':
            print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        else:
            print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd, i_bufs = tvm.build(s, inputs, args.target, binds=binds)
        # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
        run_utils.run(fadd, i_bufs, [Q, K, O], args.batch_size, args.max_batches,
                      args.dataset, args.datadir, args.target, args.debug)
