import math
import os
import utils
import run_utils
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--peel-loops', dest='peel_loops', default=False, action='store_true')
parser.add_argument('--unroll-loops', dest='unroll_loops', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

NUM_HEADS = 8
IN_SIZE = 256
OUT_SIZE = 64
QKV_NUM = 3
TILE=64
RTILE=4
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
qkv = Dim('qkv')
md = Dim('md')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_uf(name): return Uf(name, "l", (0, MAX_LEN), [bd], lambda b: lens[b])

ls =  {
    0: Uf.from_constant('qkv', QKV_NUM, "l"),
    1: Uf.from_constant('bd', args.batch_size, "l"),
    2: Uf.from_constant('md', NUM_HEADS, "l"),
    3: len_uf('s1'),
    4: Uf.from_constant('id', IN_SIZE, "l"),
    5: Uf.from_constant('od', OUT_SIZE, "l"),
}

# loop_ufs=[ls[0], ls[4], ls[1], ls[3]]
# width_ufs=loop_ufs
# QKV = te.ragged_placeholder((QKV_NUM, IN_SIZE, args.batch_size, MAX_LEN), [qkv, id, bd, s1], loop_ufs,
                            # name='QKV', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[4]]
width_ufs=loop_ufs
QKV = te.ragged_placeholder((QKV_NUM, args.batch_size, MAX_LEN, IN_SIZE), [qkv, bd, s1, id], loop_ufs,
                            name='QKV', width_ufs=width_ufs)

W = te.placeholder((QKV_NUM, IN_SIZE, NUM_HEADS, OUT_SIZE), name='W')

loop_ufs=[ls[0], ls[1], ls[2], ls[3], ls[5]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
O = te.ragged_compute((QKV_NUM, args.batch_size, NUM_HEADS, MAX_LEN, OUT_SIZE), [qkv, bd, md, s1, od], loop_ufs,
                      lambda ds: tvm.sum(W[ds[qkv], k, ds[md], ds[od]] * QKV[ds[qkv], ds[bd], ds[s1], k],
                                         axis = k, dimensions = [id]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

tile = 128
rtile = 8
nt = tile // rtile
ks = next_power_of_2(IN_SIZE / (6144 // tile))

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")
block_z = lambda: tvm.thread_axis("blockIdx.z")

Ol = s.cache_write(O, "local")
Ws = s.cache_read(W, "shared", [Ol], vanilla=True)
QKVs = s.cache_read(QKV, "shared", [Ol])

Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)
QKVl = s.cache_read(QKVs, "local", [Ol])

q, b, h, l, o = s[O].leaf_iter_vars[0:5]
s[O].reorder(b, l, h, o)
s[O].bind(q, block_z());
x = s[O].fuse(h, o)
xo, xi = s[O].split(x, factor = tile)
y = s[O].fuse(b, l)
yo, yi = s[O].split(y, factor = tile)
s[O].bind(yo, block_y())
s[O].bind(xo, block_x())

yio, yii = s[O].split(yi, factor = nt)
xio, xii = s[O].split(xi, factor = nt)
s[O].bind(yii, thread_y())
s[O].bind(xii, thread_x())
s[O].reorder(yii, xii, yio, xio)
s[O].bind(yio, te.thread_axis("vthread"))
s[O].bind(xio, te.thread_axis("vthread"))
s[Ol].compute_at(s[O], xio)

q, b, h, l, o, k = s[Ol].leaf_iter_vars
s[Ol].reorder(k, b, l, h, o)
ko, ki = s[Ol].split(k, nparts = ks)
s[Ws].compute_at(s[Ol], ko)
s[QKVs].compute_at(s[Ol], ko)
s[Wl].compute_at(s[Ol], ki)
s[QKVl].compute_at(s[Ol], ki)

f = s[Ws].fuse(*s[Ws].leaf_iter_vars)
xo, xi = s[Ws].split(f, factor = nt * nt)
xio, xii = s[Ws].split(xi, factor = nt)
s[Ws].bind(xio, thread_y())
s[Ws].bind(xii, thread_x())

q, b, l, i = s[QKVs].leaf_iter_vars
f = s[QKVs].fuse(b, l)
s[QKVs].reorder(i, f)
f = s[QKVs].fuse(i, f)
xo, xi = s[QKVs].split(f, factor = nt * nt)
xio, xii = s[QKVs].split(xi, factor = nt)
s[QKVs].bind(xio, thread_y())
s[QKVs].bind(xii, thread_x())

s.fuse_tensor_dimensions(QKVs, 1, 2)
s.reorder_tensor_dimensions(QKVs, 1, 2)



# ntx = 16
# nty = 16
# Ol = s.cache_write(O, "local")
# Ws = s.cache_read(W, "shared", [Ol], vanilla=True)
# QKVs = s.cache_read(QKV, "shared", [Ol])

# Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)
# QKVl = s.cache_read(QKVs, "local", [Ol])

# q, b, h, l, o = s[O].leaf_iter_vars[0:5]
# s[O].reorder(q, h, b, l)
# f = s[O].fuse(q, h)
# s[O].bind(f, block_y());
# f = s[O].fuse(b, l)
# fo, fi = s[O].split(f, factor = 64)
# s[O].bind(fo, block_x())
# s[Ol].compute_at(s[O], o)

# fio, fii = s[O].split(fi, factor = nty)
# oo, oi = s[O].split(o, factor = ntx)
# s[O].bind(fii, thread_y())
# s[O].bind(oi, thread_x())
# s[O].reorder(fii, oi, fio, oo)
# s[O].bind(fio, te.thread_axis("vthread"))
# s[O].bind(oo, te.thread_axis("vthread"))
# s[Ol].compute_at(s[O], oo)

# q, b, h, l, o, k = s[Ol].leaf_iter_vars
# s[Ol].reorder(k, q, o)
# ko, ki = s[Ol].split(k, nparts = 4)
# s[Ws].compute_at(s[Ol], ko)
# s[QKVs].compute_at(s[Ol], ko)
# s[Wl].compute_at(s[Ol], ki)
# s[QKVl].compute_at(s[Ol], ki)

# f = s[Ws].fuse(*s[Ws].leaf_iter_vars)
# xo, xi = s[Ws].split(f, factor = ntx * nty)
# xio, xii = s[Ws].split(xi, factor = ntx)
# s[Ws].bind(xio, thread_y())
# s[Ws].bind(xii, thread_x())

# q, b, l, i = s[QKVs].leaf_iter_vars
# f = s[QKVs].fuse(b, l)
# s[QKVs].reorder(i, f)
# f = s[QKVs].fuse(f, i)
# xo, xi = s[QKVs].split(f, factor = ntx * nty)
# xio, xii = s[QKVs].split(xi, factor = ntx)
# s[QKVs].bind(xio, thread_y())
# s[QKVs].bind(xii, thread_x())

# s.fuse_tensor_dimensions(QKVs, 1, 2)
# s.reorder_tensor_dimensions(QKVs, 1, 2)

suffix = ""
gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0] + suffix
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

inputs = [[lens], [QKV, W, O]]
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
    run_utils.run(fadd, i_bufs, [QKV, W, O], args.batch_size, args.max_batches,
                  args.dataset, args.datadir, args.target, args.debug)
