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
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
RTILE=4
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)
scale = 1/8

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len_uf(name): return Uf(name, "l", (64, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], TILE))

luf = len_uf('s')
ls =  {
    0: Uf.from_constant('bd', args.batch_size, "l"),
    1: Uf.from_constant('md', NUM_HEADS, "l"),
    2: luf,
    3: luf,
    4: Uf.from_constant('hd', HEAD_SIZE, "l"),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[4]]
width_ufs = None if args.dense_storage else loop_ufs
Q = te.ragged_placeholder((args.batch_size, NUM_HEADS, HEAD_SIZE, MAX_LEN), [bd, md, s1, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[4]]
width_ufs = None if args.dense_storage else loop_ufs
K = te.ragged_placeholder((args.batch_size, NUM_HEADS, HEAD_SIZE, MAX_LEN), [bd, md, s2, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs = None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
A = te.ragged_compute((args.batch_size, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[ds[bd], ds[md], ds[s1], k] * K[ds[bd], ds[md], ds[s2], k],
                                         axis = k, dimensions = [hd]),
                      name = 'A', width_uf_lists=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
Aexp = te.ragged_compute((args.batch_size, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                         lambda ds: tvm.exp(A[ds[bd], ds[md], ds[s1], ds[s2]] * scale), name = 'Aexp')

loop_ufs=[ls[0], ls[1], ls[2]]
Asum = te.ragged_compute((args.batch_size, NUM_HEADS, MAX_LEN), [bd, md, s1], loop_ufs,
                         lambda ds, rds: tvm.sum(Aexp[ds[bd], ds[md], ds[s1], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Asum', reduce_axis_ufs = [('k', luf)])

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[loop_ufs]
O = te.ragged_compute((args.batch_size, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                      lambda ds: Aexp[ds[bd], ds[md], ds[s1], ds[s2]] / Asum[ds[bd], ds[md], ds[s1]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")


# Al = s.cache_write(A, "local")
# Qs = s.cache_read(Q, "shared", [Al])
# Ks = s.cache_read(K, "shared", [Al])

# Ql = s.cache_read(Qs, "local", [Al])
# Kl = s.cache_read(Ks, "local", [Al])

# b, h, x, y = s[A].leaf_iter_vars[0:4]
# xo, xi = s[A].split(x, factor = 64)
# yo, yi = s[A].split(y, factor = 64)

# s[A].reorder(b, xo, yo, h, xi, yi)
# f1 = s[A].fuse(xo, yo)
# f2 = s[A].fuse(b, f1)
# s[A].bind(f2, block_x())
# s[A].bind(h, block_y())
# s[Qs].compute_at(s[A], h)
# s[Ks].compute_at(s[A], h)

# xio, xii = s[A].split(xi, factor = 16)
# yio, yii = s[A].split(yi, factor = 16)
# s[A].bind(xii, thread_y())
# s[A].bind(yii, thread_x())
# s[A].bind(xio, tvm.thread_axis("vthread"))
# s[A].bind(yio, tvm.thread_axis("vthread"))
# s[A].reorder(xio, yio, xii, yii)
# s[Al].compute_at(s[A], yii)

# x, y, k = s[Al].leaf_iter_vars[2:5]
# s[Al].reorder(k, x, y)
# s[Ql].compute_at(s[Al], k)
# s[Kl].compute_at(s[Al], k)

# x, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3]
# f = s[Ks].fuse(x, y)
# fo, fi = s[Ks].split(f, factor = 256)
# fio, fii = s[Ks].split(fi, factor = 16)
# s[Ks].bind(fio, thread_y())
# s[Ks].bind(fii, thread_x())

# x, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3]
# f = s[Qs].fuse(x, y)
# fo, fi = s[Qs].split(f, factor = 256)
# fio, fii = s[Qs].split(fi, factor = 16)
# s[Qs].bind(fio, thread_y())
# s[Qs].bind(fii, thread_x())


ko, ki = s[Asum].split(s[Asum].op.reduce_axis[0], nparts = 32)
Asum_rf = s.rfactor(Asum, ki, 1)

b, h, s1, s2 = s[O].leaf_iter_vars
s[O].reorder(s1, h)
s1o, s1i = s[O].split(s1, factor = 32)
f1 = s[O].fuse(b, s1o)
f = s[O].fuse(f1, s1i)
s[O].bind(f, block_x())
s[O].bind(h, thread_y())

xo, xi = s[O].split(s2, nparts = 32)
tx = thread_x()
s[O].bind(xo, tx)
s[Asum_rf].bind(s[Asum_rf].op.reduce_axis[0], tx)

s[Asum].compute_at(s[O], xo)
s[Asum_rf].compute_at(s[Asum], s[Asum].leaf_iter_vars[3])
s[Aexp].compute_at(s[O], xo)

s[A].set_scope('global')

s[O].set_scope('global')
s[Asum].set_scope('local')
s[Asum_rf].set_scope('local')
s[Aexp].set_scope('local')

suffix = ""
gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0] + suffix
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

inputs = [[lens], [Q, K, O]]
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
    run_utils.run(fadd, i_bufs, [Q, K, O], args.batch_size, args.max_batches,
                  args.dataset, args.datadir, args.target, args.debug)
