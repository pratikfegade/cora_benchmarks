import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 32
MAX_LEN = 128
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
RTILE=4
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len_uf(name): return Uf(name, (64, MAX_LEN), [bd], lambda b: TILE*lens[b])

luf = len_uf('s')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: Uf.from_constant('md', NUM_HEADS),
    2: luf,
    3: luf,
    4: Uf.from_constant('hd', HEAD_SIZE),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[4]]
width_ufs=loop_ufs
Q = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[4]]
width_ufs=loop_ufs
K = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s2, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
A = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[ds[bd], ds[md], ds[s1], k] * K[ds[bd], ds[md], ds[s2], k], axis = k),
                      name = 'A', width_uf_lists=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[loop_ufs]
Aexp = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                         lambda ds: tvm.exp(A[ds[bd], ds[md], ds[s1], ds[s2]] * scale), name = 'Aexp')

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=[loop_ufs]
Asum = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN), [bd, md, s1], loop_ufs,
                         lambda ds, rds: tvm.sum(Aexp[ds[bd], ds[md], ds[s1], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Asum', reduce_axis_ufs = [('k', luf)])

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[loop_ufs]
O = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                      lambda ds: Aexp[ds[bd], ds[md], ds[s1], ds[s2]] / Asum[ds[bd], ds[md], ds[s1]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Al = s.cache_write(A, "local")
Qs = s.cache_read(Q, "shared", [Al])
Ks = s.cache_read(K, "shared", [Al])

Ql = s.cache_read(Qs, "local", [Al])
Kl = s.cache_read(Ks, "local", [Al])


ko, ki = s[Asum].split(s[Asum].op.reduce_axis[0], nparts = 32)
Asum_rf = s.rfactor(Asum, ki, 1)


b, h, x, y = s[A].leaf_iter_vars[0:4]
xo, xi = s[A].split(x, factor = 64)
yo, yi = s[A].split(y, factor = 64)

s[A].reorder(b, xo, yo, h, xi, yi)
f1 = s[A].fuse(xo, yo)
f2 = s[A].fuse(b, f1)
f = s[A].fuse(f2, h)
s[A].bind(f, block_x())
s[Qs].compute_at(s[A], f)
s[Ks].compute_at(s[A], f)

xio, xii = s[A].split(xi, nparts = 16)
yio, yii = s[A].split(yi, nparts = 16)
s[A].reorder(xio, yio, xii, yii)
s[A].bind(xio, thread_y())
s[A].bind(yio, thread_x())
s[Al].compute_at(s[A], yio)

x, y, k = s[Al].leaf_iter_vars[2], s[Al].leaf_iter_vars[3], s[Al].leaf_iter_vars[4]
s[Al].reorder(k, x, y)
s[Ql].compute_at(s[Al], k)
s[Kl].compute_at(s[Al], k)

x, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3]
xo, xi = s[Qs].split(x, nparts = 16)
yo, yi = s[Qs].split(y, nparts = 16)
s[Qs].bind(xo, thread_y())
s[Qs].bind(yo, thread_x())

x, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3]
xo, xi = s[Ks].split(x, nparts = 16)
yo, yi = s[Ks].split(y, nparts = 16)
s[Ks].bind(xo, thread_y())
s[Ks].bind(yo, thread_x())

s[A].set_scope('global')





b, h, s1, s2 = s[O].leaf_iter_vars
s[O].reorder(s1, h)
s1o, s1i = s[O].split(s1, factor = 32)
f1 = s[O].fuse(b, s1o)
f = s[O].fuse(f1, s1i)
s[O].bind(f, block_x())
s[O].bind(h, thread_y())

tx = thread_x()
xo, xi = s[O].split(s2, nparts = 32)
s[O].bind(xo, tx)
s[Asum_rf].bind(s[Asum_rf].op.reduce_axis[0], tx)

s[Asum].compute_at(s[O], xo)
s[Asum_rf].compute_at(s[Asum], s[Asum].leaf_iter_vars[3])
s[Aexp].compute_at(s[O], xo)

s[Asum].set_scope('local')
s[Asum_rf].set_scope('local')
s[Aexp].set_scope('local')
s[O].set_scope('global')


tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
inputs = [lens, Q, K]
stmt = tvm.lower(s, inputs, simple_mode = True)
print(stmt)
# fadd = tvm.build(s, inputs, "cuda")
# print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
