import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 32
# MAX_LEN = te.var('max_len') - 1
MAX_LEN = 128
NUM_HEADS = 8
HEAD_SIZE = 64

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len1_uf(name): return Uf(name, (64, MAX_LEN), [bd], lambda b: 64 * tvm.floordiv(lens[b] + 63, 64))
def len2_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: lens[b])

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: Uf.from_constant('md', NUM_HEADS),
    2: len1_uf('s1'),
    3: len2_uf('s2'),
    4: Uf.from_constant('hd', HEAD_SIZE),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[4]]
width_ufs=loop_ufs
V = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s2, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[4]]
width_ufs=[loop_ufs]
O = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[md], ds[s1], rds['k']] *
                                              V(ds[bd], ds[md], rds['k'], ds[hd]), axis=rds['k']),
                      name = 'O', reduce_axis_ufs = [('k', len2_uf('k'))],
                      width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Ol = s.cache_write(O, "local")
As = s.cache_read(A, "shared", [Ol])
Vs = s.cache_read(V, "shared", [Ol])

Al = s.cache_read(As, "local", [Ol])
Vl = s.cache_read(Vs, "local", [Ol])

b, h, x, y = s[O].leaf_iter_vars[0:4]
xo, xi = s[O].split(x, factor = 64)

s[O].reorder(b, xo, h, y, xi)
f1 = s[O].fuse(b, xo)
f = s[O].fuse(f1, h)
s[O].bind(f, block_x())
s[Ol].compute_at(s[O], f)

xio, xii = s[O].split(xi, nparts = 16)
yo, yi = s[O].split(y, nparts = 16)
s[O].reorder(xio, yo, xii, yi)
s[O].bind(xio, thread_y())
s[O].bind(yo, thread_x())
s[Ol].compute_at(s[O], yo)

x, y, k = s[Ol].leaf_iter_vars[2], s[Ol].leaf_iter_vars[3], s[Ol].leaf_iter_vars[4]
s[Ol].reorder(k, x, y)
ko, ki = s[Ol].split(k, factor = 64)
s[As].compute_at(s[Ol], ko)
s[Vs].compute_at(s[Ol], ko)
s[Al].compute_at(s[Ol], ki)
s[Vl].compute_at(s[Ol], ki)
s[Ol].peel(ko)

x, y = s[As].leaf_iter_vars[2], s[As].leaf_iter_vars[3]
xo, xi = s[As].split(x, nparts = 16)
yo, yi = s[As].split(y, nparts = 16)
s[As].bind(xo, thread_y())
s[As].bind(yo, thread_x())

x, y = s[Vs].leaf_iter_vars[2], s[Vs].leaf_iter_vars[3]
xo, xi = s[Vs].split(x, nparts = 16)
yo, yi = s[Vs].split(y, nparts = 16)
s[Vs].bind(xo, thread_y())
s[Vs].bind(yo, thread_x())

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [lens, V, A]
# stmt = tvm.lower(s, inputs, simple_mode = True)
# print(stmt)
fadd = tvm.build(s, inputs, "cuda")
print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
