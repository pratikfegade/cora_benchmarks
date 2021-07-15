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
OUT_SIZE = 512

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
hd = Dim('hd')
od = Dim('od')

def len_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: 64*lens[b])

luf = len_uf('s')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: Uf.from_constant('md', NUM_HEADS),
    2: luf,
    3: Uf.from_constant('hd', HEAD_SIZE),
    4: Uf.from_constant('od', OUT_SIZE),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((OUT_SIZE, NUM_HEADS * HEAD_SIZE), name='W')

loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, NUM_HEADS * HEAD_SIZE), name = 'k')
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(A[ds[bd], tvm.floordiv(k, HEAD_SIZE), ds[s1], tvm.floormod(k, HEAD_SIZE)] *
                                         W[ds[od], k], axis=k),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Ol = s.cache_write(O, "local")
Ws = s.cache_read(W, "shared", [Ol], vanilla=True)
As = s.cache_read(A, "shared", [Ol])

Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)
Al = s.cache_read(As, "local", [Ol])

b, x, y = s[O].leaf_iter_vars
xo, xi = s[O].split(x, factor = 64)
s[O].reorder(b, xo, y, xi)
f = s[O].fuse(b, xo)
s[O].bind(f, block_x())

xio, xii = s[O].split(xi, nparts = 16)
s[O].bind(xio, thread_x())
yo, yi = s[O].split(y, nparts = 8)
yio, yii = s[O].split(yi, factor = 4)
s[O].bind(yio, thread_y())
s[O].reorder(yo, xio, yio, xii, yii)

s[Ol].compute_at(s[O], yio)

bl, xl, yl, kl = s[Ol].leaf_iter_vars
s[Ol].reorder(kl, xl, yl)
klo, kli = s[Ol].split(kl, factor = 64)
s[As].compute_at(s[Ol], klo)
s[Ws].compute_at(s[Ol], klo)
s[Al].compute_at(s[Ol], kli)
s[Wl].compute_at(s[Ol], kli)

x, y = s[As].leaf_iter_vars[2], s[As].leaf_iter_vars[3]
xo, xi = s[As].split(x, nparts = 16)
yo, yi = s[As].split(y, nparts = 16)
s[As].bind(xo, thread_y())
s[As].bind(yo, thread_x())

x, y = s[Ws].leaf_iter_vars[0], s[Ws].leaf_iter_vars[1]
xo, xi = s[Ws].split(x, nparts = 16)
yo, yi = s[Ws].split(y, nparts = 16)
s[Ws].bind(xo, thread_y())
s[Ws].bind(yo, thread_x())


tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [lens, W, A]
stmt = tvm.lower(s, inputs, simple_mode = True)
print(stmt)
# fadd = tvm.build(s, inputs, "cuda")
# print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
