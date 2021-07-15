import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 32
MAX_LEN = 128
NUM_HEADS = 8
IN_SIZE = 256
OUT_SIZE = 64
QKV_NUM = 3
TILE=64
RTILE=4

lens = te.placeholder((BATCH_SIZE + 1,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
qkv = Dim('qkv')
md = Dim('md')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: lens[b])

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: len_uf('s1'),
    2: Uf.from_constant('md', NUM_HEADS),
    3: Uf.from_constant('qkv', QKV_NUM),
    4: Uf.from_constant('id', IN_SIZE),
    5: Uf.from_constant('od', OUT_SIZE),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3], ls[4]]
width_ufs=loop_ufs
QKV = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, QKV_NUM, IN_SIZE), [bd, s1, qkv, md, id], loop_ufs,
                            name='QKV', width_ufs=width_ufs)

loop_ufs=[ls[5], ls[4]]
# W = te.ragged_placeholder((OUT_SIZE, IN_SIZE), [od, id], loop_ufs, name='W')
W = te.placeholder((OUT_SIZE, IN_SIZE), name='W')

loop_ufs=[ls[0], ls[1], ls[2], ls[3], ls[5]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, QKV_NUM, OUT_SIZE), [bd, s1, qkv, md, od], loop_ufs,
                      lambda ds: tvm.sum(W[ds[od], k] * QKV[ds[bd], ds[s1], ds[qkv], ds[md], k], axis = k),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Ol = s.cache_write(O, "local")
Ws = s.cache_read(W, "shared", [Ol], vanilla=True)
QKVs = s.cache_read(QKV, "shared", [Ol])

Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)
QKVl = s.cache_read(QKVs, "local", [Ol])

f = s[O].fuse(*s[O].leaf_iter_vars[0:2])
s[O].bind(f, block_x());

s[O].bind(s[O].leaf_iter_vars[1], thread_y())
yo, yi = s[O].split(s[O].leaf_iter_vars[3], nparts = 32)
s[O].bind(yo, thread_x())
s[O].reorder(s[O].leaf_iter_vars[1], yo, s[O].leaf_iter_vars[2], yi)
s[Ol].compute_at(s[O], yo)


s[Ol].reorder(s[Ol].leaf_iter_vars[5], s[Ol].leaf_iter_vars[3], s[Ol].leaf_iter_vars[4])
s[Ol].split(s[Ol].leaf_iter_vars[3], nparts = 4)
s[Ws].compute_at(s[Ol], s[Ol].leaf_iter_vars[3])
s[QKVs].compute_at(s[Ol], s[Ol].leaf_iter_vars[3])
s[Wl].compute_at(s[Ol], s[Ol].leaf_iter_vars[4])
s[QKVl].compute_at(s[Ol], s[Ol].leaf_iter_vars[4])

f = s[Ws].fuse(*s[Ws].leaf_iter_vars[0:2])
xo, xi = s[Ws].split(f, factor = 256)
xio, xii = s[Ws].split(xi, factor = 32)
s[Ws].bind(xio, thread_y())
s[Ws].bind(xii, thread_x())

s[QKVs].bind(s[QKVs].leaf_iter_vars[2], thread_y())
xo, xi = s[QKVs].split(s[QKVs].leaf_iter_vars[4], factor = 32)
s[QKVs].bind(xi, thread_x())

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [lens, QKV, W]
stmt = tvm.lower(s, inputs, simple_mode = True)
print(stmt)
# fadd = tvm.build(s, inputs, "cuda")
# print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
