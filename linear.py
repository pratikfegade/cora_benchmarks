import tvm
from tvm import tir, te
from tvm.te import RangeDimension as RDim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 32
MAX_LEN = te.var('max_len')
# MAX_LEN = 128
NUM_HEADS = 8
IN_SIZE = 256
HEAD_SIZE = 64
scale = 16

lens = te.placeholder((BATCH_SIZE,), name = 'seq_lens', dtype = 'int32')
I = te.placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), name = 'I')
W = te.placeholder((NUM_HEADS, IN_SIZE, HEAD_SIZE), name = 'W')


bd = RDim('bd')
md = RDim('md')
s1 = RDim('s1')
hd = RDim('hd')
id = RDim('id')

def len_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: 64 * lens[b])

lds =  {
    0: (bd, Uf.from_constant('bd', BATCH_SIZE)),
    1: (md, Uf.from_constant('md', NUM_HEADS)),
    2: (s1, len_uf('s1')),
    3: (hd, Uf.from_constant('hd', HEAD_SIZE)),
    4: (id, Uf.from_constant('id', IN_SIZE)),
}

k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
Q = te.indirect_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], [lds[0], lds[1], lds[2], lds[3]], [],
                        lambda ds: tvm.sum(I[ds[bd], ds[s1], k] * W[ds[md], k, ds[hd]], axis = k), name = 'Q')

s = tvm.create_schedule([Q.op])

s[Q].reorder(s[Q].leaf_iter_vars[1], s[Q].leaf_iter_vars[0])

# xo, xi = s[QKt].split(s[QKt].leaf_iter_vars[2], factor = 64)
# yo, yi = s[QKt].split(s[QKt].leaf_iter_vars[4], factor = 64)
# f = s[QKt].fuse(xo, yo)

inputs = [lens, W, Q, I]
stmt = tvm.lower(s, inputs, simple_mode = True)
print(stmt)
# module = tvm.build(s, inputs, "cuda")
# print(module.imported_modules[0].get_source())
