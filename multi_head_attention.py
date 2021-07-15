import tvm
from tvm import tir, te
from tvm.te import RangeDimension as RDim
from tvm.tir import TensorArray, Buffer, TECapsule
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 32
MAX_LEN = te.var('max_len')
# MAX_LEN = 128
NUM_HEADS = 8
IN_SIZE = 256
HEAD_SIZE = 64
scale = 16

lens = te.placeholder((BATCH_SIZE,), name = 'seq_lens', dtype = 'int32')
starts = te.placeholder((BATCH_SIZE,), name = 'seq_starts', dtype = 'int32')
sentence = te.placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), name = 'sentence')
Wi = te.placeholder((IN_SIZE, NUM_HEADS, 3 * HEAD_SIZE), name = 'Wi')
Wo = te.placeholder((NUM_HEADS * HEAD_SIZE, IN_SIZE), name = 'Wo')


bd = RDim('bd')
md = RDim('md')
sd1c = RDim('sd1')
sd2c = RDim('sd2')
sd1v = sd1c
sd2v = sd2c
hd = RDim('hd')
id = RDim('id')

# def len_uf_c(name): return Uf.from_constant(name, MAX_LEN)
def len_uf_c(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: lens[b])
def len_uf_v(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: lens[b])

lds =  {
    0: (bd, Uf.from_constant('bd', BATCH_SIZE)),
    1: (md, Uf.from_constant('md', NUM_HEADS)),
    2: (sd1c, len_uf_c('sd1c')),
    3: (sd2c, len_uf_c('sd2c')),
    4: (sd1v, len_uf_v('sd1v')),
    5: (sd2v, len_uf_v('sd2v')),
    6: (hd, Uf.from_constant('hd', HEAD_SIZE)),
    7: (hd, Uf.from_constant('hd', 3 * HEAD_SIZE)),
    8: (id, Uf.from_constant('id', IN_SIZE)),
}

k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
QKV = te.indirect_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, 3 * HEAD_SIZE), [bd, md, sd1v, hd], [lds[0], lds[1], lds[4], lds[7]], [],
                          lambda ds: tvm.sum(sentence[ds[bd], ds[sd1v], k] * Wi[k, ds[md], ds[hd]], axis = k), name = 'QKV')

def Q(b, h, s, i): return QKV[b, h, s, i]
def K(b, h, s, i): return QKV[b, h, s, i + HEAD_SIZE]
def V(b, h, s, i): return QKV[b, h, s, i + 2*HEAD_SIZE]

k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
QKt = te.indirect_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, sd1c, sd2c], [lds[0], lds[1], lds[2], lds[3]], [],
                          lambda ds: tvm.sum(Q(ds[bd], ds[md], ds[sd1c], k) *
                                             K(ds[bd], ds[md], ds[sd2c], k), axis = k), name = 'QKt')

QKtexp = te.indirect_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, sd1c, sd2c], [lds[0], lds[1], lds[2], lds[3]], [],
                             lambda ds: tvm.exp(QKt[ds[bd], ds[md], ds[sd1c], ds[sd2c]] / scale), name = 'QKtexp')

k = tvm.reduce_axis((0, MAX_LEN), name = 'k')
QKtsum = te.indirect_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN), [bd, md, sd1c], [lds[0], lds[1], lds[2]], [],
                             lambda ds, rds: tvm.sum(QKtexp[ds[bd], ds[md], ds[sd1c], rds['k']], axis=rds['k']), name = 'QKtsum',
                             reduce_axis_ufs = [('k', len_uf_v('k'))])

QKts = te.indirect_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, sd1c, sd2c], [lds[0], lds[1], lds[2], lds[3]], [],
                           lambda ds: QKtexp[ds[bd], ds[md], ds[sd1c], ds[sd2c]] / QKtsum[ds[bd], ds[md], ds[sd1c]], name = 'QKts')

attn = te.indirect_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, sd1c, hd], [lds[0], lds[1], lds[2], lds[6]], [],
                           lambda ds, rds: tvm.sum(QKts[ds[bd], ds[md], ds[sd1c], rds['k']] *
                                                   V(ds[bd], ds[md], rds['k'], ds[hd]), axis=rds['k']),
                           name = 'attn', reduce_axis_ufs = [('k', len_uf_v('k'))])

k = tvm.reduce_axis((0, NUM_HEADS * HEAD_SIZE), name = 'k')
attn_linear = te.indirect_compute((BATCH_SIZE, MAX_LEN, IN_SIZE), [bd, sd1v, id], [lds[0], lds[4], lds[8]], [],
                                  lambda ds: tvm.sum(attn[ds[bd], tvm.indexdiv(k.var, HEAD_SIZE),
                                                          ds[sd1v], tvm.indexmod(k.var, HEAD_SIZE)] *
                                                     Wo[k, ds[id]], axis=k),
                                  name = 'attn_linear')

s = tvm.create_schedule([attn_linear.op])

inputs = [lens, starts, sentence, Wi, Wo]
stmt = tvm.lower(s, inputs, simple_mode = True)
print(stmt)
# module = tvm.build(s, inputs, "cuda")
# print(module.imported_modules[0].get_source())
