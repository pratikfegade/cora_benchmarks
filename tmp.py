import utils
import run_utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 2
MAX_LEN = 128
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
RTILE=4

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len_uf(name): return Uf(name, (64, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], TILE))

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
O = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                        lambda ds: tvm.sum(Q[ds[bd], ds[md], ds[s1], k] * K[ds[bd], ds[md], ds[s2], k], axis = k),
                        name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

target = "cuda"
if target == "cuda":
    thread_x = lambda: tvm.thread_axis("threadIdx.x")
    thread_y = lambda: tvm.thread_axis("threadIdx.y")
    block_x = lambda: tvm.thread_axis("blockIdx.x")
    block_y = lambda: tvm.thread_axis("blockIdx.y")

    Ol = s.cache_write(O, "local")
    # Qs = s.cache_read(Q, "shared", [Ol])
    # Ks = s.cache_read(K, "shared", [Ol])

    # Ql = s.cache_read(Qs, "local", [Ol])
    # Kl = s.cache_read(Ks, "local", [Ol])

    b, h, x, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)

    s[O].reorder(b, xo, yo, h, xi, yi)
    f1 = s[O].fuse(xo, yo)
    f2 = s[O].fuse(b, f1)
    f = s[O].fuse(f2, h)
    s[O].bind(f, block_x())
    # s[Qs].compute_at(s[O], f)
    # s[Ks].compute_at(s[O], f)

    xio, xii = s[O].split(xi, nparts = 16)
    yio, yii = s[O].split(yi, nparts = 16)
    s[O].reorder(xio, yio, xii, yii)
    s[O].bind(xio, thread_y())
    s[O].bind(yio, thread_x())
    s[Ol].compute_at(s[O], yio)

    x, y, k = s[Ol].leaf_iter_vars[2], s[Ol].leaf_iter_vars[3], s[Ol].leaf_iter_vars[4]
    s[Ol].reorder(k, x, y)
    # s[Ql].compute_at(s[Ol], k)
    # s[Kl].compute_at(s[Ol], k)

    # x, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3]
    # xo, xi = s[Qs].split(x, nparts = 16)
    # yo, yi = s[Qs].split(y, nparts = 16)
    # s[Qs].bind(xo, thread_y())
    # s[Qs].bind(yo, thread_x())

    # x, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3]
    # xo, xi = s[Ks].split(x, nparts = 16)
    # yo, yi = s[Ks].split(y, nparts = 16)
    # s[Ks].bind(xo, thread_y())
    # s[Ks].bind(yo, thread_x())

    tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
else:
    Ol = s.cache_write(O, "local")
    Qs = s.cache_read(Q, "shared", [Ol])
    Ks = s.cache_read(K, "shared", [Ol])

    Ql = s.cache_read(Qs, "local", [Ol])
    Kl = s.cache_read(Ks, "local", [Ol])

    b, h, x, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)

    s[O].reorder(b, xo, yo, h, xi, yi)
    f1 = s[O].fuse(xo, yo)
    f2 = s[O].fuse(b, f1)
    f = s[O].fuse(f2, h)
    # s[O].bind(f, block_x())
    s[Qs].compute_at(s[O], f)
    s[Ks].compute_at(s[O], f)

    xio, xii = s[O].split(xi, nparts = 16)
    yio, yii = s[O].split(yi, nparts = 16)
    s[O].reorder(xio, yio, xii, yii)
    s[Ol].compute_at(s[O], yio)

    x, y, k = s[Ol].leaf_iter_vars[2], s[Ol].leaf_iter_vars[3], s[Ol].leaf_iter_vars[4]
    s[Ol].reorder(k, x, y)
    s[Ql].compute_at(s[Ol], k)
    s[Kl].compute_at(s[Ol], k)

inputs = [[lens], [Q, K, O]]
# stmt = tvm.lower(s, inputs, simple_mode = True)
# print(stmt)
fadd, i_bufs = tvm.build(s, inputs, target)
# fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
# print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
# print(fadd.get_source())
# print('-----GPU code-----\n' + str(i_bufs))
l_input = run_utils.random_lengths(BATCH_SIZE, MAX_LEN)
print(((l_input + 63) // 64) * 64)
run_utils.run(fadd, [l_input], i_bufs, [Q, K, O], target)
