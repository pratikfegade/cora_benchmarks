import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 32
MAX_LEN = 128
NUM_HEADS = 8
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')

def len_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: 32*lens[b])

luf = len_uf('s2')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: Uf.from_constant('md', NUM_HEADS),
    2: luf,
    3: luf,
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
Aexp = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                         lambda ds: tvm.exp(A[ds[bd], ds[md], ds[s1], ds[s2]] * scale), name = 'Aexp')

loop_ufs=[ls[0], ls[1], ls[2]]
Asum = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN), [bd, md, s1], loop_ufs,
                         lambda ds, rds: tvm.sum(Aexp[ds[bd], ds[md], ds[s1], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Asum', reduce_axis_ufs = [('k', luf)])

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[loop_ufs]
O = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                      lambda ds: Aexp[ds[bd], ds[md], ds[s1], ds[s2]] / Asum[ds[bd], ds[md], ds[s1]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = tvm.thread_axis("threadIdx.x")
thread_y = tvm.thread_axis("threadIdx.y")
block_x = tvm.thread_axis("blockIdx.x")
block_y = tvm.thread_axis("blockIdx.y")


ko, ki = s[Asum].split(s[Asum].op.reduce_axis[0], nparts = 32)
Asum_rf = s.rfactor(Asum, ki, 1)

b, h, s1, s2 = s[O].leaf_iter_vars
s[O].reorder(s1, h)
s1o, s1i = s[O].split(s1, factor = 32)
f1 = s[O].fuse(b, s1o)
f = s[O].fuse(f1, s1i)
s[O].bind(f, block_x)
s[O].bind(h, thread_y)

xo, xi = s[O].split(s2, nparts = 32)
s[O].bind(xo, thread_x)
s[Asum_rf].bind(s[Asum_rf].op.reduce_axis[0], thread_x)

s[Asum].compute_at(s[O], xo)
s[Asum_rf].compute_at(s[Asum], s[Asum].leaf_iter_vars[3])
s[Aexp].compute_at(s[O], xo)

s[Asum].set_scope('local')
s[Asum_rf].set_scope('local')
s[Aexp].set_scope('local')

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
inputs = [lens, A]
stmt = tvm.lower(s, inputs, simple_mode = True)
print(stmt)
# fadd = tvm.build(s, inputs, "cuda")
# print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
