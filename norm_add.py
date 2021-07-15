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

eps = 0.00001
beta = 2
gamma = 5

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
od = Dim('od')

def len_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: 64*lens[b])

luf = len_uf('s')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: luf,
    2: Uf.from_constant('od', OUT_SIZE),
}

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A1 = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                           name='A1', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A2 = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                           name='A2', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=[loop_ufs]
A = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: A1[ds[bd], ds[s1], ds[od]] + A2[ds[bd], ds[s1], ds[od]],
                      name = 'A')

loop_ufs=[ls[0], ls[1]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, OUT_SIZE), name = 'k')
Am1 = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, s1], loop_ufs,
                        lambda ds: tvm.sum(A[ds[bd], ds[s1], k], axis=k),
                        name = 'Am1')

loop_ufs=[ls[0], ls[1]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, OUT_SIZE), name = 'k')
Am2 = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, s1], loop_ufs,
                        lambda ds: tvm.sum(A[ds[bd], ds[s1], k] * A[ds[bd], ds[s1], k], axis=k),
                        name = 'Am2')

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=[loop_ufs]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: beta + gamma * (A[ds[bd], ds[s1], ds[od]] - Am1[ds[bd], ds[s1]]) /
                                    tvm.sqrt(Am2[ds[bd], ds[s1]] - Am1[ds[bd], ds[s1]]*Am1[ds[bd], ds[s1]] + eps),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = tvm.thread_axis("threadIdx.x")
thread_y = tvm.thread_axis("threadIdx.y")
block_x = tvm.thread_axis("blockIdx.x")
block_y = tvm.thread_axis("blockIdx.y")

ko, ki = s[Am1].split(s[Am1].op.reduce_axis[0], nparts = 32)
Am1_rf = s.rfactor(Am1, ko)
ko, ki = s[Am2].split(s[Am2].op.reduce_axis[0], nparts = 32)
Am2_rf = s.rfactor(Am2, ko)

b, s1, h = s[O].leaf_iter_vars
s1o, s1i = s[O].split(s1, factor = 64)
s1io, s1ii = s[O].split(s1i, factor = 8)
f1 = s[O].fuse(b, s1o)
f = s[O].fuse(f1, s1io)
s[O].bind(f, block_x)
s[O].bind(s1ii, thread_y)

ho, hi = s[O].split(h, nparts = 32)
s[O].bind(ho, thread_x)

s[Am1_rf].compute_at(s[O], ho)
s[Am2_rf].compute_at(s[O], ho)
s[Am1].compute_at(s[O], ho)
s[Am2].compute_at(s[O], ho)
s[A].compute_at(s[O], ho)

s[Am1].bind(s[Am1].op.reduce_axis[0], thread_x)
s[Am2].bind(s[Am2].op.reduce_axis[0], thread_x)


s[Am1_rf].set_scope('local')
s[Am2_rf].set_scope('local')
s[Am1].set_scope('local')
s[Am2].set_scope('local')
s[A].set_scope('local')

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [lens, A1, A2]
# stmt = tvm.lower(s, inputs, simple_mode = True)
# print(stmt)
fadd = tvm.build(s, inputs, "cuda")
print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
