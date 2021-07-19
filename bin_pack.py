import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE1 = 32
BATCH_SIZE2 = 64
HEAD_SIZE = 256

bd = Dim('bd')
hd = Dim('hd')

ls =  {
    1: Uf.from_constant('bd1', BATCH_SIZE1),
    2: Uf.from_constant('bd2', BATCH_SIZE2),
    3: Uf.from_constant('hd', HEAD_SIZE),
}

loop1_ufs=[ls[1], ls[3]]
loop2_ufs=[ls[2], ls[3]]

Q1 = te.ragged_placeholder((BATCH_SIZE1, HEAD_SIZE), [bd, hd], loop1_ufs, name='Q1', width_ufs=loop1_ufs)
Q2 = te.ragged_placeholder((BATCH_SIZE2, HEAD_SIZE), [bd, hd], loop2_ufs, name='Q2', width_ufs=loop2_ufs)

O1 = te.ragged_compute((BATCH_SIZE1, HEAD_SIZE), [bd, hd], loop1_ufs, lambda ds: Q1[ds[bd], ds[hd]], name = 'O1')
O2 = te.ragged_compute((BATCH_SIZE2, HEAD_SIZE), [bd, hd], loop2_ufs, lambda ds: Q2[ds[bd], ds[hd]], name = 'O2')

s = tvm.create_schedule([O1.op, O2.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
block_x = lambda: tvm.thread_axis("blockIdx.x")

s[O1].bind(s[O1].leaf_iter_vars[0], block_x())
s[O2].bind(s[O2].leaf_iter_vars[0], block_x())

s[O1].bind(s[O1].leaf_iter_vars[1], thread_x())
s[O2].bind(s[O2].leaf_iter_vars[1], thread_x())

s.hfuse([(O1.op, s[O1].leaf_iter_vars[0]), (O2.op, s[O2].leaf_iter_vars[0])])

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [Q1, Q2, O1, O2]
# stmt = tvm.lower(s, inputs, simple_mode = True)
# print(stmt)
fadd = tvm.build(s, inputs, "cuda")
print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
