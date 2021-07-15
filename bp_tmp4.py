import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

BATCH_SIZE = 32
MAX_LEN = 128
HEAD_SIZE = 64
TILE=64

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

max_lambda = lambda b: utils.ceilmult(lens[b], 32)
min_lambda = lambda b: utils.floormult(utils.ceilmult(lens[b], 32), 64)

def len_uf(name): return Uf(name, (64, 128), [bd], lambda b: max_lambda(b) - min_lambda(b))

luf = len_uf('s')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: luf,
    2: Uf.from_constant('hd', HEAD_SIZE),
}

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, HEAD_SIZE), [bd, s1, hd], loop_ufs, name='B', width_ufs=width_ufs)

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, HEAD_SIZE), [bd, s1, hd], loop_ufs,
                      lambda ds: A[ds[bd], ds[s1] - min_lambda(ds[bd]), ds[hd]] * 2, name = 'O',width_uf_lists=[width_ufs])

s = tvm.create_schedule([O.op])

thread_x = tvm.thread_axis("threadIdx.x")
block_x = tvm.thread_axis("blockIdx.x")

b, l, h = s[O].leaf_iter_vars
lo, li = s[O].split(l, factor = 32)
f1 = s[O].fuse(b, lo)
f1o, f1i = s[O].split(f1, factor = 2)
f2 = s[O].fuse(f1i, li)
s[O].bind(f1o, block_x)
s[O].bind(f2, thread_x)

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [lens, A, O]
stmt = tvm.lower(s, inputs, simple_mode = True)
print(stmt)
# fadd = tvm.build(s, inputs, "cuda")
# print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
