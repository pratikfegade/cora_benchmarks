import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

def intrin_wmma_load_matrix(scope):
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                n,
                n,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float16")
    B = te.placeholder((n, n), name="B", dtype="float16")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute(
        (n, n),
        lambda ii, jj: te.sum(A[ii, k].astype("float") * B[k, jj].astype("float"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=256
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=256
    )
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="BC", scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle", "tir.tvm_fill_fragment", BC.data, n, n, n, BC.elem_offset // 256, 0.0
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // 256,
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    BC.data,
                    BC.elem_offset // 256,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})

def intrin_wmma_store_matrix():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                n,
                n,
                BA.elem_offset // 256,
                BC.access_ptr("w"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


BATCH_SIZE = 32
MAX_LEN = 128
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
RTILE=4

dtype = "float16"
lens = te.placeholder((BATCH_SIZE + 1,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len_uf(name): return Uf(name, (64, MAX_LEN), [bd], lambda b: TILE*lens[b])

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
                          name='Q', width_ufs=width_ufs, dtype = dtype)

loop_ufs=[ls[0], ls[1], ls[3], ls[4]]
width_ufs=loop_ufs
K = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s2, hd], loop_ufs,
                          name='K', width_ufs=width_ufs, dtype = dtype)

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
O = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                        lambda ds: tvm.sum(Q[ds[bd], ds[md], ds[s1], k] * K[ds[bd], ds[md], ds[s2], k], axis = k),
                        name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Ol = s.cache_write(O, "wmma.accumulator")
Qs = s.cache_read(Q, "shared", [Ol])
Ks = s.cache_read(K, "shared", [Ol])

Ql = s.cache_read(Qs, "wmma.matrix_a", [Ol])
Kl = s.cache_read(Ks, "wmma.matrix_b", [Ol])

b, h, x, y = s[O].leaf_iter_vars[0:4]
xo, xi = s[O].split(x, factor = 64)
yo, yi = s[O].split(y, factor = 64)

s[O].reorder(b, xo, yo, h, xi, yi)
f1 = s[O].fuse(xo, yo)
f2 = s[O].fuse(b, f1)
f = s[O].fuse(f2, h)
s[O].bind(f, block_x())
s[Qs].compute_at(s[O], f)
s[Ks].compute_at(s[O], f)

xio, xii = s[O].split(xi, nparts = 16)
yio, yii = s[O].split(yi, nparts = 16)
s[O].reorder(xio, yio, xii, yii)
s[O].bind(xio, thread_y())
s[O].bind(yio, thread_x())
s[Ol].compute_at(s[O], yio)

x, y, k = s[Ol].leaf_iter_vars[2], s[Ol].leaf_iter_vars[3], s[Ol].leaf_iter_vars[4]
ko, ki = s[Ol].split(k, factor = 8)
s[Ol].reorder(ko, ki, x, y)
s[Qs].compute_at(s[Ol], ko)
s[Ks].compute_at(s[Ol], ko)
s[Ql].compute_at(s[Ol], ki)
s[Kl].compute_at(s[Ol], ki)

x, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3]
xo, xi = s[Qs].split(x, nparts = 16)
yo, yi = s[Qs].split(y, nparts = 16)
s[Qs].bind(xo, thread_y())
s[Qs].bind(yo, thread_x())

x, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3]
xo, xi = s[Ks].split(x, nparts = 16)
yo, yi = s[Ks].split(y, nparts = 16)
s[Ks].bind(xo, thread_y())
s[Ks].bind(yo, thread_x())

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [lens, Q, K]
# stmt = tvm.lower(s, inputs, simple_mode = True)
# print(stmt)
fadd = tvm.build(s, inputs, "cuda")
print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
