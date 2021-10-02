import numpy as np
import math
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils

parser = run_utils.get_cmd_parser()
args = parser.parse_args()

args.target = run_utils.get_arm_target()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
HEAD_SIZE = 64
OUT_SIZE = 512

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
fd = Dim('fd')
s1 = Dim('s1')
hd = Dim('hd')
od = Dim('od')
mdhd = Dim('mdhd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufw1.get_uf(),
    3: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    4: Uf.from_constant('od', OUT_SIZE, 'l'),
    5: Uf.from_constant('mh', HEAD_SIZE * NUM_HEADS, 'l'),
    6: Uf.from_constant('ml', MAX_LEN, 'l'),
}

loop_ufs=[ls[0], ls[2], ls[5]]
if args.layout_unfused:
    width_ufs=[ls[0], lufw1.get_uf(), ls[5]]
else:
    width_ufs=[ls[0], lufw64.get_uf(), ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS * HEAD_SIZE), [bd, s1, mdhd], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[4], ls[5]]
width_ufs=[ls[4], ls[5]]
W = te.ragged_placeholder((OUT_SIZE, NUM_HEADS * HEAD_SIZE), [od, mdhd], loop_ufs,
                          name='W', width_ufs=width_ufs)

B = te.placeholder((OUT_SIZE,), name='B')

loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=loop_ufs
A2 = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                           name='A2', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=[[ls[0], ls[6], ls[4]]]
k = tvm.reduce_axis((0, NUM_HEADS * HEAD_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(A[ds[bd], ds[s1], k] * W[ds[od], k],
                                         axis=k, dimensions = [mdhd]),
                      name = 'S', width_uf_lists=width_ufs)

def compute_body(ds):
    if args.skip_residual: return S[ds[bd], ds[s1], ds[od]] + B[ds[od]]
    else: return A2[ds[bd], ds[s1], ds[od]] + S[ds[bd], ds[s1], ds[od]] + B[ds[od]]
loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=[[ls[0], ls[6], ls[4]]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      compute_body, name = 'O', width_uf_lists=width_ufs)

f_ext = te.var('sum')
rmap = te.fuse_ragged_axis([A, A2, W, B], O, bd, s1, fd, f_ext * 64)
A = rmap[A.op].output(0)
W = rmap[W.op].output(0)
B = rmap[B.op].output(0)
if not args.skip_residual:
    A2 = rmap[A2.op].output(0)
S = rmap[S.op].output(0)
O = rmap[O.op].output(0)

s = tvm.create_schedule([O.op])

def intrin_gemv(m, n, r):
    a = te.placeholder((m,r), name="a")
    b = te.placeholder((n,r), name="b")

    md = Dim('md')
    nd = Dim('nd')
    kd = Dim('kd')

    k = te.reduce_axis((0,r), name="k")
    c = te.ragged_compute((m, n), [md, nd], [Uf.from_constant('m', m, 'l'), Uf.from_constant('n', n, 'l')],
                          lambda ds: tvm.sum(a[ds[md], k] * b[ds[nd], k], axis=k, dimensions=[kd]),
                          name = 'O', width_uf_lists=None)

    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A1", offset_factor=1, strides=[512, 1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B1", offset_factor=1, strides=[512, 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C1", offset_factor=1, strides=[64, 1], scope='local')

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_packed("tvm.contrib.cblas.matmul_no_thread",
                                        aa,
                                        bb,
                                        cc,
                                        False,
                                        True,
                                        1.0,
                                        1.0))
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", "gemv_reset", cc.access_ptr("w"), m, n))
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def gemv_impl():
    cc_code = """
      extern "C" int gemv_reset(float *cc, int m, int n) {
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            cc[i * n + j] = 0.0;
          }
        }
        return 0;
      }
    """
    from tvm.contrib import util, clang

    temp = util.tempdir()
    ll_path = temp.relpath("temp.ll")
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code


gemv = intrin_gemv(64, 64, 512)

if True:
    S_m_c, S_n_c, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)

    x, y = s[O].leaf_iter_vars[0:2]
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(xo, yo, xi, yi)
    f = s[O].fuse(xo, yo)
    s[O].parallel(f)

    O_m, O_n = xi, yi
    O_m_o_i, O_m_i = s[O].split(O_m, factor=64)
    O_m_o_o, O_m_o_i = s[O].split(O_m_o_i, factor=1)
    O_n_o_i, O_n_i = s[O].split(O_n, factor=64)
    O_n_o_o, O_n_o_i = s[O].split(O_n_o_i, factor=1)
    s[O].reorder(O_m_o_o, O_n_o_o, O_m_o_i, O_n_o_i, O_m_i, O_n_i)
    s[S].compute_at(s[O], O_n_o_i)

    s[O].pragma(O_n_o_i, "import_llvm", gemv_impl())
    s[S].tensorize(s[S].leaf_iter_vars[0], gemv)

    s[O].vectorize(O_n_i)

    s[S].set_scope('local')

    s[S].mark_no_bounds_check()

def size_fn(l_inputs):
    lens = l_inputs[0]

    return {
        A: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(
            len(lens), lambda b: (lufw1 if args.layout_unfused else lufw64).get_fn(lens)(b)),
        A2: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                        run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b))),
        O: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                       run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)))
    }

bA = tvm.decl_buffer((f_ext*64, NUM_HEADS*HEAD_SIZE), name="bA")
bW = tvm.decl_buffer((OUT_SIZE, NUM_HEADS*HEAD_SIZE), name="bW")
if args.skip_residual:
    inputs = [[lens], [f_ext, bA, bW, B, O]]
else:
    inputs = [[lens], [f_ext, bA, A2, bW, B, O]]
binds = {A:bA, W:bW}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR, sum_var=f_ext, sum_factor=64),
                                        prep_code_mode='no_prep_code')

# _, A, W, B, O = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]))
#     ctr += this_extent
