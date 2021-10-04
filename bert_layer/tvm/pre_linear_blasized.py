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
NUM_HEADS = 8
IN_SIZE = 512
OUT_SIZE = 64
QKV_NUM = 3
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
qkv = Dim('qkv')
md = Dim('md')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')
mdod = Dim('mdod')
fd = Dim('fd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('qkv', QKV_NUM, "l"),
    1: Uf.from_constant('bd', BATCH_SIZE, "l"),
    2: Uf.from_constant('md', NUM_HEADS, "l"),
    3: lufw1.get_uf(),
    4: Uf.from_constant('id', IN_SIZE, "l"),
    5: Uf.from_constant('od', OUT_SIZE, "l"),
    6: Uf.from_constant('mo', NUM_HEADS * OUT_SIZE, "l"),
    7: Uf.from_constant('ml', MAX_LEN, "l"),
}

loop_ufs=[ls[1], ls[3], ls[4]]
width_ufs=[ls[1], ls[3], ls[4]]
QKV = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                            name='QKV', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[6], ls[4]]
width_ufs=[ls[0], ls[6], ls[4]]
W = te.ragged_placeholder((QKV_NUM, NUM_HEADS * OUT_SIZE, IN_SIZE), [qkv, mdod, id], loop_ufs,
                          name='W', width_ufs=width_ufs)

B = te.placeholder((QKV_NUM, NUM_HEADS * OUT_SIZE), name='B')

loop_ufs=[ls[0], ls[1], ls[3], ls[6]]
width_ufs=[[ls[0], ls[1], ls[7], ls[6]]]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
S = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS * OUT_SIZE), [qkv, bd, s1, mdod], loop_ufs,
                      lambda ds: tvm.sum(QKV[ds[bd], ds[s1], k] * W[ds[qkv], ds[mdod], k],
                                         axis = k, dimensions = [id]),
                      name = 'S', width_uf_lists=width_ufs)

width_ufs=None
O = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS * OUT_SIZE), [qkv, bd, s1, mdod], loop_ufs,
                      lambda ds: S[ds[qkv], ds[bd], ds[s1], ds[mdod]] + B[ds[qkv], ds[mdod]],
                      name = 'O', width_uf_lists=width_ufs)

f_ext = te.var('sum')
rmap = te.fuse_ragged_axis([QKV, W, B], O, bd, s1, fd, f_ext * 64)
QKV = rmap[QKV.op].output(0)
W = rmap[W.op].output(0)
B = rmap[B.op].output(0)
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
                                        0.0))
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
    q, x, y = s[O].leaf_iter_vars
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(q, xo, yo, xi, yi)
    f = s[O].fuse(xo, yo)
    f = s[O].fuse(f, q)
    s[O].parallel(f)
    s[O].pragma(f, "import_llvm", gemv_impl())

    s[S].compute_at(s[O], f)
    s[S].tensorize(s[S].leaf_iter_vars[0], gemv)


    s[O].vectorize(yi)
    s[S].set_scope('local')
    s[S].mark_no_bounds_check()
else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    if args.layout_unfused: out_fn = lufw1.get_fn(lens)
    else: out_fn = lufw64.get_fn(lens)

    return {
        QKV: IN_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)),
        O: QKV_NUM * NUM_HEADS * OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                                             run_utils.prefix_sum(len(lens), lambda b: out_fn(b)))
    }

bA = tvm.decl_buffer((f_ext*64, IN_SIZE), name="bA")
bW = tvm.decl_buffer((QKV_NUM, NUM_HEADS*OUT_SIZE, IN_SIZE), name="bW")
bO = tvm.decl_buffer((QKV_NUM, f_ext*64, NUM_HEADS*OUT_SIZE), name="bO")
binds={QKV: bA, W:bW, O: bO}
inputs = [[lens], [f_ext, bA, bW, B, bO]]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR, sum_var=f_ext, sum_factor=64),
                                        prep_code_mode='no_prep_code')

q_size = 0
for length in batches[0]:
    q_size += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE

ctr = 0
_, QKV, W, B, O  = out
for length in batches[0]:
    print(length, np.mean(O[0, ctr:ctr+length, :]), np.mean(O[1, ctr:ctr+length, :]), np.mean(O[2, ctr:ctr+length, :]))
    ctr += length
