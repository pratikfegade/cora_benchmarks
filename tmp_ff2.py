import math
import os
import utils
import run_utils
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--peel-loops', dest='peel_loops', default=False, action='store_true')
parser.add_argument('--unroll-loops', dest='unroll_loops', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

IN_SIZE = 2048
OUT_SIZE = 512
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 64)

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_uf(name, pad = 1): return Uf(name, "l", (0, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], pad))

luf = len_uf('s1')
# luf = Uf.from_constant('bd', MAX_LEN, "l")
ls =  {
    0: Uf.from_constant('bd', args.batch_size, "l"),
    1: luf,
    2: Uf.from_constant('id', IN_SIZE, "l"),
    3: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((args.batch_size, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((IN_SIZE, OUT_SIZE), name='W')

loop_ufs=[ls[0], ls[1], ls[3]]
width_ufs=[[ls[0], len_uf('s132', 32), ls[3]]]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
O = te.ragged_compute((args.batch_size, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(W[k, ds[od]] * A[ds[bd], ds[s1], k], axis = k, dimensions = [id]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])






O_local, = s.cache_write([O], "local")
b, l, o, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
l = s[O_local].fuse(b, l)
loi, li = s[O_local].split(l, factor=2)

ooi, oi = s[O_local].split(o, factor=2)

koi, ki = s[O_local].split(k, factor=4)
koo, koi = s[O_local].split(koi, factor=2)

s[O_local].reorder(koo, koi, loi, ooi, ki, li, oi)

if not args.debug_code:
    s[O_local].unroll(koi)
    s[O_local].unroll(loi)
    s[O_local].unroll(ooi)
    s[O_local].unroll(ki)
    s[O_local].unroll(li)
    s[O_local].unroll(oi)

O_b, O_l, O_o, O_k = tuple(O.op.axis) + tuple(O.op.reduce_axis)
O_l = s[O].fuse(O_b, O_l)

O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=2)
O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

O_o_o_i, O_o_i = s[O].split(O_o, factor=4)
O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
s[O].reorder(O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)



A_shared = s.cache_read(A, "shared", [O_local])
A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
s[A_shared].compute_at(s[O_local], koo)

W_shared = s.cache_read(W, "shared", [O_local], vanilla=True)
W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
s[W_shared].compute_at(s[O_local], koo)

s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.y"))
s[O].bind(O_o_o_o_o, te.thread_axis("blockIdx.x"))
O_l_o_o_i_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
s[O_local].compute_at(s[O], O_l_o_i_o_o_i_fused)

A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
# s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=4)
# s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=32)
s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))


s.fuse_tensor_dimensions(O_local, 0, 1)
s.fuse_tensor_dimensions(A_shared, 0, 1)



# tile = 128
# rtile = 8
# nt = tile // rtile
# ks = utils.next_power_of_2(IN_SIZE / (6144 // tile))
# thread_x = lambda: tvm.thread_axis((0, nt), "threadIdx.x")
# thread_y = lambda: tvm.thread_axis((0, nt), "threadIdx.y")
# block_x = lambda: tvm.thread_axis("blockIdx.x")
# block_y = lambda: tvm.thread_axis("blockIdx.y")

# Ol = s.cache_write(O, 'local')
# As = s.cache_read(A, "shared", [Ol])
# # Ws = s.cache_read(W, "shared", [Ol], vanilla=True)
# # Al = s.cache_read(As, "local", [Ol])
# # Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)

# b, l, o = s[O].leaf_iter_vars[0:3]
# y = s[O].fuse(b, l)
# yo, yi = s[O].split(y, factor = tile)
# x = o
# xo, xi = s[O].split(x, factor = tile)
# s[O].bind(yo, block_y())
# s[O].bind(xo, block_x())

# yio, yii = s[O].split(yi, factor = nt)
# xio, xii = s[O].split(xi, factor = nt)
# s[O].bind(yii, thread_y())
# s[O].bind(xii, thread_x())
# s[O].reorder(yii, xii, yio, xio)
# s[O].bind(yio, te.thread_axis("vthread"), no_unroll_vthread = True)
# s[O].bind(xio, te.thread_axis("vthread"), no_unroll_vthread = True)
# s[Ol].compute_at(s[O], xio)

# b, l, o, k = s[Ol].leaf_iter_vars
# s[Ol].reorder(k, b, l, o)
# ko, ki = s[Ol].split(k, nparts = ks)
# s[As].compute_at(s[Ol], ko)
# # s[Ws].compute_at(s[Ol], ko)
# # s[Al].compute_at(s[Ol], ki)
# # s[Wl].compute_at(s[Ol], ki)

# # f = s[Ws].fuse(*s[Ws].leaf_iter_vars)
# # xo, xi = s[Ws].split(f, factor = nt * nt)
# # xio, xii = s[Ws].split(xi, factor = nt)
# # s[Ws].bind(xio, thread_y())
# # s[Ws].bind(xii, thread_x())

# b, l, i = s[As].leaf_iter_vars
# f = s[As].fuse(b, l)
# s[As].reorder(i, f)
# f = s[As].fuse(i, f)
# # xo, xi = s[As].split(f, factor = nt * nt)
# # xio, xii = s[As].split(xi, factor = nt)
# # s[As].bind(xio, thread_y())
# # s[As].bind(xii, thread_x())
# # s.fuse_tensor_dimensions(As, 0, 1)
# # s.reorder_tensor_dimensions(As, 0, 1)

# s.fuse_tensor_dimensions(A, 0, 1)
# s.fuse_tensor_dimensions(O, 0, 1)




suffix = ""
gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0] + suffix
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))


bA = tvm.decl_buffer([args.batch_size, MAX_LEN, IN_SIZE], name = "bA")
inputs = [[lens], [bA, W, O]]
if args.debug_code:
    lowered = tvm.lower(s, inputs, args.target, simple_mode = True, binds = {A: bA})
    print(lowered)
    # fadd, _ = tvm.build(s, inputs, args.target, binds = {A: bA})
    # if args.target == 'cuda':
        # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
    # else:
        # print('-----CPU code-----\n' + fadd.get_source())
else:
    fadd, i_bufs = tvm.build(s, inputs, args.target, binds = {A: bA})
    # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
    run_utils.run(fadd, i_bufs, inputs[1], args.batch_size, args.max_batches,
                  args.dataset, args.datadir, args.target, args.debug)
