import os
import argparse
import run_utils
import utils
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
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 64)
NUM_HEADS = 8
HEAD_SIZE = 64
red_tile = 16

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def ub_pad(name, pad): return Uf(name, 'l', (pad, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], pad))
def lb(name): return Uf(name, 'l', (0, MAX_LEN), [bd], lambda b: utils.floormult(lens[b], 64))

bd_uf = Uf.from_constant('bd', args.batch_size, 'l')
md_uf = Uf.from_constant('md', NUM_HEADS, 'l')
lb_uf = lb('lb')
ub_uf = ub_pad('ub', 32)
hd_uf = Uf.from_constant('hd', HEAD_SIZE, 'l')
s1_uf = ub_pad('k', red_tile)

loop_ufs=[bd_uf, ub_uf, md_uf, s1_uf]
width_ufs=loop_ufs
A = te.ragged_placeholder((args.batch_size, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s2, md, s1], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[bd_uf, s1_uf, md_uf, hd_uf]
width_ufs=loop_ufs
V = te.ragged_placeholder((args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s1, md, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[bd_uf, ub_uf, md_uf, hd_uf]
width_ufs=[loop_ufs]
O = te.ragged_compute((args.batch_size, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s2, md, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[s2], ds[md], rds['k']] *
                                              V(ds[bd], rds['k'], ds[md], ds[hd]),
                                              axis=rds['k'], dimensions=[s1]),
                      name = 'O', reduce_axis_ufs = [('k', s1_uf)],
                      width_uf_lists=width_ufs)

output_layout = O.op.output_layout(0)
s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis('threadIdx.x')
thread_y = lambda: tvm.thread_axis('threadIdx.y')
block_x = lambda: tvm.thread_axis('blockIdx.x')
block_y = lambda: tvm.thread_axis('blockIdx.y')
ntx = 16
nty = 16

def schedule_op(O, tile, suffix):
    Ol = s.cache_write(O, 'local')

    As = s.cache_read(A, 'shared', [Ol], suffix=suffix)
    Vs = s.cache_read(V, 'shared', [Ol], suffix=suffix)

    Al = s.cache_read(As, 'local', [Ol], suffix=suffix)
    Vl = s.cache_read(Vs, 'local', [Ol], suffix=suffix)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = tile)

    s[O].reorder(b, xo, h, xi, y)
    f = s[O].fuse(b, xo)
    s[O].bind(f, block_x())
    s[O].bind(h, block_y())

    xio, xii = s[O].split(xi, factor = nty)
    yo, yi = s[O].split(y, factor = ntx)
    s[O].bind(xii, thread_y())
    s[O].bind(yi, thread_x())
    s[O].bind(xio, tvm.thread_axis('vthread', name='vth1'))
    s[O].bind(yo, tvm.thread_axis('vthread', name='vth2'))
    s[Ol].compute_at(s[O], yi)

    b, x, h, y, k = s[Ol].leaf_iter_vars
    s[Ol].reorder(b, h, k, x, y)
    ko, ki = s[Ol].split(k, factor = red_tile)
    s[As].compute_at(s[Ol], ko)
    s[Vs].compute_at(s[Ol], ko)
    s[Al].compute_at(s[Ol], ki)
    s[Vl].compute_at(s[Ol], ki)
    s[Ol].peel(ko)

    _, x, h, y = s[As].leaf_iter_vars
    s[As].reorder(h, x, y)
    f = s[As].fuse(x, y)
    fo, fi = s[As].split(f, factor = ntx * nty * 4)
    fio, fii = s[As].split(fi, factor = ntx * 4)
    fiio, fiii = s[As].split(fii, factor = 4)
    s[As].bind(fio, thread_y())
    s[As].bind(fiio, thread_x())
    if not args.debug_functions: s[As].vectorize(fiii)

    _, x, h, y = s[Vs].leaf_iter_vars
    s[Vs].reorder(h, x, y)
    f = s[Vs].fuse(x, y)
    fo, fi = s[Vs].split(f, factor = ntx * nty * 4)
    fio, fii = s[Vs].split(fi, factor = ntx * 4)
    fiio, fiii = s[Vs].split(fii, factor = 4)
    s[Vs].bind(fio, thread_y())
    s[Vs].bind(fiio, thread_x())
    if not args.debug_functions: s[Vs].vectorize(fiii)

O1, O2 = s.split_for_bin_packing(O, {O.op.axis[1]: lb_uf})
schedule_op(O1, 64, '1')
schedule_op(O2, 32, '2')

s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0])])

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

bO = tvm.tir.decl_buffer(output_layout, name="bO")
binds = {O1:bO, O2:bO}
inputs = [[lens], [V, A, bO]]
with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=not args.debug_functions):
    if args.debug_code:
        # lowered = tvm.lower(s, inputs, args.target, simple_mode=True, binds=binds)
        # print(lowered)
        fadd, _ = tvm.build(s, inputs, args.target, binds=binds)
        if args.target == 'cuda':
            print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        else:
            print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd, i_bufs = tvm.build(s, inputs, args.target, binds=binds)
        # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
        run_utils.run(fadd, i_bufs, [V, A, O], args.batch_size, args.max_batches,
                      args.dataset, args.datadir, args.target, args.debug)