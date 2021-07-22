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
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
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

BATCH_SIZE = args.batch_size
NUM_HEADS = 8
IN_SIZE = 256
OUT_SIZE = 64
QKV_NUM = 3
TILE=64
RTILE=4
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
qkv = Dim('qkv')
md = Dim('md')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: lens[b])

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE),
    1: len_uf('s1'),
    2: Uf.from_constant('md', NUM_HEADS),
    3: Uf.from_constant('qkv', QKV_NUM),
    4: Uf.from_constant('id', IN_SIZE),
    5: Uf.from_constant('od', OUT_SIZE),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3], ls[4]]
width_ufs=loop_ufs
QKV = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, QKV_NUM, IN_SIZE), [bd, s1, qkv, md, id], loop_ufs,
                            name='QKV', width_ufs=width_ufs)

loop_ufs=[ls[5], ls[4]]
W = te.placeholder((IN_SIZE, OUT_SIZE), name='W')

loop_ufs=[ls[0], ls[1], ls[2], ls[3], ls[5]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, QKV_NUM, OUT_SIZE), [bd, s1, qkv, md, od], loop_ufs,
                      lambda ds: tvm.sum(W[k, ds[od]] * QKV[ds[bd], ds[s1], ds[qkv], ds[md], k], axis = k),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Ol = s.cache_write(O, "local")
Ws = s.cache_read(W, "shared", [Ol], vanilla=True)
QKVs = s.cache_read(QKV, "shared", [Ol])

Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)
QKVl = s.cache_read(QKVs, "local", [Ol])

b, l, h, q, o = s[O].leaf_iter_vars

f = s[O].fuse(b, l)
s[O].bind(f, block_x());

s[O].bind(h, thread_y())
oo, oi = s[O].split(o, factor = 32)
s[O].bind(oi, thread_x())
s[O].bind(oo, tvm.thread_axis("vthread"))
s[O].reorder(oi, oo, q)
s[Ol].compute_at(s[O], oo)

b, l, h, q, o, k = s[Ol].leaf_iter_vars
s[Ol].reorder(k, q, o)
ko, ki = s[Ol].split(k, nparts = 4)
s[Ws].compute_at(s[Ol], ko)
s[QKVs].compute_at(s[Ol], ko)
s[Wl].compute_at(s[Ol], ki)
s[QKVl].compute_at(s[Ol], ki)



f = s[Ws].fuse(*s[Ws].leaf_iter_vars)
xo, xi = s[Ws].split(f, factor = 256)
xio, xii = s[Ws].split(xi, factor = 32)
s[Ws].bind(xio, thread_y())
s[Ws].bind(xii, thread_x())

s[QKVs].bind(s[QKVs].leaf_iter_vars[2], thread_y())
xo, xi = s[QKVs].split(s[QKVs].leaf_iter_vars[4], factor = 32)
s[QKVs].bind(xi, thread_x())

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

inputs = [[lens], [QKV, W]]
if args.debug_code:
    lowered = tvm.lower(s, inputs, simple_mode = True)
    print(lowered)
    # fadd, _ = tvm.build(s, inputs, args.target)
    # if args.target == 'cuda':
        # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
    # else:
        # print('-----CPU code-----\n' + fadd.get_source())
else:
    fadd, i_bufs = tvm.build(s, inputs, args.target)
    # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
    run_utils.run(fadd, i_bufs, [QKV, W], args.batch_size, args.max_batches,
                  args.dataset, args.datadir, args.target, args.debug)
