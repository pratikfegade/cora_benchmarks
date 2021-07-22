import run_utils
import argparse
import utils
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
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
NUM_HEADS = 8
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')

def len_uf(name): return Uf(name, (0, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], 32))

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

inputs = [[lens], [A]]
if args.debug_code:
    lowered = tvm.lower(s, inputs, simple_mode = True)
    print(lowered)
    # fadd = tvm.build(s, inputs, args.target)
    # if args.target == 'cuda':
    #     print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
    # else:
    #     print('-----CPU code-----\n' + fadd.get_source())
else:
    fadd, i_bufs = tvm.build(s, inputs, args.target)
    # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
    run_utils.run(fadd, i_bufs, [A], args.batch_size, args.max_batches,
                  args.dataset, args.datadir, args.target, args.debug)
