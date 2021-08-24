import os
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

BATCH_SIZE = args.batch_size
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
OUT_SIZE = 512

eps = 0.00001
beta = 2
gamma = 5

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
od = Dim('od')

def len_uf(name): return Uf(name, 'l', (1, MAX_LEN), [bd], lambda b: lens[b])

luf = len_uf('s')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: luf,
    2: Uf.from_constant('od', OUT_SIZE, 'l'),
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
                        lambda ds: tvm.sum(A[ds[bd], ds[s1], k], axis=k, dimensions=[od]),
                        name = 'Am1')

loop_ufs=[ls[0], ls[1]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, OUT_SIZE), name = 'k')
Am2 = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, s1], loop_ufs,
                        lambda ds: tvm.sum(A[ds[bd], ds[s1], k] * A[ds[bd], ds[s1], k], axis=k, dimensions=[od]),
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

ko, ki = s[Am1].split(s[Am1].op.reduce_axis[0], factor = 32)
Am1_rf = s.rfactor(Am1, ki, 1)

ko, ki = s[Am2].split(s[Am2].op.reduce_axis[0], factor = 32)
Am2_rf = s.rfactor(Am2, ki, 1)

ntx = 32

b, l, h = s[O].leaf_iter_vars
f = s[O].fuse(b, l)
s[O].bind(f, block_x)


ho, hi = s[O].split(h, factor = ntx)
s[O].bind(hi, thread_x)

s[Am1_rf].compute_at(s[Am1], s[Am1].leaf_iter_vars[2])
s[Am2_rf].compute_at(s[Am2], s[Am2].leaf_iter_vars[2])
s[Am1].compute_at(s[O], f)
s[Am2].compute_at(s[O], f)
s[A].compute_inline()

s[Am1].bind(s[Am1].op.reduce_axis[0], thread_x)
s[Am2].bind(s[Am2].op.reduce_axis[0], thread_x)

s[Am1_rf].set_scope('local')
s[Am2_rf].set_scope('local')
s[Am1].set_scope('local')
s[Am2].set_scope('local')
s[A].set_scope('local')

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=True):
    inputs = [[lens], [A1, A2, O]]
    if args.debug_code:
        lowered = tvm.lower(s, inputs, args.target, simple_mode = True)
        print(lowered)
        # fadd, _ = tvm.build(s, inputs, args.target)
        # if args.target == 'cuda':
            # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        # else:
            # print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd, i_bufs = tvm.build(s, inputs, args.target)
        # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
        run_utils.run(fadd, i_bufs, inputs[1], args.batch_size, args.max_batches,
                      args.dataset, args.datadir, args.target, args.debug)
