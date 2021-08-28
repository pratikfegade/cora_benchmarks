import numpy as np
import os
import run_utils
import argparse
import utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw

parser = run_utils.get_cmd_parser()
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
OUT_SIZE = 512

eps = 0.001
beta = 0.2
gamma = 0.5

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))

lufw = len_ufw('s', 1)
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: lufw.get_uf(),
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

def compute_body(ds):

    mean1 = Am1[ds[bd], ds[s1]]/OUT_SIZE
    mean2 = Am2[ds[bd], ds[s1]]/OUT_SIZE
    std = tvm.sqrt(mean2 - mean1*mean1 + eps)
    normed = (A[ds[bd], ds[s1], ds[od]] - mean1) / std
    return beta + gamma * normed

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=None if args.dense_storage else [loop_ufs]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs, compute_body, name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
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

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A1: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        A2: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        O: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        A: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        Am1: run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        Am2: run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
    }

with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=True):
    inputs = [[lens], [A1, A2, O, A, Am1, Am2]]
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
        outs, batches = run_utils.run2(fadd, i_bufs, inputs[1], size_fn, args)
        # A1, A2, O = outs
        # for i in range(args.batch_size):
            # this_a1 = A1[i]
            # this_a2 = A2[i]
            # added = this_a1 + this_a2
            # mean = np.mean(added, axis=1, keepdims=True)
            # std = np.std(added, axis=1, keepdims=True)
            # res = beta + gamma * ((added - mean) / (std + eps))
            # length = batches[0][i]
            # print(length, np.mean(res), np.std(res), np.mean(O[i,0:length,:]), np.std(O[i,0:length,:]))
