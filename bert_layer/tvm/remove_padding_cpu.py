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
HEAD_SIZE = 64
NUM_HEADS = 8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
md = Dim('md')
hd = Dim('hd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: lufw1.get_uf(),
    2: Uf.from_constant('md', NUM_HEADS, 'l'),
    3: Uf.from_constant('od', HEAD_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[ls[0], lufw64.get_uf(), ls[2], ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s1, md, hd], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[[ls[0], lufw1.get_uf(), ls[2], ls[3]]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s1, md, hd], loop_ufs,
                      lambda ds: A[ds[bd], ds[s1], ds[md], ds[hd]], name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if True:
    b, l, n, h = s[O].leaf_iter_vars
    f = s[O].fuse(b, l, padding=8)
    fo, fi = s[O].split(f, nparts=8)
    s[O].parallel(fo)

def size_fn(l_inputs):
    lens = l_inputs[0]

    return {}
    # return {
    #     A: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(
    #         len(lens), lambda b: (lufw1 if args.layout_unfused else lufw64).get_fn(lens)(b)),
    #     O: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
    #                    run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)))
    # }

inputs = [[lens], [BS_VAR, A, O]]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

# _, A, W, B, O = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]))
#     ctr += this_extent
