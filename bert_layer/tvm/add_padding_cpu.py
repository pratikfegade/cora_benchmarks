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
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('qkv', QKV_NUM, "l"),
    1: Uf.from_constant('bd', BATCH_SIZE, "l"),
    2: Uf.from_constant('md', NUM_HEADS, "l"),
    3: lufw64.get_uf(),
    5: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[0], ls[1], ls[3], ls[2], ls[5]]
width_ufs=[ls[0], ls[1], lufw1.get_uf(), ls[2], ls[5]]
A = te.ragged_placeholder((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od], loop_ufs,
                          name = 'A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[2], ls[5]]
width_ufs=[[ls[0], ls[1], lufw64.get_uf(), ls[2], ls[5]]]
O = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od], loop_ufs,
                      lambda ds: A[ds[qkv], ds[bd], ds[s1], ds[md], ds[od]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if True:
    q, b, l, n, h = s[O].leaf_iter_vars
    f = s[O].fuse(b, l, padding=8)
    fo, fi = s[O].split(f, nparts=8)
    f = s[O].fuse(q, fo)
    s[O].parallel(f)
else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    if args.layout_unfused: out_fn = lufw1.get_fn(lens)
    else: out_fn = lufw64.get_fn(lens)
    return {}

inputs = [[lens], [BS_VAR, A, O]]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

q_size = 0
for length in batches[0]:
    q_size += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE

# ctr = 0
# _, QKV, W, B, O  = out
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * NUM_HEADS * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]), np.mean(O[ctr+q_size:ctr+q_size + this_extent]),
#           np.mean(O[ctr+2*q_size:ctr+2*q_size + this_extent]))
#     ctr += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE
