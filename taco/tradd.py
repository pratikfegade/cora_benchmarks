import numpy as np
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import utils
import run_utils

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--m', dest='m', default=1024, type=int)
parser.add_argument('--only-prep-code', dest='only_prep_code', default=None, type=str)
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--disable-assert', dest='disable_assert', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
args = parser.parse_args()

M = args.m
md = Dim('md')
nd = Dim('nd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.ceilmult(m + 1, pad))
luf = len_ufw('s2k', 1).get_uf()

ls =  {
    0: Uf.from_constant('md', M, 'l'),
    1: luf,
}

loop_ufs=[ls[0], ls[1]]
width_ufs=loop_ufs
A1 = te.ragged_placeholder((M, M), [md, nd], loop_ufs, name='A1', width_ufs=width_ufs)
A2 = te.ragged_placeholder((M, M), [md, nd], loop_ufs, name='A2', width_ufs=width_ufs)

O = te.ragged_compute((M, M), [md, nd], loop_ufs,
                      lambda ds: A1[ds[md], ds[nd]] * A2[ds[md], ds[nd]],
                      name = 'O', width_uf_lists=[width_ufs])

s = tvm.create_schedule([O.op])

i, j = s[O].op.axis
f = s[O].fuse(i, j, padding=128)
fo, fi = s[O].split(f, factor=128)
s[O].bind(fo, tvm.thread_axis("blockIdx.x"))
s[O].bind(fi, tvm.thread_axis("threadIdx.x"))

s.fuse_tensor_dimensions(A1, 0, 1)
s.fuse_tensor_dimensions(A2, 0, 1)
s.fuse_tensor_dimensions(O, 0, 1)

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

inputs = [[], [A1, A2, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, run_function=run_utils.run_trmm,
                               prep_code_mode='with_prep_code')
