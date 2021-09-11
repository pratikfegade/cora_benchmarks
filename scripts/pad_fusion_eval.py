import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PYTORCH_RUNNER = SCRIPT_DIR + '/../bert_layer/pytorch/layer.py'
TVM_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/masked_mha.py'
TVM_MEM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/bert_layer_memory.py'
TVM_LIB_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/gen_libs.sh'
FTRANS_RUNNER = SCRIPT_DIR + '/../bert_layer/faster_transformer/run_encoder_sample.sh'
PYTHON = 'python3'

DIR_PREFIX = SCRIPT_DIR + '/../bert_layer/tvm/'
ops = {
    'pre_linear':       (False, '--unused', DIR_PREFIX + 'pre_linear.py'),
    'add_pad64':        (True,  '--add',    DIR_PREFIX + 'padding_64to1.py'),
    'qkt':              (False, '--unused', DIR_PREFIX + 'qkt.py'),
    'change_pad_to_32': (True,  '--remove', DIR_PREFIX + 'padding_32to64.py'),
    'softmax':          (False, '--unused', DIR_PREFIX + 'softmax.py'),
    'change_pad_to_64': (True,  '--add',    DIR_PREFIX + 'padding_32to64.py'),
    'attn_v':           (False, '--unused', DIR_PREFIX + 'attn_v.py'),
    'rem_pad64':        (True,  '--remove', DIR_PREFIX + 'padding_64to1.py'),
    'post_linear':      (False, '--unused', DIR_PREFIX + 'post_linear.py'),
}

def run_cora_op(op_name, b_sizes, dataset, n_batch, err_file, args, pad_fusion):
    op_data = ops[op_name]
    pad_op = op_data[0]
    runner = op_data[2]
    cmd = ([PYTHON, runner, '--target', com.get_tvm_target(target), '--batch-sizes'] +
           [str(i) for i in b_sizes] +
           ['--max-batches', str(n_batch), '--dataset', dataset])

    if pad_fusion:
        if pad_op: return None
    else:
        if pad_op: cmd += [op_data[1]]
        else: cmd += ['--layout-unfused']

    print(' '.join(cmd))
    out, err = '', ''
    # out, err = run_cmd(cmd)
    if err: print(err, file = err_file)
    return com.extract_time_batches(out)

def run_cora(b_sizes, pad_fusion, dataset, n_batch, err_file, args):
    ret = {}
    for b_size in b_sizes: ret[b_size] = 0.0

    for op_name in ops:
        this_times = run_cora_op(op_name, b_sizes, dataset, n_batch, err_file, args, pad_fusion)
        if this_times:
            for b_size in b_sizes: ret[b_size] += this_times[b_size]

    return ret

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')

args = parser.parse_args()

b_sizes = [2, 8, 32, 128]
targets = [args.target] if args.target else ['cuda']
datasets = com.get_all_datasets() if args.dataset is None else [args.dataset]

out_prefix = 'pad_fusion'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Target,Dataset,Batch Size,Unfused,Fused'
print(header, file = results_out)

target = targets[0]
for dataset in datasets:
    exe_times = {}
    unfused_times = run_cora(b_sizes, False, dataset, args.max_batches, results_err, args)
    fused_times = run_cora(b_sizes, True, dataset, args.max_batches, results_err, args)

    for b_size in b_sizes:
        out_str = '%s,%s,%d,%g,%g' % (target, dataset, b_size, unfused_times[b_size], fused_times[b_size])
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
