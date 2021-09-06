import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
QKT_TVM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/qkt.py'
QKT_BP_TVM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/qkt_bin_packed.py'
ATTN_TVM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/attn_v.py'
ATTN_BP_TVM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/attn_v_bin_packed.py'
BP_TEST_RUNNER = SCRIPT_DIR + '/../experimental/bp_test2.py'
PYTHON = 'python3'

def get_runner_and_args(op, split_op, hfuse):
    if op == 'qkt':
        if split_op:
            return QKT_BP_TVM_RUNNER, ['--hfuse'] if hfuse else []
        else:
            return QKT_TVM_RUNNER, []
    elif op == 'attn_v':
        if split_op:
            return ATTN_BP_TVM_RUNNER, ['--hfuse'] if hfuse else []
        else:
            return ATTN_TVM_RUNNER, []
    elif op == 'bp_test':
        args = []
        if split_op: args += ['--split']
        if hfuse: args += ['--hfuse']
        return BP_TEST_RUNNER, args
    else:
        raise ValueError('No such op')

def prepare_and_execute(op, target, dataset, b_sizes, n_batch, split_op, hfuse):
    if op == 'bp_test':
        times = {}
        for b_size in b_sizes:
            runner, extra_args = get_runner_and_args(op, split_op, hfuse)
            cmd = ([PYTHON, runner, '--target', target, '--batch-size', str(b_size),
                    '--max-batches', str(n_batch), '--dataset', dataset])
            cmd += extra_args
            print(' '.join(cmd))
            out, err = '', ''
            out, err = com.run_cmd(cmd)
            if err: print(err, file = results_err)
            times[b_size] = com.extract_times(out, 1)[0]
        return times
    else:
        runner, extra_args = get_runner_and_args(op, split_op, hfuse)
        cmd = ([PYTHON, runner, '--target', target, '--batch-sizes'] + [str(i) for i in b_sizes] +
               ['--max-batches', str(n_batch), '--dataset', dataset])
        cmd += extra_args
        out, err = com.run_cmd(cmd)
        if err: print(err, file = results_err)
        return com.extract_time_batches(out)

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='cuda')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

ops = ['qkt', 'attn_v', 'bp_test']
# ops = ['bp_test']
b_sizes = [2, 8, 32, 128, 256]
# datasets = (com.get_all_datasets() if args.dataset is None else [args.dataset]).remove('cola')
datasets = ['mnli', 'random_96_128', 'random_169_192']

results_out, results_err = get_out_files(args, 'bin_packed', 'a' if args.append else 'w')
header = 'Op,Target,Dataset,Batch Size,Vanilla Time,+OpSplit,+HFuse'
print(header, file = results_out)

for op in ops:
    for dataset in datasets:
        log(args, 'Running vanilla %s %s' % (op, dataset))
        vanilla_times = prepare_and_execute(op, args.target, dataset, b_sizes, args.max_batches, False, False)
        log(args, 'Running +split %s %s' % (op, dataset))
        op_split_times = prepare_and_execute(op, args.target, dataset, b_sizes, args.max_batches, True, False)
        log(args, 'Running +fuse %s %s' % (op, dataset))
        op_split_hfuse_times = prepare_and_execute(op, args.target, dataset, b_sizes, args.max_batches, True, True)
        for i in range(len(b_sizes)):
            b_size = b_sizes[i]
            print(op, args.target, dataset, b_size, vanilla_times[b_size],
                  op_split_times[b_size], op_split_hfuse_times[b_size], sep=',')
            out_str = '%s,%s,%s,%d,%g,%g,%g' % (op, args.target, dataset, b_size, vanilla_times[b_size],
                                                op_split_times[b_size], op_split_hfuse_times[b_size])
            print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
