import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TVM_VGEMM_RUNNER = SCRIPT_DIR + '/../vbatch_gemm/tvm/vbatch_gemm.py'
TVM_QKT_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/qkt.py'
VGEMM_DATA_FILE = SCRIPT_DIR + '/../vbatch_gemm/data.txt'
PYTHON = 'python3'

def run_cbt(op, target, b_sizes, n_batch, data_file_path, err_file, args):
    exe_times = {}
    for b_size in b_sizes:
        cmd = [CBT_RUNNER, str(b_size), str(n_batch), data_file_path, str(100), str(1), op]
        # print(' '.join(cmd))
        out, err = run_cmd(cmd)
        # print(out, err)
        if err: print(err, file = err_file)
        exe_times[b_size] = com.extract_times(out, 1)[0]
    return exe_times

def run_qkt(target, b_sizes, n_batch, hoist_loads):
    cmd = ([PYTHON, TVM_QKT_RUNNER, '--target', target, '--batch-sizes'] +
           [str(i) for i in b_sizes] +
           ['--max-batches', str(n_batch), '--dataset', 'race'])
    if not hoist_loads: cmd += ['--no-hoist-loads']
    print(' '.join(cmd))
    out, err = com.run_cmd(cmd)
    if err: print(err, file = results_err)
    return com.extract_time_batches(out)

def run_vgemm(target, b_sizes, n_batch, hoist_loads):
    cmd = ([PYTHON, TVM_VGEMM_RUNNER, '--target', target, '--batch-sizes'] +
           [str(i) for i in b_sizes] +
           ['--max-batches', str(n_batch), '--data-file', VGEMM_DATA_FILE])
    if not hoist_loads: cmd += ['--no-hoist-loads']
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

# ops = ['vgemm', 'qkt']
ops = ['qkt']
b_sizes = {'vgemm': [4, 32, 256], 'qkt': [32, 64, 128]}
op_runners = {'vgemm': run_vgemm, 'qkt': run_qkt}

results_out, results_err = get_out_files(args, 'load_hoist', 'a' if args.append else 'w')
header = 'Op,Target,Batch Size,Vanilla Time,+LoadHoist'
print(header, file = results_out)

for op in ops:
    runner = op_runners[op]

    log(args, 'Running vanilla %s' % (op))
    vanilla_times = runner(args.target, b_sizes[op], args.max_batches, False)
    print(vanilla_times)

    log(args, 'Running hoisted %s' % (op))
    hoisted_times = runner(args.target, b_sizes[op], args.max_batches, True)
    print(hoisted_times)

    for b_size in b_sizes[op]:
        out_str = '%s,%s,%d,%g,%g' % (op, args.target, b_size, vanilla_times[b_size], hoisted_times[b_size])
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
