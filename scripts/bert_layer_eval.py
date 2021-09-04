import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PYTORCH_RUNNER = SCRIPT_DIR + '/../bert_layer/pytorch/layer.py'
TVM_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/bert_layer_ragged.py'
TVM_MEM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/bert_layer_memory.py'
TVM_LIB_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/gen_libs.sh'
PYTHON = 'python3'

def generate_tvm_libs(dataset, args):
    cmd = [TVM_LIB_RUNNER, dataset, '1' if args.bin_packed else '0', '0', '1' if prep_overhead else '0']
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    print(err)

def run_pytorch(b_size, dataset, n_batch, err_file, args):
    cmd = [PYTHON, PYTORCH_RUNNER, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
           '--max-batches', str(n_batch), '--dataset', dataset]
    if args.mem: cmd += ['--mem']
    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)

    if args.mem: return com.extract_mem(out, 1)[0]
    else: return com.extract_times(out, 1)[0]

def run_tvm(b_size, dataset, n_batch, err_file, args):
    runner = TVM_MEM_RUNNER if args.mem else TVM_EXE_RUNNER

    cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
           '--max-batches', str(n_batch), '--dataset', dataset]
    if args.bin_packed: cmd += ['--bin_packed']
    out, err = '', ''
    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)

    if args.mem: return com.extract_mem(out, 1)[0]
    else: return com.extract_times(out, 1)[0]

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--bin-packed', dest='bin_packed', default=False, action='store_true')
parser.add_argument('--prep-overhead', dest='prep_overhead', default=False, action='store_true')
parser.add_argument('--gen-libs', dest='gen_libs', default=False, action='store_true')
parser.add_argument('--mem', dest='mem', default=False, action='store_true')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

# batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
batch_sizes = [1, 2, 4, 8]
targets = [args.target] if args.target else ['cuda']
datasets = com.cluster_datasets_by_max_len() if args.dataset is None else {com.get_dataset_max_len(args.dataset) : [args.dataset]}

if args.prep_overhead:
    framework_funs = {
        'pytorch': lambda b_sizes, *args: com.batchify(b_sizes, run_pytorch, *args),
    }
else:
    framework_funs = {
        'pytorch': lambda b_sizes, *args: com.batchify(b_sizes, run_pytorch, *args),
        'cora': lambda b_sizes, *args: com.batchify(b_sizes, run_tvm, *args),
    }

out_prefix = 'bert_layer'
if args.prep_overhead: out_prefix += '_prelude'
if args.mem: out_prefix += '_mem'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Target,Dataset,Batch Size'
for framework, func in framework_funs.items(): header += ',' + framework + ' (ms)'
print(header, file = results_out)

for target in targets:
    for _, dataset_list in datasets.items():
        if args.gen_libs: generate_tvm_libs(dataset_list[0], args);
        for dataset in dataset_list:
            exe_times = {}
            for framework, func in framework_funs.items():
                log(args, 'Running %s %s %s' % (target, dataset, batch_sizes))
                exe_times[framework] = func(batch_sizes, dataset, args.max_batches, results_err, args)
                print(exe_times[framework])

            for b_size in batch_sizes:
                out_str = '%s,%s,%d' % (target, dataset, b_size)
                for framework, framework_exe_times in exe_times.items():
                    out_str += ',%g' % framework_exe_times[b_size]
                print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
