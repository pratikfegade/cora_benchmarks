import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PYTORCH_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/pytorch/layer_cpu_per_op.py'
TF_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tf/layer_per_op.py'
TVM_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/masked_mha_blasized.py'
TVM_LIB_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/gen_libs_cpu.sh'
FTRANS_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/faster_transformer/run_encoder_sample.sh'
PYTHON = 'python3'

def generate_tvm_libs(dataset, args):
    cmd = [TVM_LIB_RUNNER, dataset, '0', '0', '1' if args.prep_overhead else '0']
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    print(out, err)

def run_pytorch(b_size, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    num_layers = 25
    cmd = [PYTHON, PYTORCH_EXE_RUNNER, '--target', args.target, '--dataset', dataset, '--batch-size', str(b_size),
           '--max-batches', str(n_batch)]

    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)
    return com.extract_time_ops(out)

def run_tf(b_size, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    num_layers = 25
    cmd = [PYTHON, TF_EXE_RUNNER, '--dataset', dataset, '--batch-size', str(b_size), '--max-batches', str(n_batch)]
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    print(out)
    print(err)
    if err: print(err, file = err_file)
    return com.extract_time_ops(out)

def run_cora(b_size, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    cmd = [PYTHON, TVM_EXE_RUNNER, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
           '--max-batches', str(n_batch), '--dataset', dataset, '--per-op', '--masked-mha']
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)
    return com.extract_time_ops(out)

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='cuda')
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--prep-overhead', dest='prep_overhead', default=False, action='store_true')
parser.add_argument('--gen-libs', dest='gen_libs', default=False, action='store_true')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

data_points = [('race', 128), ('wiki_128', 32)]
# data_points = [('race', 128)]
# data_points = [('wiki_128', 32)]
target = 'cpu'

out_prefix = 'per_op_times'
if args.prep_overhead: out_prefix += '_prelude'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Dataset,Batch Size,Framework,Op,Time'
print(header, file = results_out)

for dataset, b_size in data_points:
    if args.gen_libs: generate_tvm_libs(dataset, args);

    # cora_times = run_cora(b_size, dataset, args.max_batches, results_err, args)
    # for op, time in cora_times.items():
    #     out_str = '%s,%d,%s,%s,%g' % (dataset, b_size, 'cora', op, time)
    #     print(out_str, file = results_out)
    # results_out.flush()

    # pt_times = run_pytorch(b_size, dataset, args.max_batches, results_err, args)
    # for op, time in pt_times.items():
        # out_str = '%s,%d,%s,%s,%g' % (dataset, b_size, 'pytorch', op, time)
        # print(out_str, file = results_out)
    # results_out.flush()

    tf_times = run_tf(b_size, dataset, args.max_batches, results_err, args)
    for op, time in tf_times.items():
        out_str = '%s,%d,%s,%s,%g' % (dataset, b_size, 'tensorflow', op, time)
        print(out_str, file = results_out)
    results_out.flush()

if not args.stdout:
    results_out.close()
    results_err.close()
