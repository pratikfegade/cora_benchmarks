import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MEM_RUNNER = SCRIPT_DIR + '/../prelude_overheads_breakdown/prelude_overheads'

def execute(b_size, op, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    cmd = [MEM_RUNNER, str(b_size), str(n_batch), com.get_dataset_file(dataset), op]
    out, err = run_cmd(cmd)
    ftime, ttime, ctime = com.extract_times(out, 3)
    fmem, tmem = com.extract_mem(out, 2)
    return (fmem, ftime, tmem, ttime, ctime)

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

b_sizes = [32, 64, 128]
args.target = 'cuda'
datasets = ['race', 'cola']

ops = ['csf_vanilla', 'cora_vanilla', 'csf_opt', 'cora_opt']
out_prefix = 'prelude_overheads_breakdown'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = (
    'Dataset,Batch Size,'
    'csf_vanilla_ftime,csf_vanilla_fmem,csf_vanilla_ttime,csf_vanilla_tmem,csf_vanilla_ctime'
    'cora_vanilla_ftime,cora_vanilla_fmem,cora_vanilla_ttime,cora_vanilla_tmem,cora_vanilla_ctime'
    'csf_opt_ftime,csf_opt_fmem,csf_opt_ttime,csf_opt_tmem,csf_opt_ctime'
    'cora_opt_ftime,cora_opt_fmem,cora_opt_ttime,cora_opt_tmem,cora_opt_ctime'
)
print(header, file = results_out)

for dataset in datasets:
    for b_size in b_sizes:
        out_str = '%s,%d' % (dataset, b_size)
        for op in ops:
            fmem, ftime, tmem, ttime, ctime = execute(b_size, op, dataset, args.max_batches, results_err, args)
            out_str += ',%g,%g,%g,%g,%g' % (ftime, fmem, ttime, tmem, ctime)
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
