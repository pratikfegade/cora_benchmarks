import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TVM_RUNNER = SCRIPT_DIR + '/../taco/tradd.py'
TACO_RUNNER = SCRIPT_DIR + '/../taco/taco_csr_tradd'
PYTHON = 'python3'

def run_tvm(m_size, err_file, args):
    runner = TVM_RUNNER
    cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--m', str(m_size)]
    out, err = run_cmd(cmd)
    print(' '.join(cmd))
    if err: print(err, file = err_file)
    return com.extract_times(out, 1)[0]

def run_taco(m_size, err_file, args):
    runner = TACO_RUNNER
    cmd = [runner, str(m_size)]
    out, err = run_cmd(cmd)
    print(' '.join(cmd))
    if err: print(err, file = err_file)
    return com.extract_times(out, 1)[0]

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

m_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

targets = [args.target] if args.target else ['cuda']

if args.target == 'cuda':
    framework_funs = {
        'cora': run_tvm,
        'taco': run_taco,
    }

results_out, results_err = get_out_files(args, 'taco_tradd', 'a' if args.append else 'w')
header = 'Target,M'
for framework, func in framework_funs.items(): header += ',' + framework + ' (ms)'
print(header, file = results_out)

for target in targets:
    exe_times = {}
    for m_size in m_sizes:
        for framework, func in framework_funs.items():
            log(args, 'Running %s %d' % (framework, m_size))
            exe_times[framework] = func(m_size, results_err, args)
            print(exe_times[framework])

        out_str = '%s,%d' % (target, m_size)
        for framework, framework_exe_time in exe_times.items():
            out_str += ',%g' % framework_exe_time
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
