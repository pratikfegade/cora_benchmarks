import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CUBLAS_RUNNER = SCRIPT_DIR + '/../trmm/cublas/gemm_cublas'
TVM_RUNNER = SCRIPT_DIR + '/../trmm/tvm/trmm.py'
PYTHON = 'python3'

def get_cublas_runner(pad):
    def run_cublas(m_size, err_file, args):
        cmd = [CUBLAS_RUNNER, str(m_size), '1' if pad else'0', str(100), str(1)]
        out, err = run_cmd(cmd)
        if err: print(err, file = err_file)
        return com.extract_times(out, 1)[0]
    return run_cublas

def run_tvm(m_size, err_file, args):
    runner = TVM_RUNNER
    cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--m', str(m_size)]
    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)
    return com.extract_times(out, 1)[0]

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

m_sizes = [128, 256, 512, 1024, 2048, 4096]
targets = [args.target] if args.target else ['cuda']

if args.target == 'cuda':
    framework_funs = {
        'cublas_nopad': get_cublas_runner(False),
        'cublas_pad': get_cublas_runner(True),
        'cora': run_tvm,
    }

results_out, results_err = get_out_files(args, 'trmm', 'a' if args.append else 'w')
header = 'Target,M'
for framework, func in framework_funs.items(): header += ',' + framework + ' (ms)'
print(header, file = results_out)

for target in targets:
    exe_times = {}
    for m_size in m_sizes:
        for framework, func in framework_funs.items():
            log(args, 'Running %s %s %d' % (target, framework, m_size))
            exe_times[framework] = func(m_size, results_err, args)
            # print(exe_times[framework])

        out_str = '%s,%d' % (target, m_size)
        for framework, framework_exe_time in exe_times.items():
            out_str += ',%g' % framework_exe_time
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()