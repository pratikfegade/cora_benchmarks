import os
import time
import tvm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils


def load_module(op_name):
    return tvm.runtime.module.load_module(run_utils.MODULE_DIR + '/' + op_name + '.so')

def load_ibuf_info(op_name):
    bufs = [[], []]
    with open(run_utils.MODULE_DIR + '/' + op_name + '_bufs.txt') as topo_file:
        for line in topo_file:
            arr = line.strip().split('|')
            arr[1] = 'lambda bs: (' + arr[1] + ')'
            data = (eval(arr[1]), arr[2])
            if arr[0] == 'h':
                bufs[0].append(data)
            else:
                bufs[1].append(data)
    return bufs

def create_ibufs(ibuf_info, batch_size, cpu_ctx, dev_ctx):
    def get_or_call(i):
        if isinstance(i, int): return i
        else:
            assert callable(i)
            return i(batch_size)
    host_bufs = [tvm.nd.array(run_utils.create_numpy_array(get_or_call(i[0]), i[1]), cpu_ctx) for i in ibuf_info[0]]
    dev_bufs = [tvm.nd.array(run_utils.create_numpy_array(get_or_call(i[0]), i[1]), dev_ctx) for i in ibuf_info[1]]
    return host_bufs, dev_bufs

class Op:
    def __init__(self, name, module_name, batch_size, tensor_inputs, cpu_ctx, dev_ctx):
        self.name = name
        self.module_name = module_name
        self.tensor_inputs = tensor_inputs
        self.module = load_module(module_name)
        ibuf_info = load_ibuf_info(module_name)
        self.host_ibufs, self.dev_ibufs = create_ibufs(ibuf_info, batch_size, cpu_ctx, dev_ctx)
        self.batch_size = batch_size

    def execute(self, l_inputs):
        inputs = [self.batch_size] + self.tensor_inputs + l_inputs + self.host_ibufs + self.dev_ibufs
        # print('Exe', self.name, len(inputs))
        # sys.stdout.flush()
        self.module(*inputs)
