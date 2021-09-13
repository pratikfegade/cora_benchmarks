import os
import time
import tvm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils


def load_module(op_name, variants=None):
    if variants:
        return [tvm.runtime.module.load_module(run_utils.MODULE_DIR + '/' + op_name + str(variant) + '.so') for variant in variants]
    else:
        return [tvm.runtime.module.load_module(run_utils.MODULE_DIR + '/' + op_name + '.so')]

def load_ibuf_info(op_name, variants=None):
    if variants:
        ret = []
        for variant in variants:
            bufs = [[], []]
            with open(run_utils.MODULE_DIR + '/' + op_name + str(variant) + '_bufs.txt') as topo_file:
                for line in topo_file:
                    arr = line.strip().split('|')
                    arr[1] = 'lambda bs: (' + arr[1] + ')'
                    data = (eval(arr[1]), arr[2])
                    if arr[0] == 'h':
                        bufs[0].append(data)
                    else:
                        bufs[1].append(data)
            ret.append(bufs)
        return ret;
    else:
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
        return [bufs]

def create_ibufs(ibuf_infos, batch_size, cpu_ctx, dev_ctx, alloc_op=None):
    def get_or_call(i):
        if isinstance(i, int): return i
        else:
            assert callable(i)
            # print(i(batch_size))
            return i(batch_size)

    ret = []
    for ibuf_info in ibuf_infos:
        host_bufs = [tvm.nd.array(run_utils.create_numpy_array(get_or_call(i[0]), i[1]), cpu_ctx) for i in ibuf_info[0]]
        dev_bufs = [tvm.nd.array(run_utils.create_numpy_array(get_or_call(i[0]), i[1]), dev_ctx) for i in ibuf_info[1]]
        if alloc_op:
            [alloc_op([get_or_call(i[0])], get_or_call(i[0]), i[1], cpu_ctx) for i in ibuf_info[0]]
            [alloc_op([get_or_call(i[0])], get_or_call(i[0]), i[1], cpu_ctx) for i in ibuf_info[1]]
        ret.append((host_bufs, dev_bufs))
    return ret

def mean(l):
    return sum(l) / len(l)

class Op:
    def __init__(self, name, module_name, batch_size, tensor_inputs, cpu_ctx, dev_ctx, alloc_op=None, variants=None):
        self.name = name
        self.module_name = module_name
        self.tensor_inputs = tensor_inputs
        self.modules = load_module(module_name, variants)
        ibuf_info = load_ibuf_info(module_name, variants)
        self.host_ibufs, self.dev_ibufs = list(zip(*create_ibufs(ibuf_info, batch_size, cpu_ctx, dev_ctx, alloc_op=alloc_op)))
        self.batch_size = batch_size

    def execute(self, l_inputs):
        inputs = [self.batch_size] + self.tensor_inputs + l_inputs + self.host_ibufs[0] + self.dev_ibufs[0]
        # print('Exe', self.name, self.batch_size)
        # sys.stdout.flush()
        self.modules[0](*inputs)

    def execute_multiple(self, l_inputs, ctx):
        # print('Exe', self.name, self.batch_size)
        # sys.stdout.flush()
        means = []
        for i in range(len(self.modules)):
            inputs = [self.batch_size] + self.tensor_inputs + l_inputs + self.host_ibufs[i] + self.dev_ibufs[i]
            # evaluator = self.modules[i].time_evaluator(self.modules[i].entry_name, ctx, number=10, repeat=10)
            evaluator = self.modules[i].time_evaluator(self.modules[i].entry_name, ctx, number=2, repeat=10)
            eval_result = evaluator(*inputs)
            means.append(mean(list(eval_result.results)[1:]))
        return min(means)
