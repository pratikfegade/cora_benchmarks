import numpy as np
import tvm

def random_lengths(batch_size, max_seq_len):
    min_seq_len = int(0.1 * max_seq_len)
    return np.random.randint(min_seq_len, max_seq_len, batch_size, "int32")

def np_arrays(shape_list):
    return [np.zeros(shape, "float32") for shape in shape_list]

def get_ctx(target):
    ctx = None
    if target.startswith('llvm') or target == 'c':
        ctx = tvm.cpu(0)
    elif target.startswith('cuda'):
        ctx = tvm.gpu(0)
    else:
        raise ValueError('Unsupported target %s' % target)
    return ctx

def execute(target, built, inputs, ctx, debug = False):
    if debug:
        if target == 'c':
            built['default_function'](*inputs)
        else:
            built(*inputs)
        ctx.sync()
        return -100000000
    else:
        if target == 'c':
            built['default_function'](*inputs)
            return -100000000
            evaluator = built.time_evaluator('default_function', ctx, 1, repeat=10)
        else:
            evaluator = built.time_evaluator(built.entry_name, ctx, number=1, repeat=1)
        eval_result = evaluator(*inputs)
        return eval_result.mean

def run(built, inputs, target):
    ctx = get_ctx(target)
    inputs = [tvm.nd.array(i, ctx) for i in inputs]
    print([i.dtype for i in inputs])
    execute(target, built, inputs, ctx)
