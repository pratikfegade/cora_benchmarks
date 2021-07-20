import numpy as np
import tvm

def random_lengths(batch_size, max_seq_len):
    min_seq_len = int(0.1 * max_seq_len)
    return np.random.randint(min_seq_len, max_seq_len, batch_size, "int32")

def np_arrays(shape_list):
    return [np.zeros(shape, "float32") for shape in shape_list]

def int_shape(expr_shape):
    return [int(i) for i in expr_shape]

def get_shape(t):
    if isinstance(t, tvm.te.Tensor):
        return int_shape(t.shape)
    elif isinstance(t, tvm.tir.Buffer):
        return int_shape(t.shape.dense_shape())
    else:
        assert False

def create_numpy_array(t, dtype):
    return np.zeros(get_shape(t), dtype)

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

def run(built, l_inputs, i_inputs_tensors, t_inputs_tensors, target):
    ctx = get_ctx(target)
    cpu_ctx = get_ctx("llvm")
    l_inputs = [tvm.nd.array(i, cpu_ctx) for i in l_inputs]
    print([(t, get_shape(t)) for t in i_inputs_tensors])
    i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors]
    t_inputs = [tvm.nd.array(create_numpy_array(i, "float32"), ctx) for i in t_inputs_tensors]
    inputs = t_inputs + l_inputs + i_inputs
    execute(target, built, inputs, ctx)
