import os
import numpy as np
import tvm

dataset_files = {
    "wiki_128": "/old_wikipedia/full_lengths_128.txt",
    "wiki_512": "/old_wikipedia/full_lengths_512.txt",
    "squadv2": "/squadv2/train_lengths.txt",
    "mnli": "/glue_data/MNLI/train_lengths.txt",
    "mrpc": "/glue_data/MRPC/train_lengths.txt",
    "cola": "/glue_data/CoLA/train_lengths.txt",
    "xnli": "/glue_data/XNLI/train_lengths.txt",
    "race": "/race/train_lengths.txt",
}

dataset_max_lens = {
    "wiki_128" : 128,
    "wiki_512" : 512,
    "squadv2" : 384,
    "mnli" : 128,
    "mrpc" : 112,
    "cola" : 48,
    "xnli" : 128,
    "race" : 512,
}

def get_dataset_max_len(dataset):
    return dataset_max_lens[dataset]

def random_lengths(batch_size, max_seq_len):
    # min_seq_len = int(0.1 * max_seq_len)
    max_seq_len = 128
    avg_seq_len = 128
    min_seq_len = 2 * avg_seq_len - max_seq_len
    return np.random.randint(min_seq_len, max_seq_len + 1, batch_size, "int32")

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
    shape = get_shape(t)
    print(t, shape)
    return np.zeros(shape, dtype)

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
            evaluator = built.time_evaluator(built.entry_name, ctx, number=10, repeat=10)
        eval_result = evaluator(*inputs)
        return eval_result.mean * 1000

def chunks(lst, n, m):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, min(m * n, len(lst)), n):
        yield np.array(lst[i:i + n], "int32")

def read_lengths(filename, skip = 0):
    data_lines = [int(line.strip()) for line in open(filename, "r", errors='replace')]
    return data_lines[skip:]

def read_and_chunk_lengths(batch_size, max_batches, lengths_file):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_lines = read_lengths(lengths_file)
    return list(chunks(data_lines, batch_size, max_batches))

def run(built, i_inputs_tensors, t_inputs_tensors, batch_size, num_batches, dataset, datadir, target, debug):
    ctx = get_ctx(target)
    cpu_ctx = get_ctx("llvm")
    host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), cpu_ctx) for i in i_inputs_tensors[0]]
    dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors[1]]
    t_inputs = [tvm.nd.array(create_numpy_array(i, "float32"), ctx) for i in t_inputs_tensors]

    if debug: num_batches = 1

    if dataset == "random":
        batches = [random_lengths(batch_size, 64) for i in range(num_batches)]
    else:
        batches = read_and_chunk_lengths(batch_size, num_batches, datadir + "/" + dataset_files[dataset])

    time = 0
    for batch in batches:
        sorted(batch)
        l_inputs = [tvm.nd.array(batch, cpu_ctx)]
        inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
        time += execute(target, built, inputs, ctx, debug)

    print(time / len(batches))
