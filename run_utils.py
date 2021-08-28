import argparse
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

def get_cmd_parser(no_options=False):
    parser = argparse.ArgumentParser()
    if not no_options:
        parser.add_argument('--target', nargs='?', default='llvm')
        parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
        parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
        parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
        parser.add_argument('--debug', dest='debug', default=False, action='store_true')
        parser.add_argument('--debug-code', dest='debug_code', default=False, action='store_true')
        parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
        parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
        parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
        parser.add_argument('--dataset', nargs='?', default='random')
        parser.add_argument('--datadir', nargs='?', default='random')
    return parser

def prefix_sum(extent, fn):
    s = 0
    for i in range(extent):
        s += fn(i)
    return s

def get_dataset_max_len(dataset):
    if dataset.startswith("random"):
        _, _, max_seq_len = dataset.split("_")
        return int(max_seq_len)
    else:
        return dataset_max_lens[dataset]

def random_lengths(batch_size, avg_seq_len, max_seq_len):
    min_seq_len = 2 * avg_seq_len - max_seq_len
    return np.random.randint(min_seq_len, max_seq_len + 1, batch_size, "int32")

def int_shape(expr_shape, rmap):
    shape = []
    for i in expr_shape:
        if i in rmap:
            i = rmap[i]
        shape.append(int(i))
    return shape

def get_shape(t, rmap):
    if isinstance(t, tvm.te.Tensor):
        return int_shape(t.shape, rmap)
    elif isinstance(t, tvm.tir.Buffer):
        return int_shape(t.shape.dense_shape(), rmap)
    else:
        print(t)
        assert False

def create_ragged_array(dense_shape, flat_size, dtype, ctx):
    src_np_array = np.random.normal(size=(flat_size,)).astype(dtype)
    tvm_array = tvm.nd.ragged_empty(dense_shape, flat_size, dtype=dtype, ctx=ctx)
    tvm_array.copyfrom(src_np_array, is_dst_ragged=True)
    return tvm_array

def create_numpy_array(t, dtype, rmap={}, lw_args=None):
    shape = get_shape(t, rmap)
    # return np.zeros(shape, dtype)
    return np.full(shape, 0.1, dtype)
    # return np.random.normal(size=shape, loc=0.5, scale=4).astype(dtype)

def create_tvm_array(t, dtype, ctx, rmap={}, lw_args=None):
    shape = get_shape(t, rmap)

    assert (lw_args is not None)
    if t in lw_args:
        flat_size = lw_args[t]
        print(t, flat_size, shape)
        return create_ragged_array(shape, flat_size, dtype, ctx)

    # return np.zeros(shape, dtype)
    return tvm.nd.array(np.full(shape, 0.1, dtype), ctx)
    # return np.random.normal(size=shape, loc=0.5, scale=4).astype(dtype)

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
    host_i_inputs, dev_i_inputs = [], []
    if len(i_inputs_tensors) == 2:
        host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), cpu_ctx) for i in i_inputs_tensors[0]]
        dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors[1]]
    t_inputs = [tvm.nd.array(create_numpy_array(i, "float32"), ctx) for i in t_inputs_tensors]
    if debug: num_batches = 1

    # print([np.mean(t.asnumpy()) for t in t_inputs])

    if dataset.startswith("random"):
        _, avg_seq_len, max_seq_len = dataset.split("_")
        batches = [random_lengths(batch_size, int(avg_seq_len), int(max_seq_len)) for i in range(num_batches)]
    else:
        batches = read_and_chunk_lengths(batch_size, num_batches, datadir + "/" + dataset_files[dataset])

    time = 0
    for batch in batches:
        sorted(batch)
        l_inputs = [tvm.nd.array(batch, cpu_ctx)]
        inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
        time += execute(target, built, inputs, ctx, debug)

    print("RESULT", time / len(batches))
    return [t.asnumpy() for t in t_inputs], batches

def run_fused(built, total_len_var, t_inputs_tensors, batch_size, num_batches, dataset, datadir, target, debug):
    ctx = get_ctx(target)
    cpu_ctx = get_ctx("llvm")

    if debug: num_batches = 1

    if dataset.startswith("random"):
        _, avg_seq_len, max_seq_len = dataset.split("_")
        batches = [random_lengths(batch_size, int(avg_seq_len), int(max_seq_len)) for i in range(num_batches)]
    else:
        batches = read_and_chunk_lengths(batch_size, num_batches, datadir + "/" + dataset_files[dataset])

    time = 0
    for batch in batches:
        sorted(batch)
        total_length = int(sum(batch))
        total_length = ((total_length + 63) // 64) * 64
        rmap = {total_len_var: total_length}
        t_inputs = [tvm.nd.array(create_numpy_array(i, "float32", rmap), ctx) for i in t_inputs_tensors]
        inputs = [total_length] + t_inputs
        time += execute(target, built, inputs, ctx, debug)

    print("RESULT", time / len(batches))

def is_ragged(t):
    if isinstance(t, tvm.te.Tensor):
        if t.op.output_layout(0) is None:
            return False
        else:
            return t.op.output_layout(0).is_ragged()
    elif isinstance(t, tvm.tir.Buffer):
        return t.shape.is_ragged()
    else:
        print(t)
        assert False

def run2(built, i_inputs_tensors, t_inputs_tensors, lw_args, args):
    ctx = get_ctx(args.target)
    cpu_ctx = get_ctx("llvm")
    host_i_inputs, dev_i_inputs = [], []
    if len(i_inputs_tensors) == 2:
        host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), cpu_ctx) for i in i_inputs_tensors[0]]
        dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors[1]]

    num_batches = args.max_batches
    if args.debug: num_batches = 1

    if args.dataset.startswith("random"):
        _, avg_seq_len, max_seq_len = args.dataset.split("_")
        batches = [random_lengths(args.batch_size, int(avg_seq_len), int(max_seq_len)) for i in range(num_batches)]
    else:
        batches = read_and_chunk_lengths(args.batch_size, args.num_batches, args.datadir + "/" + dataset_files[args.dataset])

    time = 0
    for batch in batches:
        sorted(batch)
        t_inputs = [create_tvm_array(i, "float32", ctx, lw_args=lw_args([batch])) for i in t_inputs_tensors]
        l_inputs = [tvm.nd.array(batch, cpu_ctx)]
        inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
        time += execute(args.target, built, inputs, ctx, args.debug)

    print("RESULT", time / len(batches))
    for i in range(len(t_inputs)):
        size_fn = lw_args([batch])
        target = None
        if t_inputs_tensors[i] in size_fn:
            target = np.empty(size_fn[t_inputs_tensors[i]], dtype='float32')
        t_inputs[i] = t_inputs[i].asnumpy(target=target, is_src_ragged=is_ragged(t_inputs_tensors[i]))
    return t_inputs, batches
