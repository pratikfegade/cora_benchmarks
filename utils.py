import tvm
import os
from tvm.contrib import nvcc, cublas, cblas

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

def get_tvm_callback_cuda_postproc(args, path, dirname = 'perf', fileprefix = 'dummy_file'):
    def tvm_callback_cuda_postproc(code):
        d = os.path.dirname(path)
        d = d + '/' + dirname + '/'
        if not os.path.exists(d):
            os.mkdir(d)
        write_code(code, d + fileprefix + "_gen.cu")
        if args.manual_code:
            # print("Using manual code")
            code = open(d + fileprefix + "_manual.cu").read()
        return code
    return tvm_callback_cuda_postproc

def get_tvm_callback_cuda_compile(threads, grid_sync = False):
    tvm.target.set_cuda_grid_sync_on(grid_sync)
    tvm.runtime.module.set_cuda_grid_sync_on(grid_sync)
    def tvm_callback_cuda_compile(code):
        print('Using NVCC')
        # options = ["--ptxas-options='-v -warn-lmem-usage -warn-spills' --nvlink-options='-v'", '-rdc=true']
        # options = ["-Xcompiler", "-rdynamic", "-D_FORCE_INLINES", "--use_fast_math", "-lineinfo",
                   # "--ptxas-options='-allow-expensive-optimizations'", "--maxrregcount=" + str((65536 // threads) - 1)]
        # options = ["-Xcompiler", "-rdynamic", "-D_FORCE_INLINES",
                   # "--ptxas-options='-allow-expensive-optimizations'", "--maxrregcount=" + str((65536 // threads) - 1)]
        options = ["-Xcompiler", "-rdynamic", "-D_FORCE_INLINES",
                   "--ptxas-options='-allow-expensive-optimizations'", "--use_fast_math"]
        if nvcc.have_grid_sync(grid_sync): options += ["-rdc=true", "-L /usr/lib/x86_64-linux-gnu"]
        ptx = nvcc.compile_cuda(code, target="ptx", options = options)
        return ptx
    return tvm_callback_cuda_compile

def ceildiv(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return (a + b - 1) // b
    else:
        return tvm.floordiv(a + b - 1, b)

def ceilmult(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return b * ((a + b - 1) // b)
    else:
        return b * tvm.floordiv(a + b - 1, b)

def floormult(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return b * (a // b)
    else:
        return b * tvm.floordiv(a, b)
