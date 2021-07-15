import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def get_fn(pad):
    def fn(i, j):
        pi = i // pad
        pj = j // pad
        return (pi >= pj).astype(int)
    return fn

# print(np.fromfunction(get_fn(1), (32, 32)))
# print(np.fromfunction(get_fn(2), (32, 32)))
print(np.fromfunction(get_fn(8), (32, 32)))
