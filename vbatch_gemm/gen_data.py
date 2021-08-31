import random
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--out', dest='out', default=None, type=str)
args = parser.parse_args()

multiples = list(range(4, 12))
factor = 128

with open(args.out, 'w') as outfile:
    for i in range(args.batch_size):
        M = factor * random.choice(multiples)
        N = factor * random.choice(multiples)
        K = factor * random.choice(multiples)
        print(M, N, K, file = outfile)
        # print(M, N, K)
