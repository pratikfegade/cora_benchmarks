#!/bin/bash

python3 pre_linear2.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
python3 qkt2.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
python3 softmax2.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
python3 attn_v2.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
python3 post_linear2.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
python3 norm_add.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
python3 ff1.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
python3 ff2.py --target cuda --batch-size 32 --max-batches 10 --dataset race --datadir ../data_analysis/ --gen-lib
