#!/bin/bash

DS=$1
BS=$2
MB=$3

python3 pre_linear2.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
python3 qkt2.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
python3 softmax2.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
python3 attn_v2.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
python3 post_linear2.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
python3 norm_add.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
python3 ff1.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
python3 ff2.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --datadir ../data_analysis/ --gen-lib
