#!/bin/bash

DS=$1
BS=$2
MB=$3
BP=$4

YES="1"

python3 pre_linear.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
python3 softmax.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
python3 post_linear.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
python3 norm_add.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
python3 ff1.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
python3 ff2.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib

if [ $BP == $YES ]; then
    python3 qkt_bin_packed.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
    python3 attn_v_bin_packed.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
else
    python3 qkt.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
    python3 attn_v.py --target cuda --batch-size $BS --max-batches $MB --dataset $DS --gen-lib
fi
