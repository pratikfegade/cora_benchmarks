#!/bin/bash

DS=$1
BP=$2
MS=$3
OP=$4
YES="1"
set -x
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm -f $SCRIPT_DIR/genlibs/*
EXTRA_ARGS=$EXTRA_ARGS" --skip-residual"

python3 ${SCRIPT_DIR}/pre_linear_blasized.py --dataset $DS --gen-lib $EXTRA_ARGS
python3 ${SCRIPT_DIR}/add_padding_cpu.py --dataset $DS --gen-lib $EXTRA_ARGS
python3 ${SCRIPT_DIR}/qkt_cpu.py --dataset $DS --gen-lib $EXTRA_ARGS --sched 1
python3 ${SCRIPT_DIR}/qkt_cpu.py --dataset $DS --gen-lib $EXTRA_ARGS --sched 2
python3 ${SCRIPT_DIR}/softmax_cpu.py --dataset $DS --gen-lib $EXTRA_ARGS
python3 ${SCRIPT_DIR}/attn_v_cpu.py --dataset $DS --gen-lib $EXTRA_ARGS
python3 ${SCRIPT_DIR}/remove_padding_cpu.py --dataset $DS --gen-lib $EXTRA_ARGS
python3 ${SCRIPT_DIR}/post_linear_blasized.py --dataset $DS --gen-lib $EXTRA_ARGS
