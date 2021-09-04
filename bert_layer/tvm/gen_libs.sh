#!/bin/bash

DS=$1
BP=$2
MS=$3
OP=$4
YES="1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 ${SCRIPT_DIR}/pre_linear.py --target cuda --dataset $DS --gen-lib
python3 ${SCRIPT_DIR}/post_linear.py --target cuda --dataset $DS --gen-lib
python3 ${SCRIPT_DIR}/norm_add.py --target cuda --dataset $DS --gen-lib
python3 ${SCRIPT_DIR}/ff1.py --target cuda --dataset $DS --gen-lib
python3 ${SCRIPT_DIR}/ff2.py --target cuda --dataset $DS --gen-lib

EXTRA_ARGS=""
if [ $OP == $YES ]; then
    EXTRA_ARGS="--only-prep-code"
fi

if [ $MS == $YES ]; then
    if [ $BP == $YES ]; then
	echo "Masked bin packed operators not implemented"
	exit 1
    else
	python3 ${SCRIPT_DIR}/masked_qkt.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
	python3 ${SCRIPT_DIR}/masked_attn_v.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
	python3 ${SCRIPT_DIR}/masked_softmax.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    fi
else
    python3 ${SCRIPT_DIR}/softmax.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    if [ $BP == $YES ]; then
	python3 ${SCRIPT_DIR}/qkt_bin_packed.py --hfuse --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
	python3 ${SCRIPT_DIR}/attn_v_bin_packed.py --hfuse --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    else
	python3 ${SCRIPT_DIR}/qkt.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
	python3 ${SCRIPT_DIR}/attn_v.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    fi
fi
