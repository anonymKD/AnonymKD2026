#!/bin/bash
ENTRY=$1
CONFIG=$2
MODEL=$3
PY_ARGS=${@:4}

set -x

export PYTHONPATH=$(pwd):$PYTHONPATH
torchrun --nproc_per_node=1 ${ENTRY} -c ${CONFIG} --model ${MODEL} ${PY_ARGS}

