#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python3 -m --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

