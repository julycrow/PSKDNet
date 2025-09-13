#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
#PORT=${PORT:-29542}
# 如果没有设置PORT环境变量，则生成随机端口（范围1024-65535）
if [ -z "$PORT" ]; then
    # 生成随机端口 (1024-65535)
    PORT=$(( RANDOM % 64512 + 1024 ))
    echo "使用随机端口: $PORT"
fi
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
