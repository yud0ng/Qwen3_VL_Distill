#!/usr/bin/env bash
# DeepSpeed ZeRO-2 双卡冒烟（GPU 0,1）；需已安装 deepspeed、GPU 驱动。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "[1/2] make_dummy_assets"
python scripts/make_dummy_assets.py

echo "[2/2] deepspeed ZeRO-2 smoke (2 GPUs)"
deepspeed --num_gpus=2 scripts/smoke_deepspeed_zero2.py \
  --deepspeed_config configs/deepspeed_zero2_bf16.json

echo "OK"
