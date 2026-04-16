#!/usr/bin/env bash
# 自动选 2 张空闲 GPU（利用率<=5%、空闲显存优先），用 DeepSpeed ZeRO-2 跑 train_distill.py。
# 勿使用 --num_gpus（会覆盖物理卡）；本脚本用 --include localhost:i,j 绑定真实卡号。
# 用法（在 kd_pipeline 目录）:
#   bash scripts/run_deepspeed_two_idle.sh --config configs/train_teacher_responses.yaml \\
#     --deepspeed_config configs/deepspeed_zero2_bf16.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t PICK < <(bash "$ROOT/scripts/pick_two_idle_gpus.sh")
if [[ ${#PICK[@]} -lt 2 ]]; then
  echo "pick_two_idle_gpus failed" >&2
  exit 1
fi
CUDA_PAIR="${PICK[0]}"
INCLUDE="${PICK[1]}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_PAIR} (same pair); deepspeed --include ${INCLUDE}" >&2

exec deepspeed --include "$INCLUDE" "$ROOT/scripts/train_distill.py" "$@"
