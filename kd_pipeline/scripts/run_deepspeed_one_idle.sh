#!/usr/bin/env bash
# 自动选 1 张空闲 GPU（利用率<=5%、空闲显存优先），用 DeepSpeed ZeRO-2 跑 train_distill.py。
# 用法（在 kd_pipeline 目录）:
#   bash scripts/run_deepspeed_one_idle.sh --config configs/variant_a_clean_train_lora.yaml \
#     --deepspeed_config configs/deepspeed_zero2_bf16_c_task.json --lora_r 16 --lora_alpha 32
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t PICK < <(bash "$ROOT/scripts/pick_one_idle_gpu.sh")
if [[ ${#PICK[@]} -lt 2 ]]; then
  echo "pick_one_idle_gpu failed" >&2
  exit 1
fi
CUDA_ONE="${PICK[0]}"
INCLUDE="${PICK[1]}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_ONE} (single); deepspeed --include ${INCLUDE}" >&2

exec deepspeed --include "$INCLUDE" "$ROOT/scripts/train_distill.py" "$@"
