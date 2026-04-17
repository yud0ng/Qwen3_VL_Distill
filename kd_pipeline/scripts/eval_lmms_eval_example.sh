#!/usr/bin/env bash
# 与《技术方案》7.2 / 5.1 对齐的 lmms-eval 调用示例（需自行安装 lmms-eval v0.7+）。
# 将 pretrained 指向 HF id 或本地 checkpoint / adapter 合并后的目录。
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-VL-2B-Instruct}"
OUT="${2:-logs/lmms_eval_run}"
# 设 LOG_SAMPLES=1 开启样本级 JSONL 产出（分层 Recovery% / 误差分析必需）
LOG_SAMPLES_FLAG=""
if [[ "${LOG_SAMPLES:-0}" == "1" ]]; then
  LOG_SAMPLES_FLAG="--log_samples"
fi

echo "Model: $MODEL"
echo "Output: $OUT"
echo "LOG_SAMPLES: ${LOG_SAMPLES:-0}"
echo "若使用 LoRA，请先合并："
echo "  python scripts/export_merged_model.py --adapter_dir runs/.../adapter_final --out_dir exports/merged"
echo "再将 pretrained 指向 exports/merged"

python -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=${MODEL}" \
  --tasks mmstar,mme,cv_bench \
  --batch_size 1 \
  ${LOG_SAMPLES_FLAG} \
  --output_path "${OUT}"
