#!/usr/bin/env bash
# 端到端冒烟：占位数据 → 教师 top-k → 学生训练（需 GPU + 已安装依赖）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/3] make_dummy_assets"
python scripts/make_dummy_assets.py

echo "[2/3] teacher top-k (truncated)"
rm -f "${ROOT}/data/teacher_topk.jsonl"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/gen_teacher_topk.py \
  --config configs/gen_teacher.yaml --max_samples 1

echo "[3/3] student distill"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_distill.py \
  --config configs/variant_C.yaml --max_steps 2 --save_every 1 --num_epochs 2 --out_dir runs/e2e_smoke

echo "OK -> runs/e2e_smoke/adapter_final (and checkpoint-* if save_every>0)"
