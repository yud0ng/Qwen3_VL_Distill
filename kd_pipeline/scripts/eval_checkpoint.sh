#!/usr/bin/env bash
# 供评测同学（B）对接：给定 train_distill 产出的 checkpoint 目录，写 for_eval 并调用 lmms-eval 示例。
# 用法: bash scripts/eval_checkpoint.sh <checkpoint_dir> [lmms_output_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:?usage: eval_checkpoint.sh <checkpoint_dir> [output_dir]}"
OUT="${2:-${ROOT}/logs/eval_$(basename "$CKPT")}"

mkdir -p "$OUT"
echo "checkpoint_dir=$(realpath "$CKPT")" > "$CKPT/for_eval.txt"
echo "lmms_output=$OUT" >> "$CKPT/for_eval.txt"
echo "cmd: bash $ROOT/scripts/eval_lmms_eval_example.sh $(realpath "$CKPT") $OUT" >> "$CKPT/for_eval.txt"

bash "$ROOT/scripts/eval_lmms_eval_example.sh" "$(realpath "$CKPT")" "$OUT"
