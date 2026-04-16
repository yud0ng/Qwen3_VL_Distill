#!/usr/bin/env bash
# C 任务：先拆分 teacher_responses，再依次跑 Variant A（通用）与 Variant A+（空间）。
# 用法（在 kd_pipeline 目录）:
#   bash scripts/run_c_task_variant_a_two_idle.sh
# 图像根目录约定：../datasets/coco/train2014（相对仓库 distill/distill）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

INPUT_JSONL="${INPUT_JSONL:-../teacher_responses.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-data/c_task_splits}"
LOG_DIR="${LOG_DIR:-runs/c_task_logs}"
mkdir -p "$LOG_DIR" "$SPLIT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# 若 image 为其它机器绝对路径，可在运行前二选一：
#   COCO_ROOT=/path/to/train2014  — 按文件名重写到该目录下
#   或 PREFIX_FROM + PREFIX_TO    — 整段字符串替换路径前缀
REWRITTEN="${LOG_DIR}/teacher_responses.rewritten_${TS}.jsonl"
if [[ -n "${COCO_ROOT:-}" ]]; then
  echo "Rewriting image paths with --coco-root $COCO_ROOT -> $REWRITTEN"
  python "$ROOT/scripts/rewrite_teacher_image_paths.py" \
    --input "$INPUT_JSONL" --output "$REWRITTEN" --coco-root "$COCO_ROOT"
  INPUT_JSONL="$REWRITTEN"
elif [[ -n "${PREFIX_FROM:-}" && -n "${PREFIX_TO:-}" ]]; then
  echo "Rewriting paths: ${PREFIX_FROM} -> ${PREFIX_TO}"
  python "$ROOT/scripts/rewrite_teacher_image_paths.py" \
    --input "$INPUT_JSONL" --output "$REWRITTEN" \
    --from-prefix "$PREFIX_FROM" --to-prefix "$PREFIX_TO"
  INPUT_JSONL="$REWRITTEN"
fi

MANIFEST="$LOG_DIR/split_${TS}.log"

echo "[1/3] split by source -> $SPLIT_DIR" | tee "$MANIFEST"
python "$ROOT/scripts/split_teacher_by_source.py" --input "$INPUT_JSONL" --out-dir "$SPLIT_DIR" 2>&1 | tee -a "$MANIFEST"

LOG_A="$LOG_DIR/variant_a_general_${TS}.log"
LOG_AP="$LOG_DIR/variant_a_plus_spatial_${TS}.log"

echo "[2/3] Variant A (general) -> runs/variant_a_full_sft_general" | tee -a "$MANIFEST"
bash "$ROOT/scripts/run_deepspeed_two_idle.sh" \
  --config "$ROOT/configs/variant_a_full_sft_general.yaml" \
  --deepspeed_config "$ROOT/configs/deepspeed_zero2_bf16_c_task.json" \
  2>&1 | tee "$LOG_A"

echo "[3/3] Variant A+ (spatial) -> runs/variant_a_plus_full_sft_spatial" | tee -a "$MANIFEST"
bash "$ROOT/scripts/run_deepspeed_two_idle.sh" \
  --config "$ROOT/configs/variant_a_plus_full_sft_spatial.yaml" \
  --deepspeed_config "$ROOT/configs/deepspeed_zero2_bf16_c_task.json" \
  2>&1 | tee "$LOG_AP"

echo "Done. Logs: $MANIFEST $LOG_A $LOG_AP" | tee -a "$MANIFEST"
