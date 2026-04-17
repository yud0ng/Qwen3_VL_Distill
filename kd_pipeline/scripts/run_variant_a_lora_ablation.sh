#!/usr/bin/env bash
# CP-5 LoRA 消融：Variant A + clean_train.jsonl，比对 r=16 与 r=64。
# 默认单卡（single）以提升吞吐稳定性；可切 multi 保留双卡流程。
#
# 用法（在 kd_pipeline 目录）:
#   bash scripts/run_variant_a_lora_ablation.sh
#   MODE=multi bash scripts/run_variant_a_lora_ablation.sh
#   RANKS="16 64" bash scripts/run_variant_a_lora_ablation.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-single}" # single | multi
RANKS="${RANKS:-16 64}"
BASE_CFG="${BASE_CFG:-configs/variant_a_teacher_lora.yaml}"
DS_CFG="${DS_CFG:-configs/deepspeed_zero2_bf16_c_task.json}"
TRAIN_JSONL="${TRAIN_JSONL:-data/clean_train.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-variant_a_lora}"
MAX_STEPS="${MAX_STEPS:-0}" # 0=full epoch; >0 for fast ablation

if [[ ! -f "$BASE_CFG" ]]; then
  echo "base config not found: $BASE_CFG" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "train_jsonl not found: ${TRAIN_JSONL} (请先准备 data/clean_train.jsonl，或设置 TRAIN_JSONL=...)" >&2
  exit 2
fi

case "$MODE" in
  single) RUNNER="bash $ROOT/scripts/run_deepspeed_one_idle.sh" ;;
  multi) RUNNER="bash $ROOT/scripts/run_deepspeed_two_idle.sh" ;;
  *)
    echo "unknown MODE=$MODE (use single|multi)" >&2
    exit 1
    ;;
esac

echo "[CP-5] Variant A LoRA ablation start"
echo "MODE=$MODE"
echo "RANKS=$RANKS"
echo "BASE_CFG=$BASE_CFG"
echo "DS_CFG=$DS_CFG"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "MAX_STEPS=$MAX_STEPS"

for r in $RANKS; do
  alpha=$((2 * r))
  out="runs/${OUT_PREFIX}_r${r}_${MODE}"
  log="runs/c_task_logs/variant_a_lora_r${r}_${MODE}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$(dirname "$log")"
  echo "[run] r=${r} alpha=${alpha} out=${out}"
  echo "cmd: $RUNNER --config $BASE_CFG --deepspeed_config $DS_CFG --train_jsonl $TRAIN_JSONL --max_steps $MAX_STEPS --lora_r $r --lora_alpha $alpha --out_dir $out" | tee "$log"
  $RUNNER \
    --config "$BASE_CFG" \
    --deepspeed_config "$DS_CFG" \
    --train_jsonl "$TRAIN_JSONL" \
    --max_steps "$MAX_STEPS" \
    --lora_r "$r" \
    --lora_alpha "$alpha" \
    --out_dir "$out" 2>&1 | tee -a "$log"
  echo "[done] r=${r} -> ${out}/adapter_final" | tee -a "$log"
done

echo "[CP-5] all runs done."
