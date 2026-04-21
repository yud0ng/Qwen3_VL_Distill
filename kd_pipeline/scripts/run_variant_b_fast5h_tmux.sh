#!/usr/bin/env bash
# 变体 B：tmux + timeout 5h。默认数据为本地化路径的 clean_train（避免 /ocean/... 全跳过）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CFG="${CFG:-configs/variant_b_teacher_lora_fast5h.yaml}"
DS_CFG="${DS_CFG:-configs/deepspeed_zero2_bf16_c_task.json}"
TRAIN_JSONL="${TRAIN_JSONL:-data/clean_train_teacher_cot_20260417_164544.local.jsonl}"
MAX_STEPS="${MAX_STEPS:-2400}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-$((2 * LORA_R))}"
WALL_SEC="${WALL_SEC:-18000}"
SESSION="${SESSION:-variant_b_lora_5h}"

if [[ ! -f "$CFG" ]] || [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "missing: CFG=$CFG TRAIN_JSONL=$TRAIN_JSONL" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-runs/variant_b_teacher_cot_lora_fast5h_r${LORA_R}_single}"
mkdir -p runs/c_task_logs
LOG="runs/c_task_logs/variant_b_fast5h_r${LORA_R}_$(date +%Y%m%d_%H%M%S).log"

INNER="$ROOT/scripts/_variant_b_fast5h_inner.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd \"$ROOT\""
  echo "export PYTHONPATH=\"$ROOT\${PYTHONPATH:+:\$PYTHONPATH}\""
  echo "echo \"[variant_b] start \$(date -Is) wall=${WALL_SEC}s max_steps=${MAX_STEPS} out=${OUT_DIR}\" | tee -a \"$LOG\""
  echo "timeout -k 120 ${WALL_SEC} bash scripts/run_deepspeed_one_idle.sh \\"
  echo "  --config \"$CFG\" \\"
  echo "  --deepspeed_config \"$DS_CFG\" \\"
  echo "  --train_jsonl \"$TRAIN_JSONL\" \\"
  echo "  --max_steps \"$MAX_STEPS\" \\"
  echo "  --lora_r \"$LORA_R\" \\"
  echo "  --lora_alpha \"$LORA_ALPHA\" \\"
  echo "  --out_dir \"$OUT_DIR\" \\"
  echo "  2>&1 | tee -a \"$LOG\""
  echo "echo \"[variant_b] end exit=\$? \$(date -Is)\" | tee -a \"$LOG\""
} > "$INNER"
chmod +x "$INNER"

if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
  tmux new-session -d -s "$SESSION" "$INNER"
  echo "tmux: $SESSION  |  log: $LOG  |  out: $OUT_DIR"
else
  exec "$INNER"
fi
