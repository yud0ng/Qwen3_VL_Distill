#!/usr/bin/env bash
# Variant C（CE + top-k KL）：单卡 LoRA、tmux、timeout 5h、max_steps 限制（不全量扫完 jsonl）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CFG="${CFG:-configs/variant_c_teacher_all.yaml}"
DS_CFG="${DS_CFG:-configs/deepspeed_zero2_bf16_c_task.json}"
MAX_STEPS="${MAX_STEPS:-2400}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-$((2 * LORA_R))}"
WALL_SEC="${WALL_SEC:-18000}"
SESSION="${SESSION:-variant_c_lora_5h}"

for f in "$CFG" "data/teacher_all.local.jsonl" "data/teacher_topk_from_teacher_all.jsonl" "$DS_CFG"; do
  if [[ ! -f "$f" ]]; then
    echo "missing file: $f" >&2
    exit 1
  fi
done

OUT_DIR="${OUT_DIR:-runs/variant_c_teacher_all_lora_r${LORA_R}_fast5h_single}"
mkdir -p runs/c_task_logs
LOG="runs/c_task_logs/variant_c_fast5h_r${LORA_R}_$(date +%Y%m%d_%H%M%S).log"

INNER="$ROOT/scripts/_variant_c_fast5h_inner.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd \"$ROOT\""
  echo "export PYTHONPATH=\"$ROOT\${PYTHONPATH:+:\$PYTHONPATH}\""
  echo "echo \"[variant_c] start \$(date -Is) wall=${WALL_SEC}s max_steps=${MAX_STEPS} out=${OUT_DIR}\" | tee -a \"$LOG\""
  echo "timeout -k 120 ${WALL_SEC} bash scripts/run_deepspeed_one_idle.sh \\"
  echo "  --config \"$CFG\" \\"
  echo "  --deepspeed_config \"$DS_CFG\" \\"
  echo "  --max_steps \"$MAX_STEPS\" \\"
  echo "  --lora_r \"$LORA_R\" \\"
  echo "  --lora_alpha \"$LORA_ALPHA\" \\"
  echo "  --out_dir \"$OUT_DIR\" \\"
  echo "  2>&1 | tee -a \"$LOG\""
  echo "echo \"[variant_c] end exit=\$? \$(date -Is)\" | tee -a \"$LOG\""
} > "$INNER"
chmod +x "$INNER"

if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
  tmux new-session -d -s "$SESSION" "$INNER"
  echo "tmux: $SESSION  |  log: $LOG  |  out: $OUT_DIR"
else
  exec "$INNER"
fi
