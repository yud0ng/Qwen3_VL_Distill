#!/usr/bin/env bash
# eval_checkpoint.sh — Given a checkpoint path, run MME eval and output scores to logs/
#
# Usage:
#   ./eval_checkpoint.sh <checkpoint_path> [step_label]
#
# Examples:
#   ./eval_checkpoint.sh /media/yutao/T91/checkpoints/qwen3-2b-sft-step500
#   ./eval_checkpoint.sh /media/yutao/T91/checkpoints/qwen3-2b-sft-step500 step500
#
# Output:
#   /media/yutao/T91/lmms_eval_logs/ckpt_<step_label>/  — full lmms-eval JSON results
#   /media/yutao/T91/lmms_eval_logs/ckpt_<step_label>/summary.txt — human-readable score summary

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
CKPT_PATH="${1:-}"
if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: No checkpoint path provided."
    echo "Usage: $0 <checkpoint_path> [step_label]"
    exit 1
fi

if [[ ! -d "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_PATH"
    exit 1
fi

# Default step label: basename of checkpoint dir
STEP_LABEL="${2:-$(basename "$CKPT_PATH")}"

# ── Config ────────────────────────────────────────────────────────────────────
CONDA_ENV=/media/yutao/T91/miniconda3/envs/lmms_eval
PYTHON=$CONDA_ENV/bin/python
HF_CACHE=/media/yutao/T91/hf_cache
LOG_DIR=/media/yutao/T91/lmms_eval_logs/ckpt_${STEP_LABEL}
TASKS="mme"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Checkpoint Eval — ${STEP_LABEL}"
echo "  Checkpoint : $CKPT_PATH"
echo "  Output dir : $LOG_DIR"
echo "  Tasks      : $TASKS"
echo "  Started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── Run eval ──────────────────────────────────────────────────────────────────
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=$HF_CACHE \
$PYTHON -m lmms_eval \
    --model qwen3_vl \
    --model_args "pretrained=${CKPT_PATH},load_in_4bit=True,max_pixels=401408,enable_thinking=False" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --output_path "$LOG_DIR"

# ── Parse and summarise results ───────────────────────────────────────────────
# Find the most recently written results JSON in LOG_DIR
RESULT_JSON=$(ls -t "$LOG_DIR"/*/results.json 2>/dev/null | head -1 || \
              ls -t "$LOG_DIR"/*results*.json 2>/dev/null | head -1 || true)

export RESULT_JSON
SUMMARY_FILE="$LOG_DIR/summary.txt"

{
echo "============================================================"
echo "  Eval Summary — ${STEP_LABEL}"
echo "  Checkpoint : $CKPT_PATH"
echo "  Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 2B baseline for reference (established 2026-04-15)
echo ""
echo "  Reference scores (2B baseline, 4-bit, max_pixels=401408):"
echo "    MME Perception : 1395.64"
echo "    MME Cognition  :  515.00"
echo "    MME Total      : 1910.64"
echo ""
echo "  32B teacher scores:"
echo "    MME Perception : 1787.94"
echo "    MME Cognition  :  688.57"
echo "    MME Total      : 2476.51"
echo ""

if [[ -n "$RESULT_JSON" && -f "$RESULT_JSON" ]]; then
    echo "  Results file: $RESULT_JSON"
    echo ""
    # Extract MME scores with python
    $PYTHON - <<'PYEOF'
import json, sys, os

result_file = os.environ.get("RESULT_JSON", "")
if not result_file or not os.path.isfile(result_file):
    print("  (Could not locate results JSON for parsing)")
    sys.exit(0)

with open(result_file) as f:
    data = json.load(f)

results = data.get("results", {})
mme = results.get("mme", {})

perception = mme.get("mme_perception_score,none", mme.get("mme_perception_score", None))
cognition  = mme.get("mme_cognition_score,none",  mme.get("mme_cognition_score", None))

if perception is not None and cognition is not None:
    total = perception + cognition
    baseline_total = 1910.64
    teacher_total  = 2476.51
    gap = teacher_total - baseline_total
    recovery = (total - baseline_total) / gap * 100 if gap > 0 else float("nan")

    print(f"  Current checkpoint scores:")
    print(f"    MME Perception : {perception:.2f}")
    print(f"    MME Cognition  : {cognition:.2f}")
    print(f"    MME Total      : {total:.2f}")
    print(f"")
    print(f"  Recovery% (vs 2B baseline -> 32B teacher gap):")
    print(f"    {recovery:+.1f}%  (positive = improvement over baseline)")
else:
    print("  (MME scores not found in expected keys — raw results below)")
    for k, v in mme.items():
        print(f"    {k}: {v}")
PYEOF
else
    echo "  WARNING: Could not find a results JSON file in $LOG_DIR"
    echo "  Check the lmms-eval output above for errors."
fi

echo ""
echo "============================================================"
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary written to: $SUMMARY_FILE"
