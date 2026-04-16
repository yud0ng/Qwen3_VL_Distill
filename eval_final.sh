#!/usr/bin/env bash
# eval_final.sh — Full 3-benchmark eval (MME + MMStar + CV-Bench) for a checkpoint
#
# Usage:
#   ./eval_final.sh <checkpoint_path> [run_label]
#
# Examples:
#   ./eval_final.sh /media/yutao/T91/checkpoints/variant_A_r16 variant_A_r16
#   ./eval_final.sh /media/yutao/T91/checkpoints/variant_B_r64 variant_B_r64
#
# Output:
#   <script_dir>/logs/final_<run_label>/  — full lmms-eval JSON per benchmark
#   <script_dir>/logs/final_<run_label>/summary.txt — combined scores + Recovery%

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Args ──────────────────────────────────────────────────────────────────────
CKPT_PATH="${1:-}"
if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: No checkpoint path provided."
    echo "Usage: $0 <checkpoint_path> [run_label]"
    exit 1
fi

if [[ ! -d "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_PATH"
    exit 1
fi

RUN_LABEL="${2:-$(basename "$CKPT_PATH")}"

# ── Config ────────────────────────────────────────────────────────────────────
CONDA_ENV=/media/yutao/T91/miniconda3/envs/lmms_eval
PYTHON=$CONDA_ENV/bin/python
HF_CACHE=/media/yutao/T91/hf_cache
LOG_DIR="${SCRIPT_DIR}/logs/final_${RUN_LABEL}"

mkdir -p "$LOG_DIR"

MODEL_ARGS="pretrained=${CKPT_PATH},load_in_4bit=True,max_pixels=401408,enable_thinking=False"

run_task() {
    local task="$1"
    echo ""
    echo "------------------------------------------------------------"
    echo "  Running: $task"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=$HF_CACHE \
    $PYTHON -m lmms_eval \
        --model qwen3_vl \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size 1 \
        --output_path "$LOG_DIR"
    echo "  Finished $task: $(date '+%Y-%m-%d %H:%M:%S')"
}

echo "============================================================"
echo "  Final Eval — ${RUN_LABEL}"
echo "  Checkpoint : $CKPT_PATH"
echo "  Output dir : $LOG_DIR"
echo "  Tasks      : mmstar, mme, cv_bench"
echo "  Started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Run each benchmark separately so partial results are always saved
run_task mmstar
run_task mme
run_task cv_bench

echo ""
echo "============================================================"
echo "  All tasks complete. Generating summary..."
echo "============================================================"

# ── Parse results ─────────────────────────────────────────────────────────────
# Baseline and teacher reference scores
export BASELINE_MME_PERCEP=1395.64
export BASELINE_MME_COGN=515.00
export BASELINE_MME_TOTAL=1910.64
export BASELINE_MMSTAR=0.4756
export BASELINE_CVBENCH=0.7309

export TEACHER_MME_PERCEP=1787.94
export TEACHER_MME_COGN=688.57
export TEACHER_MME_TOTAL=2476.51
export TEACHER_MMSTAR=0.7074
export TEACHER_CVBENCH=0.8817

export LOG_DIR
export RUN_LABEL
export CKPT_PATH

SUMMARY_FILE="$LOG_DIR/summary.txt"

$PYTHON - <<'PYEOF' | tee "$SUMMARY_FILE"
import json, os, glob

log_dir    = os.environ["LOG_DIR"]
run_label  = os.environ["RUN_LABEL"]
ckpt_path  = os.environ["CKPT_PATH"]

# Reference scores
baseline = {
    "mme_total":   float(os.environ["BASELINE_MME_TOTAL"]),
    "mme_percep":  float(os.environ["BASELINE_MME_PERCEP"]),
    "mme_cogn":    float(os.environ["BASELINE_MME_COGN"]),
    "mmstar":      float(os.environ["BASELINE_MMSTAR"]),
    "cvbench":     float(os.environ["BASELINE_CVBENCH"]),
}
teacher = {
    "mme_total":   float(os.environ["TEACHER_MME_TOTAL"]),
    "mme_percep":  float(os.environ["TEACHER_MME_PERCEP"]),
    "mme_cogn":    float(os.environ["TEACHER_MME_COGN"]),
    "mmstar":      float(os.environ["TEACHER_MMSTAR"]),
    "cvbench":     float(os.environ["TEACHER_CVBENCH"]),
}

def recovery(val, key):
    gap = teacher[key] - baseline[key]
    return (val - baseline[key]) / gap * 100 if gap != 0 else float("nan")

def find_results(task_name):
    """Find the results.json for a given task under log_dir."""
    # lmms-eval writes: log_dir/<model_name>/<timestamp>_results.json
    patterns = [
        f"{log_dir}/*/*results*.json",
        f"{log_dir}/*results*.json",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    # Filter by task presence inside the JSON
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            if task_name in data.get("results", {}):
                return data["results"][task_name]
        except Exception:
            continue
    return None

mme     = find_results("mme")
mmstar  = find_results("mmstar")
cvbench = find_results("cv_bench")

def get(d, *keys):
    if d is None:
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None

mme_percep  = get(mme,     "mme_perception_score,none", "mme_perception_score")
mme_cogn    = get(mme,     "mme_cognition_score,none",  "mme_cognition_score")
mmstar_avg  = get(mmstar,  "average,none", "average")
cvbench_avg = get(cvbench, "average,none", "overall,none", "average")

mme_total = (mme_percep + mme_cogn) if (mme_percep is not None and mme_cogn is not None) else None

def fmt(val, decimals=4):
    return f"{val:.{decimals}f}" if val is not None else "N/A"

def fmt_rec(val, key):
    if val is None:
        return "N/A"
    return f"{recovery(val, key):+.1f}%"

SEP = "=" * 62
sep = "-" * 62

print(SEP)
print(f"  Final Eval Summary — {run_label}")
print(f"  Checkpoint : {ckpt_path}")
print(SEP)
print()
print(f"  {'Benchmark':<20} {'2B Baseline':>12} {'32B Teacher':>12} {'This Ckpt':>12} {'Recovery%':>10}")
print(f"  {sep}")

rows = [
    ("MME Total",      baseline["mme_total"],  teacher["mme_total"],  mme_total,   "mme_total",  2),
    ("MME Perception", baseline["mme_percep"], teacher["mme_percep"], mme_percep,  "mme_percep", 2),
    ("MME Cognition",  baseline["mme_cogn"],   teacher["mme_cogn"],   mme_cogn,    "mme_cogn",   2),
    ("MMStar Avg",     baseline["mmstar"],      teacher["mmstar"],     mmstar_avg,  "mmstar",      4),
    ("CV-Bench Avg",   baseline["cvbench"],     teacher["cvbench"],    cvbench_avg, "cvbench",     4),
]

for name, base_val, teach_val, ckpt_val, key, dec in rows:
    print(f"  {name:<20} {base_val:>12.{dec}f} {teach_val:>12.{dec}f} {fmt(ckpt_val, dec):>12} {fmt_rec(ckpt_val, key):>10}")

print()
print(f"  Recovery% = (checkpoint - 2B baseline) / (32B teacher - 2B baseline) x 100")
print(f"  Positive = improvement over baseline; 100% = matched teacher")
print()

missing = []
if mme_total  is None: missing.append("MME")
if mmstar_avg is None: missing.append("MMStar")
if cvbench_avg is None: missing.append("CV-Bench")
if missing:
    print(f"  WARNING: Could not parse results for: {', '.join(missing)}")
    print(f"  Check JSON files in: {log_dir}")
    print()

print(SEP)
PYEOF

echo ""
echo "Summary written to: $SUMMARY_FILE"

# ── Append to results tracker ─────────────────────────────────────────────────
TRACKER="${SCRIPT_DIR}/logs/results_tracker.csv"
if [[ ! -f "$TRACKER" ]]; then
    echo "run_label,checkpoint,mme_total,mme_perception,mme_cognition,mmstar_avg,cvbench_avg,timestamp" > "$TRACKER"
fi

$PYTHON - <<'PYEOF' >> "$TRACKER"
import json, os, glob, datetime

log_dir   = os.environ["LOG_DIR"]
run_label = os.environ["RUN_LABEL"]
ckpt_path = os.environ["CKPT_PATH"]

def find_results(task_name):
    candidates = glob.glob(f"{log_dir}/*/*results*.json") + glob.glob(f"{log_dir}/*results*.json")
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            if task_name in data.get("results", {}):
                return data["results"][task_name]
        except Exception:
            continue
    return None

def get(d, *keys):
    if d is None: return ""
    for k in keys:
        if k in d: return round(d[k], 4)
    return ""

mme     = find_results("mme")
mmstar  = find_results("mmstar")
cvbench = find_results("cv_bench")

mme_p = get(mme,     "mme_perception_score,none", "mme_perception_score")
mme_c = get(mme,     "mme_cognition_score,none",  "mme_cognition_score")
mme_t = round(mme_p + mme_c, 4) if isinstance(mme_p, float) and isinstance(mme_c, float) else ""
ms    = get(mmstar,  "average,none", "average")
cv    = get(cvbench, "average,none", "overall,none", "average")
ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"{run_label},{ckpt_path},{mme_t},{mme_p},{mme_c},{ms},{cv},{ts}")
PYEOF

echo "Results appended to tracker: $TRACKER"
