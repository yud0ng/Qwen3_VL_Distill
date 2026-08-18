# Qwen3-VL Spatial Knowledge Distillation

Distilling spatial reasoning from **Qwen3-VL-32B-Instruct** into a
**Qwen3-VL-2B-Instruct** student for memory-constrained and edge deployment.

This repository contains the teacher-data pipeline, four configurable
distillation objectives, LoRA/full-finetuning training code, and evaluation on
MME, MMStar, and CV-Bench.

## Highlights

- Builds a roughly 50k teacher set from spatial COCO questions and general
  LLaVA-Instruct examples.
- Supports response SFT, chain-of-thought supervision, top-k logit KL, and the
  combined objective through YAML configs.
- Includes resumable vLLM teacher generation and a portable Slurm template.
- Keeps the CV-Bench evaluation customization as a local `lmms-eval` overlay;
  the third-party repository is not vendored.
- Provides per-sample CV-Bench logs for reproducible sub-task analysis.

The main experimental result is that **LoRA rank 64 preserves the 2B model's
pretrained capabilities much better than full-parameter finetuning**. Variant
A+ with LoRA reaches 75.40% on CV-Bench, compared with 73.09% for the original
2B model and 88.17% for the 32B teacher.

## Results

| Method | MME total | MMStar (%) | CV-Bench (%) |
| --- | ---: | ---: | ---: |
| Qwen3-VL-32B teacher | 2476.51 | 70.74 | 88.17 |
| Qwen3-VL-2B baseline | 1630.64 | 45.56 | 73.09 |
| Variant A, full FT, mixed data | 1259.29 | 33.24 | 51.00 |
| Variant A+, full FT, spatial-focused | 1570.51 | 32.21 | 44.39 |
| **Variant A+, LoRA r=64, spatial-focused** | **1854.75** | 49.92 | **75.40** |
| Variant B, LoRA r=64, CoT | 1977.07 | **51.95** | 46.85 |
| Variant C, LoRA r=64, top-k KL | Pending | Pending | Pending |
| Variant BC, LoRA r=64, joint | Pending | Pending | Pending |

Recovery is defined as:

```text
(distilled score - 2B baseline) / (32B teacher - 2B baseline) * 100
```

The best LoRA run recovers 15.3% of the aggregate CV-Bench gap. Its largest
sub-task improvement is Depth: 84.67% to 90.00%. CoT improves MME and MMStar,
but performs poorly on short-answer CV-Bench questions; this result should be
treated as preliminary while the trace/answer masking is refined.

## Pipeline

```text
COCO train2014 + LLaVA-Instruct-150K
                  |
                  v
       Qwen3-VL-32B teacher generation
       response / CoT / top-k logits
                  |
                  v
       confidence and CoT quality filters
                  |
                  v
       Qwen3-VL-2B training (full FT or LoRA)
                  |
                  v
          MME / MMStar / CV-Bench
```

The spatial split contains metric, relational, and egocentric questions. COCO
bounding boxes are used only to verify generated teacher answers; the student
trains on teacher responses rather than raw box coordinates.

## Repository layout

| Path | Purpose |
| --- | --- |
| `gen_teacher_data.py` | Generate response-only teacher data with vLLM |
| `gen_cot_data.py` | Generate and filter spatial reasoning traces |
| `gen_all.py` | Generate text, top-k logits, and hidden states in two phases |
| `convert_cot_to_all.py` | Convert existing CoT JSONL into the unified schema |
| `kd_pipeline/` | Training configs, losses, collator, tests, demo, and utilities |
| `evaluation/lmms_eval/` | Project-local forced-choice CV-Bench task overlay |
| `cv_bench_compare/` | Per-sample logs used for the reported CV-Bench analysis |
| `results/` | Baseline result artifacts |
| `scripts/run_teacher_generation.slurm` | Portable four-GPU teacher job template |

Detailed field definitions are in
[`kd_pipeline/docs/DATA_FORMAT.md`](kd_pipeline/docs/DATA_FORMAT.md). Variant C
implementation notes are in
[`VARIANT_C_LOGIT_KL_IMPLEMENTATION_PLAN.md`](VARIANT_C_LOGIT_KL_IMPLEMENTATION_PLAN.md).

## Setup

Training requires a recent Python environment with PyTorch and Transformers:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kd_pipeline/requirements.txt
```

Teacher generation additionally requires a Qwen3-VL-compatible vLLM build.
Evaluation dependencies are separated to keep the training environment small:

```bash
pip install -r kd_pipeline/requirements-eval.txt
```

## Data layout

Set one root for models, datasets, caches, and generated files:

```bash
export QWEN3_VL_DATA_ROOT=/path/to/qwen3-vl-data
```

The generation scripts use the following defaults, all of which can be
overridden with command-line flags:

```text
$QWEN3_VL_DATA_ROOT/
├── models/qwen3-vl-32b/
├── datasets/coco/train2014/
├── datasets/coco/annotations/instances_train2014.json
├── datasets/LLaVA-Instruct-150K/llava_instruct_150k.json
└── data/
```

Large datasets, checkpoints, generated JSONL files, and model weights are
excluded from Git.

## Generate teacher data

Response-only 50k run:

```bash
python gen_teacher_data.py --total 50000 --tp 4 --batch_size 32 --resume
```

Spatial CoT run with the built-in length, keyword-density, and pivot-word
quality gate:

```bash
python gen_cot_data.py --total 50000 --tp 4 --batch_size 16 --resume
```

Unified text plus top-k signal generation:

```bash
python gen_all.py --total 50000 --tp 4 --batch_size 16 --logit_k 20 --resume
```

To reuse an existing CoT file and run only the forward-pass phase:

```bash
python convert_cot_to_all.py \
  --input "$QWEN3_VL_DATA_ROOT/data/cot_responses.jsonl" \
  --output "$QWEN3_VL_DATA_ROOT/data/teacher_all.jsonl"

python gen_all.py \
  --output "$QWEN3_VL_DATA_ROOT/data/teacher_all.jsonl" \
  --skip_phase1 --resume
```

For Slurm, export `QWEN3_VL_DATA_ROOT`, optionally set `CONTAINER_IMAGE`, then
submit `scripts/run_teacher_generation.slurm`.

## Train the student

Run training commands from `kd_pipeline/` because config paths are relative to
that directory:

```bash
cd kd_pipeline
python scripts/train_distill.py --config configs/variant_A.yaml
python scripts/train_distill.py --config configs/variant_B.yaml
python scripts/train_distill.py --config configs/variant_C.yaml
python scripts/train_distill.py --config configs/variant_BC.yaml
```

Before a full run, replace the sample paths in the selected config with the
generated JSONL files. The default LoRA setting is rank 64 with alpha 128.

Useful utilities include:

```bash
python scripts/prepare_data_manifest.py --jsonl data/clean_train.jsonl
python scripts/export_merged_model.py \
  --adapter_dir runs/variant_a/adapter_final \
  --out_dir exports/variant_a_merged --bf16
```

## Evaluate

Standard evaluation example:

```bash
cd kd_pipeline
bash scripts/eval_lmms_eval_example.sh /path/to/checkpoint logs/eval_run
```

For the short, forced-choice CV-Bench protocol used in the project:

```bash
python -m lmms_eval \
  --include_path evaluation/lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=/path/to/checkpoint,enable_thinking=False \
  --tasks cv_bench_forced \
  --batch_size 1 \
  --log_samples \
  --output_path logs/cv_bench
```

## Validation

CPU-safe unit tests:

```bash
cd kd_pipeline
make test
```

The end-to-end smoke test creates dummy multimodal data and runs a short teacher
and student path; it requires the model dependencies and suitable hardware:

```bash
cd kd_pipeline
make smoke
```

## Current status

- Completed: teacher response/CoT generation, Variants A and B, LoRA/full-FT
  comparison, three-benchmark evaluation, and per-task CV-Bench analysis.
- In progress: Variant C top-k KL and Variant BC joint results.
- Planned: LoRA rank sensitivity, cleaner CoT loss masking, and Jetson Orin
  latency measurements.
