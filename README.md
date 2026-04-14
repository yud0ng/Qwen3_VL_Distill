# Qwen3-VL-32B Baseline Evaluation Results

**Vector Robotics · Qwen3-VL Knowledge Distillation Project**

| Field | Detail |
|-------|--------|
| Model | `Qwen3-VL-32B-Instruct` |
| Eval Framework | lmms-eval (unknown@unknown) |
| Date | 2026-04-14 |
| Eval Time | ~91 min (5466 sec) |
| Hardware | PSC (`/ocean/projects/cis220039p/yluo22/`) |

---

## Results

### MME

| Metric | Score |
|--------|-------|
| Perception Score | **1787.94** |
| Cognition Score | **688.57** |
| **Total Score** | **2476.51** |

### MMStar

| Dimension | Accuracy |
|-----------|----------|
| Coarse Perception | 76.88% |
| Fine-grained Perception | 68.30% |
| Instance Reasoning | 79.81% |
| Logical Reasoning | 73.78% |
| Science & Technology | 63.50% |
| Math | 62.15% |
| **Average** | **70.74%** |

### CV-Bench

| Metric | Score |
|--------|-------|
| **Average Acc** | **88.17%** |

---

## Summary Table (for results template)

| Benchmark | Metric | 32B Teacher | 2B Original | 2B Distilled | Recovery% |
|-----------|--------|-------------|-------------|--------------|-----------|
| MME | Total Score | 2476.51 | - | - | - |
| MMStar | Average Acc | 70.74% | - | - | - |
| CV-Bench | Average Acc | 88.17% | - | - | - |

> Recovery% = (Distilled − Original) / (Teacher − Original) × 100

---

## Eval Command

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=/ocean/projects/cis220039p/yluo22/models/qwen3-vl-32b" \
  --tasks mmstar,mme,cv_bench \
  --batch_size 1 \
  --output_path /ocean/projects/cis220039p/yluo22/logs/32b/
```

---

## Sample Counts

| Benchmark | Samples |
|-----------|---------|
| MME | 2,374 |
| MMStar | 1,500 |
| CV-Bench | 2,638 |
| **Total** | **6,512** |

---

## Throughput

| Metric | Value |
|--------|-------|
| Total tokens generated | 22,151 |
| Avg speed | 9.76 tokens/s |
| Total elapsed time | 5,466 sec |

---
