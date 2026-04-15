# Qwen3-VL-32B Baseline Evaluation Results

**Vector Robotics · Qwen3-VL Knowledge Distillation Project**

### 训练数据（50k 构成）

| 子集 | 规模 | 来源 |
|------|------|------|
| 空间推理 | 25k | CV-Bench（种子） |
| 通用 VQA | 25k | LLaVA-Instruct-150k（采样） |

说明与 Variant C 实施细节见同目录 `VARIANT_C_LOGIT_KL_IMPLEMENTATION_PLAN.md` §1.1。

---

| Model | `Qwen3-VL-32B-Instruct` |


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
