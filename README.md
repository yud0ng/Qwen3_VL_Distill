# Qwen3-VL-32B Baseline Evaluation Results

**Vector Robotics · Qwen3-VL Knowledge Distillation Project**


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
| MME | Total Score | 2476.51 | 1910.64 | - | - |
| MMStar | Average Acc | 70.74% | 47.56% | - | - |
| CV-Bench | Average Acc | 88.17% | 73.09% | - | - |

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

teacher response (baseline):

Total written: 50000

Type breakdown:
  egocentric         8221
  general           25000
  metric             8472
  relational         8307

Confidence distribution (pre-filter):
  conf=5          41163  (82.3%)
  conf=4           7572  (15.1%)
  conf=3            204  (0.4%)
  conf=2             19  (0.0%)
  conf=1              4  (0.0%)
  conf=missing     1038  (2.1%)

Samples conf>=4 (will pass filter): 48735  (97.5%)
