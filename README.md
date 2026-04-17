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

# Teacher Response (Baseline) - Statistics Report

### 1. Total Volume
- **Total Written:** 50,000

### 2. Type Breakdown
| Category | Count |
| :--- | :--- |
| **General** | 25,000 |
| **Metric** | 8,472 |
| **Relational** | 8,307 |
| **Egocentric** | 8,221 |

### 3. Confidence Distribution (Pre-filter)
| Confidence Level | Count | Percentage |
| :--- | :--- | :--- |
| **Conf=5** | 41,163 | 82.3% |
| **Conf=4** | 7,572 | 15.1% |
| **Conf=3** | 204 | 0.4% |
| **Conf=2** | 19 | 0.0% |
| **Conf=1** | 4 | 0.0% |
| **Missing** | 1,038 | 2.1% |

### 4. Filter Results
- **Samples (Conf >= 4):** 48,735
- **Pass Rate:** 97.5%

---

# Teacher Response (with CoT and filter) - Statistics Report

* **Total written:** 12,891 (resume base: 0, this run: 12,891)
* **Rejected by filter:** 7,109 | **Pass rate:** 64.5% *(this run only)*
  * ↳ *too short:* 20
  * ↳ *low density:* 526
  * ↳ *no pivot word:* 6,645
* **Has thinking:** 12,891 (100.0%)

---

### Thinking-Chain Length Distribution

| Length | Count | Percentage |
| :--- | :--- | :--- |
| **<50** | 346 | 2.7% |
| **50-99** | 4,543 | 35.2% |
| **100-199** | 5,642 | 43.8% |
| **200-499** | 2,044 | 15.9% |
| **500+** | 316 | 2.5% |

---

### Confidence Distribution

| Confidence Level | Count | Percentage |
| :--- | :--- | :--- |
| **conf=5** | 10,337 | 80.2% |
| **conf=4** | 2,309 | 17.9% |
| **conf=3** | 177 | 1.4% |
| **conf=2** | 37 | 0.3% |
| **conf=1** | 9 | 0.1% |
| **conf=missing**| 22 | 0.2% |
