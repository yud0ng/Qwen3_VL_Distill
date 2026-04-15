# Variant C（Logit KL 蒸馏）实施计划

面向《技术方案文档》模块二 **4.2 变体 C**：  
\(L = \lambda_1 L_{\text{CE}}(\text{answer}) + \lambda_3 \cdot \text{KL}(p_{\text{student}} \,\|\, p_{\text{teacher\_topk}}) \cdot T^2\)。

本文档包含：**分步任务**、**仓库内可复用代码盘点**、**缺口与对接项**。

---

## 一、目标与验收标准

| 项目 | 约定（与技术方案一致） |
|------|------------------------|
| 学生 | `Qwen/Qwen3-VL-2B-Instruct` |
| 教师 | `Qwen3-VL-32B-Instruct`（或团队固定路径） |
| KL 子集 | 空间推理子集约 **10k** 条，每条带 **top-k** 教师分布 |
| k | 默认 **50**，可降至 **20** 以节省存储 |
| 温度 T | **4.0** |
| 默认权重 | \(\lambda_1=0.5,\ \lambda_3=0.5\)（联合 B+C 时再按方案改为 0.5/0.3/0.2） |
| 实验行 | **M2-C**（Logit KL，全参数）；后续 **M2-BC** 需本分支 + CoT 分支同时开启 |

**最小验收**：在含 logits 的子集上训练能稳定降 loss；checkpoint 在 MME 或 CV-Bench 上相对 Variant A 有可报告的差异（哪怕小幅）。

---

## 1.1 数据集构成（项目约定 · 50k）

`clean_train.jsonl` 共 **约 50,000** 条，由两部分组成，与《技术方案》3.1、3.4 一致：

| 子集 | 规模 | 来源 | 说明 |
|------|------|------|------|
| 空间推理 | **25k** | **CV-Bench**（作种子） | 覆盖 Count / Depth / Spatial Relation 等；可再按方案做问题扩增、教师重写答案 |
| 通用 VQA | **25k** | **LLaVA-Instruct-150k**（采样） | 从 150k 中 **随机或分层采样 25k**，维持通用多模态能力，避免灾难性遗忘 |

**字段建议**：每条 JSONL 增加 `data_source`: `"cv_bench"` | `"llava_instruct"`，便于分层统计与 Logit 子集抽取（例如 **仅对空间子集约 10k 条**存教师 top-k，通用 25k 可走纯 CE）。

**Logit KL 子集**：在 **空间推理相关**样本上优先生成 `data/logits/`（技术方案：约 10k 条 top-k）；通用 25k 可不存 logits，训练时对该部分 \(\lambda_3=0\) 或整条样本跳过 KL。

---

## 二、仓库内现有代码盘点

### 2.1 与本项目直接相关

| 路径 | 内容 | 对 Variant C 的用处 |
|------|------|------------------------|
| `intro2dl-proj/Qwen3_VL_Distill/README.md` | 32B 基线评测结果模板 | 结果表、Recovery% 叙事对齐 |
| `intro2dl-proj/技术方案文档.docx.pdf`（若在同一目录） | 完整 loss 公式、数据规模、目录结构 | 唯一需求源 |

**进展**：`Qwen3_VL_Distill/kd_pipeline/` 已提供可运行骨架：`scripts/gen_teacher_topk.py`、`scripts/train_distill.py`、`src/losses.py`、`configs/variant_*.yaml`、YAML 合并、`--save_every`/`--seed`、pytest、`docs/DATA_FORMAT.md`、`scripts/smoke_end_to_end.sh`。全量 50k + 32B 教师仍需按下面章节接入真实数据与资源。

### 2.2 可借鉴的现有实现（Road_sign_recognition_Qwen）

以下代码任务不同（交通标志 **粗分类 + 6 类 softmax**），但**工程模式可复用**：

| 路径 | 可复用点 |
|------|-----------|
| `Road_sign_recognition_Qwen/02_train_student_kd.py` | 组合损失：`loss = λ * CE + (1-λ) * KD`；`T**2 * F.kl_div(...)`；LoRA（`peft.LoraConfig`）；`gradient_checkpointing`；保存 adapter |
| `Road_sign_recognition_Qwen/01_cache_teacher_soft_labels.py` | 教师离线缓存 JSONL、断点续跑（`done_images`）、温度 softmax、版本元数据思路 |
| `Road_sign_recognition_Qwen/final_qwen25_scoring.py` | `logits → log_softmax`、多模态 `forward` 传 `pixel_values` / `image_grid_thw` / `mm_token_type_ids` 的写法；**按 token 位置取 log prob** 的模式 |
| `Road_sign_recognition_Qwen/milestone/04_train_gtsrb_lora_full.py` | **Qwen3-VL** + `Trainer` + LoRA；batch=1 等与视觉 token 相关的约束说明 |

**重要差异（勿照搬逻辑）**：

- 现有 KD 是对 **6 个固定 completion** 打分再 softmax，属于**极小闭集上的分布**。
- Variant C 需要 **生成序列每个位置** 在 **词表 top-k** 上的分布（或方案约定的 answer 段上），并在学生 **同一位置** 取 logits 做 **子空间 KL**。

因此：**复用「脚本结构 + KL 缩放习惯 + 模型加载模式」**；**重写「数据字段 + 逐 token top-k 对齐」**。

### 2.3 本仓库其他目录

- `sglang/`、`xinyu/agenticTraining/` 等体量大，与「Qwen3-VL + HF/DeepSpeed 学生微调」主线不一致，**不建议**作为首选依赖；若日后用 vLLM 批量出教师 top-k，再单独评估。

---

## 三、分阶段实施步骤

### 阶段 0：契约冻结（与数据负责人 / M1）

1. **样本 ID**：每条训练样本唯一 `id`（或 `hash(image_path, user_text)`）。
2. **对齐粒度**：KL 计算在 **哪些 token 位置**（建议先定：**仅 answer / assistant 段**，与 CE mask 一致）。
3. **落盘格式**（示例，可二选一）  
   - **A**：每样本一个 JSON 数组 `per_step: [{ "pos": int, "ids": [k], "logits": [k] }]`（`pos` 为在完整 `input_ids` 中的下标）。  
   - **B**：单独 `numpy`/torch 文件，用 `sample_id` 索引。  
4. **与 `clean_train.jsonl` 的关系**：logits 子集是其中子集（约 10k），行级引用 `id`，避免重复存图路径。

**产出物**：`docs/` 或 README 片段「Logits Schema v1」。

---

### 阶段 1：教师 top-k 生成脚本（32B，离线）

1. 读取 M1 产出的 **空间子集**（或临时用全量中的一万条试跑）。
2. 对每条样本：构造与训练时一致的 **messages → processor → input_ids**。
3. **Teacher forward**（可用 HF 或 vLLM，团队已定则用 vLLM 提吞吐）：对每个需要监督的位置 \(t\)，取 `logits[t]` 的 **top-k indices + values**（存 logits 或存已除 T 的 log_softmax 需统一，建议存 **未除 T 的 logits** 以便重算 T）。
4. 写入 `data/logits/`（或技术方案中的 `project/data/logits/`）。
5. **数值自检**：随机抽 10 条，对第一个监督位置手算 `softmax(logits/T)`，与存储一致。

**依赖**：transformers ≥ 4.57、32B 显存方案（量化 / 多卡）。

**仓库参考**：`01_cache_teacher_soft_labels.py` 的断点续跑与 JSONL 行格式；**逻辑需新建** `scripts/gen_teacher_topk.py`（名称自定）。

---

### 阶段 2：Dataset 与 Collator

1. 从 `clean_train.jsonl` 读入对话与图像。
2. 若样本 **无** logits 字段：仅参与 CE（\(\lambda_3=0\) 或跳过 KL）；若有：加载 `teacher_topk_ids`, `teacher_topk_logits`，长度与监督位置数一致。
3. `labels`：标准因果 LM mask（prompt 为 -100）。
4. **监督位置 mask**：与方案一致（通常 answer 段 `labels != -100` 的移位对齐需注意：预测第 \(t\) 个 token 用 `logits[:, t-1, :]` 还是 `t`，必须与教师存 logits 时的步对齐，**全队统一**）。

**产出物**：`dataset.py` 或内嵌在 `train.py`；单元测试：batch 形状、`kl` 非 NaN。

---

### 阶段 3：实现 `distillation_loss`（Variant C 核心）

按技术方案 **4.4** 伪代码：

1. `loss_ce = CrossEntropy(student_logits, answer_labels)`（仅对有效 label 位）。
2. 对学生 logits 在教师 **top-k id** 上 gather，softmax 后 **重归一化** 得 `s_topk`。
3. 教师 `t_probs = softmax(teacher_topk_logits / T)`。
4. `loss_kl = kl_div(log(s_topk), t_probs, reduction='batchmean') * (T**2)`。
5. `total = lam1 * loss_ce + lam3 * loss_kl`（Variant C 无 trace 时 `lam2=0`）。

**注意**：

- `F.kl_div` 第一个参数需为 **log-prob**（对 `s_topk` 用 `.log()` 前确保 >0，加小 `eps` 或 `clamp`）。
- 若 batch 内序列长度不同，需按 mask 对有效 KL 位求平均，与 CE 的 reduction 一致。

**仓库参考**：`02_train_student_kd.py` 中 \(T^2\) 与 `kl_div` 习惯；**须改为** 张量 batch 的逐位置实现，而非 6 维向量。

---

### 阶段 4：接入统一训练脚本（config 切换）

1. 单一 `train.py`（或 HF `Trainer` 子类）支持 YAML/flags：`use_kl`, `lam1`, `lam3`, `T`, `topk`, `full_finetune` / `lora_r`。
2. **`lam3=0` 时**：不加载 logits 或不算 KL（省显存、少分支 bug）。
3. DeepSpeed ZeRO-2 / bf16 / checkpoint 频率与方案 **4.1** 对齐。
4. 先跑 **Variant A** 同数据子集，再打开 KL，确保 **唯一变量** 是 KL。

**仓库参考**：`04_train_gtsrb_lora_full.py` 的 Trainer 与 LoRA 挂载方式；KD 部分无现成 Qwen3-VL 序列版，需新写 `compute_loss`。

---

### 阶段 5：实验与记录（M2-C）

1. 固定种子与数据顺序；保存完整 `config.yaml` 快照。
2. 跑 **M2-C** 一行；填结果表（MME / MMStar / CV-Bench）。
3. 可选：扫 **k∈{20,50}**、**λ₃∈{0.3,0.5}** 的小网格（时间允许时）。

---

### 阶段 6：与 Variant B+C 合并（若你负责总闸）

1. 在同一 `forward` 中：`loss = lam1*ce + lam2*ce_trace + lam3*kl`（trace mask 独立）。
2. 默认 **0.5 / 0.3 / 0.2**；先单独验证 B 与 C 再合并。

---

## 四、风险与缓解

| 风险 | 缓解 |
|------|------|
| 教师存 logits 与学生 forward **位置不对齐** | 阶段 0 冻结；用短序列 golden 样本人工核对 |
| KL 与 CE **量级差一个数量级** | 先记录三者 raw loss；调 \(\lambda\) 或 T；必要时 KL 只对部分层/部分步 |
| 存储过大 | k=20；只存 answer 段；fp16 存 logits |
| 全参 OOM | 先 LoRA 验证 KL 曲线，再全参跑正式表 |

---

## 五、建议目录（与技术方案 7.3 对齐）

在 `Qwen3_VL_Distill/`（或团队统一 `project/`）逐步补齐：

```text
Qwen3_VL_Distill/kd_pipeline/   # 已实现冒烟版
├── data/
│   ├── sample_train.jsonl
│   └── teacher_topk.jsonl
├── docs/DATA_FORMAT.md
├── scripts/
│   ├── train_distill.py        # A/C；YAML；save_every；resume_adapter
│   ├── gen_teacher_topk.py
│   ├── export_merged_model.py  # merge_and_unload → 全量目录（lmms-eval）
│   ├── merge_jsonl.py
│   ├── prepare_data_manifest.py
│   ├── eval_lmms_eval_example.sh
│   └── smoke_end_to_end.sh
├── tests/                      # pytest
├── Makefile                    # make test / make smoke
└── README.md
```

---

## 六、检查清单（Definition of Done）

- [ ] Logits schema 文档 + 样例文件已评审  
- [ ] 教师 top-k 脚本可断点续跑，10k 子集跑完  
- [ ] 训练脚本 `use_kl=true` 时 loss 无 NaN，tensorboard/wandb 有 `ce` / `kl`  
- [ ] M2-C 至少 1 个完整 checkpoint + 评测日志路径可复现  
- [ ] 与 M2-A（同数据预算）对比表已填 Recovery%

---

## 七、相关文件快速索引

| 文件 | 说明 |
|------|------|
| `intro2dl-proj/Qwen3_VL_Distill/README.md` | 基线与结果模板 |
| `intro2dl-proj/Road_sign_recognition_Qwen/02_train_student_kd.py` | CE+KL 参考（分类式 KD） |
| `intro2dl-proj/Road_sign_recognition_Qwen/01_cache_teacher_soft_labels.py` | 教师缓存与 softmax 流程 |
| `intro2dl-proj/Road_sign_recognition_Qwen/final_qwen25_scoring.py` | logits / 多模态 forward 参考 |
| `intro2dl-proj/Road_sign_recognition_Qwen/milestone/04_train_gtsrb_lora_full.py` | Qwen3-VL + Trainer + LoRA |
| `intro2dl-proj/Qwen3_VL_Distill/kd_pipeline/` | **本仓库主实现**：训练 / 教师 top-k / 合并数据 / 导出合并权重 / 评测示例 |

---

*文档版本：2026-04-15（§1.1：50k = CV-Bench 25k + LLaVA 采样 25k；§五：kd_pipeline 目录已更新）· `intro2dl-proj/Qwen3_VL_Distill/`。*
