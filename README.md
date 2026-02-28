# Intro2DL OCR-VLM Milestone (Zero-shot Baseline)

## 1) 当前 milestone 目标

在单卡小算力预算下，先搭建一个 **最小可跑通** 的 OCR-oriented VLM baseline：
- 固定一个小模型、一个数据格式、一个 prompt 模板
- 跑通 image+text 推理
- 跑通最小评测
- 跑通 sanity 子集 zero-shot 闭环

当前阶段不做蒸馏、不做 QLoRA、不做 full finetuning。

---

## 2) 选用模型与原因

### 首选模型
- `HuggingFaceTB/SmolVLM-Instruct`

### 备选模型
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`

### 选择理由（以“易跑通 milestone”为主）
- 模型体量小，适合小算力环境先打通流程
- 支持 image + text 输入，符合 OCR QA 任务形态
- 可直接走 Hugging Face `transformers` 推理链路
- 后续可在 `peft` 生态继续做 LoRA 微调（当前里程碑先不做）

---

## 3) 目录结构

```text
intro2dl/
  README.md
  requirements.txt
  eval.py
  data/
    train.json
    val.json
    test.json
    sanity.json
    images/              # synthetic 或导出的图像
  scripts/
    vlm_utils.py
    test_infer.py
    prepare_data.py
    run_zero_shot.py
  results/
```

---

## 4) 统一数据格式说明

每条样本为：

```json
{
  "id": "sample_0001",
  "image_path": "data/images/xxx.png",
  "question": "What is the exact text in the box?",
  "answer": "total-1234"
}
```

`data/*.json` 都是由上述对象构成的 JSON list。

---

## 5) 如何准备数据

### 方案 A（推荐先跑通）：synthetic OCR 数据
先生成可控小样本，快速验证闭环：

```bash
python scripts/prepare_data.py \
  --mode synthetic \
  --sanity_size 20 \
  --train_size 1000 \
  --val_size 200 \
  --test_size 200
```

会生成：
- `data/sanity.json`
- `data/train.json`
- `data/val.json`
- `data/test.json`

### 方案 B：本地 OCR 标注导入
如果你已有本地 benchmark 子集（例如 OCR-VQA / DocVQA 导出结果），准备 JSONL：

```json
{"id":"1","image_path":"...","question":"...","answer":"..."}
```

然后运行：

```bash
python scripts/prepare_data.py \
  --mode local_jsonl \
  --input_jsonl /path/to/your_data.jsonl \
  --id_key id \
  --image_key image_path \
  --question_key question \
  --answer_key answer
```

### 方案 C：Hugging Face dataset 导入（框架已支持）

```bash
python scripts/prepare_data.py \
  --mode hf_dataset \
  --hf_dataset_name your_dataset_name \
  --hf_split train \
  --id_key id \
  --image_key image \
  --question_key question \
  --answer_key answer
```

> 注意：不同 benchmark 字段名不同，按实际字段覆盖 `*_key` 参数。

---

## 6) 最小 inference demo

先安装依赖：

```bash
pip install -r requirements.txt
```

运行单图推理：

```bash
python scripts/test_infer.py \
  --image_path data/images/synthetic_00000.png \
  --question "What is the exact text in the box?" \
  --model_id HuggingFaceTB/SmolVLM-Instruct \
  --device auto
```

输出即模型生成答案文本。

---

## 7) 运行 zero-shot sanity check

固定 prompt 模板已经内置在 `scripts/vlm_utils.py`（`DEFAULT_PROMPT_TEMPLATE`）。

运行命令：

```bash
python scripts/run_zero_shot.py \
  --input_json data/sanity.json \
  --pred_out results/pred_zero_shot_sanity.json \
  --metrics_out results/zero_shot_sanity_metrics.json \
  --model_id HuggingFaceTB/SmolVLM-Instruct \
  --max_samples 20
```

输出：
- `results/pred_zero_shot_sanity.json`
- `results/zero_shot_sanity_metrics.json`

---

## 8) eval.py 用法

### prediction 文件格式

```json
[
  {"id": "sample_0001", "prediction": "total-1234"},
  {"id": "sample_0002", "prediction": "invoice-9876"}
]
```

### 运行示例

```bash
python eval.py \
  --gt_json data/sanity.json \
  --pred_json results/pred_zero_shot_sanity.json \
  --metric accuracy \
  --save_path results/zero_shot_sanity_metrics.json
```

支持指标：
- `accuracy`（默认）
- `anls`

评测内置 normalization：
- lower case
- 去首尾空格
- 去标点

---

## 9) 当前完成状态与下一步

### 当前已完成
- 最小 VLM 推理脚本：`scripts/test_infer.py`
- 数据处理脚本：`scripts/prepare_data.py`
- 统一评测脚本：`eval.py`
- zero-shot 执行脚本：`scripts/run_zero_shot.py`
- 项目目录与文档规范化

### 下一步（LoRA 微调）
- 基于同一数据格式接入 `peft` LoRA 训练脚本
- 在同一 `val/test` 上对比 `zero-shot vs LoRA`
- 再考虑是否引入 QLoRA / 蒸馏（非当前阶段）
