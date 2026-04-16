# 训练 JSONL 字段约定（50k：CV-Bench 25k + LLaVA-Instruct 采样 25k）

与 `scripts/train_distill.py` / `src/qwen3_vl_collator.py` 兼容的最小字段如下。

## `teacher_responses.jsonl`（教师 M1 产物）

与 `--input_format teacher_responses` 配合使用（见 `src/teacher_responses.py`）。

| 字段 | 说明 |
|------|------|
| `id` | 样本 ID |
| `question` | 用户问题 → 作为 user prompt |
| `response` | 32B 完整输出；若含 `<answer>...</answer>`，默认 **`--teacher_target answer_only`** 只监督答案内文本 |
| `image` | 图像**绝对或相对路径**（需在本机可读） |
| `confidence` | 可选；`--min_confidence N` 低于则跳过 |
| `source` / `type` | 用于内部 `data_source` 映射（统计用） |

示例命令：

```bash
cd kd_pipeline
python scripts/train_distill.py --config configs/train_teacher_responses.yaml
```

若图片路径在别的机器（如 `/ocean/...`），需同步数据或改路径；默认 **`skip_missing_image: true`** 会跳过缺失文件。

## 必填（`chat` / clean_train 格式）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 全局唯一，用于对齐 `teacher_topk.jsonl` 与日志 |
| `data_source` | string | 建议：`cv_bench`（空间推理子集）或 `llava_instruct`（通用 VQA） |
| `assistant_text` | string | 教师生成或标注的**最终回答**（训练目标） |

## 二选一：用户侧输入

**方式 A（推荐，与占位样例一致）**

- `messages`：OpenAI 风格列表；`user` 含 `image`（本地路径字符串）+ `text`。

**方式 B**

- `user`：纯文本 prompt  
- `image`：图像路径（可选；无图则仅文本）

## 教师 top-k 产物（`gen_teacher_topk.py` 输出）

每行一个 JSON 对象：

- `id`：与训练 JSONL 一致  
- `teacher_model_id`、`topk`  
- `kl_steps`：`[{ "t": int, "ids": [K], "logits": [K] }, ...]`  
  - `t` 为 **shift 后**下标（与 `shift_logits[0, t]` 对齐）

## 验收

```bash
python scripts/prepare_data_manifest.py --jsonl data/clean_train.jsonl
```

期望（全量就绪后）：`data_source[cv_bench] ≈ 25000`，`data_source[llava_instruct] ≈ 25000`。

## 合并两路 JSONL

若 CV-Bench 与 LLaVA 各生成一个文件，可用：

```bash
python scripts/merge_jsonl.py --out data/clean_train.jsonl \
  --inputs data/cv_bench_25k.jsonl:cv_bench data/llava_25k.jsonl:llava_instruct
```
