# 训练 JSONL 字段约定（50k：CV-Bench 25k + LLaVA-Instruct 采样 25k）

与 `scripts/train_distill.py` / `src/qwen3_vl_collator.py` 兼容的最小字段如下。

## 必填

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
