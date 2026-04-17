# Qwen3-VL KD Pipeline（含 Variant C：top-k Logit KL）

本目录实现 **教师离线 top-k** + **学生 CE / CE+KL** 训练骨架，与 `../VARIANT_C_LOGIT_KL_IMPLEMENTATION_PLAN.md` 对齐。

## 数据约定（50k）

| 子集 | 规模 | 来源 |
|------|------|------|
| 空间推理 | 25k | CV-Bench |
| 通用 VQA | 25k | LLaVA-Instruct-150k 采样 |

全量训练将 `sample_train.jsonl` 换为 M1 产出的 `clean_train.jsonl`；**含 KL 的 ~10k** 优先覆盖 `data_source=cv_bench` 样本。字段约定见 **`docs/DATA_FORMAT.md`**。

**教师 `teacher_responses.jsonl`**：使用 `--input_format teacher_responses` 与 `configs/train_teacher_responses.yaml`（解析 `question` + `response` 中的 `<answer>`）。

## 环境

```bash
pip install -r requirements.txt
# 若 peft 报 torchao 版本冲突：pip uninstall -y torchao
```

部分环境 SDPA/cuDNN 报错时，脚本已默认 **`attn_implementation="eager"`**。

## 快速冒烟（单机单卡）

```bash
cd kd_pipeline
python scripts/make_dummy_assets.py
CUDA_VISIBLE_DEVICES=0 python scripts/gen_teacher_topk.py --config configs/gen_teacher.yaml --max_samples 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_distill.py --config configs/variant_C.yaml --max_steps 3 --out_dir runs/smoke_c
```

- **Variant A**（仅 CE）：`--variant A` 且无需 `teacher_topk_jsonl`（或见 `configs/variant_A.yaml`）。
- **32B 教师**：在 `configs/gen_teacher.yaml` 或命令行将 `teacher_model_id` 改为 `Qwen/Qwen3-VL-32B-Instruct`（需足够显存或量化方案）。
- **数据验收**：`python scripts/prepare_data_manifest.py --jsonl data/clean_train.jsonl` 查看 `data_source` 计数（目标 **cv_bench 25k + llava_instruct 25k**）。
- **合并子集**：`python scripts/merge_jsonl.py --out data/clean_train.jsonl --inputs a.jsonl:cv_bench b.jsonl:llava_instruct`
- **断点续训 LoRA**：`--resume_adapter runs/.../checkpoint-500`
- **评测示例**：`bash scripts/eval_lmms_eval_example.sh`（需 `pip install -r requirements-eval.txt`）
- **合并 LoRA 供评测**：`python scripts/export_merged_model.py --adapter_dir runs/.../adapter_final --out_dir exports/merged --bf16`

## CP-5 LoRA 消融（单卡/多卡）

在相同数据与 Variant A 下，对比 `r=16` 与 `r=64`（全参数结果复用 CP-4）。

```bash
cd kd_pipeline
# 默认单卡（推荐先试，通常比双卡更稳）
bash scripts/run_variant_a_lora_ablation.sh

# 双卡版本（保留原流程）
MODE=multi bash scripts/run_variant_a_lora_ablation.sh

# teacher_responses 格式（你当前手头 JSONL）
BASE_CFG=configs/variant_a_teacher_lora.yaml TRAIN_JSONL=/path/to/teacher_or_merged.jsonl \
  bash scripts/run_variant_a_lora_ablation.sh
```

- 默认读取 `data/clean_train.jsonl`；若路径不同可设：`TRAIN_JSONL=/path/to/clean_train.jsonl`
- 每个 rank 产物输出到：`runs/${OUT_PREFIX}_r{16|64}_{single|multi}`（默认 `OUT_PREFIX=variant_a_lora`）
- 日志输出到：`runs/c_task_logs/variant_a_lora_r*`

## 一键冒烟

```bash
bash scripts/smoke_end_to_end.sh
```

会清理并重建 `data/teacher_topk.jsonl`、训练 2 step、并写出 `checkpoint-1` / `checkpoint-2`（`--save_every 1`）。

## 目录

| 路径 | 说明 |
|------|------|
| `docs/DATA_FORMAT.md` | 50k 数据 JSONL 与 top-k 对齐字段 |
| `src/losses.py` | top-k 子空间 KL（§4.4） |
| `src/qwen3_vl_collator.py` | 多模态 chat 拼 `labels` |
| `scripts/gen_teacher_topk.py` | 离线写 `data/teacher_topk.jsonl` |
| `scripts/train_distill.py` | LoRA + CE / CE+KL；`--save_every` / `--seed` / `--resume_adapter` |
| `scripts/merge_jsonl.py` | 多文件合并并打 `data_source` |
| `scripts/export_merged_model.py` | LoRA `merge_and_unload` 导出全量权重 |
| `scripts/eval_lmms_eval_example.sh` | lmms-eval 三任务示例 |
| `Makefile` | `make test` / `make smoke` |
| `scripts/smoke_end_to_end.sh` | 端到端自检 |
| `data/sample_train.jsonl` | 占位样例（`make_dummy_assets.py` 生成） |

## 产物

- 训练日志：`runs/.../train_log.jsonl`
- Adapter：`runs/.../adapter_final/`

---

*最后更新：以 `RUN_STATUS.md` 为准。*
