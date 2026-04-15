# 运行状态 / 时间线

| UTC 时间 | 动作 |
|----------|------|
| 2026-04-15 | 初始化 `kd_pipeline/`：`gen_teacher_topk.py`、`train_distill.py`、`src/losses.py`、占位数据；`gen_teacher_topk` + `train_distill --variant C` 冒烟通过；适配环境：卸载冲突 `torchao`、`attn_implementation=eager` 规避 cuDNN SDPA 错误。 |
| 2026-04-15 | 增加：`src/config_utils.py`；`--config` 合并 YAML（`configs/variant_C.yaml`、`configs/gen_teacher.yaml`）；`num_epochs`；`resolved_args.json`；`tests/test_*.py`（pytest 4 passed）；`prepare_data_manifest.py`；`.gitignore`。 |
| 2026-04-15 | `train_distill.py`：`--save_every`、`--seed`；`docs/DATA_FORMAT.md`；`scripts/smoke_end_to_end.sh`（清 teacher_topk 避免断点跳过；`num_epochs=2` 配合 `max_steps=2` 验证 `checkpoint-1/2`）。 |
| 2026-04-15 | `dtype=` 替代弃用 `torch_dtype`；`--resume_adapter` + `PeftModel.from_pretrained`；`merge_jsonl.py`；`eval_lmms_eval_example.sh`；`tests/test_merge_jsonl.py`。 |
| 2026-04-15 | `export_merged_model.py`（merge_and_unload）；`requirements-eval.txt`；`Makefile`（test/smoke）。 |

**下一步（建议）**

1. 接入真实 **CV-Bench 25k + LLaVA 25k** JSONL（字段 `id`, `data_source`, `messages` / `assistant_text`）。
2. 32B 教师批量跑 top-k（或 vLLM）；空间子集约 **10k** 写 `teacher_topk.jsonl`。
3. 多卡 / DeepSpeed ZeRO-2、有效 batch、与 `lmms-eval` 评测脚本对齐。
