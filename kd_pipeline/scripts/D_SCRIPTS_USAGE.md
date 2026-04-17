# D 角色交付脚本速查

> 版本 2.0 · 2026-04-16（SDD refactor 后）
> 所有脚本均在 `kd_pipeline/` 目录下运行。

## 一览

| 脚本 | 职责 | 依赖（新增）| 消费者 |
|---|---|---|---|
| `classify_spatial_level.py` | L1/L2/L3 分层 + 每层 Recovery% | `src/spatial_vocab`, `src/lmms_eval_io` | B（评测） |
| `filter_cot_quality.py` | CoT 推理链质量门控 | `src/spatial_vocab` | A（数据生成后）|
| `select_logit_subset.py` | 选 ~10k 空间样本 ID | `src/spatial_vocab` | A（logit 生成前）|
| `sample_error_cases.py` | 50 条 2B 错/32B 对样本抽样 | `src/lmms_eval_io`, `src/csv_safe` | B / D（误差分析）|

Demo:
- `demo/build_teacher_cache.py` — 构建 demo 用 32B 缓存
- `demo/app.py` — Gradio 三列 UI

## 前置条件

**B 跑 lmms-eval 时必须加 `--log_samples`**（否则 M2/M6 读不到样本级数据）：

```bash
LOG_SAMPLES=1 bash scripts/eval_lmms_eval_example.sh <model> <out_dir>
# 或
bash eval_final.sh <ckpt> <label>   # eval_final.sh 默认已开启
```

## 常用命令

### A 角色：取 logit 子集 ID
```bash
python scripts/select_logit_subset.py \
    --input ../teacher_responses.jsonl \
    --n 10000 \
    --out_ids data/logit_subset_ids.txt \
    --out_manifest data/logit_subset_ids.manifest.json

# 然后
python scripts/gen_teacher_topk.py --id_list data/logit_subset_ids.txt ...
```

### A 角色（thinking=True 重跑后）：过滤 CoT
```bash
python scripts/filter_cot_quality.py \
    --input ../teacher_responses_cot.jsonl \
    --out_pass data/clean_train_cot.jsonl \
    --out_fail data/cot_rejected.jsonl \
    --report data/cot_filter_report.json

# 阈值调参
python scripts/filter_cot_quality.py ... \
    --min_confidence 4 \
    --min_trace_words 30 \
    --min_spatial_keywords 2 \
    --min_pivots 1
```

注：`--min_trace_words` 是按空白分词的**词数**，不是 BPE token。
英文推理文本 BPE token 比词数多约 30-50%，需按词数设计阈值。
旧 `--min_trace_tokens` 保留为别名。

### B 角色：分层准确率 + Recovery%
```bash
# 仅本 checkpoint 的分层准确率
python scripts/classify_spatial_level.py \
    --samples_jsonl logs/final_variantA/samples_cv_bench.jsonl \
    --out_csv logs/final_variantA/by_level.csv

# 完整分层 Recovery%
python scripts/classify_spatial_level.py \
    --samples_jsonl logs/final_variantA/samples_cv_bench.jsonl \
    --baseline_samples logs/2b_baseline/samples_cv_bench.jsonl \
    --teacher_samples logs/32b/samples_cv_bench.jsonl \
    --out_csv logs/final_variantA/recovery_by_level.csv
```

### B / D 角色：误差抽样
```bash
python scripts/sample_error_cases.py \
    --distilled_samples logs/final_variantBC/samples_cv_bench.jsonl \
    --teacher_samples logs/32b/samples_cv_bench.jsonl \
    --n 50 \
    --out_csv logs/final_variantBC/error_samples.csv
```

输出 CSV 含 `error_category` 空列，由人工填写
（`direction_confusion` / `distance_estimation` / `occlusion_reasoning`）。

## 输出契约

### `logit_subset_ids.txt`
每行一个 sample id，无 header。同目录 `.manifest.json` 含统计：
```json
{
  "requested": 10000,
  "selected": 10000,
  "skipped_no_id": 0,
  "total_by_bucket": {"B1_spatial_conf5": 19910, ...},
  "chosen_by_bucket": {"B1_spatial_conf5": 10000}
}
```

### `clean_train_cot.jsonl`
输入行的超集（保持原字段），仅通过质量门控的行。对应 `.fail.jsonl` 附
`_filter: {reasons: [...], metrics: {...}}`。

### `error_samples.csv`
列：`sample_id, level, image, question, distilled_answer, teacher_answer, error_category`。
**注意**：模型输出以 `=/+/-/@` 开头会被加 `'` 前缀（防 Excel formula injection）。

### `per_level.csv`
列：`level, n, acc_eval [, acc_baseline, acc_teacher, recovery_pct]`。

## 测试

```bash
make test                              # 全部 pytest（210+）
make d_pipeline                        # 端到端验证 + 产出 logit_ids + cache
pytest tests/test_e2e_d_pipeline.py    # 仅 D 端到端
pytest tests/ -k "spatial_vocab"       # 某一模块
make coverage                          # 带覆盖率报告
```

## SDD 规范

本目录由 `.sdd/spec.md` + `.sdd/plan.md` + `.sdd/task.md` 约束。修改脚本
接口前请先更新 spec 并通知 A/B/C。见 `.sdd/reviews.md` 了解 v1 → v2 重构涉及的
7 个 CRITICAL 修复（XSS / state race / 空间关键词 word-boundary / CoT token
语义 / 等）。
