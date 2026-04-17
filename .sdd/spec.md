# Role D Data Pipeline + Demo — Specification

**Version**: 1.0 · 2026-04-16
**Owner**: D (data engineer + demo builder)
**Status**: DRAFT — pending Owner approval

---

## 1. Overview

Build the data-plumbing and demo layer for the Qwen3-VL 32B→2B distillation project, such that teammates A (32B generation), B (evaluation), and C (training) can consume D's outputs without extra engineering. All code runs on a CPU-only box for scripts and a single 24GB GPU for demo.

## 2. Background & Motivation

Role D owns:
- Data format conversion / confidence filtering / CoT quality gating (§3.3 of technical plan)
- Logit subset selection for Variant C top-k generation (§4.4)
- L1/L2/L3 stratified classifier used by evaluation (§5.2)
- Error case sampling for §5.3 qualitative analysis
- Gradio three-column demo (§CP-10)

Teammates block on D's contracts when they finish their training runs. D needs to finish plug-and-play deliverables before the distillation race so later collaboration is low-friction.

## 3. Goals

### MUST
- **M1 CoT quality filter**: Given `teacher_responses_cot.jsonl` (with `<think>` tags), write `clean_train_cot.jsonl` filtered by confidence≥4 + keyword density≥2 + trace tokens≥30 + ≥1 pivot word. Configurable via CLI flags.
- **M2 L1/L2/L3 classifier**: Pure function `classify_level(question) -> "L1"|"L2"|"L3"` with word-boundary matching. CLI for computing per-level accuracy + Recovery% from **two supported lmms-eval output formats** (aggregate JSON + per-sample JSONL).
- **M3 Logit subset selector**: Deterministic priority-based sampler emitting ~10k sample IDs to `logit_subset_ids.txt` for A's `gen_teacher_topk.py --id_list`.
- **M4 Teacher cache builder**: 30-50 curated entries covering Count/Depth/Relational/Egocentric/Metric from `teacher_responses.jsonl`, with paths rewritten to a configurable local root.
- **M5 Gradio demo**: Three-column comparison UI (2B raw / 2B distilled / 32B cached). `--no_load` mode for UI self-test without GPU.
- **M6 Error sampler**: Given distilled + teacher per-sample output, sample 50 "2B wrong & 32B right" cases stratified across L2/L3 (L1 fallback). CSV with a blank `error_category` column for manual tagging.
- **Security**: No XSS in demo, no CSV injection, no path leakage, no raw exception strings in frontend.
- **Tests**: ≥90% line coverage on refactored modules; ≥95 tests across all D deliverables.
- **Zero runtime dependency on A/B/C work**: Scripts ship with synthetic fixture data for end-to-end tests.

### SHOULD
- Consolidate shared parsing utilities (answer/trace/confidence extraction, lmms-eval sample loader) into a single internal module.
- Log unknown-schema samples to stderr instead of silent drop.
- Provide a `Makefile` / npm-script equivalent that runs all 6 modules end-to-end on fixture data.
- Emit JSON manifest alongside every CLI output for reproducibility.

### MAY
- Cache image thumbnails for faster Gradio startup.
- Accept `teacher_responses.jsonl.gz` as input.

## 4. Non-Goals

- **NOT** implementing 32B teacher inference (that's A's scope).
- **NOT** implementing training loss / LoRA (that's C's scope).
- **NOT** implementing lmms-eval invocation (that's B's scope) — D only consumes its output.
- **NOT** running the distilled model in demo on headless server — 24GB GPU assumption holds.
- **NOT** internationalization. English + Chinese mixed output is acceptable; no i18n framework.

## 5. User Scenarios

### Scenario A: A wants a logit subset ID list
```
A: python scripts/select_logit_subset.py --input ../teacher_responses.jsonl --n 10000 --out_ids data/logit_subset_ids.txt
A: python scripts/gen_teacher_topk.py --id_list data/logit_subset_ids.txt ...
```
D's deliverable: deterministic `logit_subset_ids.txt` + manifest JSON showing bucket breakdown.

### Scenario B: B needs L1/L2/L3 Recovery% for a checkpoint
```
B: bash eval_final.sh runs/.../adapter_final label    # now passes --log_samples
B: python scripts/classify_spatial_level.py \
     --samples_jsonl logs/final_label/samples_cv_bench.jsonl \
     --baseline_samples logs/2b_baseline/samples_cv_bench.jsonl \
     --teacher_samples logs/32b/samples_cv_bench.jsonl \
     --out_csv runs/label/per_level.csv
```
D's deliverable: per-level table with Recovery% and confidence intervals.

### Scenario C: B triggers error analysis after a training run
```
B: python scripts/sample_error_cases.py \
     --distilled_samples logs/final_label/samples_cv_bench.jsonl \
     --teacher_samples logs/32b/samples_cv_bench.jsonl \
     --n 50 --out_csv runs/label/errors.csv
```
D's deliverable: CSV ready for manual classification; `error_category` column left blank.

### Scenario D: Final presentation demo
```
D: python demo/app.py \
     --original_model_path Qwen/Qwen3-VL-2B-Instruct \
     --distilled_model_path runs/.../adapter_final \
     --port 7860
```
D's deliverable: three-column UI, preset questions, live L1/L2/L3 tag, diff highlight.

## 6. Technical Constraints

- Python 3.10+ (3.13 on Windows dev box; Linux cluster Python 3.10).
- No DeepSpeed / CUDA requirement for scripts M1-M4, M6.
- M5 needs `transformers>=4.57`, `torch`, `gradio>=4.0`.
- All CLIs MUST work from the `kd_pipeline/` working directory.
- All test fixtures MUST be synthetic (no reference to `teacher_responses.jsonl` in unit tests).
- Image paths MUST be portable — scripts MUST NOT hardcode PSC paths.
- No secrets, API keys, or hardcoded user home paths in code or data.

## 7. Interface Definitions

### 7.1 CLI contracts

| Script | Input | Output | Exit codes |
|---|---|---|---|
| `filter_cot_quality.py` | `--input <jsonl>` | `--out_pass`, `--out_fail`, `--report` | 0 ok, 1 input missing, 2 no samples passed |
| `classify_spatial_level.py` | `--samples_jsonl` OR `--question` | `--out_csv`, stdout table | 0 ok, 1 no samples, 2 schema unknown |
| `select_logit_subset.py` | `--input <jsonl>` | `--out_ids`, `--out_manifest` | 0 ok, 2 requested > available |
| `sample_error_cases.py` | `--distilled_samples`, `--teacher_samples` | `--out_csv`, `--out_summary` | 0 ok, 1 no overlap, 2 zero errors |
| `build_teacher_cache.py` | `--input <jsonl>`, `--image_root` | `--out <json>` | 0 ok |
| `app.py` | `--original_model_path`, `--distilled_model_path`, `--cache` | Gradio server on `--port` | 0 ok, 1 model load failed (unless `--no_load`) |

### 7.2 Data contracts

**`clean_train_cot.jsonl`** (M1 output): superset of input plus pass (no extra fields).
**`logit_subset_ids.txt`** (M3 output): one sample `id` per line, trailing newline.
**`logit_subset_ids.manifest.json`** (M3 companion): `{"selected": N, "chosen_by_bucket": {...}}`.
**`teacher_cache.json`** (M4 output): list of objects with `id / image / question / level / category / teacher_answer / teacher_full_response / confidence`. `image` MUST be a path relative to `--image_root` or a repo-relative path.
**`error_samples.csv`** (M6 output): columns `sample_id, level, image, question, distilled_answer, teacher_answer, error_category`.
**`per_level.csv`** (M2 output): columns `level, n, acc_eval, acc_baseline?, acc_teacher?, recovery_pct?`.

### 7.3 lmms-eval sample schema (consumed contract)

Per-sample JSONL from `lmms_eval --log_samples` is expected to contain at least:
- `doc_id` (int) — stable sample identifier
- `doc` (dict) — raw dataset row, typically with `question` / `image` / metadata keys
- `target` (str) — ground-truth answer
- `resps` (list[list[str]]) — model raw generations (outer = samples, inner = replicate)
- `filtered_resps` (list[str]) — parsed final answers
- `<metric>,none` (numeric) — correctness (e.g. `cv_bench_acc,none`)

Our scripts MUST tolerate the `,none` suffix and fall back gracefully when `doc`/`resps` are absent.

## 8. Test Contracts

### 8.1 Unit test contracts (per module)

**M1 filter_cot_quality**
- [ ] `test_extract_think_basic` — happy path
- [ ] `test_extract_think_multiline` — DOTALL regex works
- [ ] `test_extract_think_missing` — returns empty string, not None
- [ ] `test_count_spatial_keywords_word_boundary` — "behindsight" ≠ "behind"
- [ ] `test_count_pivots_distinct_patterns` — counts distinct patterns
- [ ] `test_evaluate_passes_good_sample` — all gates green
- [ ] `test_evaluate_fails_each_gate_independently` — parametrized over 4 failure modes
- [ ] `test_run_end_to_end_writes_expected_files` — pass/fail/report files
- [ ] `test_run_respects_threshold_overrides` — `FilterThresholds` customization
- [ ] `test_run_handles_empty_input` — no crash, pass_rate=0

**M2 classify_spatial_level**
- [ ] `test_classify_level_l1_l2_l3_parametrized` — ≥8 cases
- [ ] `test_classify_level_l3_wins_over_l2` — precedence
- [ ] `test_classify_level_word_boundary_safety` — no substring false positives
- [ ] `test_classify_level_empty_and_none_input` — defensive
- [ ] `test_load_samples_jsonl_autodetect` — JSONL auto-detected
- [ ] `test_load_samples_json_wrapped_variants` — 3+ wrapping shapes
- [ ] `test_load_samples_lmms_eval_log_samples_shape` — real schema (`doc_id`/`doc`/`filtered_resps`/`cv_bench_acc,none`)
- [ ] `test_extract_question_from_doc_nested` — nested `doc.question`
- [ ] `test_extract_correctness_from_metric_with_none_suffix` — `cv_bench_acc,none`
- [ ] `test_level_stats_distribution` — L1/L2/L3 counts + accs
- [ ] `test_level_stats_unknown_correctness_counted` — tracked in meta
- [ ] `test_recovery_formula_and_nan` — happy + zero-gap

**M3 select_logit_subset**
- [ ] `test_bucket_key_each_case_parametrized` — 5 buckets
- [ ] `test_select_priority_order` — fills B1 before B2
- [ ] `test_select_caps_at_request` — n capping
- [ ] `test_select_insufficient_returns_all_available`
- [ ] `test_select_deterministic_with_seed`
- [ ] `test_select_skips_rows_missing_id` — silent drop but counted in manifest

**M4 build_teacher_cache**
- [ ] `test_infer_category_parametrized` — 5 categories
- [ ] `test_extract_answer_tagged_and_untagged`
- [ ] `test_build_cache_per_category_cap`
- [ ] `test_build_cache_skips_low_confidence`
- [ ] `test_build_cache_rewrites_image_path` — with `--image_root`
- [ ] `test_build_cache_strips_absolute_cluster_paths_optionally`

**M5 demo/app**
- [ ] `test_diff_highlight_identical`
- [ ] `test_diff_highlight_different`
- [ ] `test_diff_highlight_html_escape_script_tag` — XSS prevention
- [ ] `test_diff_highlight_html_escape_angle_brackets`
- [ ] `test_find_cached_by_question_case_insensitive`
- [ ] `test_find_cached_by_image_basename`
- [ ] `test_find_cached_missing_returns_none`
- [ ] `test_load_cache_missing_file_returns_empty`
- [ ] `test_model_runner_placeholder_when_no_path` — `[not loaded]`
- [ ] `test_safe_error_message_hides_exception_details` — post-fix

**M6 sample_error_cases**
- [ ] `test_sample_id_from_id_field`
- [ ] `test_sample_id_from_doc_id_lmms_eval`
- [ ] `test_sample_id_fallback_hash`
- [ ] `test_answer_text_from_resps_nested_list`
- [ ] `test_answer_text_from_filtered_resps`
- [ ] `test_image_path_from_doc`
- [ ] `test_join_by_id_unmatched_logged_to_stderr`
- [ ] `test_stratified_sample_targets_wrong_right_only`
- [ ] `test_stratified_sample_respects_right_right_exclusion`
- [ ] `test_stratified_sample_falls_back_to_l1`
- [ ] `test_stratified_sample_deterministic`
- [ ] `test_write_csv_sanitizes_formula_injection` — `=`/`+`/`-`/`@` prefix
- [ ] `test_write_csv_expected_columns`

**Shared utilities (new)**
- [ ] `test_lmms_eval_loader_real_sample_shape`
- [ ] `test_lmms_eval_loader_metric_suffix_stripping`
- [ ] `test_html_escape_helper`
- [ ] `test_csv_cell_escape_helper`

### 8.2 Integration test contracts

- [ ] `test_e2e_m1_m3_m4_on_synthetic_50_samples` — runs 3 pipelines, validates outputs
- [ ] `test_e2e_m2_m6_on_mock_lmms_eval_samples` — uses synthetic per-sample JSONL
- [ ] `test_demo_app_ui_builds_without_model` — `--no_load` smoke

## 9. Acceptance Criteria

1. `pytest kd_pipeline/tests/` passes with ≥95 tests.
2. Coverage for `scripts/filter_cot_quality.py`, `classify_spatial_level.py`, `select_logit_subset.py`, `sample_error_cases.py`, `demo/build_teacher_cache.py`, `demo/app.py` each ≥90% line coverage.
3. `bandit -r kd_pipeline/scripts/ kd_pipeline/demo/` reports 0 HIGH / 0 MEDIUM severity.
4. Running Gradio demo with `--no_load` does not execute any user input as HTML/JS (XSS-safe).
5. CSV outputs do not start with `=`, `+`, `-`, or `@` for any model-generated cell.
6. Every CLI script exposes `--help` with usage examples.
7. `make d_pipeline` (or equivalent) runs all 6 modules on shipped fixture data end-to-end.
8. Existing 77 tests continue to pass (no regression in repo-level `pytest tests/`).

## 10. Open Questions

1. Does `eval_final.sh` need amending to add `--log_samples` so M2/M6 work? **Proposed**: yes, send PR hint in `.sdd/open_questions.md`.
2. Should D's scripts live under `kd_pipeline/scripts/d/` sub-folder for namespacing? **Proposed**: keep flat to match existing convention.
3. Internal shared module name — `src/d_utils.py`? `src/eval_io.py`? **Proposed**: split into `src/lmms_eval_io.py` + `src/safe_io.py`.
4. Do we need GitHub Actions CI for these tests? **Proposed**: out of scope; add to OPEN_QUESTIONS.
