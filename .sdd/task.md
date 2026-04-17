# Role D Data Pipeline + Demo — Task Breakdown (TDD)

**Version**: 1.0 · 2026-04-16
**Depends on**: `.sdd/spec.md`, `.sdd/plan.md`, `.sdd/reviews.md`

All tasks use strict RED → GREEN → REFACTOR cycle. Each task names the test file to write first.

---

## Wave 1 — Shared Utilities (parallel)

### T1: `src/spatial_vocab.py`
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: none
- **Addresses**: C1, C2 (BLOCKING)
- **TDD Deliverables**:
  - **RED**: `tests/test_spatial_vocab.py` — word-boundary safety, L3>L2 precedence, parametrized over ≥12 questions incl. "sidewalk"/"research"/"unapproachable"/"reaching"
  - **GREEN**: implement `classify_level`, `count_spatial_keywords`, `count_pivot_patterns` with `\b` guards for single-word, phrase matching for multi-word
  - **REFACTOR**: extract constants, freeze tuples
- **Verify**: `pytest tests/test_spatial_vocab.py`

### T2: `src/lmms_eval_io.py`
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: none
- **Addresses**: R1 (refactor)
- **TDD Deliverables**:
  - **RED**: `tests/test_lmms_eval_io.py` — load JSON/JSONL autodetect, `,none` suffix strip, `resps` list-of-list extraction, `doc.*` nested, unknown-schema fallback
  - **GREEN**: implement `load_samples`, `extract_sample_id`, `extract_question`, `extract_correctness`, `extract_answer_text`, `extract_image_path`
  - **REFACTOR**: docstrings + type hints
- **Verify**: `pytest tests/test_lmms_eval_io.py`

### T3: `src/safe_html.py`
- **Status**: [ ] pending
- **Agent**: Gamma
- **Depends**: none
- **Addresses**: S1 (BLOCKING)
- **TDD Deliverables**:
  - **RED**: `tests/test_safe_html.py` — escape `<script>`, `<img onerror>`, preserved content, diff mark on changes, empty inputs, unicode
  - **GREEN**: implement `escape`, `diff_wrap`
  - **REFACTOR**: comment threat model inline
- **Verify**: `pytest tests/test_safe_html.py`

### T4: `src/csv_safe.py`
- **Status**: [ ] pending
- **Agent**: Gamma
- **Depends**: none
- **Addresses**: S4
- **TDD Deliverables**:
  - **RED**: `tests/test_csv_safe.py` — formula prefix `=/+/-/@/\t`, CRLF strip, benign values unchanged
  - **GREEN**: implement `escape_cell`, `sanitize_row`
- **Verify**: `pytest tests/test_csv_safe.py`

---

## Wave 2 — Script Refactors (parallel, depends on Wave 1)

### T5: Refactor `classify_spatial_level.py`
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: T1, T2
- **Addresses**: C1 + partial R1
- **TDD Deliverables**:
  - **RED**: update `tests/test_classify_spatial_level.py` — add L3 substring false-positive cases ("sidewalk", "research"), lmms-eval `,none` suffix, `resps` extraction
  - **GREEN**: delete local `SPATIAL_L2/L3` constants, `_extract_*` functions; import from `src.spatial_vocab` + `src.lmms_eval_io`
  - **REFACTOR**: move `level_stats` to `src/` for reuse (optional)
- **Verify**: `pytest tests/test_classify_spatial_level.py tests/test_spatial_vocab.py`

### T6: Refactor `filter_cot_quality.py`
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: T1
- **Addresses**: C3, H4, M3
- **TDD Deliverables**:
  - **RED**: update `tests/test_filter_cot_quality.py`:
    - rename `count_tokens_approx` → `count_words_approx` tests
    - add test for file handle opened inside try-block (simulate out_fail path permission error)
    - add test that `_filter` metadata exists on every fail row
    - add test that `obj` isn't mutated (deep copy)
  - **GREEN**:
    - replace local keyword lists with `src.spatial_vocab` imports
    - rename token→words; keep `--min_trace_tokens` CLI alias for backcompat
    - move file opens into `try`, use `ExitStack`
    - `copy.deepcopy` when adding `_filter`
  - **REFACTOR**: extract `FilterThresholds` factory from CLI args
- **Verify**: `pytest tests/test_filter_cot_quality.py`

### T7: Refactor `select_logit_subset.py`
- **Status**: [ ] pending
- **Agent**: Gamma
- **Depends**: T1
- **Addresses**: H3
- **TDD Deliverables**:
  - **RED**: update `tests/test_select_logit_subset.py`:
    - add test for row missing `id` → counted in `skipped_no_id`
    - add test verifying shuffle doesn't mutate input list
  - **GREEN**: use `list(buckets[bk])` before shuffle; add `skipped_no_id` counter; import vocab from `src.spatial_vocab`
- **Verify**: `pytest tests/test_select_logit_subset.py`

### T8: Refactor `sample_error_cases.py`
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: T2, T4
- **Addresses**: C4, S4
- **TDD Deliverables**:
  - **RED**: update `tests/test_sample_error_cases.py`:
    - add test for idempotent `stratified_sample` on same `paired` (no mutation)
    - add test for CSV formula injection: answer `"=1+1"` must become `"'=1+1"`
    - add test for `lmms-eval` schema variants (`filtered_resps`, `doc_id`)
  - **GREEN**:
    - copy pools before shuffle
    - import `src.lmms_eval_io` for extraction
    - import `src.csv_safe.escape_cell` in `write_csv`
- **Verify**: `pytest tests/test_sample_error_cases.py`

---

## Wave 3 — Demo (depends on Wave 1)

### T9: Refactor `demo/app.py`
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: T3
- **Addresses**: C5, S1, S2, S3 (accept), S5
- **TDD Deliverables**:
  - **RED**: update `tests/test_demo_app.py`:
    - XSS: question `<script>alert(1)</script>` → rendered HTML has no raw `<script>`
    - exception hiding: simulate model load failure → frontend string has no file path
    - preset→run state test: `on_preset` sets state, `on_run` reads state without re-querying
    - log injection: model name with `\n\rEVIL` → sanitized before log
  - **GREEN**:
    - `diff_highlight` → `src.safe_html.diff_wrap`
    - `gr.State` for cached teacher answer
    - Generic user error + server-side `logging.exception`
    - Strip CR/LF before logging
- **Verify**: `pytest tests/test_demo_app.py`

### T10: Refactor `demo/build_teacher_cache.py`
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: T1
- **Addresses**: M1, R2, INFO-7
- **TDD Deliverables**:
  - **RED**: new `tests/test_build_teacher_cache.py`:
    - `--image_root` rewrites `/ocean/...` → `LOCAL/...`
    - confidence=None counted in skip summary
    - reuses `src.teacher_responses.normalize_teacher_text` (assertion by monkey-patch)
  - **GREEN**:
    - delete local `extract_answer`; call `src.teacher_responses._extract_answer_tag`
    - add `--image_root` arg
    - add skipped counter to printed summary
- **Verify**: `pytest tests/test_build_teacher_cache.py`

---

## Wave 4 — Infra / Docs (parallel, depends on Wave 2+3)

### T11: eval script `--log_samples` patch
- **Status**: [ ] pending
- **Agent**: Gamma
- **Depends**: none (can be done any time)
- **Addresses**: Schema-1
- **TDD Deliverables**:
  - **RED**: `tests/test_eval_scripts_log_samples.py` — grep scripts for `--log_samples` when `LOG_SAMPLES=1`
  - **GREEN**: edit `scripts/eval_lmms_eval_example.sh` and `../eval_final.sh` to include `--log_samples` flag behind `${LOG_SAMPLES:-0}`
  - **REFACTOR**: document in `demo/README.md` and `kd_pipeline/README.md`
- **Verify**: `pytest tests/test_eval_scripts_log_samples.py`

### T12: Integration test (end-to-end)
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: T5-T10
- **TDD Deliverables**:
  - **RED**: `tests/test_e2e_d_pipeline.py` — golden-file test:
    - synth 20-sample teacher_responses jsonl (10 general + 10 spatial)
    - run M1 filter (using synthetic CoT)
    - run M3 select_logit_subset
    - run M4 build_teacher_cache
    - synth per-sample cv_bench JSONL
    - run M2 classify_spatial_level
    - run M6 sample_error_cases
    - assert every output has expected shape + row counts
  - **GREEN**: nothing (just assembly)
- **Verify**: `pytest tests/test_e2e_d_pipeline.py`

### T13: `Makefile` target + README updates
- **Status**: [ ] pending
- **Agent**: Gamma (Scribe role)
- **Depends**: T12
- **Deliverables**:
  - Append `d_pipeline` target to `Makefile`: runs fixtures + pytest
  - Update `kd_pipeline/README.md` with D's section
  - Update `demo/README.md` with security notes
- **Verify**: `make d_pipeline`

### T14: Coverage gate
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: T12
- **Deliverables**:
  - `pytest --cov=src --cov=scripts --cov=demo --cov-report=term-missing`
  - Any module <90% coverage: add targeted tests
- **Verify**: ≥90% coverage on all D modules

---

## Execution Waves

| Wave | Tasks | Agents | Gate (GREEN) |
|---|---|---|---|
| 1 | T1 T2 T3 T4 | Alpha Beta Gamma | 4 new test files pass; new `src/*.py` usable |
| 2 | T5 T6 T7 T8 | Alpha Beta Gamma Alpha | refactored scripts + updated tests pass; full suite pass |
| 3 | T9 T10 | Alpha Beta | demo tests pass incl. security |
| 4 | T11 T12 T13 T14 | Gamma Beta Gamma Alpha | e2e + coverage pass |

**Parallel launch strategy**: in this session, I execute all tasks sequentially as Dispatcher+Engineer (no external team available). Waves 1 and 2 still run test-first (RED before GREEN per task).

## Definition of Done

- [ ] 7 CRITICAL issues resolved (verified by new tests)
- [ ] ≥95 tests pass (target 103)
- [ ] ≥90% line coverage on D modules
- [ ] `bandit -r src/ scripts/ demo/` 0 HIGH/MEDIUM
- [ ] Existing 77 tests still pass (no regression)
- [ ] `.sdd/status.json` updated to `phase_5_verify`
- [ ] Completion report written to `.sdd/completion_report.md`
