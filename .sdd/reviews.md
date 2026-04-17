# Review Findings — D's Deliverables v1

**Date**: 2026-04-16 · **Reviewers**: code-reviewer, security-reviewer, Explore(schema)

## Summary

| Severity | Code Review | Security | Schema | Total |
|---|---|---|---|---|
| CRITICAL | 5 | 2 | 0 | **7** |
| HIGH / MEDIUM | 7 | 3 | 1 | 11 |
| INFO | 0 | 3 | 0 | 3 |

**Gate: BLOCK** — 7 criticals must be fixed before shipping contracts to teammates.

---

## CRITICAL (must fix)

### C1 — L3 keywords use bare substring match
**File**: `scripts/classify_spatial_level.py:~69`
**Bug**: `any(kw in q for kw in SPATIAL_L3)` matches "sidewalk" → L3, "research" → L3, "unapproachable" → L3.
**Fix**: Apply `\b...\b` guard to single-word keywords; keep phrase matching for multi-word keys.

### C2 — Duplicate / inconsistent spatial keyword lists
**Files**: `classify_spatial_level.py` (SPATIAL_L2) and `filter_cot_quality.py` (SPATIAL_KEYWORDS) maintained independently.
**Fix**: Extract to `src/spatial_vocab.py`; both scripts import from there.

### C3 — `count_tokens_approx` confuses words with BPE tokens
**File**: `filter_cot_quality.py:~76-77`
**Bug**: Spec says "token ≥ 30"; implementation counts whitespace-delimited words (~1.3× less than real BPE tokens). Threshold is effectively ~40 BPE tokens, stricter than intended.
**Fix**: Rename to `count_words` + update threshold name in docs; OR switch to tokenizer-based count (heavier dependency).
**Decision**: Rename + document — teammates can tune threshold rather than adding HF dep to a pure-text filter.

### C4 — `stratified_sample` mutates input via `rng.shuffle(...)` on shared refs
**File**: `scripts/sample_error_cases.py:~131,145`
**Bug**: `by_level.get("L1", [])` returns the internal list; `rng.shuffle` mutates it in place. Subsequent calls see reordered input.
**Fix**: `pool = list(by_level.get(lvl, []))` before shuffle. Per project `coding-style.md`: "ALWAYS create new objects".

### C5 — Gradio `teacher_box` dual-write race
**File**: `demo/app.py:~198-206`
**Bug**: Both `on_preset` and `on_run` write to `teacher_box`. Preset selection → Run click may race and overwrite correct cached answer with `"[cached reference only]"`.
**Fix**: Use a hidden `gr.State` to carry preset-resolved teacher answer; `on_run` reads state instead of re-querying.

### S1 — XSS via `diff_highlight` into `gr.HTML`
**File**: `demo/app.py:~55-73,193,196`
**Bug**: Model output + user question passed through `<mark>` wrapping, then rendered by `gr.HTML`. `<script>` / `<img onerror>` execute in browser.
**Fix**: `html.escape()` every chunk before wrapping; use a dedicated helper in `src/safe_html.py`.

### S2 — Full exception strings leaked to frontend
**File**: `demo/app.py:~109,142`
**Bug**: `return f"[{self.name} generate error: {e}]"` exposes filesystem paths, stack metadata, CUDA internals.
**Fix**: Generic user message; log full exception server-side via `logging`.

---

## HIGH / MEDIUM

| ID | File | Issue | Fix |
|---|---|---|---|
| H3 | `select_logit_subset.py` | Rows with missing `id` silently dropped | Count in manifest as `skipped_no_id` |
| H4 | `filter_cot_quality.py` | shallow copy on `obj` for `_filter` metadata | `copy.deepcopy` or keep separate |
| M1 | `build_teacher_cache.py` | confidence=None dropped without log | Add counter, print summary |
| M2 | `demo/app.py` | `diff_highlight` collapses whitespace | Preserve original spans or use ndiff |
| M3 | `filter_cot_quality.py` | File handles opened before try block | Move into try / ExitStack |
| S3 | `demo/app.py` | `trust_remote_code=True` on CLI-supplied path | Document risk; acceptable for local demo |
| S4 | `sample_error_cases.py` | CSV formula injection in answers | Prefix `=+-@` with `'` |
| S5 | `demo/app.py` | Log injection via unsanitized model_name/path | Strip CR/LF before logging |
| Schema-1 | (all) | `eval_final.sh` missing `--log_samples` | Patch shell scripts; document flag |

---

## Test Gaps (from code review)

- T1: No L3 substring false-positive test ("sidewalk" → L1)
- T2: No token-count vs word-count documentation test
- T3: No "missing id" test in logit subset
- T4: No `on_preset`/`on_run` interaction test
- T5: `test_run_end_to_end` doesn't verify `_filter` metadata shape
- T6: No test for `stratified_sample` idempotency on double call

---

## Refactor Candidates

| R | Change |
|---|---|
| R1 | Extract lmms-eval schema probing → `src/lmms_eval_io.py` |
| R2 | Reuse `src/teacher_responses._extract_answer_tag` in demo/build_teacher_cache (currently reimplemented) |
| R3 | Add `__init__.py` to `scripts/` or move utils to `src/` — drop `sys.path` hacks |
| R4 | Extract spatial vocab → `src/spatial_vocab.py` (consumed by M1 + M2) |
| R5 | New `src/safe_html.py` (for demo) + `src/csv_safe.py` (for M6) |

---

## Schema verification (Explore)

Confirmed lmms-eval `--log_samples` JSONL schema:
- `doc_id`, `doc`, `target`, `resps: list[list[str]]`, `filtered_resps`, `arguments`
- CV-Bench correctness: `cv_bench_acc` binary 1.0/0.0
- MMStar: `average`, plus per-category `coarse_perception`, etc.
- MME: aggregate `mme_perception_score` etc.; per-sample structure likely similar (not directly verified)

**Action**: our `_extract_question` / `_extract_correctness` / `_sample_id` schema probing is compatible. But `eval_final.sh` and `eval_lmms_eval_example.sh` do NOT currently pass `--log_samples`. Need to patch both.
