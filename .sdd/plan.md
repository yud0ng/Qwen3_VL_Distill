# Role D Data Pipeline + Demo — Technical Plan

**Version**: 1.0 · 2026-04-16
**Depends on**: `.sdd/spec.md` v1.0
**Incorporates**: `.sdd/reviews.md` findings (7 CRITICAL, 11 HIGH/MEDIUM)

---

## 1. Architecture Overview

One internal Python package (`kd_pipeline.src`) exposes pure utilities. CLI scripts in `kd_pipeline/scripts/` and `kd_pipeline/demo/` are thin adapters around these utilities. Tests live in `kd_pipeline/tests/`.

```
kd_pipeline/
├── src/                       # importable, testable, no side effects
│   ├── __init__.py
│   ├── spatial_vocab.py        # (NEW)  single source for L1/L2/L3 + pivot keywords
│   ├── lmms_eval_io.py         # (NEW)  schema probing: extract_question / _correctness / _sample_id / _answer / load
│   ├── safe_html.py            # (NEW)  html.escape + diff highlighting
│   ├── csv_safe.py             # (NEW)  prevent formula injection
│   ├── teacher_responses.py    # (existing)  answer/trace/confidence extraction — REUSED
│   ├── qwen3_vl_collator.py    # (existing)  unaffected
│   ├── losses.py               # (existing)  unaffected
│   └── config_utils.py         # (existing)  unaffected
├── scripts/
│   ├── classify_spatial_level.py   # imports src.spatial_vocab, src.lmms_eval_io
│   ├── filter_cot_quality.py       # imports src.spatial_vocab, src.teacher_responses
│   ├── select_logit_subset.py      # imports src.spatial_vocab
│   ├── sample_error_cases.py       # imports src.lmms_eval_io, src.csv_safe
│   └── (existing scripts unaffected)
├── demo/
│   ├── app.py                   # imports src.safe_html (XSS fix) + src.teacher_responses
│   ├── build_teacher_cache.py   # imports src.teacher_responses
│   ├── teacher_cache.json       # regenerated with safe paths
│   └── README.md
├── tests/
│   ├── test_spatial_vocab.py           (NEW)
│   ├── test_lmms_eval_io.py            (NEW)
│   ├── test_safe_html.py               (NEW)
│   ├── test_csv_safe.py                (NEW)
│   ├── test_classify_spatial_level.py  (refactored)
│   ├── test_filter_cot_quality.py      (refactored)
│   ├── test_select_logit_subset.py     (refactored)
│   ├── test_sample_error_cases.py      (refactored)
│   ├── test_demo_app.py                (refactored)
│   ├── test_build_teacher_cache.py     (NEW)
│   └── test_e2e_d_pipeline.py          (NEW integration test)
└── Makefile                      # (append d_pipeline target)
```

## 2. Technical Decisions

| Component | Choice | Rationale |
|---|---|---|
| Shared vocab | `src/spatial_vocab.py` frozen tuples | Single source, easy to diff in PR |
| Schema probing | `src/lmms_eval_io.py` module | Centralizes lmms-eval v0.7 output quirks (`,none` suffix, `resps` list-of-list) |
| HTML escape | `html.escape` stdlib (no bleach dep) | No allowlist needed — we only emit `<mark>` which we control |
| CSV safe cell | Prefix `=+-@` with `'` + strip CR/LF | Prevents Excel/Sheets formula execution and log injection |
| Token count | Rename to `count_words` + document | Avoid adding `transformers` dep to a pure-text filter |
| Import mechanism | Add `scripts/__init__.py` + convert CLI to `python -m scripts.X` **OR** keep thin `sys.path` shims | Keep shims; making scripts a package conflicts with the existing `train_distill.py` and other CLIs. Only add `kd_pipeline/src/__init__.py` and absolute imports (which already works via the `sys.path.insert(ROOT)`). |
| Gradio race fix | `gr.State` holds preset-resolved teacher answer; `on_run` reads from state or live lookup | Avoids double-query, atomic |
| Logging | `logging` stdlib to stderr in scripts; `gr.Warning` for user-facing messages | Keeps secrets out of browser |

## 3. Module Design

### 3.1 `src/spatial_vocab.py`
```python
# Exports (frozen tuples):
L3_PHRASES          # multi-word spatial phrases (egocentric / navigation)
L3_WORDS            # single-word spatial terms requiring \b guard
L2_WORDS            # relational spatial terms with \b guard
FILTER_KEYWORDS     # union used by CoT keyword density count
PIVOT_PATTERNS      # regex fragments for CoT pivot words

# Functions:
def match_l3(text: str) -> bool
def match_l2(text: str) -> bool
def classify_level(question: str) -> Literal["L1","L2","L3"]
def count_spatial_keywords(text: str) -> int
def count_pivot_patterns(text: str) -> int
```

### 3.2 `src/lmms_eval_io.py`
```python
def load_samples(path: Path) -> list[dict]     # JSON / JSONL autodetect
def extract_sample_id(sample: dict) -> str      # doc_id → id → sample_id → hash
def extract_question(sample: dict) -> str       # doc.question → question → prompt → input
def extract_correctness(sample: dict) -> bool | None  # cv_bench_acc,none → correct → metrics.acc
def extract_answer_text(sample: dict) -> str    # filtered_resps → resps[0][0] → prediction → response
def extract_image_path(sample: dict) -> str     # doc.image → image_path → input_media[0]
```

All functions robust to missing keys. Unknown samples return empty string / None; caller decides whether to count as unknown.

### 3.3 `src/safe_html.py`
```python
def escape(s: str) -> str                       # stdlib html.escape wrapper; single source
def diff_wrap(a: str, b: str) -> tuple[str,str] # XSS-safe diff highlight — ALL content escaped pre-wrap
```

### 3.4 `src/csv_safe.py`
```python
FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r", "\n")

def escape_cell(value: str) -> str  # prefix with "'" if triggers; strip \r\n
def sanitize_row(row: dict[str,str]) -> dict[str,str]
```

### 3.5 Refactored CLI scripts

Each script after refactor:
1. Imports one or more `src/*` modules (no more local duplicates).
2. CLI layer only: argparse → call pure functions → write output.
3. Has a `_main` function testable without `subprocess`.
4. Logs warnings/errors via `logging` (not `print(..., file=sys.stderr)` ad hoc).
5. Returns integer exit code defined in spec §7.1.

### 3.6 `demo/app.py` — state fix

```python
with gr.Blocks() as app:
    cached_teacher_state = gr.State(value="")   # NEW: carries preset-resolved answer

    def on_preset(label):
        e = cache[label]
        return e["question"], e["image_show"], e["teacher_answer"], e["level"], e["teacher_answer"]
                                                                              # ^ state

    def on_run(question, image, state_teacher_answer):
        # Use state if still matching; else re-query
        teacher_out = state_teacher_answer or "[cached reference only]"
        # ... generate from 2B models ...
        raw_html, distilled_html = safe_html.diff_wrap(raw_out, distilled_out)
        return level, raw_html, distilled_html, teacher_out

    preset.change(on_preset, [preset],
                  [question, image, teacher_box, level_badge, cached_teacher_state])
    run_btn.click(on_run, [question, image, cached_teacher_state],
                  [level_badge, raw_box, distilled_box, teacher_box])
```

## 4. Data Flow

```
teacher_responses.jsonl ─────┬── filter_cot_quality → clean_train_cot.jsonl  (when A rerun with thinking=True)
                             ├── select_logit_subset → logit_subset_ids.txt  (for A's gen_teacher_topk)
                             └── build_teacher_cache → demo/teacher_cache.json

lmms-eval --log_samples JSONL ──┬── classify_spatial_level → per_level.csv   (B consumes)
                                └── sample_error_cases    → errors.csv      (B/D consume)

demo/teacher_cache.json + 2B ckpt ──→ demo/app.py → Gradio UI
```

## 5. Directory Structure

See §1.

## 6. Key Implementation Details

### 6.1 Spatial vocab split
- L3 = **phrases** (always substring-matched, case-insensitive) ∪ **words** (always `\b` guarded).
- L2 = **words only**, `\b` guarded.
- CoT keyword density uses the union of L2_WORDS + selected L3_WORDS (for consistency with the filter).

### 6.2 `count_words` rename
- Drop function `count_tokens_approx`; rename to `count_words_approx`.
- Update threshold variable `MIN_TRACE_TOKENS` → `MIN_TRACE_WORDS` with default 30 (same behavior, clearer semantics).
- Backward-compat alias `--min_trace_tokens` CLI flag stays to avoid breaking any pre-existing invocations.

### 6.3 Deterministic sampling
- Every `random.Random` seeded from CLI `--seed`.
- Copy any collection before `shuffle`: `pool = list(x); rng.shuffle(pool)`.

### 6.4 HTML escape strategy
- `safe_html.diff_wrap(a, b)`: split, diff, **escape** each chunk, wrap changed chunks in `<mark>`, join with `" "`. Control chars (`\r\n\t`) stripped or replaced pre-escape.
- Test cases include `<script>alert(1)</script>`, `<img src=x onerror=y>`, zero-width and Unicode edge chars.

### 6.5 lmms-eval `--log_samples`
- Patch `scripts/eval_lmms_eval_example.sh` and `../eval_final.sh` to include `--log_samples` flag (opt-in via `LOG_SAMPLES=1` env var to avoid breaking existing runs).
- Document the expected JSONL output location in `demo/README.md` and `kd_pipeline/README.md`.

### 6.6 `build_teacher_cache.py` path handling
- New `--image_root LOCAL` arg; if given, rewrite `/ocean/...` → `LOCAL/...` using the existing `rewrite_teacher_image_paths` logic (or delegate to it).
- When `--image_root` is omitted, keep original path but warn: some entries may not render in demo.

## 7. Test Strategy

| Layer | Scope | Count target |
|---|---|---|
| Unit — src/ | new shared modules (vocab, io, html, csv) | 30 tests |
| Unit — scripts/ | refactored CLIs; all CLI error-exit paths | 40 tests |
| Unit — demo/ | app.py helpers + build_teacher_cache | 20 tests |
| Integration — e2e | run all 6 scripts on fixtures end-to-end | 5 tests |
| Security | XSS + CSV injection + log injection | 8 tests |
| **Total** | | **≥103 tests** (current: 77) |

Tests MUST pass `bandit -r scripts/ demo/ src/` with 0 HIGH/MEDIUM.

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Refactor breaks existing 77 tests | regression | Run full suite after every module |
| `sys.path` insertion order flips test imports | flaky tests | Pin shared-util paths; use absolute `from kd_pipeline.src...` only if we package; otherwise `sys.path.insert(0, str(ROOT))` at top and `from src.X import ...` |
| lmms-eval schema drift in v0.8 | brittle extractors | `extract_*` functions take an unknown-key path; surfaces via `_meta.unknown` counts |
| Gradio state API changes | demo breaks | Pin `gradio>=4.0,<5.0` in requirements |
| CSV escaping breaks downstream analysis | false positives | Document escaping convention in `error_samples.csv` header row |
| `trust_remote_code=True` risk | accepted for local demo | Document in README; recommend not binding to `0.0.0.0` |

## 9. Dependency & Backward Compatibility

- No new runtime deps for scripts M1-M4, M6. Demo keeps existing (`gradio`, `transformers`, `torch`).
- CLI arg names preserved (add new aliases, don't rename existing).
- Manifest JSON key names preserved.
- Existing `tests/test_config_utils.py`, `test_losses.py`, `test_merge_jsonl.py`, `test_teacher_responses.py` MUST continue to pass (they don't touch D's modules).

## 10. Rollback Plan

- Every refactor commit is atomic and reverts cleanly.
- If a refactor causes regression, revert one module at a time using git.
- Contract JSON/CSV output formats unchanged — downstream consumers unaffected by internal refactor.
