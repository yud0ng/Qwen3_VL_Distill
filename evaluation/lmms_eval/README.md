# Project-local lmms-eval task

The server upload contained a modified `lmms-eval` checkout. The repository is
not vendored here. Instead, install upstream `lmms-eval` and load this small,
portable task overlay:

```bash
python -m lmms_eval \
  --include_path evaluation/lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=/path/to/checkpoint,enable_thinking=False \
  --tasks cv_bench_forced \
  --batch_size 1 \
  --log_samples \
  --output_path logs/cv_bench
```

The overlay caps generation at eight tokens, requires a choice-letter answer,
and retains a conservative fallback for models that return choice text.
