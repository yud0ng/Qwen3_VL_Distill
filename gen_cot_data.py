"""
gen_cot_data.py
===============
Generate spatial CoT samples using Qwen3-VL-32B in thinking mode.

Builds directly on gen_teacher_data.py — reuses SYSTEM_PROMPT, load_coco_samples,
and parse_confidence without modification.  The only prompt change is appending
'<think>\\n' to the assistant prefix so the model is forced into CoT mode.

Only samples that pass all three quality filters are written to disk:

  1. Thinking chain length  ≥ --min_think_tokens  (default 30, whitespace-split)
  2. Spatial keyword density ≥ --min_density       (default 0.01 = 1 % of words)
     keywords: left / right / above / below / distance / depth /
               near / far / between / closer / farther / behind /
               in front / beside / under / over
  3. Thinking chain contains ≥ 1 pivot word:
     "therefore" | "because" | "since" | "first … then"

Output schema (one JSON per line):
  {
    "id":        "coco_cot_0000001",
    "source":    "coco_spatial",
    "type":      "metric" | "relational" | "egocentric",
    "image":     "/absolute/path/to/image.jpg",
    "question":  "...",
    "thinking":  "<think>\\n...reasoning...\\n</think>",
    "response":  "<answer>...</answer><confidence>N</confidence>",
    "bbox_gt":   "left",
    "confidence": 4,
    "think_len": 142
  }

Usage:
  python gen_cot_data.py --total 50000 --tp 4 --batch_size 16
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── reuse everything from gen_teacher_data (no modifications there) ─
sys.path.insert(0, str(Path(__file__).parent))
from gen_teacher_data import (
    SYSTEM_PROMPT,           # noqa: F401 – used inside build_prompt (via base)
    METRIC_TEMPLATES,        # noqa: F401 – used inside load_coco_samples
    RELATIONAL_TEMPLATES,    # noqa: F401 – used inside load_coco_samples
    EGOCENTRIC_TEMPLATES,    # noqa: F401 – used inside load_coco_samples
    build_prompt as _teacher_build_prompt,
    load_coco_samples,
    parse_confidence,
)

FLUSH_INTERVAL = 500


# ── Thinking-mode prompt ──────────────────────────────────────────
# Extends gen_teacher_data.build_prompt by appending '<think>\n' to the
# assistant prefix, which forces Qwen3 into CoT mode.  The rest of the
# template (system prompt, vision tokens, chat format) is unchanged.

def build_prompt(question: str) -> str:
    return _teacher_build_prompt(question) + "<think>\n"


# ── CoT quality filters ───────────────────────────────────────────

_COT_SPATIAL_KW = frozenset({
    "left", "right", "above", "below", "distance", "depth",
    "near", "far", "between", "closer", "farther", "behind",
    "in front", "beside", "under", "over",
})

# Pivot words that signal logical progression in the reasoning chain.
# "first … then" allows up to 150 chars between the two words.
_PIVOT_RE = re.compile(
    r"\b(therefore|because|since)\b"
    r"|first\b.{1,150}?\bthen\b",
    re.IGNORECASE | re.DOTALL,
)


def _strip_tags(thinking: str) -> str:
    """Remove <think> / </think> wrapper before analysis."""
    return re.sub(r"</?think>", "", thinking)


def _think_tokens(thinking: str) -> int:
    """Approximate token count via whitespace split (no tokenizer needed)."""
    return len(_strip_tags(thinking).split())


def _spatial_density(thinking: str) -> float:
    """
    Fraction of whitespace-separated words that overlap with a spatial keyword.
    Multi-word phrases ("in front", "closer", …) are matched as substrings of
    each word token so they are caught even if split differently.
    """
    inner = _strip_tags(thinking).lower()
    words = inner.split()
    if not words:
        return 0.0
    # Count each word that contains at least one keyword as a substring
    hits = sum(
        1 for w in words
        if any(kw in w for kw in _COT_SPATIAL_KW)
    )
    # Also credit phrase-level matches that span token boundaries
    for kw in _COT_SPATIAL_KW:
        if " " in kw:
            hits += inner.count(kw)
    return hits / len(words)


def is_cot_quality(
    thinking: str,
    min_tokens: int,
    min_density: float,
) -> tuple[bool, dict]:
    """
    Return (passes_all_filters, stats_dict).

    stats_dict contains per-filter diagnostics and is always returned
    regardless of outcome, so callers can aggregate rejection reasons.
    """
    n_tok    = _think_tokens(thinking)
    density  = _spatial_density(thinking)
    has_piv  = bool(_PIVOT_RE.search(thinking))

    ok_len  = n_tok   >= min_tokens
    ok_den  = density >= min_density
    ok_piv  = has_piv

    return (ok_len and ok_den and ok_piv), {
        "think_len":   n_tok,
        "density":     round(density, 4),
        "has_pivot":   has_piv,
        "fail_len":    not ok_len,
        "fail_density": not ok_den,
        "fail_pivot":  not ok_piv,
    }


# ── Parse model output into (thinking, response) ──────────────────

def parse_thinking(raw: str) -> tuple[str, str]:
    """
    The model output starts AFTER the '<think>\\n' prefix injected into the
    prompt.  Re-attach the tag so we can parse uniformly.

    Returns:
        thinking : full '<think>…</think>' block (or best-effort if unclosed)
        response : text after </think>, stripped
    """
    full = "<think>\n" + raw
    m = re.search(r"(<think>.*?</think>)\s*(.*)", full, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fallback: model never closed </think> — treat everything as thinking
    return full.strip(), ""


# ── Batch inference ───────────────────────────────────────────────

def run_batch(
    llm: LLM,
    batch: list[dict],
    sampling_params: SamplingParams,
) -> list[dict]:
    inputs, valid = [], []
    for item in batch:
        try:
            img = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {item['image']}: {e}", file=sys.stderr)
            continue
        inputs.append({
            "prompt":           build_prompt(item["question"]),
            "multi_modal_data": {"image": img},
        })
        valid.append(item)

    if not inputs:
        return []

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for item, out in zip(valid, outputs):
        raw = out.outputs[0].text.strip()
        thinking, response = parse_thinking(raw)
        results.append({**item, "thinking": thinking, "response": response})
    return results


# ── Argument parsing ──────────────────────────────────────────────

def parse_args():
    BASE = "/ocean/projects/cis220039p/yluo22"
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model_path",       default=f"{BASE}/models/qwen3-vl-32b")
    p.add_argument("--coco_dir",         default=f"{BASE}/datasets/coco/train2014")
    p.add_argument("--coco_ann",         default=f"{BASE}/datasets/coco/annotations/instances_train2014.json")
    p.add_argument("--output",           default=f"{BASE}/data/cot_responses.jsonl")
    p.add_argument("--total",            type=int,   default=50_000)
    p.add_argument("--tp",               type=int,   default=4)
    p.add_argument("--batch_size",       type=int,   default=16,
                   help="Keep small — CoT responses are long (~1-3 k tokens)")
    p.add_argument("--max_tokens",       type=int,   default=3000,
                   help="Must be large enough for thinking chain + answer")
    p.add_argument("--min_think_tokens", type=int,   default=30,
                   help="Filter: thinking chain must have ≥ this many tokens")
    p.add_argument("--min_density",      type=float, default=0.01,
                   help="Filter: spatial keyword density ≥ this fraction")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--resume",           action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    for label, path in [
        ("model_path", args.model_path),
        ("coco_dir",   args.coco_dir),
        ("coco_ann",   args.coco_ann),
    ]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load spatial samples ────────────────────────────────────
    # load_coco_samples is unchanged from gen_teacher_data; it already
    # generates bbox-grounded spatial questions (metric/relational/egocentric).
    samples = load_coco_samples(args.coco_ann, args.coco_dir, args.total)
    random.shuffle(samples)
    print(f"Spatial samples ready: {len(samples)}")

    # ── 1b. Resume ────────────────────────────────────────────────
    done_images: set[str] = set()
    resume_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_images.add(r["image"])
                    resume_count += 1
                except Exception:
                    pass
        samples = [s for s in samples if s["image"] not in done_images]
        print(f"[RESUME] {resume_count} done, {len(samples)} remaining")

    # ── 2. Init vLLM ──────────────────────────────────────────────
    print(f"\nLoading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        max_num_seqs=args.batch_size,
        mm_processor_kwargs={"min_pixels": 28*28, "max_pixels": 1280*28*28},
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_custom_all_reduce=True,
    )
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
    )

    # ── 3. Inference + filtering ───────────────────────────────────
    written   = resume_count
    sample_id = resume_count
    n_rejected = 0
    reject_reasons = {"fail_len": 0, "fail_density": 0, "fail_pivot": 0}

    out_f = open(args.output, "a" if args.resume else "w", encoding="utf-8")

    def write(item: dict) -> bool:
        """Apply quality filters; write if passing. Returns True if written."""
        nonlocal written, sample_id, n_rejected

        thinking = item.get("thinking", "")
        passes, stats = is_cot_quality(
            thinking,
            min_tokens=args.min_think_tokens,
            min_density=args.min_density,
        )

        if not passes:
            n_rejected += 1
            for k in reject_reasons:
                if stats[k]:
                    reject_reasons[k] += 1
            return False

        out_f.write(json.dumps({
            "id":         f"coco_cot_{sample_id:07d}",
            "source":     item["source"],
            "type":       item["type"],
            "image":      item["image"],
            "question":   item["question"],
            "thinking":   thinking,
            "response":   item["response"],
            "bbox_gt":    item.get("bbox_gt"),
            "confidence": parse_confidence(item["response"]),
            "think_len":  stats["think_len"],
        }, ensure_ascii=False) + "\n")
        written   += 1
        sample_id += 1
        return True

    print(
        f"\nStarting CoT inference  "
        f"total={len(samples)}  batch={args.batch_size}  "
        f"max_tokens={args.max_tokens}\n"
        f"Filters: think_len≥{args.min_think_tokens}  "
        f"density≥{args.min_density}  pivot=True"
    )
    pbar = tqdm(total=len(samples), unit="samples", dynamic_ncols=True)

    consecutive_failures = 0
    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]
        try:
            results = run_batch(llm, batch, sampling_params)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"\n[WARN] Batch {i}-{i+len(batch)} failed: {e}", file=sys.stderr)
            pbar.update(len(batch))
            if consecutive_failures >= 3:
                print(
                    f"\n[FATAL] Engine dead after {consecutive_failures} failures. "
                    f"Re-run with --resume (written={written}).",
                    file=sys.stderr,
                )
                break
            continue

        for item in results:
            write(item)

        pbar.update(len(batch))
        pbar.set_postfix(written=written, rejected=n_rejected)

        if written % FLUSH_INTERVAL == 0 and written > resume_count:
            out_f.flush()

    out_f.flush()
    out_f.close()
    pbar.close()

    # ── 4. Summary ────────────────────────────────────────────────
    total_seen = written - resume_count + n_rejected
    pass_rate  = written / total_seen * 100 if total_seen else 0

    has_think = 0
    conf_dist = {}
    len_buckets = {"<50": 0, "50-99": 0, "100-199": 0, "200-499": 0, "500+": 0}
    with open(args.output, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("thinking"):
                has_think += 1
            c = str(r.get("confidence") or "missing")
            conf_dist[c] = conf_dist.get(c, 0) + 1
            tl = r.get("think_len", 0)
            if   tl < 50:   len_buckets["<50"]     += 1
            elif tl < 100:  len_buckets["50-99"]   += 1
            elif tl < 200:  len_buckets["100-199"] += 1
            elif tl < 500:  len_buckets["200-499"] += 1
            else:           len_buckets["500+"]     += 1

    print("\n" + "=" * 60)
    print(f"Output           : {args.output}")
    print(f"Total written    : {written}  (resume base: {resume_count})")
    print(f"Rejected by filter: {n_rejected}  pass rate: {pass_rate:.1f}%")
    print(f"  ↳ too short    : {reject_reasons['fail_len']}")
    print(f"  ↳ low density  : {reject_reasons['fail_density']}")
    print(f"  ↳ no pivot word: {reject_reasons['fail_pivot']}")
    print(f"Has thinking     : {has_think}  ({has_think/written*100:.1f}%)" if written else "")
    print(f"\nThinking-chain length distribution:")
    for bucket, count in len_buckets.items():
        pct = count / written * 100 if written else 0
        print(f"  {bucket:<10}  {count:>7}  ({pct:.1f}%)")
    print(f"\nConfidence distribution:")
    for k in ["5", "4", "3", "2", "1", "missing"]:
        v   = conf_dist.get(k, 0)
        pct = v / written * 100 if written else 0
        print(f"  conf={k:<7}  {v:>7}  ({pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
