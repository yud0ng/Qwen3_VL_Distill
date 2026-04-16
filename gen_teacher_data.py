"""
gen_teacher_data.py
===================
Batch inference with Qwen3-VL-32B (vLLM offline) to generate ~50k teacher responses.

Dataset layout assumed on PSC:
  COCO images   : /ocean/projects/cis220039p/yluo22/datasets/coco/train2014/
  COCO annots   : /ocean/projects/cis220039p/yluo22/datasets/coco/annotations/instances_train2014.json
  LLaVA JSON    : /ocean/projects/cis220039p/yluo22/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json
  LLaVA images  : LLaVA reuses COCO images -> same coco/train2014/ directory
  Model         : /ocean/projects/cis220039p/yluo22/models/qwen3-vl-32b
  Output        : /ocean/projects/cis220039p/yluo22/data/teacher_responses.jsonl

Data split:
  25k spatial reasoning  <- COCO train2014 (bbox-grounded question generation)
  25k general VQA        <- LLaVA-Instruct-150k random sample

Spatial question types:
  Metric      : quantitative distance / depth estimation
  Relational  : left/right/above/below/behind/in-front-of
  Egocentric  : camera/robot-viewpoint inference (hardest, L3)

Output schema (one JSON per line):
  {
    "id":         "coco_spatial_0000001",
    "source":     "coco_spatial" | "llava_general",
    "type":       "metric" | "relational" | "egocentric" | "general",
    "image":      "/absolute/path/to/image.jpg",
    "question":   "...",
    "response":   "<answer>...</answer><confidence>N</confidence>",
    "bbox_gt":    "left",   # derived from COCO bbox coords; null for LLaVA
    "confidence": 4         # parsed int; null if missing from response
  }

Streaming flush:
  First flush at 20k written  -> unblocks filter_data.py immediately
  Then every 2k written

Usage:
  python gen_teacher_data.py \
      --model_path       /ocean/projects/cis220039p/yluo22/models/qwen3-vl-32b \
      --coco_dir         /ocean/projects/cis220039p/yluo22/datasets/coco/train2014 \
      --coco_ann         /ocean/projects/cis220039p/yluo22/datasets/coco/annotations/instances_train2014.json \
      --llava_json       /ocean/projects/cis220039p/yluo22/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json \
      --llava_image_dir  /ocean/projects/cis220039p/yluo22/datasets/coco/train2014 \
      --output           /ocean/projects/cis220039p/yluo22/data/teacher_responses.jsonl \
      --total            50000 \
      --tp               4 \
      --batch_size       256

Note on --llava_image_dir:
  LLaVA-Instruct-150k stores filenames like "COCO_train2014_000000XXXXXX.jpg".
  These images live in coco/train2014/, so --llava_image_dir defaults to the
  same path as --coco_dir.  If your copy has a separate images/ subfolder,
  point --llava_image_dir there instead.
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

# ── Constants ─────────────────────────────────────────────────────
FLUSH_EARLY_AT = 20_000   # first flush -> unblocks filter_data.py
FLUSH_INTERVAL  = 2_000   # subsequent flush cadence

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise spatial reasoning assistant. "
    "Carefully examine the image, then answer the question with concrete spatial details. "
    "End every response with this exact format: "
    "<answer>YOUR ANSWER HERE</answer><confidence>N</confidence> "
    "where N is an integer 1 (very uncertain) to 5 (very confident) "
    "about the spatial relationships in your answer."
)

# ── Spatial prompt templates ──────────────────────────────────────

METRIC_TEMPLATES = [
    "Which object appears closer to the camera, the {A} or the {B}? Use visual depth cues.",
    "Estimate the approximate distance ratio between the {A} and the {B} relative to the camera.",
    "Which is farther from the camera: the {A} or the {B}? By roughly how much?",
    "Rank these objects from nearest to farthest from the camera: {obj_list}.",
    "Is the {A} closer to the camera than the {B}? Describe the depth relationship.",
    "Which would you reach first walking toward the scene: the {A} or the {B}?",
]

RELATIONAL_TEMPLATES = [
    "Is the {A} to the left or to the right of the {B}?",
    "What object is directly above the {A} in this image?",
    "Describe the spatial arrangement: where is the {A} relative to the {B}?",
    "Is the {A} above, below, or at the same height as the {B}?",
    "What lies between the {A} and the {B}?",
    "Which object is in the top-left region of the image?",
    "Is the {A} in front of or behind the {B} in this scene?",
    "Describe the relative positions of the {A}, the {B}, and the {C}.",
    "Is the {A} closer to the left edge or the right edge of the image?",
    "Which is higher in the frame: the {A} or the {B}?",
]

EGOCENTRIC_TEMPLATES = [
    "If you were standing at the camera position, which would you reach first: the {A} or the {B}?",
    "From the camera's perspective, which direction would you turn to face the {A}?",
    "Walking straight forward from the camera into this scene, what objects would you encounter first?",
    "From your current viewpoint, is the {A} on your left side or your right side?",
    "Standing at the camera position, which object is easiest to pick up without moving?",
    "If you approached from the camera's position, which object would block your path first?",
    "From the camera's point of view, is the {A} within arm's reach or far away?",
    "Imagine you are the camera. Describe the spatial layout of the objects around you.",
]


# ── COCO loading ──────────────────────────────────────────────────

def load_coco_samples(ann_path: str, images_dir: str, n: int) -> list[dict]:
    """
    Load COCO train2014 and generate spatially-grounded questions from bounding boxes.
    Only images with >= 2 distinct object categories are used.
    bbox_gt stores a coordinate-derived ground truth for downstream validation.
    """
    print(f"Loading COCO annotations: {ann_path}")
    with open(ann_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    img_to_objs: dict[int, list] = {}
    for ann in coco["annotations"]:
        name = cat_map.get(ann["category_id"], "object")
        img_to_objs.setdefault(ann["image_id"], []).append((name, ann["bbox"]))

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    eligible = [iid for iid, objs in img_to_objs.items() if len(objs) >= 2]
    random.shuffle(eligible)

    samples = []
    for iid in eligible:
        if len(samples) >= n:
            break
        img_path = os.path.join(images_dir, id_to_file[iid])
        if not os.path.exists(img_path):
            continue

        # one representative per category
        seen, unique_objs = set(), []
        for name, bbox in img_to_objs[iid]:
            if name not in seen:
                seen.add(name)
                unique_objs.append((name, bbox))
        if len(unique_objs) < 2:
            continue

        q_type            = random.choice(["metric", "relational", "egocentric"])
        question, bbox_gt = _build_spatial_question(unique_objs, q_type)
        samples.append({
            "image":   img_path,
            "question": question,
            "type":    q_type,
            "source":  "coco_spatial",
            "bbox_gt": bbox_gt,
        })

    print(f"  -> {len(samples)} COCO spatial samples ready")
    return samples


def _build_spatial_question(objs: list[tuple], q_type: str) -> tuple[str, str]:
    """
    Instantiate a template and compute a coordinate-derived ground-truth answer.
    """
    random.shuffle(objs)
    A_name, A_bbox = objs[0]
    B_name, B_bbox = objs[1]
    C_name         = objs[2][0] if len(objs) >= 3 else A_name

    cx_A = A_bbox[0] + A_bbox[2] / 2
    cy_A = A_bbox[1] + A_bbox[3] / 2
    cx_B = B_bbox[0] + B_bbox[2] / 2

    area_A = A_bbox[2] * A_bbox[3]
    area_B = B_bbox[2] * B_bbox[3]
    closer = A_name if area_A >= area_B else B_name

    if q_type == "metric":
        tmpl = random.choice(METRIC_TEMPLATES)
        q    = tmpl.format(A=A_name, B=B_name, obj_list=", ".join(o[0] for o in objs[:4]))
        gt   = closer

    elif q_type == "relational":
        tmpl = random.choice(RELATIONAL_TEMPLATES)
        q    = tmpl.format(A=A_name, B=B_name, C=C_name) if "{C}" in tmpl \
               else tmpl.format(A=A_name, B=B_name) if "{A}" in tmpl \
               else tmpl
        gt   = "left" if cx_A < cx_B else "right"

    else:  # egocentric
        tmpl = random.choice(EGOCENTRIC_TEMPLATES)
        q    = tmpl.format(A=A_name, B=B_name) if "{A}" in tmpl and "{B}" in tmpl \
               else tmpl.format(A=A_name)       if "{A}" in tmpl \
               else tmpl
        gt   = closer

    return q, gt


# ── Spatial keyword filter ────────────────────────────────────────
# Used to exclude spatial questions from the LLaVA general split,
# keeping it truly complementary to the COCO spatial split.
# LLaVA-Instruct-150k contains ~8-12% spatial questions (GPT-4 generated);
# filtering them avoids double-counting spatial supervision.

_SPATIAL_KEYWORDS = {
    "left", "right", "above", "below", "behind", "in front",
    "closer", "farther", "distance", "between", "next to",
    "beside", "under", "over", "near", "far",
    "top", "bottom", "front", "back",
    "where is", "which side", "how far", "how close",
    "to the left", "to the right", "on top", "at the bottom",
}

def _is_spatial(question: str) -> bool:
    """Return True if the question contains spatial reasoning keywords."""
    q = question.lower()
    return any(kw in q for kw in _SPATIAL_KEYWORDS)


# ── LLaVA loading ─────────────────────────────────────────────────

def load_llava_samples(llava_json: str, images_dir: str, n: int) -> list[dict]:
    """
    Sample n non-spatial items from LLaVA-Instruct-150k.

    Spatial questions are skipped so this split stays complementary
    to the COCO spatial split (no double-counting of spatial supervision).

    Notes on image availability:
      LLaVA-Instruct-150k references three image sources:
        ~80k  COCO train2014      <- available at images_dir
        ~40k  Visual Genome       <- NOT downloaded, silently skipped
        ~30k  OCR-VQA / other     <- NOT downloaded, silently skipped
      In practice we draw from the ~80k COCO subset, which is enough.

    JSON format:
      [{"id":..., "image": "COCO_train2014_000000XXXXXX.jpg",
        "conversations": [{"from":"human","value":"<image>\nQuestion"},
                          {"from":"gpt",  "value":"Answer"}]}]
    """
    print(f"Loading LLaVA-Instruct-150k: {llava_json}")
    with open(llava_json) as f:
        data = json.load(f)

    random.shuffle(data)
    samples      = []
    n_skipped_spatial  = 0
    n_skipped_missing  = 0

    for item in data:
        if len(samples) >= n:
            break

        basename = os.path.basename(item.get("image", ""))
        img_path = os.path.join(images_dir, basename)
        if not os.path.exists(img_path):
            # LLaVA JSON stores bare IDs like "000000033471.jpg" but COCO
            # train2014 images may be named "COCO_train2014_000000033471.jpg"
            alt = os.path.join(images_dir, "COCO_train2014_" + basename)
            if os.path.exists(alt):
                img_path = alt
            else:
                n_skipped_missing += 1
                continue

        question = ""
        for turn in item.get("conversations", []):
            if turn.get("from") == "human":
                question = re.sub(r"<image>\s*", "", turn["value"]).strip()
                break
        if not question:
            continue

        # Skip spatial questions — covered by the COCO spatial split
        if _is_spatial(question):
            n_skipped_spatial += 1
            continue

        samples.append({
            "image":    img_path,
            "question": question,
            "type":     "general",
            "source":   "llava_general",
            "bbox_gt":  None,
        })

    print(f"  -> {len(samples)} LLaVA general samples ready")
    print(f"     skipped {n_skipped_spatial} spatial questions (handed to COCO split)")
    print(f"     skipped {n_skipped_missing} items (image not found — VG / OCR-VQA)")
    return samples


# ── vLLM helpers ──────────────────────────────────────────────────

def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_batch(llm: LLM, batch: list[dict], sampling_params: SamplingParams) -> list[dict]:
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
    return [{**item, "response": out.outputs[0].text.strip()}
            for item, out in zip(valid, outputs)]


def parse_confidence(response: str) -> int | None:
    m = re.search(r"<confidence>(\d)</confidence>", response)
    return int(m.group(1)) if m else None


# ── Argument parsing ──────────────────────────────────────────────

def parse_args():
    BASE = "/ocean/projects/cis220039p/yluo22"
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model_path",      default=f"{BASE}/models/qwen3-vl-32b")
    p.add_argument("--coco_dir",        default=f"{BASE}/datasets/coco/train2014",
                   help="COCO train2014 images directory")
    p.add_argument("--coco_ann",        default=f"{BASE}/datasets/coco/annotations/instances_train2014.json",
                   help="COCO instances_train2014.json")
    p.add_argument("--llava_json",      default=f"{BASE}/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json")
    p.add_argument("--llava_image_dir", default=f"{BASE}/datasets/coco/train2014",
                   help="LLaVA image directory (defaults to coco/train2014 since LLaVA reuses COCO images)")
    p.add_argument("--output",          default=f"{BASE}/data/teacher_responses.jsonl")
    p.add_argument("--total",      type=int, default=50_000)
    p.add_argument("--resume",     action="store_true",
                   help="Resume from existing output file, skipping already-processed images")
    p.add_argument("--tp",         type=int, default=4,   help="tensor_parallel_size")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    # Fail fast on missing paths
    for label, path in [
        ("model_path",     args.model_path),
        ("coco_dir",       args.coco_dir),
        ("coco_ann",       args.coco_ann),
        ("llava_json",     args.llava_json),
        ("llava_image_dir",args.llava_image_dir),
    ]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Prepare samples ────────────────────────────────────────
    n_spatial = args.total // 2
    n_general = args.total - n_spatial

    spatial = load_coco_samples(args.coco_ann, args.coco_dir, n_spatial)
    general = load_llava_samples(args.llava_json, args.llava_image_dir, n_general)

    all_samples = spatial + general
    random.shuffle(all_samples)
    print(f"\nTotal: {len(spatial)} spatial + {len(general)} general = {len(all_samples)}")

    # ── 1b. Resume: skip already-processed images ─────────────────
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
        all_samples = [s for s in all_samples if s["image"] not in done_images]
        print(f"[RESUME] {resume_count} already done, {len(all_samples)} remaining")

    # ── 2. Init vLLM ─────────────────────────────────────────────
    print(f"\nLoading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        max_num_seqs=args.batch_size,
        mm_processor_kwargs={"min_pixels": 28*28, "max_pixels": 1280*28*28},
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
    )

    # ── 3. Batch inference + streaming flush ──────────────────────
    written       = resume_count
    sample_id     = resume_count
    early_flushed = written >= FLUSH_EARLY_AT

    out_f = open(args.output, "a" if args.resume else "w", encoding="utf-8")

    def write(item: dict):
        nonlocal written, sample_id
        out_f.write(json.dumps({
            "id":         f"{item['source']}_{sample_id:07d}",
            "source":     item["source"],
            "type":       item["type"],
            "image":      item["image"],
            "question":   item["question"],
            "response":   item["response"],
            "bbox_gt":    item.get("bbox_gt"),
            "confidence": parse_confidence(item["response"]),
        }, ensure_ascii=False) + "\n")
        written   += 1
        sample_id += 1

    print(f"\nStarting inference  total={len(all_samples)}  batch={args.batch_size}")
    pbar = tqdm(total=len(all_samples), unit="samples", dynamic_ncols=True)

    for i in range(0, len(all_samples), args.batch_size):
        batch = all_samples[i : i + args.batch_size]
        try:
            results = run_batch(llm, batch, sampling_params)
        except Exception as e:
            print(f"\n[WARN] Batch {i}-{i+len(batch)} failed: {e}", file=sys.stderr)
            pbar.update(len(batch))
            continue

        for item in results:
            write(item)

        pbar.update(len(batch))
        pbar.set_postfix(written=written)

        if not early_flushed and written >= FLUSH_EARLY_AT:
            out_f.flush()
            early_flushed = True
            print(f"\n[FLUSH] {written} samples written -> filter_data.py can start now.")
        elif early_flushed and written % FLUSH_INTERVAL == 0:
            out_f.flush()

    out_f.flush()
    out_f.close()
    pbar.close()

    # ── 4. Summary ───────────────────────────────────────────────
    conf_dist, type_dist = {}, {}
    with open(args.output) as f:
        for line in f:
            r = json.loads(line)
            c = str(r.get("confidence") or "missing")
            t = r.get("type", "unknown")
            conf_dist[c] = conf_dist.get(c, 0) + 1
            type_dist[t] = type_dist.get(t, 0) + 1

    print("\n" + "=" * 54)
    print(f"Output       : {args.output}")
    print(f"Total written: {written}")
    print(f"\nType breakdown:")
    for k, v in sorted(type_dist.items()):
        print(f"  {k:<14}  {v:>7}")
    print(f"\nConfidence distribution (pre-filter):")
    for k in ["5", "4", "3", "2", "1", "missing"]:
        v   = conf_dist.get(k, 0)
        pct = v / written * 100 if written else 0
        print(f"  conf={k:<7}  {v:>7}  ({pct:.1f}%)")
    hi = conf_dist.get("5", 0) + conf_dist.get("4", 0)
    print(f"\nSamples conf>=4 (will pass filter): {hi}  ({hi/written*100:.1f}%)")
    print("=" * 54)


if __name__ == "__main__":
    main()
