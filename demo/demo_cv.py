#!/usr/bin/env python3
"""
Webcam Spatial Reasoning — OpenCV desktop demo
Vector Robotics · 2026

Controls:
    SPACE      — capture current frame
    Any key    — type your question directly in the window
    ENTER      — submit question
    BACKSPACE  — delete last char
    Q / ESC    — quit
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
import logging
import threading

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("demo_cv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

HF_CACHE = "/media/yutao/T91/hf_cache"
MAX_DISPLAY_SIZE = 448
MAX_NEW_TOKENS = 128
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HOME"] = HF_CACHE

# ── Model registry ────────────────────────────────────────────────────────────
# "type": "base" → load directly from HF model ID or local path
# "type": "lora" → load base + apply LoRA adapter from adapter_path
#
# To add a fine-tuned variant, uncomment or add an entry below, then run:
#   python demo_cv.py --model "Variant A"
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "baseline": {
        "type": "base",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "description": "2B Baseline (no fine-tuning)",
    },
    "variant_a_plus": {
        "type": "base",  # full fine-tune (lora_r=0), loads as a standalone model
        "model_id": "/media/yutao/T91/variant_a_plus_full_sft_spatial/adapter_final",
        "description": "Variant A+ — Full SFT spatial COCO data",
    },
    # "variant_a": {
    #     "type": "base",  # full fine-tune, point to the correct dated folder
    #     "model_id": "/media/yutao/T91/variant_a_full_sft_general-<date>/variant_a_full_sft_general/adapter_final",
    #     "description": "Variant A — Full SFT general LLaVA data",
    # },
}

DEFAULT_MODEL = "baseline"

# ── Model loader ──────────────────────────────────────────────────────────────

_model = None
_processor = None
_loaded_key = None

def load_model(model_key: str = DEFAULT_MODEL):
    global _model, _processor, _loaded_key
    if _loaded_key == model_key:
        return
    cfg = MODEL_REGISTRY[model_key]
    logger.info("Loading model: %s", cfg["description"])
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _processor = AutoProcessor.from_pretrained(
        cfg["model_id"], trust_remote_code=True, cache_dir=HF_CACHE
    )
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa", cache_dir=HF_CACHE,
    )
    if cfg["type"] == "lora":
        from peft import PeftModel
        logger.info("Applying LoRA adapter: %s", cfg["adapter_path"])
        _model = PeftModel.from_pretrained(_model, cfg["adapter_path"])
        _model = _model.merge_and_unload()
    _model.eval()
    _loaded_key = model_key


def ask(pil_image: Image.Image, question: str) -> str:
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": question},
    ]}]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(_model.device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        out_ids = _model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
        )
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    response = _processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response


# ── Drawing helpers ────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_text_box(canvas, lines, x, y, w, font_scale=0.55, pad=8,
                  bg=(30, 30, 30), fg=(220, 220, 220)):
    line_h = int(font_scale * 28)
    box_h = pad * 2 + line_h * len(lines)
    cv2.rectangle(canvas, (x, y), (x + w, y + box_h), bg, -1)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (x + pad, y + pad + line_h * (i + 1) - 4),
                    FONT, font_scale, fg, 1, cv2.LINE_AA)
    return y + box_h


def draw_input_bar(canvas, prompt_text, h, w):
    bar_h = 36
    y = h - bar_h
    cv2.rectangle(canvas, (0, y), (w, h), (50, 50, 50), -1)
    cv2.putText(canvas, prompt_text + "|", (8, h - 10),
                FONT, 0.6, (0, 255, 180), 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    choices=list(MODEL_REGISTRY.keys()),
                    help="Model key from MODEL_REGISTRY (default: %(default)s)")
    ap.add_argument("--list-models", action="store_true", help="Print available models and exit")
    args = ap.parse_args()

    if args.list_models:
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:20s}  {v['description']}")
        return

    model_key = args.model
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")

    WIN = "Spatial Q&A  [SPACE=capture  ENTER=ask  Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 520)

    captured_frame = None   # BGR numpy
    typed_question = ""
    model_desc = MODEL_REGISTRY[model_key]["description"]
    answer_lines: list[str] = [f"Model: {model_desc}", "Press SPACE to capture a frame, then type your question."]
    status = "ready"        # ready | loading | thinking
    typing_mode = False     # True after capture, space = space char not capture

    # Background inference thread
    result_queue: list[str] = []

    def run_inference(pil_img, question):
        nonlocal status
        try:
            if _loaded_key != model_key:
                status = "loading"
                load_model(model_key)
            status = "thinking"
            ans = ask(pil_img, question)
            result_queue.append(ans)
        except Exception as e:
            result_queue.append(f"Error: {e}")
        status = "ready"

    while True:
        ret, live = cap.read()
        if not ret:
            continue

        h, w = live.shape[:2]
        canvas = live.copy()

        # Thumbnail of captured frame (top-right)
        if captured_frame is not None:
            th = cv2.resize(captured_frame, (180, 135))
            canvas[8:143, w - 188:w - 8] = th
            cv2.rectangle(canvas, (w - 189, 7), (w - 7, 144), (0, 220, 80), 2)
            cv2.putText(canvas, "captured", (w - 185, 158),
                        FONT, 0.45, (0, 220, 80), 1, cv2.LINE_AA)

        # Answer / status box
        if status == "loading":
            display_lines = ["Loading model... (~30s first time)"]
        elif status == "thinking":
            display_lines = ["Thinking..."]
        else:
            display_lines = answer_lines
        draw_text_box(canvas, display_lines, 8, 8, w - 210)

        # Input bar at bottom
        if typing_mode:
            hint = "[ENTER=ask  ESC=cancel]"
            prompt = f"Q: {typed_question}  {hint}"
        elif status == "ready":
            prompt = "SPACE=capture frame"
        else:
            prompt = ""
        draw_input_bar(canvas, prompt, h, w)

        cv2.imshow(WIN, canvas)

        # Poll for inference result
        if result_queue:
            raw = result_queue.pop(0)
            answer_lines = textwrap.wrap(f"A: {raw}", width=60) or ["(no response)"]
            typing_mode = False

        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue  # no key

        # ESC: cancel typing or quit
        if key == 27:
            if typing_mode:
                typing_mode = False
                typed_question = ""
            else:
                break

        elif key in (ord('q'), ord('Q')) and not typing_mode:
            break

        elif key == ord(' ') and not typing_mode:
            # Capture frame, enter typing mode
            captured_frame = live.copy()
            typed_question = ""
            typing_mode = True
            answer_lines = ["Frame captured — type question, press ENTER to ask."]

        elif key == 13:  # Enter — submit
            q = typed_question.strip()
            if not q:
                answer_lines = ["Type a question first."]
            elif captured_frame is None:
                answer_lines = ["Press SPACE to capture a frame first."]
            elif status == "ready":
                pil_img = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                pil_img.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.LANCZOS)
                typed_question = ""
                typing_mode = False
                threading.Thread(target=run_inference, args=(pil_img, q), daemon=True).start()

        elif key == 8:  # Backspace
            typed_question = typed_question[:-1]

        elif 32 <= key < 127 and typing_mode:
            typed_question += chr(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
