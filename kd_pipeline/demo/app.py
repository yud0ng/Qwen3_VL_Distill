#!/usr/bin/env python3
"""
Gradio 三列对比 Demo：2B 原始 / 2B 蒸馏 / 32B 参考（预缓存）。

Security (see reviews.md S1-S5 fixes):
  - All HTML content goes through src.safe_html (XSS防御)
  - Model name / path sanitized before logging (log injection防御)
  - Generic user-facing error messages; full exception logged server-side
  - cached teacher answer carried via gr.State to avoid dual-write race on teacher_box

Demo 硬件：RTX 3090 24GB（加载两个 2B 模型，合计约 8GB）

用法：
  python demo/app.py \\
      --original_model_path Qwen/Qwen3-VL-2B-Instruct \\
      --distilled_model_path ../runs/variant_a_full_sft_general/adapter_final \\
      --cache demo/teacher_cache.json \\
      --port 7860

  # UI 自检（不加载模型）
  python demo/app.py --no_load --port 7860
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safe_html import diff_wrap  # noqa: E402
from src.spatial_vocab import classify_level  # noqa: E402

logger = logging.getLogger("demo.app")


def sanitize_for_log(s: Any) -> str:
    """Strip CR/LF to prevent log injection. None -> '' ."""
    if s is None:
        return ""
    return re.sub(r"[\r\n]+", " ", str(s))


def load_cache(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("cache parse error: %s", sanitize_for_log(e))
        return []


def find_cached(
    cache: list[dict],
    question: str,
    image_hint: str | None,
) -> dict | None:
    q = (question or "").strip().lower()
    for e in cache:
        if (e.get("question") or "").strip().lower() == q:
            if not image_hint:
                return e
            if os.path.basename(str(e.get("image") or "")) == os.path.basename(image_hint):
                return e
    return None


def diff_highlight(a: str, b: str) -> tuple[str, str]:
    """XSS-safe thin wrapper around src.safe_html.diff_wrap."""
    return diff_wrap(a, b)


class ModelRunner:
    """Lazy Qwen3-VL wrapper.

    - If ``model_path`` is None → placeholder text (UI self-test mode).
    - All exceptions logged server-side; user sees generic string.
    """

    def __init__(self, name: str, model_path: str | None):
        self.name = name
        self.model_path = model_path
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        if self._model is not None or not self.model_path:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            # Operator-supplied path; demo is bound to 127.0.0.1 only. See demo/README.md
            # "Security notes" for when NOT to run this demo. nosec B615 accepted risk.
            self._processor = AutoProcessor.from_pretrained(  # nosec B615
                self.model_path, trust_remote_code=True
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(  # nosec B615
                self.model_path,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self._model.eval()
        except Exception:
            self._model = None
            logger.exception(
                "[%s] model load failed (path=%s)",
                sanitize_for_log(self.name),
                sanitize_for_log(self.model_path),
            )

    def generate(self, question: str, image_path: str | None, max_new: int = 256) -> str:
        if not self.model_path:
            return "[not loaded]"
        if self._model is None:
            self.load()
        if self._model is None:
            return "[model unavailable; see server log]"
        try:
            import torch

            user_content: list[dict[str, Any]] = []
            if image_path and os.path.exists(image_path):
                user_content.append({"type": "image", "image": image_path})
            user_content.append({"type": "text", "text": question})
            messages = [{"role": "user", "content": user_content}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images = [image_path] if image_path and os.path.exists(image_path) else None
            inputs = self._processor(
                text=[text], images=images, return_tensors="pt"
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False
                )
            gen_ids = out[:, inputs["input_ids"].shape[1]:]
            return self._processor.batch_decode(
                gen_ids, skip_special_tokens=True
            )[0].strip()
        except Exception:
            logger.exception("[%s] generation failed", sanitize_for_log(self.name))
            return "[generation failed; see server log]"


def build_ui(
    cache: list[dict],
    runner_raw: ModelRunner,
    runner_distilled: ModelRunner,
):
    """Build Gradio Blocks app. Keeps cached teacher answer in gr.State to
    avoid the on_preset / on_run race on teacher_box (C5 fix).
    """
    import gradio as gr

    preset_choices = [
        f"[{e['category']}/{e['level']}] {e['question']}" for e in cache
    ]

    def on_preset(label: str):
        idx = preset_choices.index(label) if label in preset_choices else -1
        if idx < 0:
            return "", None, "", "", ""
        e = cache[idx]
        img = e.get("image") or ""
        img_show = img if img and os.path.exists(img) else None
        teacher_ans = e.get("teacher_answer", "")
        return (
            e.get("question", ""),  # question
            img_show,               # image
            teacher_ans,            # teacher_box
            e.get("level", ""),     # level_badge
            teacher_ans,            # cached_teacher_state
        )

    def on_run(question: str, image_path: str | None, state_teacher: str):
        level = classify_level(question)
        cached = find_cached(cache, question, image_path) if cache else None
        if cached:
            teacher_out = cached.get("teacher_answer", "")
        elif state_teacher:
            teacher_out = state_teacher
        else:
            teacher_out = "[cached reference only]"
        raw_out = runner_raw.generate(question, image_path)
        distilled_out = runner_distilled.generate(question, image_path)
        raw_html, distilled_html = diff_highlight(raw_out, distilled_out)
        return level, raw_html, distilled_html, teacher_out

    with gr.Blocks(title="Qwen3-VL 2B vs Distilled vs 32B") as app:
        gr.Markdown(
            "## Qwen3-VL Distill Demo  \n"
            "2B Original · 2B Distilled · 32B Reference (cached)"
        )
        cached_teacher_state = gr.State(value="")
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=preset_choices,
                    label="Preset questions (CV-Bench spatial)",
                )
                image = gr.Image(type="filepath", label="Image")
                question = gr.Textbox(label="Question", lines=2)
                level_badge = gr.Textbox(label="Auto level (L1/L2/L3)", interactive=False)
                run_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("**2B Original**")
                raw_box = gr.HTML()
                gr.Markdown("**2B Distilled**")
                distilled_box = gr.HTML()
                gr.Markdown("**32B Reference (cached)**")
                teacher_box = gr.Textbox(lines=4, interactive=False)
        preset.change(
            on_preset,
            inputs=[preset],
            outputs=[question, image, teacher_box, level_badge, cached_teacher_state],
        )
        run_btn.click(
            on_run,
            inputs=[question, image, cached_teacher_state],
            outputs=[level_badge, raw_box, distilled_box, teacher_box],
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--distilled_model_path", type=str, default=None)
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).parent / "teacher_cache.json",
    )
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--no_load", action="store_true", help="不加载模型（仅 UI 自检）")
    return ap


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = build_parser()
    args = ap.parse_args(argv)

    cache = load_cache(args.cache)
    if not cache:
        logger.warning("empty cache at %s", sanitize_for_log(args.cache))

    raw = ModelRunner(
        "2B-original",
        None if args.no_load else args.original_model_path,
    )
    dist = ModelRunner(
        "2B-distilled",
        None if args.no_load else args.distilled_model_path,
    )
    app = build_ui(cache, raw, dist)
    app.launch(server_name=args.host, server_port=args.port)
    return 0


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
