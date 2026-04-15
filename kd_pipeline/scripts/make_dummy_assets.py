#!/usr/bin/env python3
"""Create a tiny PNG + sample JSONL for smoke tests (no external dataset)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

img_path = DATA / "dummy_space.png"
Image.new("RGB", (256, 256), color=(120, 80, 40)).save(img_path)

sample = {
    "id": "dummy_cv_bench_001",
    "data_source": "cv_bench",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {
                    "type": "text",
                    "text": 'Is the brown region to the left or right of the center? Answer briefly.',
                },
            ],
        }
    ],
    "assistant_text": "The brown region fills the frame; there is no left/right split in this image.",
}

out_jsonl = DATA / "sample_train.jsonl"
out_jsonl.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {img_path} and {out_jsonl}")
