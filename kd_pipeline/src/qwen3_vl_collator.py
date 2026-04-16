"""Build Qwen3-VL multimodal inputs + labels (prompt masked -100)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass
class Qwen3VLChatCollator:
    processor: Any
    max_length: int = 512
    max_image_side: int = 448

    def _load_image(self, spec: str | dict) -> Image.Image:
        if isinstance(spec, dict) and "path" in spec:
            p = spec["path"]
        else:
            p = spec
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        img.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)
        return img

    def build_one(
        self,
        *,
        user_text: str,
        assistant_text: str,
        image_path: str | None,
    ) -> dict[str, torch.Tensor]:
        if image_path:
            img = self._load_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": user_text}]

        prompt_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_inputs.pop("token_type_ids", None)

        target_inputs = self.processor(
            text=assistant_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = torch.cat([prompt_inputs["input_ids"], target_inputs["input_ids"]], dim=-1)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        plen = prompt_inputs["input_ids"].shape[-1]
        labels[:, plen:] = target_inputs["input_ids"]

        if "mm_token_type_ids" in prompt_inputs:
            prompt_mm = prompt_inputs["mm_token_type_ids"]
            tgt_len = target_inputs["input_ids"].shape[-1]
            target_mm = torch.zeros((1, tgt_len), dtype=prompt_mm.dtype, device=prompt_mm.device)
            mm_token_type_ids = torch.cat([prompt_mm, target_mm], dim=-1)
        else:
            mm_token_type_ids = None

        if input_ids.shape[-1] > self.max_length:
            input_ids = input_ids[:, -self.max_length :]
            attention_mask = attention_mask[:, -self.max_length :]
            labels = labels[:, -self.max_length :]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[:, -self.max_length :]

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": torch.tensor([plen], dtype=torch.long),
        }
        if mm_token_type_ids is not None:
            out["mm_token_type_ids"] = mm_token_type_ids
        pv = prompt_inputs.get("pixel_values")
        if pv is not None:
            out["pixel_values"] = pv
        ig = prompt_inputs.get("image_grid_thw")
        if ig is not None:
            out["image_grid_thw"] = ig
        return out

    def build_trace_answer(
        self,
        *,
        user_text: str,
        trace: str,
        answer: str,
        image_path: str | None,
    ) -> dict[str, Any]:
        """
        assistant = trace + \"\\n\\n\" + answer（分两段 tokenize 再拼接，便于区分 trace / answer 的 CE）。
        trace 可为空（仅训 answer 段）。
        """
        sep = "\n\n"
        if trace.strip():
            tok_tr = self.processor(
                text=trace + sep,
                return_tensors="pt",
                add_special_tokens=False,
            )
            tr_ids = tok_tr["input_ids"]
            trace_tok_len = int(tr_ids.shape[-1])
        else:
            tr_ids = None
            trace_tok_len = 0

        tok_ans = self.processor(
            text=answer,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ans_ids = tok_ans["input_ids"]
        answer_tok_len = int(ans_ids.shape[-1])

        if tr_ids is not None:
            tgt_ids = torch.cat([tr_ids, ans_ids], dim=-1)
        else:
            tgt_ids = ans_ids

        if image_path:
            img = self._load_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": user_text}]

        prompt_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_inputs.pop("token_type_ids", None)

        input_ids = torch.cat([prompt_inputs["input_ids"], tgt_ids], dim=-1)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        plen = prompt_inputs["input_ids"].shape[-1]
        labels[:, plen:] = tgt_ids

        if "mm_token_type_ids" in prompt_inputs:
            prompt_mm = prompt_inputs["mm_token_type_ids"]
            tgt_len = tgt_ids.shape[-1]
            target_mm = torch.zeros((1, tgt_len), dtype=prompt_mm.dtype, device=prompt_mm.device)
            mm_token_type_ids = torch.cat([prompt_mm, target_mm], dim=-1)
        else:
            mm_token_type_ids = None

        segment_mask_valid = True
        if input_ids.shape[-1] > self.max_length:
            input_ids = input_ids[:, -self.max_length :]
            attention_mask = attention_mask[:, -self.max_length :]
            labels = labels[:, -self.max_length :]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[:, -self.max_length :]
            segment_mask_valid = False

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": torch.tensor([plen], dtype=torch.long),
            "trace_tok_len": torch.tensor([trace_tok_len], dtype=torch.long),
            "answer_tok_len": torch.tensor([answer_tok_len], dtype=torch.long),
            "segment_mask_valid": torch.tensor([segment_mask_valid]),
        }
        if mm_token_type_ids is not None:
            out["mm_token_type_ids"] = mm_token_type_ids
        pv = prompt_inputs.get("pixel_values")
        if pv is not None:
            out["pixel_values"] = pv
        ig = prompt_inputs.get("image_grid_thw")
        if ig is not None:
            out["image_grid_thw"] = ig
        return out


def _user_image_from_messages(messages: list) -> tuple[str, str | None]:
    user_txt = ""
    img = None
    for m in messages:
        if m.get("role") != "user":
            continue
        for part in m.get("content", []):
            if part.get("type") == "text":
                user_txt = part.get("text", "")
            if part.get("type") == "image":
                im = part.get("image")
                if isinstance(im, str):
                    img = im
    return user_txt, img


def row_from_jsonl(obj: dict) -> tuple[str, str, str | None]:
    """Expect keys: user (str), assistant (str), image (optional path), or OpenAI-style messages."""
    if obj.get("assistant_text") is not None and obj.get("assistant_text") != "":
        user_txt = obj.get("user") or ""
        img = obj.get("image")
        if not user_txt and "messages" in obj:
            user_txt, img_m = _user_image_from_messages(obj["messages"])
            if img is None:
                img = img_m
        return user_txt, str(obj["assistant_text"]), img
    if "messages" in obj:
        # minimal parser: first user text + image; first assistant string
        user_txt = ""
        img = None
        asst = obj.get("assistant_text") or ""
        for m in obj["messages"]:
            if m["role"] == "user":
                for part in m.get("content", []):
                    if part.get("type") == "text":
                        user_txt = part.get("text", "")
                    if part.get("type") == "image":
                        im = part.get("image")
                        if isinstance(im, str):
                            img = im
            if m["role"] == "assistant" and not asst:
                c = m.get("content", "")
                asst = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        return user_txt, asst, img
    user_txt = obj.get("user", "")
    asst = obj.get("assistant", obj.get("answer", ""))
    img = obj.get("image")
    return user_txt, str(asst), img
