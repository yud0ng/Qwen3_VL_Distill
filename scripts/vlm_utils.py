import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DEFAULT_PROMPT_TEMPLATE = (
    "You are an OCR assistant. Read text in the image carefully.\n"
    "Question: {question}\n"
    "Answer with the shortest exact text only."
)


def _pick_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_model_and_processor(
    model_id: str,
    device: str = "auto",
    dtype: str = "auto",
    cache_dir: str = "models/hf_cache",
) -> Tuple[object, object, str]:
    torch_dtype = _pick_dtype(dtype)
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = None
    last_error = None

    for cls in [AutoModelForImageTextToText, AutoModelForVision2Seq]:
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
            )
            break
        except Exception as exc:  # fallback to next class
            last_error = exc

    if model is None:
        raise RuntimeError(
            f"Failed to load model '{model_id}'. Last error: {last_error}"
        )

    model.to(resolved_device)
    model.eval()
    return model, processor, resolved_device


def run_single_inference(
    model,
    processor,
    image_path: str,
    question: str,
    device: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int = 64,
) -> str:
    image = Image.open(image_path).convert("RGB")
    user_prompt = prompt_template.format(question=question)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    model_inputs = processor(
        text=prompt_text,
        images=[image],
        return_tensors="pt",
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    # Some processors add helper keys that certain models do not consume in generate().
    model_inputs.pop("batch_num_images", None)
    model_inputs.pop("num_images", None)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Keep the final assistant answer only, while remaining robust across model formats.
    if "Assistant:" in decoded:
        return decoded.split("Assistant:")[-1].strip()
    if "assistant" in decoded.lower():
        lower = decoded.lower()
        idx = lower.rfind("assistant")
        return decoded[idx + len("assistant") :].lstrip(": ").strip()
    return decoded.strip()


def add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for loading model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="models/hf_cache",
        help="Local cache directory for Hugging Face model files.",
    )
