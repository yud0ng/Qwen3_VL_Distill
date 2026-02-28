import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vlm_utils import (
    DEFAULT_PROMPT_TEMPLATE,
    add_common_infer_args,
    load_model_and_processor,
    run_single_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot inference on OCR sanity set."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="data/sanity.json",
        help="Sanity data JSON path.",
    )
    parser.add_argument(
        "--pred_out",
        type=str,
        default="results/pred_zero_shot_sanity.json",
        help="Prediction output JSON path.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="results/zero_shot_sanity_metrics.json",
        help="Metrics output JSON path.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template containing {question}.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Run at most N samples.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "anls"],
        help="Primary metric for eval.py.",
    )
    parser.add_argument(
        "--allow_dummy_on_error",
        action="store_true",
        help="If set, fall back to empty predictions when model loading/inference fails.",
    )
    add_common_infer_args(parser)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    return data


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input_json))
    rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No samples found in input_json")

    model = None
    processor = None
    device = args.device
    try:
        model, processor, device = load_model_and_processor(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        if not args.allow_dummy_on_error:
            raise
        print(f"[WARN] Model load failed, using dummy predictions: {exc}")

    preds: List[Dict] = []
    for row in tqdm(rows, desc="zero-shot"):
        if model is None or processor is None:
            pred = ""
        else:
            try:
                pred = run_single_inference(
                    model=model,
                    processor=processor,
                    image_path=str(row["image_path"]),
                    question=str(row["question"]),
                    device=device,
                    prompt_template=args.prompt_template,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                if not args.allow_dummy_on_error:
                    raise
                print(f"[WARN] Inference failed for id={row['id']}: {exc}")
                pred = ""
        preds.append({"id": str(row["id"]), "prediction": pred})

    pred_out_path = Path(args.pred_out)
    save_json(pred_out_path, preds)
    print(f"[INFO] Saved predictions to {pred_out_path}")

    metrics_out_path = Path(args.metrics_out)
    eval_script = str(PROJECT_ROOT / "eval.py")
    cmd = [
        sys.executable,
        eval_script,
        "--gt_json",
        args.input_json,
        "--pred_json",
        args.pred_out,
        "--metric",
        args.metric,
        "--save_path",
        args.metrics_out,
    ]
    subprocess.run(cmd, check=True)
    print(f"[INFO] Saved metrics to {metrics_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
