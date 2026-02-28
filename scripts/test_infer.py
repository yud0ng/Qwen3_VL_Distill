import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vlm_utils import (
    add_common_infer_args,
    load_model_and_processor,
    run_single_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal image+text inference demo for small VLM."
    )
    parser.add_argument("--image_path", type=str, required=True, help="Input image path.")
    parser.add_argument("--question", type=str, required=True, help="OCR question.")
    add_common_infer_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model, processor, device = load_model_and_processor(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            cache_dir=args.cache_dir,
        )
        answer = run_single_inference(
            model=model,
            processor=processor,
            image_path=args.image_path,
            question=args.question,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        print(answer)
        return 0
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
