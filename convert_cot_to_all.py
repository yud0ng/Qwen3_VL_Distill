"""Convert CoT teacher responses to the unified distillation JSONL schema.

The output can be resumed by ``gen_all.py --skip_phase1 --resume`` to add
top-k logits and hidden states without regenerating text responses.
"""

import argparse
import json
from pathlib import Path

from cot_quality import is_cot_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="CoT response JSONL")
    parser.add_argument("--output", required=True, help="Unified output JSONL")
    parser.add_argument("--min_think_tokens", type=int, default=30)
    parser.add_argument("--min_density", type=float, default=0.01)
    return parser.parse_args()


def convert(args: argparse.Namespace) -> tuple[int, int]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = passed = 0
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {input_path}:{line_number}: {exc}"
                ) from exc

            thinking = record.get("thinking") or ""
            passes, stats = is_cot_quality(
                thinking,
                min_tokens=args.min_think_tokens,
                min_density=args.min_density,
            )
            passed += int(passes)

            unified = {
                "id": record.get("id"),
                "source": record.get("source"),
                "type": record.get("type"),
                "image": record.get("image"),
                "question": record.get("question"),
                "thinking": thinking or None,
                "response": record.get("response", ""),
                "bbox_gt": record.get("bbox_gt"),
                "confidence": record.get("confidence"),
                "think_len": record.get("think_len") or stats["think_len"],
                "cot_quality": passes,
                "token_ids": None,
                "logit_probs": None,
                "teacher_hidden": None,
            }
            destination.write(json.dumps(unified, ensure_ascii=False) + "\n")
            written += 1

    return written, passed


def main() -> None:
    args = parse_args()
    written, passed = convert(args)
    rate = passed / written * 100 if written else 0.0
    print(f"Converted: {written} records")
    print(f"cot_quality=True: {passed} ({rate:.1f}%)")
    print(f"Output: {args.output}")
    print(
        "Next: python gen_all.py --resume --skip_phase1 "
        f"--output {args.output}"
    )


if __name__ == "__main__":
    main()
