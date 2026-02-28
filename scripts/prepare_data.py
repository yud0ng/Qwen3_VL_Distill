import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OCR-oriented data into unified JSON format."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "local_jsonl", "hf_dataset"],
        help="Data source mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for train/val/test/sanity json files.",
    )
    parser.add_argument(
        "--image_output_dir",
        type=str,
        default="data/images",
        help="Output directory for generated synthetic images.",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="",
        help="Local JSONL file path (used when mode=local_jsonl).",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="",
        help="Hugging Face dataset name (used when mode=hf_dataset).",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Hugging Face split name.",
    )
    parser.add_argument(
        "--id_key", type=str, default="id", help="Key for sample id."
    )
    parser.add_argument(
        "--image_key", type=str, default="image_path", help="Key for image path."
    )
    parser.add_argument(
        "--question_key", type=str, default="question", help="Key for question."
    )
    parser.add_argument(
        "--answer_key", type=str, default="answer", help="Key for answer."
    )
    parser.add_argument("--sanity_size", type=int, default=20)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def split_dataset(
    rows: List[Dict], train_size: int, val_size: int, test_size: int, sanity_size: int
) -> Dict[str, List[Dict]]:
    n = len(rows)
    sanity = rows[: min(sanity_size, n)]
    rest = rows[min(sanity_size, n) :]
    train = rest[: min(train_size, len(rest))]
    rest = rest[len(train) :]
    val = rest[: min(val_size, len(rest))]
    rest = rest[len(val) :]
    test = rest[: min(test_size, len(rest))]
    return {"sanity": sanity, "train": train, "val": val, "test": test}


def load_local_jsonl(
    input_path: Path,
    id_key: str,
    image_key: str,
    question_key: str,
    answer_key: str,
) -> List[Dict]:
    rows: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample = {
                "id": str(obj.get(id_key, f"sample_{line_idx}")),
                "image_path": str(obj[image_key]),
                "question": str(obj[question_key]),
                "answer": str(obj[answer_key]),
            }
            rows.append(sample)
    return rows


def _safe_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "EMPTY"
    return text


def build_synthetic_dataset(
    image_output_dir: Path, total_size: int, seed: int
) -> List[Dict]:
    rng = random.Random(seed)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    words = [
        "invoice",
        "total",
        "order",
        "amount",
        "receipt",
        "number",
        "date",
        "paid",
        "tax",
        "id",
    ]

    rows: List[Dict] = []
    for i in range(total_size):
        token = f"{rng.choice(words)}-{rng.randint(1000, 9999)}"
        target_text = _safe_text(token)
        image_name = f"synthetic_{i:05d}.png"
        image_path = image_output_dir / image_name

        img = Image.new("RGB", (512, 160), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle((8, 8, 504, 152), outline=(0, 0, 0), width=2)
        draw.text((24, 60), target_text, fill=(0, 0, 0))
        img.save(image_path)

        rows.append(
            {
                "id": f"syn_{i:05d}",
                "image_path": str(image_path),
                "question": "What is the exact text in the box?",
                "answer": target_text,
            }
        )
    return rows


def load_hf_dataset(
    dataset_name: str,
    split: str,
    id_key: str,
    image_key: str,
    question_key: str,
    answer_key: str,
    image_output_dir: Path,
) -> List[Dict]:
    from datasets import load_dataset

    image_output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_name, split=split)

    rows: List[Dict] = []
    for idx, item in enumerate(ds):
        sample_id = str(item.get(id_key, idx))
        question = str(item.get(question_key, "What is the text in the image?"))
        answer = item.get(answer_key, "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer)

        image_obj = item.get(image_key)
        if image_obj is None:
            continue
        image_path = image_output_dir / f"{sample_id}.png"
        image_obj.save(image_path)

        rows.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "question": question,
                "answer": answer,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    image_output_dir = Path(args.image_output_dir)

    target_total = args.sanity_size + args.train_size + args.val_size + args.test_size

    if args.mode == "synthetic":
        rows = build_synthetic_dataset(
            image_output_dir=image_output_dir,
            total_size=target_total,
            seed=args.seed,
        )
    elif args.mode == "local_jsonl":
        if not args.input_jsonl:
            raise ValueError("--input_jsonl is required when mode=local_jsonl")
        rows = load_local_jsonl(
            input_path=Path(args.input_jsonl),
            id_key=args.id_key,
            image_key=args.image_key,
            question_key=args.question_key,
            answer_key=args.answer_key,
        )
    else:
        if not args.hf_dataset_name:
            raise ValueError("--hf_dataset_name is required when mode=hf_dataset")
        rows = load_hf_dataset(
            dataset_name=args.hf_dataset_name,
            split=args.hf_split,
            id_key=args.id_key,
            image_key=args.image_key,
            question_key=args.question_key,
            answer_key=args.answer_key,
            image_output_dir=image_output_dir,
        )

    random.shuffle(rows)
    splits = split_dataset(
        rows=rows,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        sanity_size=args.sanity_size,
    )

    for split_name, split_rows in splits.items():
        write_json(output_dir / f"{split_name}.json", split_rows)
        print(f"[INFO] Wrote {split_name}.json with {len(split_rows)} samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
