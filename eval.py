import argparse
import json
import string
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR QA predictions.")
    parser.add_argument("--gt_json", type=str, required=True, help="Ground truth JSON path.")
    parser.add_argument("--pred_json", type=str, required=True, help="Prediction JSON path.")
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "anls"],
        help="Primary metric to report.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional path to save metrics JSON.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (ca != cb)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def compute_anls(pred: str, gt: str, threshold: float = 0.5) -> float:
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if not pred_norm and not gt_norm:
        return 1.0
    if not pred_norm or not gt_norm:
        return 0.0
    dist = levenshtein_distance(pred_norm, gt_norm)
    nl = dist / max(len(pred_norm), len(gt_norm))
    score = 1.0 - nl
    return score if score >= threshold else 0.0


def load_json_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def build_pred_map(pred_rows: List[Dict]) -> Dict[str, str]:
    pred_map: Dict[str, str] = {}
    for row in pred_rows:
        sid = str(row["id"])
        pred_map[sid] = str(row["prediction"])
    return pred_map


def main() -> int:
    args = parse_args()
    gt_rows = load_json_rows(Path(args.gt_json))
    pred_rows = load_json_rows(Path(args.pred_json))
    pred_map = build_pred_map(pred_rows)

    total = len(gt_rows)
    hit = 0
    anls_scores: List[float] = []

    for row in gt_rows:
        sid = str(row["id"])
        gt = str(row["answer"])
        pred = pred_map.get(sid, "")

        gt_norm = normalize_text(gt)
        pred_norm = normalize_text(pred)
        if gt_norm == pred_norm:
            hit += 1

        anls_scores.append(compute_anls(pred, gt))

    accuracy = hit / total if total else 0.0
    anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0

    primary_score = accuracy if args.metric == "accuracy" else anls

    metrics = {
        "num_samples": total,
        "num_hits": hit,
        "accuracy": accuracy,
        "anls": anls,
        "metric": args.metric,
        "score": primary_score,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Metrics saved to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
