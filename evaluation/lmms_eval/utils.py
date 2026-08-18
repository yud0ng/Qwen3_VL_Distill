"""Project-local CV-Bench parsing used through lmms-eval --include_path."""

import re
from typing import Any, Dict, List, Optional


def _extract_answer_letter(text: str) -> str:
    text = text.strip()
    match = re.match(
        r"[\(\s]*([A-Z])(?:[\)\.\s]|$)", text, flags=re.IGNORECASE
    )
    if not match:
        match = re.search(
            r"\b([A-Z])(?:[\)\.\s]|$)", text, flags=re.IGNORECASE
        )
    return match.group(1).upper() if match else ""


def _match_response_to_choice(response: str, choices: list) -> str:
    response_lower = response.lower()
    for index, choice in enumerate(choices):
        choice_text = str(choice).strip().lower()
        if choice_text and choice_text in response_lower:
            return chr(65 + index)
    return ""


def cv_bench_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    letters = ", ".join(chr(65 + i) for i in range(len(doc["choices"])))
    return kwargs.get("pre_prompt", "").format(letters) + doc["prompt"]


def cv_bench_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def cv_bench_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    response = result[0]
    prediction = _extract_answer_letter(response)
    if not prediction:
        prediction = _match_response_to_choice(response, doc.get("choices", []))
    target = doc["answer"].strip("()")
    sample = {
        "id": doc["idx"],
        "gt_content": target,
        "pred_parsed": prediction,
        "pred": response,
        "type": doc["type"],
        "task": doc["task"],
        "source": doc["source"],
        "is_correct": prediction == target,
    }
    return {"cv_bench_acc": sample}


def cv_bench_aggregate_results(results: List[Dict]) -> float:
    return (
        sum(sample["is_correct"] for sample in results) / len(results)
        if results
        else 0.0
    )
