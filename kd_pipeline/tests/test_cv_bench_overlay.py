import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[2] / "evaluation" / "lmms_eval" / "utils.py"
)
SPEC = importlib.util.spec_from_file_location("cv_bench_overlay", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_answer_letter_accepts_short_and_verbose_answers():
    assert MODULE._extract_answer_letter("A") == "A"
    assert MODULE._extract_answer_letter("(b)") == "B"
    assert MODULE._extract_answer_letter("The answer is C.") == "C"


def test_choice_text_is_used_only_when_no_letter_is_found():
    assert MODULE._match_response_to_choice("The object is on the left", [
        "right", "left"
    ]) == "B"
