import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

from cot_quality import is_cot_quality


def test_quality_filter_accepts_spatial_reasoning_with_a_pivot():
    thinking = (
        "<think>First the left object is closer to the camera, "
        "then the right object appears farther away.</think>"
    )
    passed, stats = is_cot_quality(thinking, min_tokens=8, min_density=0.1)
    assert passed
    assert stats["has_pivot"]


def test_quality_filter_reports_each_failure_reason():
    passed, stats = is_cot_quality("<think>uncertain</think>", 8, 0.1)
    assert not passed
    assert stats["fail_len"]
    assert stats["fail_density"]
    assert stats["fail_pivot"]
