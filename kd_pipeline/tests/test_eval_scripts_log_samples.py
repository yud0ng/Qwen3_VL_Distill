"""T11 · verify eval shell scripts support --log_samples for D's pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "kd_pipeline"


@pytest.mark.parametrize(
    "script_path",
    [
        PIPE_ROOT / "scripts" / "eval_lmms_eval_example.sh",
        REPO_ROOT / "eval_final.sh",
    ],
)
def test_script_mentions_log_samples_flag(script_path: Path):
    assert script_path.is_file(), f"{script_path} not found"
    text = script_path.read_text(encoding="utf-8")
    assert "--log_samples" in text, (
        f"{script_path.name} missing --log_samples flag "
        f"(required for D's classify_spatial_level and sample_error_cases)"
    )


@pytest.mark.parametrize(
    "script_path",
    [
        PIPE_ROOT / "scripts" / "eval_lmms_eval_example.sh",
        REPO_ROOT / "eval_final.sh",
    ],
)
def test_script_log_samples_gated_by_env_var(script_path: Path):
    text = script_path.read_text(encoding="utf-8")
    assert "LOG_SAMPLES" in text, (
        f"{script_path.name} should gate log_samples behind LOG_SAMPLES env var"
    )
