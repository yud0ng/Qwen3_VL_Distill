import json
import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_merge_jsonl_script():
    import subprocess

    a = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    b = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    try:
        a.write(json.dumps({"id": "1", "assistant_text": "x"}) + "\n")
        a.close()
        b.write(json.dumps({"id": "2", "assistant_text": "y"}) + "\n")
        b.close()
        out.close()
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "merge_jsonl.py"),
                "--out",
                out.name,
                "--inputs",
                f"{a.name}:cv_bench",
                f"{b.name}:llava_instruct",
            ]
        )
        lines = Path(out.name).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        o0 = json.loads(lines[0])
        o1 = json.loads(lines[1])
        assert o0["data_source"] == "cv_bench"
        assert o1["data_source"] == "llava_instruct"
    finally:
        Path(a.name).unlink(missing_ok=True)
        Path(b.name).unlink(missing_ok=True)
        Path(out.name).unlink(missing_ok=True)
