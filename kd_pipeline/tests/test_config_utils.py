import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_utils import load_yaml, merge_defaults


def test_load_yaml_roundtrip():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("variant: C\nlam1: 0.5\n")
        p = f.name
    try:
        d = load_yaml(p)
        assert d["variant"] == "C"
        assert d["lam1"] == 0.5
    finally:
        Path(p).unlink()


def test_merge_defaults():
    base = merge_defaults({"variant": "A", "lr": 1e-4}, None)
    assert base["variant"] == "A"
