"""Load optional YAML defaults before argparse (CLI overrides YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    yaml = None  # type: ignore
    _yaml_err = e
else:
    _yaml_err = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml") from _yaml_err
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data)}")
    return data


def merge_defaults(cli_args: dict[str, Any], yaml_path: str | None) -> dict[str, Any]:
    """CLI wins over YAML for keys present in both (non-None CLI)."""
    out = dict(cli_args)
    if not yaml_path:
        return out
    cfg = load_yaml(yaml_path)
    for k, v in cfg.items():
        if k not in out:
            out[k] = v
        elif out[k] is None:
            out[k] = v
    return out
