"""Small argument-file helper used by the package entry point."""

from __future__ import annotations

from pathlib import Path
import shlex


def load_arg_file(path: str | Path) -> list[str]:
    """Load shell-like arguments while allowing comments and quoted values."""
    arg_path = Path(path)
    if not arg_path.exists():
        raise FileNotFoundError(f"Argument file does not exist: {arg_path}")
    return shlex.split(arg_path.read_text(encoding="utf-8"), comments=True, posix=True)

