from __future__ import annotations

from pathlib import Path
from typing import Tuple


def ensure_io(raw_dir: Path, output_dir: Path) -> Tuple[Path, Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Expected raw dataset at {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, output_dir
