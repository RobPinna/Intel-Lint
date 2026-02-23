from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    src = str(src_dir)
    if src not in sys.path:
        sys.path.insert(0, src)
