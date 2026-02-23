from __future__ import annotations

import sys
from pathlib import Path


def _candidate_dist_dirs(repo_root: Path) -> list[Path]:
    candidates = [repo_root / "frontend" / "dist", repo_root / "dist"]

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(str(meipass))
        candidates.extend(
            [
                base / "webui_dist",
                base / "frontend" / "dist",
                base / "dist",
            ]
        )

    return candidates


def locate_frontend_dist(repo_root: Path) -> Path | None:
    for candidate in _candidate_dist_dirs(repo_root):
        index_file = candidate / "index.html"
        if index_file.exists() and index_file.is_file():
            return candidate
    return None
