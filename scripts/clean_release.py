from __future__ import annotations

import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink()
    print(f"removed: {path}")


def main() -> int:
    targets = [
        ROOT_DIR / "dist_release",
        ROOT_DIR / "build",
        ROOT_DIR / "dist",
        ROOT_DIR / "frontend" / "dist",
    ]
    for target in targets:
        _remove_path(target)
    print("release clean complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
