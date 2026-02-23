from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
BACKEND_DIR = ROOT_DIR / "backend"

for path in (SRC_DIR, BACKEND_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

# Tests default to placeholder mode so Ollama is never required for pytest.
os.environ.setdefault("ENGINE", "placeholder")
