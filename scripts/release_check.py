from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath


SENSITIVE_BASENAMES = {".env", "settings.json"}
RUNTIME_OR_BUILD_DIRS = (
    "frontend/node_modules/",
    "frontend/dist/",
    "dist/",
    "dist_release/",
    "build/",
    "outputs/",
    "logs/",
    ".local_data/",
    ".intel-lint/",
    "intel-lint-data/",
    "cache/",
    "caches/",
)
MODEL_OR_VECTOR_DIRS = (
    "llm_models/",
    "vectorstore/",
    "vectorstores/",
    "chroma/",
    ".chroma/",
    "faiss_index/",
)
BINARY_OR_MODEL_SUFFIXES = (
    ".gguf",
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
    ".pkl",
    ".pickle",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".faiss",
    ".ann",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".zip",
)


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _git_lines(args: list[str]) -> list[str]:
    proc = _run_git(args)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _inside_git_repo() -> bool:
    proc = _run_git(["rev-parse", "--is-inside-work-tree"])
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _has_path_fragment(path: str, fragments: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/").lower()
    probe = f"/{normalized}"
    return any(probe.startswith(f"/{frag}") or f"/{frag}" in probe for frag in fragments)


def _staged_size_bytes(path: str) -> int | None:
    proc = _run_git(["cat-file", "-s", f":{path}"])
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def _check(max_bytes: int) -> list[str]:
    errors: list[str] = []

    staged_files = _git_lines(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    tracked_files = _git_lines(["ls-files"])

    forbidden_tracked = [path for path in tracked_files if PurePosixPath(path).name.lower() in SENSITIVE_BASENAMES]
    if forbidden_tracked:
        errors.append(
            "tracked sensitive config files detected: "
            + ", ".join(sorted(forbidden_tracked))
            + " (remove from Git history/index)"
        )

    tracked_runtime = [path for path in tracked_files if _has_path_fragment(path, RUNTIME_OR_BUILD_DIRS)]
    if tracked_runtime:
        errors.append(
            "tracked runtime/build artifacts detected: " + ", ".join(sorted(tracked_runtime[:20]))
        )

    for path in staged_files:
        name_lower = PurePosixPath(path).name.lower()
        path_lower = path.replace("\\", "/").lower()

        if name_lower in SENSITIVE_BASENAMES:
            errors.append(f"staged sensitive config file: {path}")

        if _has_path_fragment(path, RUNTIME_OR_BUILD_DIRS):
            errors.append(f"staged runtime/build artifact path: {path}")

        if _has_path_fragment(path, MODEL_OR_VECTOR_DIRS):
            errors.append(f"staged model/vector artifact path: {path}")

        if any(path_lower.endswith(ext) for ext in BINARY_OR_MODEL_SUFFIXES):
            errors.append(f"staged binary/model extension: {path}")

        size_bytes = _staged_size_bytes(path)
        if size_bytes is not None and size_bytes > max_bytes:
            errors.append(f"staged file too large: {path} ({size_bytes} bytes > {max_bytes})")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight release safety check for staged files.")
    parser.add_argument(
        "--max-mb",
        type=float,
        default=5.0,
        help="Maximum allowed staged file size in MB (default: 5.0).",
    )
    args = parser.parse_args()

    if args.max_mb <= 0:
        print("error: --max-mb must be > 0", file=sys.stderr)
        return 2

    if not _inside_git_repo():
        print("error: not inside a Git repository", file=sys.stderr)
        return 2

    max_bytes = int(args.max_mb * 1024 * 1024)
    try:
        errors = _check(max_bytes=max_bytes)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("RELEASE CHECK: FAIL")
        for issue in errors:
            print(f"- {issue}")
        return 1

    print("RELEASE CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
