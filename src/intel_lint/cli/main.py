from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..core.engine import analyze_with_selected_engine
from ..io.outputs import write_latest_outputs
from ..models.schemas import AnalyzeRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intel-lint",
        description="Analyze CTI text and export claims.json, annotated.md, and rewrite.md",
    )
    parser.add_argument("input", help="Path to an input .txt or .md file")
    parser.add_argument(
        "--out",
        default="outputs/latest",
        help="Output directory for generated files (default: outputs/latest)",
    )
    parser.add_argument(
        "--engine",
        choices=("placeholder", "ollama"),
        default=None,
        help="Optional engine override for this command",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable rewrite generation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: unable to read input file: {exc}", file=sys.stderr)
        return 2

    if not text.strip():
        print("error: input file is empty", file=sys.stderr)
        return 2

    if args.engine:
        os.environ["ENGINE"] = args.engine

    request = AnalyzeRequest(
        text=text,
        sample_name=input_path.name,
        generate_rewrite=not args.no_rewrite,
    )

    try:
        response = analyze_with_selected_engine(request)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.out).resolve()
    write_latest_outputs(response, output_dir)
    print(f"wrote outputs to {output_dir}")
    print(f"claims={len(response.claims.claims)} engine={response.claims.engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
