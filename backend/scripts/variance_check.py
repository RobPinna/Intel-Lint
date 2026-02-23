#!/usr/bin/env python
"""
Run N analyses against the local API and report variance in claims/bias outputs.

Usage:
  python backend/scripts/variance_check.py --text-file report.md --runs 10 --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

def load_text(path: Path | None) -> str:
    if path is None:
        data = sys.stdin.read()
    else:
        data = path.read_text(encoding="utf-8")
    return data


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only claims+bias evidence in a stable order for hashing."""
    claims = payload.get("claims", {}).get("claims", [])

    def norm_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
        ev = sorted(
            claim.get("evidence", []),
            key=lambda e: (int(e.get("start", 0)), int(e.get("end", 0)), e.get("quote", "")),
        )
        flags = sorted(
            claim.get("bias_flags", []),
            key=lambda f: (f.get("tag", ""), f.get("suggested_fix", "")),
        )
        for f in flags:
            f["evidence"] = sorted(
                f.get("evidence", []),
                key=lambda e: (int(e.get("start", 0)), int(e.get("end", 0)), e.get("quote", "")),
            )
        return {
            "claim_id": claim.get("claim_id"),
            "text": claim.get("text"),
            "score_label": claim.get("score_label"),
            "evidence": ev,
            "bias_flags": flags,
        }

    norm = {
        "claims": [norm_claim(c) for c in sorted(claims, key=lambda c: c.get("claim_id", ""))],
    }
    return norm


def hash_payload(payload: Dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_once(api_base: str, text: str, generate_rewrite: bool = False) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/analyze"
    import urllib.error
    import urllib.request

    payload = json.dumps({"text": text, "generate_rewrite": generate_rewrite}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, help="Path to report text. If omitted, read stdin.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="API base URL.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/variance"), help="Where to save run artifacts.")
    args = parser.parse_args()

    text = load_text(args.text_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    digests: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for i in range(args.runs):
        start = time.perf_counter()
        try:
            payload = run_once(args.api, text, generate_rewrite=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Run {i+1} failed: {exc}", file=sys.stderr)
            sys.exit(1)
        elapsed = time.perf_counter() - start
        norm = normalize_payload(payload)
        digest = hash_payload(norm)
        digests.append(digest)
        payloads.append(payload)
        (args.out_dir / f"run_{i+1:02d}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Run {i+1}/{args.runs}: digest={digest[:12]} time={elapsed:.2f}s")

    unique = set(digests)
    print(f"\nUnique digests: {len(unique)} of {args.runs}")
    if len(unique) > 1:
        print("Digests observed:")
        for d in sorted(unique):
            print(f"  {d}")

    # Simple diff summary: counts per bias tag and claim_id set
    def bias_counts(payload: Dict[str, Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for claim in payload.get("claims", {}).get("claims", []):
            for flag in claim.get("bias_flags", []):
                tag = flag.get("tag") or "unknown"
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    baseline_bias = bias_counts(payloads[0]) if payloads else {}
    baseline_claim_ids = sorted(c.get("claim_id") for c in payloads[0].get("claims", {}).get("claims", [])) if payloads else []

    for idx, payload in enumerate(payloads, start=1):
        bc = bias_counts(payload)
        claims_ids = sorted(c.get("claim_id") for c in payload.get("claims", {}).get("claims", []))
        if bc != baseline_bias or claims_ids != baseline_claim_ids:
            print(f"\nRun {idx} diverges:")
            if bc != baseline_bias:
                print(f"  bias counts {bc} vs baseline {baseline_bias}")
            if claims_ids != baseline_claim_ids:
                print(f"  claim ids {claims_ids} vs baseline {baseline_claim_ids}")


if __name__ == "__main__":
    main()
