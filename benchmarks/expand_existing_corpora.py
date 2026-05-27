#!/usr/bin/env python3
"""Expand existing real-web corpora to a larger target length by cycling blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.real_web_corpus import corpus_paths, cycle_join


def split_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in text.split("\n\n")]
    return [block for block in blocks if block]


def expand_language_corpus(
    *,
    language: str,
    output_dir: Path,
    target_chars: int,
) -> Path:
    text_path, meta_path = corpus_paths(output_dir, language)
    if not text_path.exists():
        raise FileNotFoundError(f"Seed corpus does not exist: {text_path}")

    seed_text = text_path.read_text(encoding="utf-8")
    if len(seed_text) >= target_chars:
        return text_path

    blocks = split_blocks(seed_text)
    expanded = cycle_join(blocks, target_chars)
    text_path.write_text(expanded, encoding="utf-8")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "language": language,
            "target_chars": target_chars,
            "final_chars": len(expanded),
            "seed_chars": len(seed_text),
            "expanded_from_existing_seed": True,
            "expansion_block_count": len(blocks),
        }
    )
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return text_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-chars", type=int, default=1024 * 1024)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh"],
        choices=["en", "zh"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    results = []
    for language in args.languages:
        path = expand_language_corpus(
            language=language,
            output_dir=output_dir,
            target_chars=args.target_chars,
        )
        results.append(f"{language}: {path}")
    print("\n".join(results))


if __name__ == "__main__":
    main()
