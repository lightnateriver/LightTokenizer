#!/usr/bin/env python3
"""Replay the best LoPT configurations and generate final benchmark tables."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.lopt_tokenizer import LoPTConfig, LoPTParallelTokenizer
from benchmarks.vllm_tokenizer_bench import (
    NativeBenchmarkHarness,
    insert_vllm_src,
    load_text,
    sha256_token_ids,
)


VALID_FAMILIES = ("DeepSeek-V4-Pro", "Qwen3.5")


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def round_ms(value: float) -> float:
    return round(value, 3)


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


async def replay_case(
    *,
    native: NativeBenchmarkHarness,
    lopt: LoPTParallelTokenizer,
    text: str,
    repeats: int,
    chunk_count: int,
    overlap_chars: int,
) -> dict[str, Any]:
    native_runs = []
    native_token_ids: list[int] | None = None
    for _ in range(repeats):
        metrics, native_token_ids = await native.run_once(text)
        native_runs.append(metrics)
    assert native_token_ids is not None

    lopt_runs = []
    lopt_token_ids: list[int] | None = None
    for _ in range(repeats):
        result = lopt.tokenize(
            text,
            add_special_tokens=True,
            chunk_count=chunk_count,
            overlap_chars=overlap_chars,
            chat_template_time_s=0.0,
        )
        lopt_token_ids = result.token_ids
        lopt_runs.append(result)
    assert lopt_token_ids is not None

    exact_match = native_token_ids == lopt_token_ids
    if not exact_match:
        raise AssertionError("Replay exact-match check failed.")

    native_e2e_ms = round_ms(median([run.e2e_ms for run in native_runs]))
    native_tokenizer_ms = round_ms(median([run.inner_ms for run in native_runs]))
    chat_template_time_ms = round_ms(
        median([run.chat_template_time_s * 1000.0 for run in lopt_runs])
    )
    mp_dispatch_process_collect_time_ms = round_ms(
        median(
            [run.dispatch_process_collect_time_s * 1000.0 for run in lopt_runs]
        )
    )
    chunk_dedup_time_ms = round_ms(
        median([run.chunk_dedup_time_s * 1000.0 for run in lopt_runs])
    )
    lopt_e2e_time_ms = round_ms(
        chat_template_time_ms
        + mp_dispatch_process_collect_time_ms
        + chunk_dedup_time_ms
    )
    tokenizer_speedup_x = round_ms(
        native_tokenizer_ms / mp_dispatch_process_collect_time_ms
    ) if mp_dispatch_process_collect_time_ms else None
    e2e_speedup_x = round_ms(
        native_e2e_ms / lopt_e2e_time_ms
    ) if lopt_e2e_time_ms else None
    tokenizer_time_drop_pct = round_ms(
        (1.0 - (mp_dispatch_process_collect_time_ms / native_tokenizer_ms)) * 100.0
    ) if native_tokenizer_ms else None

    return {
        "native_e2e_time_ms": native_e2e_ms,
        "native_tokenizer_time_ms": native_tokenizer_ms,
            "native_output_tokens": native_runs[-1].token_count,
            "native_token_hash": native_runs[-1].token_hash,
        "chat_template_time_ms": chat_template_time_ms,
        "mp_dispatch_process_collect_time_ms": (
            mp_dispatch_process_collect_time_ms
        ),
        "chunk_dedup_time_ms": chunk_dedup_time_ms,
            "lopt_e2e_time_ms": lopt_e2e_time_ms,
            "lopt_output_tokens": len(lopt_token_ids),
            "lopt_token_hash": sha256_token_ids(lopt_token_ids),
            "exact_match": exact_match,
        "e2e_speedup_x": e2e_speedup_x,
        "tokenizer_speedup_x": tokenizer_speedup_x,
        "tokenizer_time_drop_pct": tokenizer_time_drop_pct,
        "actual_chunk_count": lopt_runs[-1].chunk_count,
        "actual_chunk_chars": lopt_runs[-1].chunk_chars,
    }


def build_markdown(records: list[dict[str, Any]]) -> str:
    sections = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (record["tokenizer_family"], record["language"]),
            [],
        ).append(record)

    for (tokenizer_family, language), items in sorted(grouped.items()):
        items.sort(key=lambda item: item["input_chars"])
        baseline_rows = [
            [
                item["length_label"],
                item["input_chars"],
                item["native_output_tokens"],
                item["native_e2e_time_ms"],
                item["native_tokenizer_time_ms"],
            ]
            for item in items
        ]
        lopt_rows = [
            [
                item["length_label"],
                item["input_chars"],
                item["worker_processes"],
                item["chunk_count"],
                item["overlap_chars"],
                item["chat_template_time_ms"],
                item["mp_dispatch_process_collect_time_ms"],
                item["chunk_dedup_time_ms"],
                item["lopt_e2e_time_ms"],
            ]
            for item in items
        ]
        compare_rows = [
            [
                item["length_label"],
                item["native_e2e_time_ms"],
                item["native_tokenizer_time_ms"],
                item["lopt_e2e_time_ms"],
                item["mp_dispatch_process_collect_time_ms"],
                item["chunk_dedup_time_ms"],
                item["e2e_speedup_x"],
                item["tokenizer_speedup_x"],
                item["tokenizer_time_drop_pct"],
                item["exact_match"],
            ]
            for item in items
        ]
        sections.extend(
            [
                f"## {tokenizer_family} / {language.upper()} Native Baseline",
                format_markdown_table(
                    [
                        "length",
                        "input_chars",
                        "output_tokens",
                        "e2e_ms",
                        "tokenizer_ms",
                    ],
                    baseline_rows,
                ),
                "",
                f"## {tokenizer_family} / {language.upper()} LoPT",
                format_markdown_table(
                    [
                        "length",
                        "input_chars",
                        "workers",
                        "chunk_count",
                        "overlap_chars",
                        "chat_template_ms",
                        "mp_dispatch_process_collect_ms",
                        "chunk_dedup_ms",
                        "lopt_e2e_ms",
                    ],
                    lopt_rows,
                ),
                "",
                f"## {tokenizer_family} / {language.upper()} Comparison",
                format_markdown_table(
                    [
                        "length",
                        "native_e2e_ms",
                        "native_tokenizer_ms",
                        "lopt_e2e_ms",
                        "lopt_mp_ms",
                        "dedup_ms",
                        "e2e_speedup_x",
                        "tokenizer_speedup_x",
                        "tokenizer_time_drop_pct",
                        "exact_match",
                    ],
                    compare_rows,
                ),
                "",
            ]
        )
    return "\n".join(sections).strip()


async def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    best_configs = json.loads(args.best_configs_json.read_text(encoding="utf-8"))
    selected_families = set(args.families)
    selected_languages = set(args.languages)
    selected_lengths = set(args.lengths)
    best_configs = [
        record
        for record in best_configs
        if record["tokenizer_family"] in selected_families
        and record["language"] in selected_languages
        and record["length_label"] in selected_lengths
    ]
    corpus_dir = args.corpus_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for base_record in best_configs:
        tokenizer_path = Path(base_record["tokenizer_path"]).resolve()
        native = NativeBenchmarkHarness(
            model_name=base_record["model_name"],
            tokenizer_path=tokenizer_path,
            renderer_workers=args.renderer_workers,
            tokenizer_mode=base_record.get("tokenizer_mode", "hf"),
        )
        lopt = LoPTParallelTokenizer(
            LoPTConfig(
                tokenizer_path=str(tokenizer_path),
                processes=int(base_record["worker_processes"]),
                overlap_chars=int(base_record["overlap_chars"]),
                min_match_tokens=args.min_match_tokens,
                max_retry_rounds=0,
            )
        )
        try:
            text = load_text(
                corpus_dir / f"{base_record['language']}_web_corpus.txt",
                int(base_record["input_chars"]),
            )
            warmup_prompt = "warmup " * 128
            await native.warmup(warmup_prompt)
            _ = lopt.tokenize(
                warmup_prompt,
                add_special_tokens=True,
                chunk_count=max(1, min(int(base_record["worker_processes"]), 4)),
                overlap_chars=min(32, max(1, len(warmup_prompt) - 1)),
            )

            replay_metrics = await replay_case(
                native=native,
                lopt=lopt,
                text=text,
                repeats=args.repeats,
                chunk_count=int(base_record["chunk_count"]),
                overlap_chars=int(base_record["overlap_chars"]),
            )
            record = dict(base_record)
            record.update(
                {
                    "tokenizer_path": str(tokenizer_path),
                    "tokenizer_mode": base_record.get("tokenizer_mode", "hf"),
                    **replay_metrics,
                }
            )
            records.append(record)
        finally:
            lopt.close()
            native.close()

    records = sorted(
        records,
        key=lambda record: (
            record["tokenizer_family"],
            record["language"],
            record["input_chars"],
        ),
    )

    json_path = output_dir / "final_replay_results.json"
    csv_path = output_dir / "final_replay_results.csv"
    md_path = output_dir / "final_replay_tables.md"

    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "tokenizer_family",
            "model_name",
            "language",
            "length_label",
            "input_chars",
            "worker_processes",
            "chunk_count",
            "overlap_chars",
            "native_e2e_time_ms",
            "native_tokenizer_time_ms",
            "chat_template_time_ms",
            "mp_dispatch_process_collect_time_ms",
            "chunk_dedup_time_ms",
            "lopt_e2e_time_ms",
            "e2e_speedup_x",
            "tokenizer_speedup_x",
            "tokenizer_time_drop_pct",
            "native_output_tokens",
            "lopt_output_tokens",
            "exact_match",
            "native_token_hash",
            "lopt_token_hash",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    md_path.write_text(build_markdown(records) + "\n", encoding="utf-8")
    return {
        "record_count": len(records),
        "selected_families": args.families,
        "selected_languages": args.languages,
        "selected_lengths": args.lengths,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-src", type=Path, required=True)
    parser.add_argument("--best-configs-json", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(VALID_FAMILIES),
        choices=list(VALID_FAMILIES),
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh"],
        choices=["en", "zh"],
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=[
            "1k",
            "4k",
            "8k",
            "16k",
            "32k",
            "64k",
            "128k",
            "256k",
            "512k",
            "720k",
            "880k",
            "1024k",
        ],
    )
    parser.add_argument("--renderer-workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-match-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    insert_vllm_src(args.vllm_src)
    result = asyncio.run(run_replay(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
