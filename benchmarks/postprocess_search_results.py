#!/usr/bin/env python3
"""Rebuild best-config tables from one or more existing search_detail.jsonl files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DETAIL_FIELDNAMES = [
    "tokenizer_family",
    "model_name",
    "tokenizer_path",
    "tokenizer_mode",
    "language",
    "length_label",
    "input_chars",
    "worker_processes",
    "chunk_count",
    "chunk_chars",
    "overlap_chars",
    "chat_template_time_ms",
    "mp_dispatch_process_collect_time_ms",
    "chunk_dedup_time_ms",
    "lopt_e2e_time_ms",
    "native_e2e_time_ms",
    "native_tokenizer_time_ms",
    "native_output_tokens",
    "native_token_hash",
    "lopt_output_tokens",
    "lopt_token_hash",
    "exact_match",
    "fallback_used",
    "fallback_serial_time_ms",
    "fallback_token_hash",
    "error_message",
    "candidate_status",
    "actual_chunk_count",
]

BEST_FIELDNAMES = [
    "tokenizer_family",
    "model_name",
    "tokenizer_path",
    "tokenizer_mode",
    "language",
    "length_label",
    "input_chars",
    "worker_processes",
    "chunk_count",
    "chunk_chars",
    "overlap_chars",
    "chat_template_time_ms",
    "mp_dispatch_process_collect_time_ms",
    "chunk_dedup_time_ms",
    "lopt_e2e_time_ms",
    "native_e2e_time_ms",
    "native_tokenizer_time_ms",
    "native_output_tokens",
    "native_token_hash",
    "lopt_output_tokens",
    "lopt_token_hash",
    "exact_match",
    "fallback_used",
    "fallback_serial_time_ms",
    "fallback_token_hash",
    "error_message",
    "candidate_status",
    "actual_chunk_count",
    "e2e_speedup_x",
    "tokenizer_speedup_x",
    "tokenizer_time_drop_pct",
]


def round_ms(value: float) -> float:
    return round(value, 3)


def build_markdown(best_records: list[dict[str, Any]]) -> str:
    headers = [
        "tokenizer",
        "language",
        "input_length",
        "workers",
        "chunk_count",
        "overlap_chars",
        "native_e2e_ms",
        "native_tokenizer_ms",
        "lopt_mp_ms",
        "dedup_ms",
        "lopt_e2e_ms",
        "e2e_speedup_x",
        "tokenizer_speedup_x",
        "exact_match",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in best_records:
        lines.append(
            "| "
            + " | ".join(
                str(record.get(key, ""))
                for key in [
                    "tokenizer_family",
                    "language",
                    "length_label",
                    "worker_processes",
                    "chunk_count",
                    "overlap_chars",
                    "native_e2e_time_ms",
                    "native_tokenizer_time_ms",
                    "mp_dispatch_process_collect_time_ms",
                    "chunk_dedup_time_ms",
                    "lopt_e2e_time_ms",
                    "e2e_speedup_x",
                    "tokenizer_speedup_x",
                    "exact_match",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def enrich(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    native_e2e_time_ms = enriched["native_e2e_time_ms"]
    native_tokenizer_time_ms = enriched["native_tokenizer_time_ms"]
    lopt_e2e_time_ms = enriched["lopt_e2e_time_ms"]
    lopt_process_time_ms = enriched["mp_dispatch_process_collect_time_ms"]

    enriched["e2e_speedup_x"] = round_ms(
        native_e2e_time_ms / lopt_e2e_time_ms
    ) if lopt_e2e_time_ms else None
    enriched["tokenizer_speedup_x"] = round_ms(
        native_tokenizer_time_ms / lopt_process_time_ms
    ) if lopt_process_time_ms else None
    enriched["tokenizer_time_drop_pct"] = round_ms(
        (1.0 - (lopt_process_time_ms / native_tokenizer_time_ms)) * 100.0
    ) if native_tokenizer_time_ms else None
    return enriched


def write_csv(
    path: Path,
    fieldnames: list[str],
    records: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dir]
    default_output_dir = input_dirs[0] if len(input_dirs) == 1 else Path.cwd()
    output_dir = (args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_paths = []
    merged_failures = []
    for input_dir in input_dirs:
        detail_jsonl = input_dir / "search_detail.jsonl"
        if not detail_jsonl.exists():
            raise FileNotFoundError(detail_jsonl)
        detail_paths.append(detail_jsonl)
        failures_json = input_dir / "case_failures.json"
        if failures_json.exists():
            merged_failures.extend(
                json.loads(failures_json.read_text(encoding="utf-8"))
            )

    rows = []
    for detail_path in detail_paths:
        rows.extend(load_rows(detail_path))
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("tokenizer_family", ""),
            row.get("language", ""),
            row.get("input_chars", 0),
            row.get("worker_processes", 0),
            row.get("chunk_count", 0),
            row.get("overlap_chars", 0),
        ),
    )
    valid_rows = [
        row for row in rows
        if row.get("candidate_status") == "valid" and not row.get("fallback_used")
    ]

    worker_best_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in valid_rows:
        key = (
            row["tokenizer_family"],
            row["language"],
            row["length_label"],
            row["input_chars"],
            row["worker_processes"],
        )
        best = worker_best_map.get(key)
        if best is None or row["lopt_e2e_time_ms"] < best["lopt_e2e_time_ms"]:
            worker_best_map[key] = dict(row)

    worker_best_records = sorted(
        (enrich(row) for row in worker_best_map.values()),
        key=lambda row: (
            row["tokenizer_family"],
            row["language"],
            row["input_chars"],
            row["worker_processes"],
        ),
    )

    final_best_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in worker_best_records:
        key = (
            row["tokenizer_family"],
            row["language"],
            row["length_label"],
            row["input_chars"],
        )
        best = final_best_map.get(key)
        if best is None or row["lopt_e2e_time_ms"] < best["lopt_e2e_time_ms"]:
            final_best_map[key] = dict(row)

    final_best_records = sorted(
        final_best_map.values(),
        key=lambda row: (
            row["tokenizer_family"],
            row["language"],
            row["input_chars"],
        ),
    )

    detail_json = output_dir / "search_detail.jsonl"
    detail_csv = output_dir / "search_detail.csv"
    worker_best_json = output_dir / "worker_best_configs.json"
    worker_best_csv = output_dir / "worker_best_configs.csv"
    best_json = output_dir / "best_configs.json"
    best_csv = output_dir / "best_configs.csv"
    best_md = output_dir / "best_configs.md"
    failures_json = output_dir / "case_failures.json"

    write_jsonl(detail_json, rows)
    write_csv(detail_csv, DETAIL_FIELDNAMES, rows)

    worker_best_json.write_text(
        json.dumps(worker_best_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best_json.write_text(
        json.dumps(final_best_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(worker_best_csv, BEST_FIELDNAMES, worker_best_records)
    write_csv(best_csv, BEST_FIELDNAMES, final_best_records)
    best_md.write_text(build_markdown(final_best_records) + "\n", encoding="utf-8")
    failures_json.write_text(
        json.dumps(merged_failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "detail_rows": len(rows),
                "valid_rows": len(valid_rows),
                "input_dirs": [str(path) for path in input_dirs],
                "detail_jsonl": str(detail_json),
                "detail_csv": str(detail_csv),
                "worker_best_count": len(worker_best_records),
                "best_count": len(final_best_records),
                "worker_best_json": str(worker_best_json),
                "best_json": str(best_json),
                "best_markdown": str(best_md),
                "case_failures_json": str(failures_json),
                "case_failure_count": len(merged_failures),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
