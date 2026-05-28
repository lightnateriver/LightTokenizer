#!/usr/bin/env python3
"""Summarize fixed-config v2 compare outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def round3(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def safe_speedup(reference_ms: float | None, candidate_ms: float | None) -> float | None:
    if reference_ms in (None, 0) or candidate_ms in (None, 0):
        return None
    return round(reference_ms / candidate_ms, 3)


def safe_delta(reference_ms: float | None, candidate_ms: float | None) -> float | None:
    if reference_ms is None or candidate_ms is None:
        return None
    return round(candidate_ms - reference_ms, 3)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 3)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def count_true(values: list[bool]) -> int:
    return sum(1 for value in values if value)


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["v2_vs_v1_mp_speedup_x"] = safe_speedup(
        row.get("v1_mp_time_ms"),
        row.get("v2_mp_time_ms"),
    )
    enriched["v2_vs_v1_mp_delta_ms"] = safe_delta(
        row.get("v1_mp_time_ms"),
        row.get("v2_mp_time_ms"),
    )
    enriched["v2_vs_v1_dedup_speedup_x"] = safe_speedup(
        row.get("v1_dedup_time_ms"),
        row.get("v2_dedup_time_ms"),
    )
    enriched["v2_vs_v1_dedup_delta_ms"] = safe_delta(
        row.get("v1_dedup_time_ms"),
        row.get("v2_dedup_time_ms"),
    )
    enriched["v2_vs_native_tokenizer_speedup_x"] = safe_speedup(
        row.get("native_tokenizer_time_ms"),
        row.get("v2_mp_time_ms"),
    )
    enriched["v2_vs_native_tokenizer_delta_ms"] = safe_delta(
        row.get("native_tokenizer_time_ms"),
        row.get("v2_mp_time_ms"),
    )
    enriched["v2_vs_native_e2e_speedup_x"] = safe_speedup(
        row.get("native_e2e_time_ms"),
        row.get("v2_e2e_time_ms"),
    )
    enriched["v2_vs_native_e2e_delta_ms"] = safe_delta(
        row.get("native_e2e_time_ms"),
        row.get("v2_e2e_time_ms"),
    )
    enriched["v1_vs_native_tokenizer_speedup_x"] = safe_speedup(
        row.get("native_tokenizer_time_ms"),
        row.get("v1_mp_time_ms"),
    )
    enriched["v1_vs_native_e2e_speedup_x"] = safe_speedup(
        row.get("native_e2e_time_ms"),
        row.get("v1_e2e_time_ms"),
    )
    return enriched


def collect_metric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "v1_exact_match_cases": count_true([bool(row.get("v1_exact_match")) for row in rows]),
        "v2_exact_match_cases": count_true([bool(row.get("v2_exact_match")) for row in rows]),
        "all_v1_exact_match": all(bool(row.get("v1_exact_match")) for row in rows),
        "all_v2_exact_match": all(bool(row.get("v2_exact_match")) for row in rows),
        "v2_better_than_v1_e2e_cases": count_true([
            row.get("v2_e2e_time_ms") is not None
            and row.get("v1_e2e_time_ms") is not None
            and row["v2_e2e_time_ms"] < row["v1_e2e_time_ms"]
            for row in rows
        ]),
        "v2_better_than_v1_mp_cases": count_true([
            row.get("v2_mp_time_ms") is not None
            and row.get("v1_mp_time_ms") is not None
            and row["v2_mp_time_ms"] < row["v1_mp_time_ms"]
            for row in rows
        ]),
        "v2_better_than_v1_dedup_cases": count_true([
            row.get("v2_dedup_time_ms") is not None
            and row.get("v1_dedup_time_ms") is not None
            and row["v2_dedup_time_ms"] < row["v1_dedup_time_ms"]
            for row in rows
        ]),
        "avg_v2_vs_v1_e2e_speedup_x": average(collect_metric(rows, "v2_vs_v1_e2e_speedup_x")),
        "median_v2_vs_v1_e2e_speedup_x": median(collect_metric(rows, "v2_vs_v1_e2e_speedup_x")),
        "avg_v2_vs_v1_mp_speedup_x": average(collect_metric(rows, "v2_vs_v1_mp_speedup_x")),
        "median_v2_vs_v1_mp_speedup_x": median(collect_metric(rows, "v2_vs_v1_mp_speedup_x")),
        "avg_v2_vs_v1_dedup_speedup_x": average(collect_metric(rows, "v2_vs_v1_dedup_speedup_x")),
        "median_v2_vs_v1_dedup_speedup_x": median(collect_metric(rows, "v2_vs_v1_dedup_speedup_x")),
        "avg_v2_vs_native_tokenizer_speedup_x": average(
            collect_metric(rows, "v2_vs_native_tokenizer_speedup_x")
        ),
        "median_v2_vs_native_tokenizer_speedup_x": median(
            collect_metric(rows, "v2_vs_native_tokenizer_speedup_x")
        ),
        "avg_v2_vs_native_e2e_speedup_x": average(
            collect_metric(rows, "v2_vs_native_e2e_speedup_x")
        ),
        "median_v2_vs_native_e2e_speedup_x": median(
            collect_metric(rows, "v2_vs_native_e2e_speedup_x")
        ),
        "avg_v2_vs_v1_e2e_delta_ms": average(collect_metric(rows, "v2_vs_v1_e2e_delta_ms")),
        "avg_v2_vs_v1_mp_delta_ms": average(collect_metric(rows, "v2_vs_v1_mp_delta_ms")),
        "avg_v2_vs_v1_dedup_delta_ms": average(collect_metric(rows, "v2_vs_v1_dedup_delta_ms")),
        "avg_v2_vs_native_tokenizer_delta_ms": average(
            collect_metric(rows, "v2_vs_native_tokenizer_delta_ms")
        ),
        "avg_v2_vs_native_e2e_delta_ms": average(
            collect_metric(rows, "v2_vs_native_e2e_delta_ms")
        ),
    }


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def build_markdown(rows: list[dict[str, Any]], overall: dict[str, Any]) -> str:
    sections = []
    sections.append("# v2 Stage1 Full Compare Summary")
    sections.append("")
    sections.append("## Overall")
    sections.append("")
    overall_rows = [
        ["case_count", overall["case_count"]],
        ["all_v1_exact_match", overall["all_v1_exact_match"]],
        ["all_v2_exact_match", overall["all_v2_exact_match"]],
        ["v2_better_than_v1_e2e_cases", overall["v2_better_than_v1_e2e_cases"]],
        ["v2_better_than_v1_mp_cases", overall["v2_better_than_v1_mp_cases"]],
        ["v2_better_than_v1_dedup_cases", overall["v2_better_than_v1_dedup_cases"]],
        ["avg_v2_vs_v1_e2e_speedup_x", overall["avg_v2_vs_v1_e2e_speedup_x"]],
        ["avg_v2_vs_v1_mp_speedup_x", overall["avg_v2_vs_v1_mp_speedup_x"]],
        ["avg_v2_vs_v1_dedup_speedup_x", overall["avg_v2_vs_v1_dedup_speedup_x"]],
        ["avg_v2_vs_native_tokenizer_speedup_x", overall["avg_v2_vs_native_tokenizer_speedup_x"]],
        ["avg_v2_vs_native_e2e_speedup_x", overall["avg_v2_vs_native_e2e_speedup_x"]],
    ]
    sections.append(format_markdown_table(["metric", "value"], overall_rows))
    sections.append("")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["tokenizer_family"], row["language"]), []).append(row)

    sections.append("## Group Summary")
    sections.append("")
    group_headers = [
        "family",
        "language",
        "cases",
        "all_exact",
        "avg_v2_vs_v1_e2e_x",
        "avg_v2_vs_v1_mp_x",
        "avg_v2_vs_v1_dedup_x",
        "avg_v2_vs_native_tok_x",
        "avg_v2_vs_native_e2e_x",
    ]
    group_rows = []
    for (family, language), items in sorted(grouped.items()):
        summary = summarize_rows(items)
        group_rows.append([
            family,
            language,
            summary["case_count"],
            summary["all_v2_exact_match"],
            summary["avg_v2_vs_v1_e2e_speedup_x"],
            summary["avg_v2_vs_v1_mp_speedup_x"],
            summary["avg_v2_vs_v1_dedup_speedup_x"],
            summary["avg_v2_vs_native_tokenizer_speedup_x"],
            summary["avg_v2_vs_native_e2e_speedup_x"],
        ])
    sections.append(format_markdown_table(group_headers, group_rows))
    sections.append("")

    sections.append("## Per Case")
    sections.append("")
    case_headers = [
        "family",
        "lang",
        "length",
        "native_tok_ms",
        "native_e2e_ms",
        "v1_mp_ms",
        "v1_dedup_ms",
        "v1_e2e_ms",
        "v2_mp_ms",
        "v2_dedup_ms",
        "v2_e2e_ms",
        "v2_vs_v1_mp_x",
        "v2_vs_v1_dedup_x",
        "v2_vs_v1_e2e_x",
        "v2_vs_native_tok_x",
        "v2_vs_native_e2e_x",
        "v2_exact",
    ]
    case_rows = []
    for row in sorted(
        rows,
        key=lambda item: (item["tokenizer_family"], item["language"], item["input_chars"]),
    ):
        case_rows.append([
            row["tokenizer_family"],
            row["language"],
            row["length_label"],
            row["native_tokenizer_time_ms"],
            row["native_e2e_time_ms"],
            row["v1_mp_time_ms"],
            row["v1_dedup_time_ms"],
            row["v1_e2e_time_ms"],
            row["v2_mp_time_ms"],
            row["v2_dedup_time_ms"],
            row["v2_e2e_time_ms"],
            row["v2_vs_v1_mp_speedup_x"],
            row["v2_vs_v1_dedup_speedup_x"],
            row["v2_vs_v1_e2e_speedup_x"],
            row["v2_vs_native_tokenizer_speedup_x"],
            row["v2_vs_native_e2e_speedup_x"],
            row["v2_exact_match"],
        ])
    sections.append(format_markdown_table(case_headers, case_rows))
    sections.append("")
    sections.append(
        "注：原始 native 链路没有单独的 dedup 阶段，因此与 native 的直接可比项为 "
        "`native_tokenizer_time_ms <-> v2_mp_time_ms` 和 `native_e2e_time_ms <-> v2_e2e_time_ms`。"
    )
    return "\n".join(sections).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="v2_stage2_full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.compare_json.read_text(encoding="utf-8"))
    enriched_rows = [enrich_row(row) for row in rows]
    overall = summarize_rows(enriched_rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix
    summary_path = output_dir / f"{output_prefix}_summary.json"
    enriched_path = output_dir / f"{output_prefix}_compare_enriched.json"
    markdown_path = output_dir / f"{output_prefix}_summary.md"

    summary_path.write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    enriched_path.write_text(
        json.dumps(enriched_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_markdown(enriched_rows, overall),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "summary": overall,
            "outputs": {
                "summary_json": str(summary_path),
                "enriched_json": str(enriched_path),
                "markdown": str(markdown_path),
            },
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
