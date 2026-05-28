#!/usr/bin/env python3
"""Compare native vLLM, LoPT v1, and LoPT v2 on fixed best configs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.lopt_tokenizer import LoPTConfig as LoPTV1Config
from benchmarks.lopt_tokenizer import LoPTParallelTokenizer as LoPTV1ParallelTokenizer
from benchmarks.lopt_tokenizer_v2 import (
    LoPTV2Config,
    LoPTV2ParallelTokenizer,
    explain_token_mismatch,
    token_ids_sha256,
)
from benchmarks.vllm_tokenizer_bench import (
    NativeBenchmarkHarness,
    insert_vllm_src,
    load_text,
)


VALID_FAMILIES = ("DeepSeek-V4-Pro", "Qwen3.5")


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 3)


def resolve_tokenizer_path(
    record: dict[str, Any],
    *,
    deepseek_tokenizer_path: Path | None,
    qwen_tokenizer_path: Path | None,
) -> Path:
    family = record["tokenizer_family"]
    if family == "DeepSeek-V4-Pro" and deepseek_tokenizer_path is not None:
        return deepseek_tokenizer_path.resolve()
    if family == "Qwen3.5" and qwen_tokenizer_path is not None:
        return qwen_tokenizer_path.resolve()
    return Path(record["tokenizer_path"]).resolve()


async def measure_native_case(
    native: NativeBenchmarkHarness,
    text: str,
    repeats: int,
) -> tuple[dict[str, Any], list[int]]:
    runs = []
    token_ids: list[int] | None = None
    for _ in range(repeats):
        metrics, token_ids = await native.run_once(text)
        runs.append(metrics)
    assert token_ids is not None
    return (
        {
            "native_e2e_time_ms": round_ms(median([run.e2e_ms for run in runs])),
            "native_tokenizer_time_ms": round_ms(
                median([run.inner_ms for run in runs])
            ),
            "native_output_tokens": runs[-1].token_count,
            "native_token_hash": runs[-1].token_hash,
        },
        token_ids,
    )


def summarize_v1_case(
    *,
    lopt: LoPTV1ParallelTokenizer,
    text: str,
    repeats: int,
    chunk_count: int,
    overlap_chars: int,
    native_token_ids: list[int],
) -> dict[str, Any]:
    runs = []
    token_ids: list[int] | None = None
    error_message = ""
    status = "valid"
    try:
        for _ in range(repeats):
            result = lopt.tokenize(
                text,
                add_special_tokens=True,
                chunk_count=chunk_count,
                overlap_chars=overlap_chars,
                chat_template_time_s=0.0,
            )
            token_ids = result.token_ids
            runs.append(result)
    except Exception as exc:
        status = "exception"
        error_message = f"{type(exc).__name__}: {exc}"

    if not runs or token_ids is None:
        return {
            "v1_status": status,
            "v1_exact_match": False,
            "v1_e2e_time_ms": None,
            "v1_mp_time_ms": None,
            "v1_dedup_time_ms": None,
            "v1_token_hash": "",
            "v1_error_message": error_message,
        }

    exact_match = token_ids == native_token_ids
    if not exact_match and status == "valid":
        status = "mismatch"
        error_message = explain_token_mismatch(native_token_ids, token_ids)

    chat_template_time_ms = round_ms(
        median([run.chat_template_time_s * 1000.0 for run in runs])
    )
    dispatch_submit_time_ms = round_ms(
        median([run.dispatch_submit_time_s * 1000.0 for run in runs])
    )
    dispatch_collect_time_ms = round_ms(
        median([run.dispatch_collect_time_s * 1000.0 for run in runs])
    )
    mp_dispatch_process_collect_time_ms = round_ms(
        median([run.dispatch_process_collect_time_s * 1000.0 for run in runs])
    )
    chunk_dedup_time_ms = round_ms(
        median([run.chunk_dedup_time_s * 1000.0 for run in runs])
    )
    worker_encode_time_ms_sum = round_ms(
        median([run.worker_encode_time_s_sum * 1000.0 for run in runs])
    )
    worker_encode_time_ms_max = round_ms(
        median([run.worker_encode_time_s_max * 1000.0 for run in runs])
    )
    worker_materialize_time_ms_sum = round_ms(
        median([run.worker_materialize_time_s_sum * 1000.0 for run in runs])
    )
    worker_materialize_time_ms_max = round_ms(
        median([run.worker_materialize_time_s_max * 1000.0 for run in runs])
    )
    collect_child_compute_makespan_ms = round_ms(
        median([run.collect_child_compute_makespan_s * 1000.0 for run in runs])
    )
    collect_result_return_tail_ms = round_ms(
        median([run.collect_result_return_tail_s * 1000.0 for run in runs])
    )
    collect_result_receive_lag_max_ms = round_ms(
        median([run.collect_result_receive_lag_max_s * 1000.0 for run in runs])
    )
    collect_result_receive_lag_avg_ms = round_ms(
        median([run.collect_result_receive_lag_avg_s * 1000.0 for run in runs])
    )
    lopt_e2e_time_ms = round_ms(
        (chat_template_time_ms or 0.0)
        + (mp_dispatch_process_collect_time_ms or 0.0)
        + (chunk_dedup_time_ms or 0.0)
    )
    return {
        "v1_status": status,
        "v1_exact_match": exact_match,
        "v1_e2e_time_ms": lopt_e2e_time_ms,
        "v1_dispatch_submit_time_ms": dispatch_submit_time_ms,
        "v1_dispatch_collect_time_ms": dispatch_collect_time_ms,
        "v1_mp_time_ms": mp_dispatch_process_collect_time_ms,
        "v1_dedup_time_ms": chunk_dedup_time_ms,
        "v1_worker_encode_time_ms_sum": worker_encode_time_ms_sum,
        "v1_worker_encode_time_ms_max": worker_encode_time_ms_max,
        "v1_worker_materialize_time_ms_sum": worker_materialize_time_ms_sum,
        "v1_worker_materialize_time_ms_max": worker_materialize_time_ms_max,
        "v1_token_hash": token_ids_sha256(token_ids),
        "v1_error_message": error_message,
    }


def summarize_v2_case(
    *,
    lopt: LoPTV2ParallelTokenizer,
    text: str,
    repeats: int,
    chunk_count: int,
    overlap_chars: int,
    native_token_ids: list[int],
) -> dict[str, Any]:
    runs = []
    token_ids: list[int] | None = None
    error_message = ""
    status = "valid"
    boundary_failure = ""
    boundary_count = None
    actual_chunk_count = None
    try:
        for _ in range(repeats):
            result = lopt.tokenize(
                text,
                add_special_tokens=True,
                chunk_count=chunk_count,
                overlap_chars=overlap_chars,
                chat_template_time_s=0.0,
            )
            token_ids = result.token_ids
            actual_chunk_count = result.chunk_count
            boundary_count = len(result.boundary_matches)
            runs.append(result)
    except Exception as exc:
        status = "exception"
        error_message = f"{type(exc).__name__}: {exc}"

    if not runs or token_ids is None:
        return {
            "v2_status": status,
            "v2_exact_match": False,
            "v2_e2e_time_ms": None,
            "v2_mp_time_ms": None,
            "v2_dedup_time_ms": None,
            "v2_token_hash": "",
            "v2_error_message": error_message,
            "v2_boundary_count": boundary_count,
            "v2_actual_chunk_count": actual_chunk_count,
            "v2_first_boundary_issue": boundary_failure,
        }

    exact_match = token_ids == native_token_ids
    if not exact_match and status == "valid":
        status = "mismatch"
        error_message = explain_token_mismatch(native_token_ids, token_ids)

    first_run = runs[-1]
    invalid_boundaries = [
        match for match in first_run.boundary_matches if not match.is_valid
    ]
    if invalid_boundaries:
        first = invalid_boundaries[0]
        boundary_failure = (
            f"{first.left_chunk_index}->{first.right_chunk_index} "
            f"{first.status}: {first.message}"
        )

    chat_template_time_ms = round_ms(
        median([run.chat_template_time_s * 1000.0 for run in runs])
    )
    dispatch_submit_time_ms = round_ms(
        median([run.dispatch_submit_time_s * 1000.0 for run in runs])
    )
    dispatch_collect_time_ms = round_ms(
        median([run.dispatch_collect_time_s * 1000.0 for run in runs])
    )
    mp_dispatch_process_collect_time_ms = round_ms(
        median([run.dispatch_process_collect_time_s * 1000.0 for run in runs])
    )
    chunk_dedup_time_ms = round_ms(
        median([run.chunk_dedup_time_s * 1000.0 for run in runs])
    )
    worker_encode_time_ms_sum = round_ms(
        median([run.worker_encode_time_s_sum * 1000.0 for run in runs])
    )
    worker_encode_time_ms_max = round_ms(
        median([run.worker_encode_time_s_max * 1000.0 for run in runs])
    )
    worker_materialize_time_ms_sum = round_ms(
        median([run.worker_materialize_time_s_sum * 1000.0 for run in runs])
    )
    worker_materialize_time_ms_max = round_ms(
        median([run.worker_materialize_time_s_max * 1000.0 for run in runs])
    )
    collect_child_compute_makespan_ms = round_ms(
        median([run.collect_child_compute_makespan_s * 1000.0 for run in runs])
    )
    collect_result_return_tail_ms = round_ms(
        median([run.collect_result_return_tail_s * 1000.0 for run in runs])
    )
    collect_result_receive_lag_max_ms = round_ms(
        median([run.collect_result_receive_lag_max_s * 1000.0 for run in runs])
    )
    collect_result_receive_lag_avg_ms = round_ms(
        median([run.collect_result_receive_lag_avg_s * 1000.0 for run in runs])
    )
    lopt_e2e_time_ms = round_ms(
        (chat_template_time_ms or 0.0)
        + (mp_dispatch_process_collect_time_ms or 0.0)
        + (chunk_dedup_time_ms or 0.0)
    )
    return {
        "v2_status": status,
        "v2_exact_match": exact_match,
        "v2_e2e_time_ms": lopt_e2e_time_ms,
        "v2_dispatch_submit_time_ms": dispatch_submit_time_ms,
        "v2_dispatch_collect_time_ms": dispatch_collect_time_ms,
        "v2_mp_time_ms": mp_dispatch_process_collect_time_ms,
        "v2_dedup_time_ms": chunk_dedup_time_ms,
        "v2_worker_encode_time_ms_sum": worker_encode_time_ms_sum,
        "v2_worker_encode_time_ms_max": worker_encode_time_ms_max,
        "v2_worker_materialize_time_ms_sum": worker_materialize_time_ms_sum,
        "v2_worker_materialize_time_ms_max": worker_materialize_time_ms_max,
        "v2_collect_child_compute_makespan_ms": collect_child_compute_makespan_ms,
        "v2_collect_result_return_tail_ms": collect_result_return_tail_ms,
        "v2_collect_result_receive_lag_max_ms": collect_result_receive_lag_max_ms,
        "v2_collect_result_receive_lag_avg_ms": collect_result_receive_lag_avg_ms,
        "v2_token_hash": token_ids_sha256(token_ids),
        "v2_error_message": error_message,
        "v2_boundary_count": len(first_run.boundary_matches),
        "v2_actual_chunk_count": first_run.chunk_count,
        "v2_first_boundary_issue": boundary_failure,
    }


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def build_markdown(records: list[dict[str, Any]]) -> str:
    sections = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (record["tokenizer_family"], record["language"]),
            [],
        ).append(record)

    headers = [
        "length",
        "input_chars",
        "workers",
        "chunks",
        "overlap",
        "native_tok_ms",
        "v1_e2e_ms",
        "v2_e2e_ms",
        "v2_vs_v1_x",
        "v1_exact",
        "v2_exact",
        "v2_status",
    ]
    for (family, language), items in sorted(grouped.items()):
        items.sort(key=lambda item: item["input_chars"])
        rows = [
            [
                item["length_label"],
                item["input_chars"],
                item["worker_processes"],
                item["chunk_count"],
                item["overlap_chars"],
                item["native_tokenizer_time_ms"],
                item["v1_e2e_time_ms"],
                item["v2_e2e_time_ms"],
                item["v2_vs_v1_e2e_speedup_x"],
                item["v1_exact_match"],
                item["v2_exact_match"],
                item["v2_status"],
            ]
            for item in items
        ]
        sections.append(f"## {family} / {language.upper()}")
        sections.append(format_markdown_table(headers, rows))
        sections.append("")
    return "\n".join(sections).strip()


async def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    best_records = json.loads(args.best_configs_json.read_text(encoding="utf-8"))
    selected_families = set(args.families)
    selected_languages = set(args.languages)
    selected_lengths = set(args.lengths)
    best_records = [
        record
        for record in best_records
        if record["tokenizer_family"] in selected_families
        and record["language"] in selected_languages
        and record["length_label"] in selected_lengths
    ]

    corpus_dir = args.corpus_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for base_record in best_records:
        tokenizer_path = resolve_tokenizer_path(
            base_record,
            deepseek_tokenizer_path=args.deepseek_tokenizer_path,
            qwen_tokenizer_path=args.qwen_tokenizer_path,
        )
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer path does not exist for "
                f"{base_record['tokenizer_family']}: {tokenizer_path}"
            )
        text = load_text(
            corpus_dir / f"{base_record['language']}_web_corpus.txt",
            int(base_record["input_chars"]),
        )
        native = NativeBenchmarkHarness(
            model_name=base_record["model_name"],
            tokenizer_path=tokenizer_path,
            renderer_workers=args.renderer_workers,
            tokenizer_mode=base_record.get("tokenizer_mode", "hf"),
        )
        v1 = LoPTV1ParallelTokenizer(
            LoPTV1Config(
                tokenizer_path=str(tokenizer_path),
                processes=int(base_record["worker_processes"]),
                overlap_chars=int(base_record["overlap_chars"]),
                min_match_tokens=args.min_match_tokens,
                max_retry_rounds=0,
            )
        )
        v2 = LoPTV2ParallelTokenizer(
            LoPTV2Config(
                tokenizer_path=str(tokenizer_path),
                processes=int(base_record["worker_processes"]),
                overlap_chars=int(base_record["overlap_chars"]),
                min_match_tokens=args.min_match_tokens,
                max_retry_rounds=0,
                record_boundary_diagnostics=args.record_boundary_diagnostics,
            )
        )
        try:
            warmup_prompt = "warmup " * 128
            await native.warmup(warmup_prompt)
            _ = v1.tokenize(
                warmup_prompt,
                add_special_tokens=True,
                chunk_count=max(1, min(int(base_record["worker_processes"]), 4)),
                overlap_chars=min(
                    32,
                    max(
                        1,
                        int(base_record["overlap_chars"]),
                    ),
                ),
            )
            _ = v2.tokenize(
                warmup_prompt,
                add_special_tokens=True,
                chunk_count=max(1, min(int(base_record["worker_processes"]), 4)),
                overlap_chars=min(
                    32,
                    max(
                        1,
                        int(base_record["overlap_chars"]),
                    ),
                ),
            )
            native_summary, native_token_ids = await measure_native_case(
                native,
                text,
                args.repeats,
            )
            v1_summary = summarize_v1_case(
                lopt=v1,
                text=text,
                repeats=args.repeats,
                chunk_count=int(base_record["chunk_count"]),
                overlap_chars=int(base_record["overlap_chars"]),
                native_token_ids=native_token_ids,
            )
            v2_summary = summarize_v2_case(
                lopt=v2,
                text=text,
                repeats=args.repeats,
                chunk_count=int(base_record["chunk_count"]),
                overlap_chars=int(base_record["overlap_chars"]),
                native_token_ids=native_token_ids,
            )
            record = {
                "tokenizer_family": base_record["tokenizer_family"],
                "model_name": base_record["model_name"],
                "tokenizer_path": str(tokenizer_path),
                "language": base_record["language"],
                "length_label": base_record["length_label"],
                "input_chars": int(base_record["input_chars"]),
                "worker_processes": int(base_record["worker_processes"]),
                "chunk_count": int(base_record["chunk_count"]),
                "overlap_chars": int(base_record["overlap_chars"]),
                **native_summary,
                **v1_summary,
                **v2_summary,
            }
            record["v2_vs_v1_e2e_speedup_x"] = ratio(
                record["v1_e2e_time_ms"],
                record["v2_e2e_time_ms"],
            )
            record["v2_vs_v1_e2e_delta_ms"] = round_ms(
                (
                    record["v2_e2e_time_ms"] - record["v1_e2e_time_ms"]
                    if record["v1_e2e_time_ms"] is not None
                    and record["v2_e2e_time_ms"] is not None
                    else None
                )
            )
            record["v2_vs_v1_dedup_delta_ms"] = round_ms(
                (
                    record["v2_dedup_time_ms"] - record["v1_dedup_time_ms"]
                    if record["v1_dedup_time_ms"] is not None
                    and record["v2_dedup_time_ms"] is not None
                    else None
                )
            )
            records.append(record)
        finally:
            v2.close()
            v1.close()
            native.close()

    csv_fields = [
        "tokenizer_family",
        "model_name",
        "tokenizer_path",
        "language",
        "length_label",
        "input_chars",
        "worker_processes",
        "chunk_count",
        "overlap_chars",
        "native_e2e_time_ms",
        "native_tokenizer_time_ms",
        "native_output_tokens",
        "native_token_hash",
        "v1_status",
        "v1_exact_match",
        "v1_e2e_time_ms",
        "v1_dispatch_submit_time_ms",
        "v1_dispatch_collect_time_ms",
        "v1_mp_time_ms",
        "v1_dedup_time_ms",
        "v1_worker_encode_time_ms_sum",
        "v1_worker_encode_time_ms_max",
        "v1_worker_materialize_time_ms_sum",
        "v1_worker_materialize_time_ms_max",
        "v1_token_hash",
        "v1_error_message",
        "v2_status",
        "v2_exact_match",
        "v2_e2e_time_ms",
        "v2_dispatch_submit_time_ms",
        "v2_dispatch_collect_time_ms",
        "v2_mp_time_ms",
        "v2_dedup_time_ms",
        "v2_worker_encode_time_ms_sum",
        "v2_worker_encode_time_ms_max",
        "v2_worker_materialize_time_ms_sum",
        "v2_worker_materialize_time_ms_max",
        "v2_collect_child_compute_makespan_ms",
        "v2_collect_result_return_tail_ms",
        "v2_collect_result_receive_lag_max_ms",
        "v2_collect_result_receive_lag_avg_ms",
        "v2_token_hash",
        "v2_error_message",
        "v2_boundary_count",
        "v2_actual_chunk_count",
        "v2_first_boundary_issue",
        "v2_vs_v1_e2e_speedup_x",
        "v2_vs_v1_e2e_delta_ms",
        "v2_vs_v1_dedup_delta_ms",
    ]
    output_prefix = args.output_prefix
    json_path = output_dir / f"{output_prefix}.json"
    csv_path = output_dir / f"{output_prefix}.csv"
    md_path = output_dir / f"{output_prefix}.md"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    md_path.write_text(build_markdown(records), encoding="utf-8")

    summary = {
        "case_count": len(records),
        "v1_exact_match_cases": sum(1 for record in records if record["v1_exact_match"]),
        "v2_exact_match_cases": sum(1 for record in records if record["v2_exact_match"]),
        "v2_better_cases": sum(
            1
            for record in records
            if record["v1_e2e_time_ms"] is not None
            and record["v2_e2e_time_ms"] is not None
            and record["v2_e2e_time_ms"] < record["v1_e2e_time_ms"]
        ),
        "average_v2_vs_v1_speedup_x": round(
            sum(
                record["v2_vs_v1_e2e_speedup_x"]
                for record in records
                if record["v2_vs_v1_e2e_speedup_x"] is not None
            )
            / max(
                1,
                sum(
                    1
                    for record in records
                    if record["v2_vs_v1_e2e_speedup_x"] is not None
                ),
            ),
            3,
        ),
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "records": records,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "summary": str(summary_path),
        },
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
    parser.add_argument(
        "--output-prefix",
        default="v2_stage2_compare",
    )
    parser.add_argument("--record-boundary-diagnostics", action="store_true")
    parser.add_argument("--deepseek-tokenizer-path", type=Path, default=None)
    parser.add_argument("--qwen-tokenizer-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    insert_vllm_src(args.vllm_src)
    result = asyncio.run(run_compare(args))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
