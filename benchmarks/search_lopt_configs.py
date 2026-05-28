#!/usr/bin/env python3
"""Search LoPT configurations across tokenizer families, languages, and lengths."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.lopt_tokenizer import (
    LoPTConfig as LoPTV1Config,
    LoPTParallelTokenizer as LoPTV1ParallelTokenizer,
)
from benchmarks.lopt_tokenizer_v2 import (
    LoPTConfig as LoPTV2Config,
    LoPTParallelTokenizer as LoPTV2ParallelTokenizer,
)
from benchmarks.real_web_corpus import ensure_corpora
from benchmarks.vllm_tokenizer_bench import (
    NativeBenchmarkHarness,
    insert_vllm_src,
    load_text,
    parse_length_spec,
    sha256_token_ids,
)


@dataclass(frozen=True)
class TokenizerFamily:
    family_name: str
    model_name: str
    tokenizer_path: Path
    tokenizer_mode: str = "hf"


VALID_FAMILIES = ("DeepSeek-V4-Pro", "Qwen3.5")
VALID_LOPT_VERSIONS = ("v1", "v2")


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def candidate_chunk_counts(
    worker_processes: int,
    chunk_multipliers: list[float],
) -> list[int]:
    counts = {1, max(1, worker_processes)}
    for multiplier in chunk_multipliers:
        counts.add(max(1, int(round(worker_processes * multiplier))))
    return sorted(counts)


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def round_ms(value: float) -> float:
    return round(value, 3)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(path: Path, fieldnames: list[str], record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def write_csv(
    path: Path,
    fieldnames: list[str],
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


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


def create_lopt_tokenizer(
    *,
    lopt_version: str,
    tokenizer_path: Path,
    worker_processes: int,
    overlap_chars: int,
    min_match_tokens: int,
    record_boundary_diagnostics: bool,
) -> Any:
    if lopt_version == "v2":
        return LoPTV2ParallelTokenizer(
            LoPTV2Config(
                tokenizer_path=str(tokenizer_path),
                processes=worker_processes,
                overlap_chars=overlap_chars,
                min_match_tokens=min_match_tokens,
                max_retry_rounds=0,
                record_boundary_diagnostics=record_boundary_diagnostics,
            )
        )
    return LoPTV1ParallelTokenizer(
        LoPTV1Config(
            tokenizer_path=str(tokenizer_path),
            processes=worker_processes,
            overlap_chars=overlap_chars,
            min_match_tokens=min_match_tokens,
            max_retry_rounds=0,
        )
    )


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


def evaluate_lopt_candidate(
    *,
    native: NativeBenchmarkHarness,
    native_summary: dict[str, Any],
    native_token_ids: list[int],
    lopt: Any,
    text: str,
    family: TokenizerFamily,
    language: str,
    length_label: str,
    input_chars: int,
    lopt_version: str,
    worker_processes: int,
    chunk_count: int,
    overlap_chars: int,
    repeats: int,
) -> dict[str, Any]:
    chunk_chars = max(1, math.ceil(input_chars / chunk_count))
    record: dict[str, Any] = {
        "tokenizer_family": family.family_name,
        "model_name": family.model_name,
        "tokenizer_path": str(family.tokenizer_path),
        "tokenizer_mode": family.tokenizer_mode,
        "lopt_version": lopt_version,
        "language": language,
        "length_label": length_label,
        "input_chars": input_chars,
        "worker_processes": worker_processes,
        "chunk_count": chunk_count,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "chat_template_time_ms": 0.0,
        "dispatch_submit_time_ms": None,
        "dispatch_collect_time_ms": None,
        "mp_dispatch_process_collect_time_ms": None,
        "chunk_dedup_time_ms": None,
        "worker_encode_time_ms_sum": None,
        "worker_encode_time_ms_max": None,
        "worker_materialize_time_ms_sum": None,
        "worker_materialize_time_ms_max": None,
        "lopt_e2e_time_ms": None,
        "native_e2e_time_ms": native_summary["native_e2e_time_ms"],
        "native_tokenizer_time_ms": native_summary["native_tokenizer_time_ms"],
        "native_output_tokens": native_summary["native_output_tokens"],
        "native_token_hash": native_summary["native_token_hash"],
        "lopt_output_tokens": None,
        "lopt_token_hash": "",
        "exact_match": False,
        "fallback_used": False,
        "fallback_serial_time_ms": None,
        "fallback_token_hash": "",
        "error_message": "",
        "candidate_status": "invalid",
        "actual_chunk_count": None,
    }

    try:
        runs = []
        token_ids: list[int] | None = None
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

        assert token_ids is not None
        chat_template_time_ms = round_ms(
            median([run.chat_template_time_s * 1000.0 for run in runs])
        )
        mp_dispatch_process_collect_time_ms = round_ms(
            median(
                [
                    run.dispatch_process_collect_time_s * 1000.0
                    for run in runs
                ]
            )
        )
        chunk_dedup_time_ms = round_ms(
            median([run.chunk_dedup_time_s * 1000.0 for run in runs])
        )
        lopt_e2e_time_ms = round_ms(
            chat_template_time_ms
            + mp_dispatch_process_collect_time_ms
            + chunk_dedup_time_ms
        )
        lopt_token_hash = sha256_token_ids(token_ids)

        record.update(
            {
                "chat_template_time_ms": chat_template_time_ms,
                "dispatch_submit_time_ms": (
                    round_ms(
                        median(
                            [run.dispatch_submit_time_s * 1000.0 for run in runs]
                        )
                    )
                ),
                "dispatch_collect_time_ms": (
                    round_ms(
                        median(
                            [run.dispatch_collect_time_s * 1000.0 for run in runs]
                        )
                    )
                ),
                "mp_dispatch_process_collect_time_ms": (
                    mp_dispatch_process_collect_time_ms
                ),
                "chunk_dedup_time_ms": chunk_dedup_time_ms,
                "worker_encode_time_ms_sum": round_ms(
                    median(
                        [run.worker_encode_time_s_sum * 1000.0 for run in runs]
                    )
                ),
                "worker_encode_time_ms_max": round_ms(
                    median(
                        [run.worker_encode_time_s_max * 1000.0 for run in runs]
                    )
                ),
                "worker_materialize_time_ms_sum": round_ms(
                    median(
                        [
                            run.worker_materialize_time_s_sum * 1000.0
                            for run in runs
                        ]
                    )
                ),
                "worker_materialize_time_ms_max": round_ms(
                    median(
                        [
                            run.worker_materialize_time_s_max * 1000.0
                            for run in runs
                        ]
                    )
                ),
                "lopt_e2e_time_ms": lopt_e2e_time_ms,
                "lopt_output_tokens": len(token_ids),
                "lopt_token_hash": lopt_token_hash,
                "actual_chunk_count": runs[-1].chunk_count,
            }
        )

        if token_ids == native_token_ids:
            record["exact_match"] = True
            record["candidate_status"] = "valid"
            return record

        fallback_ids, fallback_time_ms = native.serial_encode(
            text,
            add_special_tokens=True,
        )
        record.update(
            {
                "fallback_used": True,
                "fallback_serial_time_ms": round_ms(fallback_time_ms),
                "fallback_token_hash": sha256_token_ids(fallback_ids),
                "exact_match": fallback_ids == native_token_ids,
                "error_message": (
                    "LoPT token mismatch; fell back to direct serial tokenizer."
                ),
                "candidate_status": "fallback_token_mismatch",
            }
        )
        return record
    except Exception as exc:
        fallback_ids, fallback_time_ms = native.serial_encode(
            text,
            add_special_tokens=True,
        )
        record.update(
            {
                "fallback_used": True,
                "fallback_serial_time_ms": round_ms(fallback_time_ms),
                "fallback_token_hash": sha256_token_ids(fallback_ids),
                "exact_match": fallback_ids == native_token_ids,
                "error_message": f"{type(exc).__name__}: {exc}",
                "candidate_status": "fallback_exception",
            }
        )
        return record


def enrich_best_record(record: dict[str, Any]) -> dict[str, Any]:
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


async def run_search(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import PreTrainedTokenizerFast

    lengths = [parse_length_spec(spec) for spec in args.lengths]
    max_chars = max(length for _, length in lengths)

    corpus_dir = args.corpus_dir.resolve()
    if args.build_corpus_if_missing:
        ensure_corpora(
            output_dir=corpus_dir,
            target_chars=max_chars,
            languages=args.languages,
            timeout_s=args.corpus_timeout_s,
            force=False,
        )

    families = [
        TokenizerFamily(
            family_name="DeepSeek-V4-Pro",
            model_name=args.deepseek_model_name,
            tokenizer_path=args.deepseek_tokenizer_path.resolve(),
        ),
        TokenizerFamily(
            family_name="Qwen3.5",
            model_name=args.qwen_model_name,
            tokenizer_path=args.qwen_tokenizer_path.resolve(),
        ),
    ]
    selected_families = set(args.families)
    families = [
        family for family in families
        if family.family_name in selected_families
    ]
    if not families:
        raise ValueError(
            f"No tokenizer families selected. Valid choices: {VALID_FAMILIES!r}"
        )

    for family in families:
        if not family.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer path does not exist for {family.family_name}: "
                f"{family.tokenizer_path}"
            )
        _ = PreTrainedTokenizerFast.from_pretrained(str(family.tokenizer_path))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_jsonl = output_dir / "search_detail.jsonl"
    detail_csv = output_dir / "search_detail.csv"
    worker_best_json = output_dir / "worker_best_configs.json"
    worker_best_csv = output_dir / "worker_best_configs.csv"
    best_json = output_dir / "best_configs.json"
    best_csv = output_dir / "best_configs.csv"
    markdown_path = output_dir / "best_configs.md"
    meta_path = output_dir / "search_meta.json"
    failures_path = output_dir / "case_failures.json"

    detail_jsonl.unlink(missing_ok=True)
    detail_csv.unlink(missing_ok=True)

    meta = {
        "lopt_version": args.lopt_version,
        "cpuset_policy": args.cpuset_note,
        "selected_families": [family.family_name for family in families],
        "languages": args.languages,
        "lengths": args.lengths,
        "worker_values": args.worker_values,
        "chunk_multipliers": args.chunk_multipliers,
        "overlap_values": args.overlap_values,
        "repeats": args.repeats,
        "min_match_tokens": args.min_match_tokens,
        "families": [
            asdict(family) | {"tokenizer_path": str(family.tokenizer_path)}
            for family in families
        ],
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    detail_fieldnames = [
        "tokenizer_family",
        "model_name",
        "lopt_version",
        "language",
        "length_label",
        "input_chars",
        "worker_processes",
        "chunk_count",
        "chunk_chars",
        "overlap_chars",
        "chat_template_time_ms",
        "dispatch_submit_time_ms",
        "dispatch_collect_time_ms",
        "mp_dispatch_process_collect_time_ms",
        "chunk_dedup_time_ms",
        "worker_encode_time_ms_sum",
        "worker_encode_time_ms_max",
        "worker_materialize_time_ms_sum",
        "worker_materialize_time_ms_max",
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
    best_fieldnames = detail_fieldnames + [
        "e2e_speedup_x",
        "tokenizer_speedup_x",
        "tokenizer_time_drop_pct",
    ]

    worker_best_records: list[dict[str, Any]] = []
    final_best_records: list[dict[str, Any]] = []
    case_failures: list[dict[str, Any]] = []

    for family in families:
        native = NativeBenchmarkHarness(
            model_name=family.model_name,
            tokenizer_path=family.tokenizer_path,
            renderer_workers=args.renderer_workers,
            tokenizer_mode=family.tokenizer_mode,
        )
        try:
            warmup_prompt = "warmup " * 128
            await native.warmup(warmup_prompt)

            for language in args.languages:
                corpus_path = corpus_dir / f"{language}_web_corpus.txt"
                for length_label, input_chars in lengths:
                    text = load_text(corpus_path, input_chars)
                    native_summary, native_token_ids = await measure_native_case(
                        native,
                        text,
                        args.repeats,
                    )

                    branch_worker_bests: list[dict[str, Any]] = []
                    previous_worker_best: dict[str, Any] | None = None

                    for worker_processes in args.worker_values:
                        worker_best: dict[str, Any] | None = None
                        lopt = create_lopt_tokenizer(
                            lopt_version=args.lopt_version,
                            tokenizer_path=family.tokenizer_path,
                            worker_processes=worker_processes,
                            overlap_chars=max(args.overlap_values),
                            min_match_tokens=args.min_match_tokens,
                            record_boundary_diagnostics=(
                                args.record_boundary_diagnostics
                            ),
                        )
                        try:
                            warmup_chunk_count = max(1, min(worker_processes, 4))
                            warmup_chunk_chars = max(
                                1,
                                math.ceil(len(warmup_prompt) / warmup_chunk_count),
                            )
                            warmup_overlap = min(
                                32,
                                max(1, warmup_chunk_chars - 1),
                            )
                            _ = lopt.tokenize(
                                warmup_prompt,
                                add_special_tokens=True,
                                chunk_count=warmup_chunk_count,
                                overlap_chars=warmup_overlap,
                            )

                            for chunk_count in candidate_chunk_counts(
                                worker_processes,
                                args.chunk_multipliers,
                            ):
                                chunk_chars = max(
                                    1,
                                    math.ceil(input_chars / chunk_count),
                                )
                                for overlap_chars in args.overlap_values:
                                    if overlap_chars >= chunk_chars:
                                        continue

                                    candidate_record = evaluate_lopt_candidate(
                                        native=native,
                                        native_summary=native_summary,
                                        native_token_ids=native_token_ids,
                                        lopt=lopt,
                                        text=text,
                                        family=family,
                                        language=language,
                                        length_label=length_label,
                                        input_chars=input_chars,
                                        lopt_version=args.lopt_version,
                                        worker_processes=worker_processes,
                                        chunk_count=chunk_count,
                                        overlap_chars=overlap_chars,
                                        repeats=args.repeats,
                                    )
                                    append_jsonl(detail_jsonl, candidate_record)
                                    append_csv(
                                        detail_csv,
                                        detail_fieldnames,
                                        candidate_record,
                                    )

                                    if (
                                        candidate_record["candidate_status"] == "valid"
                                        and not candidate_record["fallback_used"]
                                    ):
                                        if (
                                            worker_best is None
                                            or candidate_record["lopt_e2e_time_ms"]
                                            < worker_best["lopt_e2e_time_ms"]
                                        ):
                                            worker_best = dict(candidate_record)
                        finally:
                            lopt.close()

                        if worker_best is not None:
                            worker_best_records.append(enrich_best_record(worker_best))
                            branch_worker_bests.append(dict(worker_best))
                            if (
                                previous_worker_best is not None
                                and worker_best["lopt_e2e_time_ms"]
                                > previous_worker_best["lopt_e2e_time_ms"]
                            ):
                                break
                            previous_worker_best = worker_best

                    if branch_worker_bests:
                        final_best_records.append(
                            enrich_best_record(
                                min(
                                    branch_worker_bests,
                                    key=lambda record: record["lopt_e2e_time_ms"],
                                )
                            )
                        )
                    else:
                        case_failures.append(
                            {
                                "tokenizer_family": family.family_name,
                                "model_name": family.model_name,
                                "language": language,
                                "length_label": length_label,
                                "input_chars": input_chars,
                                "native_e2e_time_ms": native_summary[
                                    "native_e2e_time_ms"
                                ],
                                "native_tokenizer_time_ms": native_summary[
                                    "native_tokenizer_time_ms"
                                ],
                                "native_token_hash": native_summary[
                                    "native_token_hash"
                                ],
                                "error_message": (
                                    "No exact-match non-fallback LoPT candidate "
                                    "was found for this case."
                                ),
                            }
                        )
        finally:
            native.close()

    worker_best_records = sorted(
        worker_best_records,
        key=lambda record: (
            record["tokenizer_family"],
            record["language"],
            record["input_chars"],
            record["worker_processes"],
        ),
    )
    final_best_records = sorted(
        final_best_records,
        key=lambda record: (
            record["tokenizer_family"],
            record["language"],
            record["input_chars"],
        ),
    )

    worker_best_json.write_text(
        json.dumps(worker_best_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best_json.write_text(
        json.dumps(final_best_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(worker_best_csv, best_fieldnames, worker_best_records)
    write_csv(best_csv, best_fieldnames, final_best_records)
    markdown_path.write_text(
        build_markdown(final_best_records) + "\n",
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps(case_failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "meta": meta,
        "detail_jsonl": str(detail_jsonl),
        "detail_csv": str(detail_csv),
        "worker_best_json": str(worker_best_json),
        "worker_best_csv": str(worker_best_csv),
        "best_json": str(best_json),
        "best_csv": str(best_csv),
        "best_markdown": str(markdown_path),
        "case_failures_json": str(failures_path),
        "case_failure_count": len(case_failures),
        "final_best_record_count": len(final_best_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-src", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--build-corpus-if-missing", action="store_true")
    parser.add_argument("--corpus-timeout-s", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh"],
        choices=["en", "zh"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(VALID_FAMILIES),
        choices=list(VALID_FAMILIES),
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
    parser.add_argument(
        "--worker-values",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8,16,32,64"),
    )
    parser.add_argument(
        "--chunk-multipliers",
        type=parse_float_list,
        default=parse_float_list("0.5,1,2,4"),
    )
    parser.add_argument(
        "--overlap-values",
        type=parse_int_list,
        default=parse_int_list("32,64,128,256,512,1024,2048,4096,8192"),
    )
    parser.add_argument("--renderer-workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-match-tokens", type=int, default=2)
    parser.add_argument(
        "--lopt-version",
        default="v1",
        choices=list(VALID_LOPT_VERSIONS),
    )
    parser.add_argument("--record-boundary-diagnostics", action="store_true")
    parser.add_argument("--cpuset-note", default="29-31,40-79,155-159")
    parser.add_argument(
        "--deepseek-model-name",
        default="deepseek-ai/DeepSeek-V4-Pro",
    )
    parser.add_argument("--deepseek-tokenizer-path", type=Path, required=True)
    parser.add_argument("--qwen-model-name", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--qwen-tokenizer-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    insert_vllm_src(args.vllm_src)
    result = asyncio.run(run_search(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
