#!/usr/bin/env python3
"""Offline native-vLLM vs LoPT tokenizer benchmark."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.lopt_tokenizer import (  # noqa: E402
    LoPTConfig,
    LoPTParallelTokenizer,
    recommended_process_count,
)
from benchmarks.real_web_corpus import ensure_corpora  # noqa: E402


def insert_vllm_src(vllm_src: Path) -> None:
    vllm_src = vllm_src.resolve()
    if str(vllm_src) not in sys.path:
        sys.path.insert(0, str(vllm_src))


@dataclass
class MockHFConfig:
    model_type: str = "any"
    use_unified_vision_chunk: bool = False


@dataclass
class MockModelConfig:
    task: str
    runner_type: str
    model: str
    tokenizer: str
    trust_remote_code: bool
    tokenizer_mode: str
    max_model_len: int
    tokenizer_revision: str | None
    multimodal_config: Any
    hf_config: MockHFConfig
    hf_text_config: MockHFConfig
    logits_processors: list[str] | None = None
    diff_sampling_param: dict[str, Any] | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config: dict[str, Any] | None = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 8
    enable_prompt_embeds: bool = False
    override_generation_config: dict[str, Any] = field(default_factory=dict)

    def get_diff_sampling_param(self) -> dict[str, Any]:
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig
    cache_config: Any
    observability_config: Any


@dataclass
class RunMetrics:
    e2e_ms: float
    inner_ms: float
    token_count: int
    token_hash: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseSummary:
    language: str
    length_label: str
    input_chars: int
    output_tokens: int
    native_e2e_ms: float
    native_tokenizer_ms: float
    lopt_e2e_ms: float
    lopt_process_ms: float
    e2e_speedup_x: float
    tokenizer_speedup_x: float
    exact_match: bool
    native_token_hash: str
    lopt_token_hash: str
    retries: int
    chunk_chars: int
    chunk_count: int


def sha256_token_ids(token_ids: Iterable[int]) -> str:
    from hashlib import sha256

    values = list(token_ids)
    return sha256(array("Q", values).tobytes()).hexdigest()


def parse_length_spec(spec: str) -> tuple[str, int]:
    suffix = spec[-1].lower()
    if suffix != "k":
        raise ValueError(f"Unsupported length spec: {spec!r}")
    number = float(spec[:-1])
    return spec, int(number * 1024)


class NativeTokenizerTimer:
    def __init__(self, async_cls: type[Any]) -> None:
        self._async_cls = async_cls
        self._original = async_cls.encode
        self.records: list[float] = []

    def install(self) -> None:
        original = self._original
        records = self.records

        async def wrapped(instance, prompt, **kwargs):
            start = time.perf_counter()
            result = await original(instance, prompt, **kwargs)
            records.append((time.perf_counter() - start) * 1000.0)
            return result

        self._async_cls.encode = wrapped

    def uninstall(self) -> None:
        self._async_cls.encode = self._original

    def slice_sum(self, start_index: int) -> float:
        return sum(self.records[start_index:])


class NativeBenchmarkHarness:
    def __init__(
        self,
        model_name: str,
        tokenizer_path: Path,
        renderer_workers: int,
        tokenizer_mode: str = "hf",
    ) -> None:
        from transformers import PreTrainedTokenizerFast
        from vllm.config.cache import CacheConfig
        from vllm.config.multimodal import MultiModalConfig
        from vllm.config.observability import ObservabilityConfig
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.serve.render.serving import OpenAIServingRender
        from vllm.entrypoints.serve.tokenize.protocol import TokenizeCompletionRequest
        from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
        from vllm.renderers.hf import HfRenderer
        from vllm.tokenizers import get_tokenizer
        from vllm.utils.async_utils import AsyncMicrobatchTokenizer

        self.TokenizeCompletionRequest = TokenizeCompletionRequest
        self._raw_request = SimpleNamespace(headers={})
        self._model_name = model_name
        self.serial_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(tokenizer_path)
        )

        tokenizer = get_tokenizer(
            tokenizer_name=str(tokenizer_path),
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=False,
        )
        model_config = MockModelConfig(
            task="generate",
            runner_type="generate",
            model=model_name,
            tokenizer=str(tokenizer_path),
            trust_remote_code=False,
            tokenizer_mode=tokenizer_mode,
            max_model_len=2_000_000,
            tokenizer_revision=None,
            multimodal_config=MultiModalConfig(),
            hf_config=MockHFConfig(),
            hf_text_config=MockHFConfig(),
            renderer_num_workers=renderer_workers,
        )
        config = MockVllmConfig(
            model_config=model_config,
            parallel_config=MockParallelConfig(),
            cache_config=CacheConfig(),
            observability_config=ObservabilityConfig(),
        )
        renderer = HfRenderer(config, tokenizer=tokenizer)

        engine = SimpleNamespace(
            errored=False,
            model_config=model_config,
            input_processor=SimpleNamespace(),
            renderer=renderer,
        )
        models = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=[BaseModelPath(name=model_name, model_path=str(tokenizer_path))],
        )
        serving_render = OpenAIServingRender(
            model_config=model_config,
            renderer=renderer,
            model_registry=models.registry,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        self.service = OpenAIServingTokenization(
            engine,
            models,
            openai_serving_render=serving_render,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        self.timer = NativeTokenizerTimer(AsyncMicrobatchTokenizer)
        self.timer.install()

    async def warmup(self, text: str) -> None:
        await self.run_once(text)

    async def run_once(self, text: str) -> tuple[RunMetrics, list[int]]:
        request = self.TokenizeCompletionRequest(
            model=self._model_name,
            prompt=text,
            add_special_tokens=True,
            return_token_strs=False,
        )
        start_index = len(self.timer.records)
        start = time.perf_counter()
        response = await self.service.create_tokenize(request, self._raw_request)
        e2e_ms = (time.perf_counter() - start) * 1000.0
        token_ms = self.timer.slice_sum(start_index)
        if not hasattr(response, "tokens"):
            raise RuntimeError(f"Unexpected response: {response!r}")
        token_ids = list(response.tokens)
        metrics = RunMetrics(
            e2e_ms=e2e_ms,
            inner_ms=token_ms,
            token_count=len(token_ids),
            token_hash=sha256_token_ids(token_ids),
        )
        return metrics, token_ids

    def serial_encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> tuple[list[int], float]:
        start = time.perf_counter()
        encoding = self.serial_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return list(encoding["input_ids"]), elapsed_ms

    def close(self) -> None:
        self.timer.uninstall()
        self.service.renderer.shutdown()


def load_text(corpus_path: Path, target_chars: int) -> str:
    text = corpus_path.read_text(encoding="utf-8")
    if len(text) < target_chars:
        raise RuntimeError(
            f"Corpus {corpus_path} only has {len(text)} chars, "
            f"but the benchmark needs {target_chars}."
        )
    return text[:target_chars]


def median_ms(values: list[float]) -> float:
    return round(statistics.median(values), 3)


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
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

    tokenizer_path = args.tokenizer_path.resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer path does not exist: {tokenizer_path}. "
            "Prepare a local DeepSeek-V4-Pro tokenizer directory first."
        )

    probe_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    native = NativeBenchmarkHarness(
        model_name=args.model_name,
        tokenizer_path=tokenizer_path,
        renderer_workers=args.renderer_workers,
        tokenizer_mode=args.tokenizer_mode,
    )
    lopt = LoPTParallelTokenizer(
        LoPTConfig(
            tokenizer_path=str(tokenizer_path),
            processes=args.processes,
            overlap_chars=args.overlap_chars,
            min_match_tokens=args.min_match_tokens,
            initial_chunk_chars=args.initial_chunk_chars,
        )
    )

    try:
        warmup_prompt = "warmup " * 128
        await native.warmup(warmup_prompt)
        lopt.tokenize(warmup_prompt, add_special_tokens=True)

        case_summaries: list[CaseSummary] = []
        raw_results: dict[str, Any] = {
            "config": {
                "model_name": args.model_name,
                "tokenizer_path": str(tokenizer_path),
                "renderer_workers": args.renderer_workers,
                "processes": args.processes,
                "overlap_chars": args.overlap_chars,
                "min_match_tokens": args.min_match_tokens,
                "initial_chunk_chars": args.initial_chunk_chars,
                "repeats": args.repeats,
                "length_base": 1024,
                "tokenizer_probe": {
                    "is_fast": probe_tokenizer.is_fast,
                    "vocab_size": getattr(probe_tokenizer, "vocab_size", None),
                    "max_chars_per_token": getattr(
                        probe_tokenizer, "max_chars_per_token", None
                    ),
                },
            },
            "cases": [],
        }

        for language in args.languages:
            corpus_path = corpus_dir / f"{language}_web_corpus.txt"
            for length_label, input_chars in lengths:
                text = load_text(corpus_path, input_chars)

                native_runs: list[RunMetrics] = []
                native_token_ids: list[int] | None = None
                for _ in range(args.repeats):
                    metrics, token_ids = await native.run_once(text)
                    native_runs.append(metrics)
                    native_token_ids = token_ids

                lopt_runs: list[RunMetrics] = []
                lopt_last_extra: dict[str, Any] = {}
                lopt_token_ids: list[int] | None = None
                for _ in range(args.repeats):
                    result = lopt.tokenize(text, add_special_tokens=True)
                    lopt_token_ids = result.token_ids
                    lopt_last_extra = {
                        "retry_rounds": result.retry_rounds,
                        "chunk_chars": result.chunk_chars,
                        "chunk_count": result.chunk_count,
                        "merge_ms": round(result.merge_time_s * 1000.0, 3),
                    }
                    lopt_runs.append(
                        RunMetrics(
                            e2e_ms=round(result.e2e_time_s * 1000.0, 3),
                            inner_ms=round(result.process_only_time_s * 1000.0, 3),
                            token_count=len(result.token_ids),
                            token_hash=sha256_token_ids(result.token_ids),
                            extra=lopt_last_extra,
                        )
                    )

                assert native_token_ids is not None
                assert lopt_token_ids is not None

                exact_match = native_token_ids == lopt_token_ids
                if not exact_match:
                    raise AssertionError(
                        f"Token mismatch for language={language}, length={length_label}."
                    )

                native_e2e_ms = median_ms([run.e2e_ms for run in native_runs])
                native_tokenizer_ms = median_ms([run.inner_ms for run in native_runs])
                lopt_e2e_ms = median_ms([run.e2e_ms for run in lopt_runs])
                lopt_process_ms = median_ms([run.inner_ms for run in lopt_runs])
                output_tokens = native_runs[-1].token_count

                summary = CaseSummary(
                    language=language,
                    length_label=length_label,
                    input_chars=input_chars,
                    output_tokens=output_tokens,
                    native_e2e_ms=native_e2e_ms,
                    native_tokenizer_ms=native_tokenizer_ms,
                    lopt_e2e_ms=lopt_e2e_ms,
                    lopt_process_ms=lopt_process_ms,
                    e2e_speedup_x=round(native_e2e_ms / lopt_e2e_ms, 3),
                    tokenizer_speedup_x=round(
                        native_tokenizer_ms / lopt_process_ms, 3
                    ),
                    exact_match=True,
                    native_token_hash=native_runs[-1].token_hash,
                    lopt_token_hash=lopt_runs[-1].token_hash,
                    retries=int(lopt_last_extra["retry_rounds"]),
                    chunk_chars=int(lopt_last_extra["chunk_chars"]),
                    chunk_count=int(lopt_last_extra["chunk_count"]),
                )
                case_summaries.append(summary)
                raw_results["cases"].append(
                    {
                        "language": language,
                        "length_label": length_label,
                        "input_chars": input_chars,
                        "native_runs": [asdict(run) for run in native_runs],
                        "lopt_runs": [asdict(run) for run in lopt_runs],
                        "summary": asdict(summary),
                    }
                )

        by_language: dict[str, list[CaseSummary]] = {}
        for summary in case_summaries:
            by_language.setdefault(summary.language, []).append(summary)

        markdown_sections = []
        for language, summaries in sorted(by_language.items()):
            summaries.sort(key=lambda item: item.input_chars)
            baseline_rows = [
                [
                    item.length_label,
                    item.input_chars,
                    item.output_tokens,
                    item.native_e2e_ms,
                    item.native_tokenizer_ms,
                ]
                for item in summaries
            ]
            lopt_rows = [
                [
                    item.length_label,
                    item.input_chars,
                    item.output_tokens,
                    item.lopt_e2e_ms,
                    item.lopt_process_ms,
                    item.retries,
                    item.chunk_chars,
                    item.chunk_count,
                ]
                for item in summaries
            ]
            compare_rows = [
                [
                    item.length_label,
                    item.native_e2e_ms,
                    item.native_tokenizer_ms,
                    item.lopt_e2e_ms,
                    item.lopt_process_ms,
                    item.e2e_speedup_x,
                    item.tokenizer_speedup_x,
                    item.exact_match,
                ]
                for item in summaries
            ]
            markdown_sections.extend(
                [
                    f"## {language.upper()} Native Baseline",
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
                    f"## {language.upper()} LoPT",
                    format_markdown_table(
                        [
                            "length",
                            "input_chars",
                            "output_tokens",
                            "e2e_ms",
                            "lopt_process_ms",
                            "retries",
                            "chunk_chars",
                            "chunk_count",
                        ],
                        lopt_rows,
                    ),
                    "",
                    f"## {language.upper()} Comparison",
                    format_markdown_table(
                        [
                            "length",
                            "native_e2e_ms",
                            "native_tokenizer_ms",
                            "lopt_e2e_ms",
                            "lopt_process_ms",
                            "e2e_speedup_x",
                            "tokenizer_speedup_x",
                            "exact_match",
                        ],
                        compare_rows,
                    ),
                    "",
                ]
            )

        raw_results["markdown"] = "\n".join(markdown_sections).strip()
        return raw_results
    finally:
        lopt.close()
        native.close()


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_results.json"
    md_path = output_dir / "benchmark_tables.md"
    csv_path = output_dir / "benchmark_summary.csv"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(result["markdown"] + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "language",
                "length_label",
                "input_chars",
                "output_tokens",
                "native_e2e_ms",
                "native_tokenizer_ms",
                "lopt_e2e_ms",
                "lopt_process_ms",
                "e2e_speedup_x",
                "tokenizer_speedup_x",
                "exact_match",
                "retries",
                "chunk_chars",
                "chunk_count",
            ]
        )
        for case in result["cases"]:
            summary = case["summary"]
            writer.writerow(
                [
                    summary["language"],
                    summary["length_label"],
                    summary["input_chars"],
                    summary["output_tokens"],
                    summary["native_e2e_ms"],
                    summary["native_tokenizer_ms"],
                    summary["lopt_e2e_ms"],
                    summary["lopt_process_ms"],
                    summary["e2e_speedup_x"],
                    summary["tokenizer_speedup_x"],
                    summary["exact_match"],
                    summary["retries"],
                    summary["chunk_chars"],
                    summary["chunk_count"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-src",
        type=Path,
        required=True,
        help="Path to the latest local vLLM source tree copied to the run host.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="Local directory containing the DeepSeek-V4-Pro tokenizer files.",
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-V4-Pro",
        help="Model name used in the mocked OpenAI-compatible request path.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        default="hf",
        help="Tokenizer mode used by the native vLLM baseline.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("benchmarks/assets"),
        help="Directory containing en_web_corpus.txt and zh_web_corpus.txt.",
    )
    parser.add_argument(
        "--build-corpus-if-missing",
        action="store_true",
        help="Build the real-web corpora if the corpus files are missing.",
    )
    parser.add_argument(
        "--corpus-timeout-s",
        type=float,
        default=20.0,
        help="Timeout used only when corpus files need to be fetched.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh"],
        choices=["en", "zh"],
        help="Languages to benchmark.",
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
        help="Character-length cases to benchmark. 'k' means 1024 characters.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured runs per case.",
    )
    parser.add_argument(
        "--renderer-workers",
        type=int,
        default=8,
        help="Renderer thread-pool size for the native vLLM baseline.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=recommended_process_count(),
        help="Process-pool size used by LoPT.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=1024,
        help="Character overlap between adjacent LoPT chunks.",
    )
    parser.add_argument(
        "--min-match-tokens",
        type=int,
        default=2,
        help="Minimum overlap token run length required by LoPT.",
    )
    parser.add_argument(
        "--initial-chunk-chars",
        type=int,
        default=None,
        help="Optional initial chunk size for LoPT. Defaults to input_len / processes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/out"),
        help="Directory where JSON/CSV/Markdown outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    insert_vllm_src(args.vllm_src)
    result = asyncio.run(run_benchmark(args))
    write_outputs(args.output_dir.resolve(), result)
    print(result["markdown"])


if __name__ == "__main__":
    main()
