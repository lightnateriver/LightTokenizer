#!/usr/bin/env python3
"""First-pass LoPT implementation for long-text tokenizer acceleration."""

from __future__ import annotations

import math
import os
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Iterable

from transformers import PreTrainedTokenizerFast

_WORKER_TOKENIZER: PreTrainedTokenizerFast | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def _tokenize_chunk(task: tuple[int, int, str]) -> "ChunkResult":
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker is not initialized.")

    start, end, text = task
    encoding = _WORKER_TOKENIZER(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    return ChunkResult(start=start, end=end, input_ids=input_ids, offsets=offsets)


@dataclass(frozen=True)
class ChunkResult:
    start: int
    end: int
    input_ids: list[int]
    offsets: list[tuple[int, int]]


@dataclass(frozen=True)
class MatchResult:
    left_start_index: int
    right_start_index: int
    length: int


@dataclass(frozen=True)
class LoPTConfig:
    tokenizer_path: str
    processes: int
    overlap_chars: int = 1024
    min_match_tokens: int = 2
    initial_chunk_chars: int | None = None
    max_retry_rounds: int = 12


@dataclass(frozen=True)
class LoPTResult:
    token_ids: list[int]
    raw_token_ids: list[int]
    chat_template_time_s: float
    dispatch_submit_time_s: float
    dispatch_collect_time_s: float
    dispatch_process_collect_time_s: float
    collect_child_compute_makespan_s: float
    collect_result_return_tail_s: float
    collect_result_receive_lag_max_s: float
    collect_result_receive_lag_avg_s: float
    chunk_dedup_time_s: float
    worker_encode_time_s_sum: float
    worker_encode_time_s_max: float
    worker_materialize_time_s_sum: float
    worker_materialize_time_s_max: float
    process_only_time_s: float
    e2e_time_s: float
    merge_time_s: float
    retry_rounds: int
    chunk_chars: int
    chunk_count: int


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    from hashlib import sha256

    values = list(token_ids)
    return sha256(array("Q", values).tobytes()).hexdigest()


def split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive.")
    if overlap_chars <= 0:
        raise ValueError("overlap_chars must be positive.")

    tasks = []
    text_len = len(text)
    start = 0
    while start < text_len:
        end = min(text_len, start + chunk_chars + overlap_chars)
        tasks.append((start, end, text[start:end]))
        if end == text_len and start + chunk_chars >= text_len:
            break
        start += chunk_chars
    return tasks


def overlap_window(left: ChunkResult, right: ChunkResult) -> tuple[int, int]:
    return right.start, min(left.end, right.end)


def collect_overlap_tokens(
    result: ChunkResult,
    overlap_start: int,
    overlap_end: int,
) -> list[tuple[int, int, int, int]]:
    collected: list[tuple[int, int, int, int]] = []
    for index, ((local_start, local_end), token_id) in enumerate(
        zip(result.offsets, result.input_ids, strict=True)
    ):
        if local_end <= local_start:
            continue
        global_start = result.start + local_start
        global_end = result.start + local_end
        if global_start >= overlap_start and global_end <= overlap_end:
            collected.append((global_start, global_end, token_id, index))
    return collected


def find_position_aware_match(left: ChunkResult, right: ChunkResult) -> MatchResult:
    overlap_start, overlap_end = overlap_window(left, right)
    left_tokens = collect_overlap_tokens(left, overlap_start, overlap_end)
    right_tokens = collect_overlap_tokens(right, overlap_start, overlap_end)

    matched_pairs: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(left_tokens) and j < len(right_tokens):
        left_key = left_tokens[i][:3]
        right_key = right_tokens[j][:3]
        if left_key == right_key:
            matched_pairs.append((left_tokens[i][3], right_tokens[j][3]))
            i += 1
            j += 1
        elif left_key[:2] < right_key[:2]:
            i += 1
        else:
            j += 1

    if not matched_pairs:
        return MatchResult(-1, -1, 0)

    best_start = 0
    best_len = 1
    run_start = 0
    run_len = 1

    for idx in range(1, len(matched_pairs)):
        prev_left, prev_right = matched_pairs[idx - 1]
        cur_left, cur_right = matched_pairs[idx]
        if cur_left == prev_left + 1 and cur_right == prev_right + 1:
            run_len += 1
        else:
            if run_len > best_len:
                best_start = run_start
                best_len = run_len
            run_start = idx
            run_len = 1

    if run_len > best_len:
        best_start = run_start
        best_len = run_len

    left_start, right_start = matched_pairs[best_start]
    return MatchResult(left_start, right_start, best_len)


def merge_chunk_results(results: list[ChunkResult], matches: list[MatchResult]) -> list[int]:
    if len(results) == 1:
        return list(results[0].input_ids)

    merged: list[int] = []
    for index, result in enumerate(results):
        tokens = result.input_ids
        if index == 0:
            merged.extend(tokens[: matches[0].left_start_index + matches[0].length])
        elif index == len(results) - 1:
            prev = matches[index - 1]
            merged.extend(tokens[prev.right_start_index + prev.length :])
        else:
            prev = matches[index - 1]
            current = matches[index]
            merged.extend(
                tokens[
                    prev.right_start_index
                    + prev.length : current.left_start_index
                    + current.length
                ]
            )
    return merged


class LoPTParallelTokenizer:
    def __init__(self, config: LoPTConfig) -> None:
        self.config = config
        self._pool = ProcessPoolExecutor(
            max_workers=config.processes,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(config.tokenizer_path,),
        )
        self._special_token_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            config.tokenizer_path
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "LoPTParallelTokenizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _initial_chunk_chars(self, text: str) -> int:
        configured = self.config.initial_chunk_chars
        if configured is not None:
            return max(1, configured)
        return max(1, math.ceil(len(text) / max(self.config.processes, 1)))

    def _dispatch(
        self,
        tasks: list[tuple[int, int, str]],
    ) -> tuple[list[ChunkResult], float, float]:
        submit_start = time.perf_counter()
        futures = [self._pool.submit(_tokenize_chunk, task) for task in tasks]
        submit_time_s = time.perf_counter() - submit_start
        collect_start = time.perf_counter()
        results = [future.result() for future in futures]
        collect_time_s = time.perf_counter() - collect_start
        return results, submit_time_s, collect_time_s

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        *,
        chunk_count: int | None = None,
        overlap_chars: int | None = None,
        chat_template_time_s: float = 0.0,
    ) -> LoPTResult:
        if not text:
            raw_ids: list[int] = []
            final_ids = (
                self._special_token_tokenizer.build_inputs_with_special_tokens(raw_ids)
                if add_special_tokens
                else raw_ids
            )
            return LoPTResult(
                token_ids=list(final_ids),
                raw_token_ids=raw_ids,
                chat_template_time_s=chat_template_time_s,
                dispatch_submit_time_s=0.0,
                dispatch_collect_time_s=0.0,
                dispatch_process_collect_time_s=0.0,
                collect_child_compute_makespan_s=0.0,
                collect_result_return_tail_s=0.0,
                collect_result_receive_lag_max_s=0.0,
                collect_result_receive_lag_avg_s=0.0,
                chunk_dedup_time_s=0.0,
                worker_encode_time_s_sum=0.0,
                worker_encode_time_s_max=0.0,
                worker_materialize_time_s_sum=0.0,
                worker_materialize_time_s_max=0.0,
                process_only_time_s=0.0,
                e2e_time_s=0.0,
                merge_time_s=0.0,
                retry_rounds=0,
                chunk_chars=0,
                chunk_count=0,
            )

        overall_start = time.perf_counter()
        active_overlap_chars = (
            self.config.overlap_chars if overlap_chars is None else overlap_chars
        )
        if chunk_count is not None:
            if chunk_count <= 0:
                raise ValueError("chunk_count must be positive.")
            chunk_chars = max(1, math.ceil(len(text) / chunk_count))
        else:
            chunk_chars = self._initial_chunk_chars(text)

        tasks = split_text(text, chunk_chars, active_overlap_chars)
        results, dispatch_submit_time_s, dispatch_collect_time_s = self._dispatch(tasks)
        dispatch_process_collect_time_s = (
            dispatch_submit_time_s + dispatch_collect_time_s
        )

        if len(results) == 1:
            raw_ids = list(results[0].input_ids)
            chunk_dedup_time_s = 0.0
        else:
            merge_start = time.perf_counter()
            matches = [
                find_position_aware_match(left, right)
                for left, right in zip(results, results[1:])
            ]
            if not all(
                match.length >= self.config.min_match_tokens for match in matches
            ):
                raise RuntimeError(
                    "LoPT failed to find sufficient overlap matches with "
                    f"{chunk_chars=}, {active_overlap_chars=}, "
                    f"actual_chunk_count={len(tasks)}."
                )
            raw_ids = merge_chunk_results(results, matches)
            chunk_dedup_time_s = time.perf_counter() - merge_start

        final_ids = (
            self._special_token_tokenizer.build_inputs_with_special_tokens(raw_ids)
            if add_special_tokens
            else raw_ids
        )
        e2e_time_s = (
            chat_template_time_s
            + dispatch_process_collect_time_s
            + chunk_dedup_time_s
        )
        return LoPTResult(
            token_ids=list(final_ids),
            raw_token_ids=raw_ids,
            chat_template_time_s=chat_template_time_s,
            dispatch_submit_time_s=dispatch_submit_time_s,
            dispatch_collect_time_s=dispatch_collect_time_s,
            dispatch_process_collect_time_s=dispatch_process_collect_time_s,
            collect_child_compute_makespan_s=0.0,
            collect_result_return_tail_s=0.0,
            collect_result_receive_lag_max_s=0.0,
            collect_result_receive_lag_avg_s=0.0,
            chunk_dedup_time_s=chunk_dedup_time_s,
            worker_encode_time_s_sum=0.0,
            worker_encode_time_s_max=0.0,
            worker_materialize_time_s_sum=0.0,
            worker_materialize_time_s_max=0.0,
            process_only_time_s=dispatch_process_collect_time_s,
            e2e_time_s=e2e_time_s,
            merge_time_s=chunk_dedup_time_s,
            retry_rounds=0,
            chunk_chars=chunk_chars,
            chunk_count=len(tasks),
        )


def recommended_process_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(32, cpu_count))
