"""Worker-pool management for the LoPT v2 tokenizer."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from os import getpid

from transformers import PreTrainedTokenizerFast

from .types import ChunkResult, ChunkTask

_WORKER_TOKENIZER: PreTrainedTokenizerFast | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER

    _WORKER_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def _collect_left_boundary_entries(
    offsets: list[tuple[int, int]],
    input_ids: list[int],
    task: ChunkTask,
) -> list[tuple[int, int, int, int]]:
    entries: list[tuple[int, int, int, int]] = []
    local_boundary_end = min(task.end, task.start + task.overlap_chars) - task.start
    for token_index, ((local_start, local_end), token_id) in enumerate(
        zip(offsets, input_ids, strict=True)
    ):
        if local_end <= local_start:
            continue
        if local_start >= local_boundary_end:
            break
        if local_end <= local_boundary_end:
            entries.append(
                (
                    task.start + local_start,
                    task.start + local_end,
                    token_id,
                    token_index,
                )
            )
    return entries


def _collect_right_boundary_entries(
    offsets: list[tuple[int, int]],
    input_ids: list[int],
    task: ChunkTask,
) -> list[tuple[int, int, int, int]]:
    local_boundary_start = task.core_end - task.start
    local_boundary_end = min(task.end, task.core_end + task.overlap_chars) - task.start
    entries_reversed: list[tuple[int, int, int, int]] = []
    for token_index in range(len(offsets) - 1, -1, -1):
        local_start, local_end = offsets[token_index]
        if local_end <= local_start:
            continue
        if local_start < local_boundary_start:
            break
        if local_end <= local_boundary_end:
            entries_reversed.append(
                (
                    task.start + local_start,
                    task.start + local_end,
                    input_ids[token_index],
                    token_index,
                )
            )
    entries_reversed.reverse()
    return entries_reversed


def _tokenize_chunk(task: ChunkTask) -> ChunkResult:
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker is not initialized.")

    worker_started_at_s = time.perf_counter()
    encode_start = time.perf_counter()
    encoding = _WORKER_TOKENIZER(
        task.text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
    )
    encode_time_s = time.perf_counter() - encode_start
    materialize_start = time.perf_counter()
    input_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    left_boundary_entries = _collect_left_boundary_entries(offsets, input_ids, task)
    right_boundary_entries = _collect_right_boundary_entries(offsets, input_ids, task)
    materialize_time_s = time.perf_counter() - materialize_start
    worker_finished_at_s = time.perf_counter()
    return ChunkResult(
        index=task.index,
        start=task.start,
        end=task.end,
        input_ids=input_ids,
        left_boundary_entries=left_boundary_entries,
        right_boundary_entries=right_boundary_entries,
        worker_pid=getpid(),
        worker_started_at_s=worker_started_at_s,
        worker_finished_at_s=worker_finished_at_s,
        worker_encode_time_s=encode_time_s,
        worker_materialize_time_s=materialize_time_s,
    )


class TokenizerWorkerPool:
    def __init__(self, tokenizer_path: str, processes: int) -> None:
        ctx = get_context("spawn")
        self._pool = ProcessPoolExecutor(
            max_workers=processes,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(tokenizer_path,),
        )

    def cleanup_results(self, results: list[ChunkResult]) -> None:
        del results

    def dispatch(
        self,
        tasks: list[ChunkTask],
    ) -> tuple[list[ChunkResult], float, float]:
        submit_start = time.perf_counter()
        futures = [self._pool.submit(_tokenize_chunk, task) for task in tasks]
        submit_time_s = time.perf_counter() - submit_start

        collect_start = time.perf_counter()
        results: list[ChunkResult] = []
        for future in as_completed(futures):
            result = future.result()
            result.parent_received_at_s = time.perf_counter()
            results.append(result)
        results.sort(key=lambda item: item.index)
        collect_time_s = time.perf_counter() - collect_start
        return results, submit_time_s, collect_time_s

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "TokenizerWorkerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
