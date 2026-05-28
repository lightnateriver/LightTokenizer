"""Chunk planning for LoPT v2 stage1."""

from __future__ import annotations

import math

from .types import ChunkTask


def resolve_chunk_chars(
    text_length: int,
    *,
    chunk_count: int | None,
    chunk_chars: int | None,
) -> int:
    if text_length < 0:
        raise ValueError("text_length must be non-negative.")
    if chunk_count is not None:
        if chunk_count <= 0:
            raise ValueError("chunk_count must be positive.")
        return max(1, math.ceil(text_length / chunk_count))
    if chunk_chars is None:
        raise ValueError("Either chunk_count or chunk_chars must be provided.")
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive.")
    return max(1, chunk_chars)


def build_chunk_plan(
    text: str,
    *,
    chunk_count: int | None = None,
    chunk_chars: int | None = None,
    overlap_chars: int,
) -> tuple[list[ChunkTask], int]:
    if overlap_chars <= 0:
        raise ValueError("overlap_chars must be positive.")

    resolved_chunk_chars = resolve_chunk_chars(
        len(text),
        chunk_count=chunk_count,
        chunk_chars=chunk_chars,
    )
    text_len = len(text)
    tasks: list[ChunkTask] = []
    start = 0
    index = 0
    while start < text_len:
        end = min(text_len, start + resolved_chunk_chars + overlap_chars)
        tasks.append(
            ChunkTask(
                index=index,
                start=start,
                core_end=min(text_len, start + resolved_chunk_chars),
                end=end,
                overlap_chars=overlap_chars,
                text=text[start:end],
            )
        )
        if end == text_len and start + resolved_chunk_chars >= text_len:
            break
        start += resolved_chunk_chars
        index += 1
    return tasks, resolved_chunk_chars
