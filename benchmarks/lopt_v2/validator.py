"""Validation helpers for LoPT v2 stage1."""

from __future__ import annotations

from array import array
from hashlib import sha256
from typing import Iterable


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    values = list(token_ids)
    return sha256(array("Q", values).tobytes()).hexdigest()


def explain_token_mismatch(
    reference: list[int],
    candidate: list[int],
    *,
    preview: int = 6,
) -> str:
    mismatch_index = None
    for idx, (ref_token, cand_token) in enumerate(
        zip(reference, candidate, strict=False)
    ):
        if ref_token != cand_token:
            mismatch_index = idx
            break
    if mismatch_index is None:
        if len(reference) != len(candidate):
            return (
                "Token length mismatch: "
                f"reference={len(reference)} candidate={len(candidate)}."
            )
        return "Token IDs are identical."

    start = max(0, mismatch_index - preview)
    end = mismatch_index + preview + 1
    return (
        f"First token mismatch at index {mismatch_index}; "
        f"reference_window={reference[start:end]!r}; "
        f"candidate_window={candidate[start:end]!r}."
    )
