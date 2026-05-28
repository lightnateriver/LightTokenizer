"""Fallback classification helpers for LoPT v2 stage1."""

from __future__ import annotations


def classify_failure(exc: BaseException | None, *, mismatched: bool = False) -> str:
    if mismatched:
        return "fallback_token_mismatch"
    if exc is None:
        return "valid"
    return "fallback_exception"


def failure_message(exc: BaseException | None, *, mismatch_detail: str = "") -> str:
    if exc is not None:
        return f"{type(exc).__name__}: {exc}"
    return mismatch_detail
