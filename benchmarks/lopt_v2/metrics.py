"""Metric helpers for LoPT v2 stage1."""

from __future__ import annotations

import statistics


def median(values: list[float]) -> float:
    return statistics.median(values)


def round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def seconds_to_ms(value: float) -> float:
    return round(value * 1000.0, 3)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 3)


def delta_ms(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    return round(candidate - reference, 3)
