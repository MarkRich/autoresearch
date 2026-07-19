"""Small pure helpers shared by training and its CPU-only tests."""

from __future__ import annotations


def attention_window_covers_sequence(window_size: tuple[int, int], sequence_len: int) -> bool:
    """Return whether an attention window covers the current token sequence."""
    return window_size[0] >= sequence_len
