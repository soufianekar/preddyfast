"""Small text helpers for messages built from settings labels."""

from __future__ import annotations


def in_sentence(label: str) -> str:
    """A label as it reads mid-sentence: capitalised words lowered, units and acronyms kept.

    `str.lower()` turns "500 hPa" into "500 hpa" and "MSLP" into "mslp"; only
    words that are merely capitalised ("Max", "Temperature") are lowered.
    """
    return " ".join(
        w.lower() if w[:1].isupper() and (w[1:].islower() or len(w) == 1) else w
        for w in label.split(" ")
    )
