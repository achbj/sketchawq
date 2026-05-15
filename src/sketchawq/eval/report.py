from __future__ import annotations

from typing import Any


def merge_reports(*sections: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for section in sections:
        merged.update(section)
    return merged
