"""Geometry primitives shared by ingestion, citations, and PDF rendering."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned rectangle in PDF page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def serialize_bboxes(bboxes: Sequence[BBox]) -> str:
    return json.dumps([list(box.as_tuple()) for box in bboxes])


def parse_bboxes(raw: Any) -> list[BBox]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        payload = raw
    else:
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            return []
    boxes: list[BBox] = []
    if not isinstance(payload, list):
        return []
    for item in payload:
        if not isinstance(item, list | tuple) or len(item) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in item)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append(BBox(x0=x0, y0=y0, x1=x1, y1=y1))
    return boxes
