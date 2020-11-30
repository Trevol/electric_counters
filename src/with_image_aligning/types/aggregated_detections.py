from dataclasses import dataclass
from typing import List

from core.rect import Rect


@dataclass
class DigitCount:
    digit: int
    count: int


@dataclass
class AggregatedDetections:
    box: Rect
    score: float
    digit_counts: List[DigitCount]