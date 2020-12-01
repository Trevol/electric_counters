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

    def totalCount(self):
        return sum(d.count for d in self.digit_counts)

    @staticmethod
    def _countAccessor(digitCount):
        return digitCount.count

    def digitWithMaxCount(self):
        return max(self.digit_counts, key=self._countAccessor).digit
