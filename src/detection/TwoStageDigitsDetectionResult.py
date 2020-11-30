from dataclasses import dataclass, field
from typing import List

import numpy as np
from trvo_utils.box_utils import xyxy2xywh

from core.rect import Rect


@dataclass
class DigitDetection:
    digit: int
    score: float
    xyxyBoxInImage: np.ndarray
    # redudancy for speedup box grouping
    xywhBoxInImage: List = None
    boxInImage: Rect = None

    def __post_init__(self):
        self.xywhBoxInImage = xyxy2xywh(self.xyxyBoxInImage).tolist()
        self.boxInImage = Rect(self.xywhBoxInImage)


@dataclass
class TwoStageDigitsDetectionResult:
    empty: bool = False
    counterBox: np.ndarray = None
    counterScore: float = None
    screenBox: np.ndarray = None
    screenScore: float = None
    digitDetections: List[DigitDetection] = field(default_factory=list)
