from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class DigitDetection:
    digit: int
    score: float
    boxInImage: np.ndarray


@dataclass
class TwoStageDigitsDetectionResult:
    counterBox: np.ndarray = None
    counterScore: float = None
    screenBox: np.ndarray = None
    screenScore: float = None
    digitDetections: List[DigitDetection] = field(default_factory=list)
