from dataclasses import dataclass
import numpy as np


@dataclass
class TwoStageDigitsDetectionResult:
    counterBox: np.ndarray = None
    screenBox: np.ndarray = None
