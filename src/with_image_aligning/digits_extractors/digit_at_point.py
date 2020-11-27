from dataclasses import dataclass

import numpy as np


@dataclass
class DigitAtPoint:
    digit: int
    point: np.ndarray
    xyxyBox: np.ndarray