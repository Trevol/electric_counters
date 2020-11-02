from dataclasses import dataclass
from typing import List
import numpy as np
from detection.TwoStageDigitsDetectionResult import DigitDetection


@dataclass
class DigitAtPoint:
    digit: int
    point: np.ndarray


class DigitsExtractorClustering:
    def extract(self, detections: List[DigitDetection]):
        pass
