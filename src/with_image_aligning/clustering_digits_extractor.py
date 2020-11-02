from dataclasses import dataclass
from typing import List

import hdbscan
import numpy as np

from detection.TwoStageDigitsDetectionResult import DigitDetection


@dataclass
class DigitAtPoint:
    digit: int
    point: np.ndarray


class DigitsExtractorClustering:
    def __init__(self):
        self.clusterer = hdbscan.HDBSCAN()

    def extract(self, detections: List[DigitDetection]) -> List[DigitAtPoint]:
        # cluster
        pass
