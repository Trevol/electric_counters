from dataclasses import dataclass
from typing import List

import hdbscan
import numpy as np
from trvo_utils.box_utils import boxCenter

from detection.TwoStageDigitsDetectionResult import DigitDetection


@dataclass
class DigitAtPoint:
    digit: int
    point: np.ndarray


class ClusteringDigitsExtractor:
    def __init__(self):
        self.clusterer = hdbscan.HDBSCAN()

    def extract(self, detections: List[DigitDetection]) -> List[DigitAtPoint]:
        # cluster by box centers
        centers = [boxCenter(d.boxInImage) for d in detections]
        centers = np.float32(centers)
        self.clusterer.fit(centers)

        clustersIds = self.clusterer.labels_
        probabilities = self.clusterer.probabilities_
        clustersCenters = {}  # clusterId: (center, probability)

        for center, clusterId, probability in zip(centers, clustersIds, probabilities):
            if clusterId == -1:  # skip noise
                continue
            # for each cluster select most prominent point (with highest probability)
            prevCenter, prevProba = clustersCenters.get(clusterId, (None, None))
            if prevCenter is None or probability > prevProba:
                clustersCenters[clusterId] = center, probability

        return [center for center, proba in clustersCenters.values()]
