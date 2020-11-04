from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
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
        self.clusterer = hdbscan.HDBSCAN(metric="l2", cluster_selection_epsilon=5, cluster_selection_method='eom')

    @staticmethod
    def _cluster(cluster_obj):
        # cluster_obj is tuple(cluster, some_object)
        return cluster_obj[0]

    def extract(self, detections: List[DigitDetection], numOfObservations) -> List[DigitAtPoint]:
        # cluster by box centers
        centers = [boxCenter(d.boxInImage) for d in detections]
        centers = np.float32(centers)
        self.clusterer.fit(centers)

        clusters = self.clusterer.labels_
        probs = self.clusterer.probabilities_

        result = list()
        clusteredDetections = sorted(
            (o for o in zip(clusters, detections, probs) if self._cluster(o) != -1),
            key=self._cluster)
        for cluster, detectionsGroup in groupby(clusteredDetections, key=self._cluster):
            # count detection for each digit, find cluster "center"
            digit_count = {}
            center, center_probability = None, 0
            detection: DigitDetection
            for _, detection, prob in detectionsGroup:
                if prob > center_probability:
                    center = boxCenter(detection.boxInImage)
                    center_probability = prob
                digit = detection.digit
                digit_count[digit] = digit_count.get(digit, 0) + 1
                # TODO: can we track digit with max count here???

            assert center is not None
            assert len(digit_count) > 0

            digit = max(digit_count.items(), key=itemgetter(1))[0]
            result.append(DigitAtPoint(digit, center))

        return result

    def calc_digit_at_point(self):
        pass

    def extractCenters_(self, detections: List[DigitDetection]) -> List[DigitAtPoint]:
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


def iterCount(iter):
    return sum(1 for _ in iter)
