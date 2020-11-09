from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import List, Iterable, Tuple

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
        self.clusterer = hdbscan.HDBSCAN(metric="l2", cluster_selection_epsilon=10, cluster_selection_method='eom')

    @staticmethod
    def _cluster(cluster_obj):
        # cluster_obj is tuple(cluster, some_object)
        return cluster_obj[0]

    def extract(self, detections: List[DigitDetection], numOfObservations) -> List[DigitAtPoint]:
        if len(detections) < 3:
            return list()
        centers = np.float32([boxCenter(d.boxInImage) for d in detections])
        self.clusterer.fit(centers)  # cluster by box centers

        # filter outliers (noise, cluster = -1) and sort for grouping
        clusters_detections_probs = zip(self.clusterer.labels_, detections, self.clusterer.probabilities_)
        sorted_denoised_clusters_detections_probs = sorted(
            (o for o in clusters_detections_probs
             if self._cluster(o) != -1),
            key=self._cluster)

        def computeDigitAtClusterCenter(detectionsCluster: Iterable[Tuple[int, DigitDetection, float]]) -> DigitAtPoint:
            # count detection for each digit, find cluster "center"
            digit_count = np.zeros(10, np.int32)  # index is digit, value is count
            center, center_probability = None, 0
            for _, detection, prob in detectionsCluster:
                if prob > center_probability:  # track point with max probability. It will be used as cluster center
                    center = boxCenter(detection.boxInImage)
                    center_probability = prob
                digit_count[detection.digit] += 1

            digit = digit_count.argmax()  # index with max count
            return DigitAtPoint(digit, center)

        digitsAtPoints = [
            computeDigitAtClusterCenter(detectionsCluster) for _, detectionsCluster in
            groupby(sorted_denoised_clusters_detections_probs, key=self._cluster)
        ]

        # sort by point.x
        digitsAtPoints.sort(key=lambda d: d.point[0])
        return digitsAtPoints

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
