from itertools import groupby
from typing import List, Tuple

from trvo_utils.box_utils import xyxy2xywh, boxCenter
from trvo_utils.iter_utils import groupBy
from trvo_utils.timer import timeit

from core.rect import Rect
from detection.TwoStageDigitsDetectionResult import DigitDetection
from with_image_aligning.digits_extractors.digit_at_point import DigitAtPoint
from with_image_aligning.digits_extractors.group_boxes import groupBoxes
from with_image_aligning.types.aggregated_detections import AggregatedDetections, DigitCount


class AggregatingBoxGroupingDigitExtractor:
    minBoxesInGroup = 3

    @staticmethod
    def _boxes_scores_digitsCounts(currentDetections: List[DigitDetection],
                                   prevDetections: List[AggregatedDetections]) -> \
            Tuple[
                List[Rect], List[float], List[List[DigitCount]]
            ]:
        boxes = [d.boxInImage for d in currentDetections]
        prevBoxes = [d.box for d in prevDetections]
        boxes.extend(prevBoxes)

        scores = [d.score for d in currentDetections]
        prevScores = [d.score for d in prevDetections]
        scores.extend(prevScores)

        digitsCounts = [[DigitCount(d.digit, 1)] for d in currentDetections]
        prevDigitsCounts = [d.digit_counts for d in prevDetections]
        digitsCounts.extend(prevDigitsCounts)
        return boxes, scores, digitsCounts

    def extract(self, currentDetections: List[DigitDetection], prevDetections: List[AggregatedDetections],
                numOfObservations) -> Tuple[List[DigitAtPoint], List[AggregatedDetections]]:
        boxes, scores, digitsCounts = self._boxes_scores_digitsCounts(currentDetections, prevDetections)
        groupIndices, keptIndices = groupBoxes(
            boxes=boxes,
            scores=scores,
            overlap_threshold=.04)

        aggregatedDetections = []
        digits = []

        # zip(groupIndices, detections).groupBy(key=ind_det=>ind_det.index, groupSelect=ind_det=>ind_det.detection)
        for index, detections_by_index in groupBy(zip(groupIndices, detections),
                                                  key=_index_accessor,
                                                  groupSelect=lambda ix_det: ix_det[1]):
            detections_by_index = list(detections_by_index)
            if len(detections_by_index) < self.minBoxesInGroup:
                continue
            # detections.groupBy(d=>d.digit).select(gr=>gr.Count())
            digit_count = groupBy(detections_by_index,
                                  key=lambda d: d.digit,
                                  groupAggregate=lambda det_by_digit: count(det_by_digit))
            # digit with max count of detections
            digit = max(digit_count, key=lambda dig_cnt: dig_cnt[1])[0]
            xyxyBox = detections[index].xyxyBoxInImage
            point = boxCenter(xyxyBox)
            digits.append(DigitAtPoint(digit, point, xyxyBox))

        return digits, aggregatedDetections


def _index_accessor(indexed_items):
    return indexed_items[0]


def count(iterable):
    return sum(1 for _ in iterable)
