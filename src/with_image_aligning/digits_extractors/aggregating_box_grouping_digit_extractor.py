from itertools import groupby
from typing import List, Tuple, Iterable

from trvo_utils.box_utils import xyxy2xywh, boxCenter
from trvo_utils.iter_utils import groupBy, flatten
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

    @staticmethod
    def merge(digitsCountsByBox: Iterable[List[DigitCount]]) -> List[DigitCount]:
        result = []
        for digit, digitCount_by_digit in groupBy(flatten(digitsCountsByBox),
                                                  key=lambda dc: dc.digit):
            cnt = sum(d.count for d in digitCount_by_digit)
            result.append(DigitCount(digit, cnt))
        return result

    def extract(self, currentDetections: List[DigitDetection], prevDetections: List[AggregatedDetections],
                numOfObservations) -> Tuple[List[DigitAtPoint], List[AggregatedDetections]]:
        boxes, scores, digitsCounts = self._boxes_scores_digitsCounts(currentDetections, prevDetections)
        groupIndices, keptIndices = groupBoxes(
            boxes=boxes,
            scores=scores,
            overlap_threshold=.04)

        aggregatedDetections: List[AggregatedDetections] = []
        for index, digitsCounts_by_index in groupBy(zip(groupIndices, digitsCounts),
                                                    key=_index_accessor,
                                                    groupSelect=lambda ix_digitsCounts: ix_digitsCounts[1]):
            box = boxes[index]
            score = scores[index]
            digitsCounts = self.merge(digitsCounts_by_index)
            aggregatedDetections.append(AggregatedDetections(box, score, digitsCounts))

        digits: List[DigitAtPoint] = []
        for aggDet in aggregatedDetections:
            if aggDet.totalCount() < self.minBoxesInGroup:
                continue
            digit = aggDet.digitWithMaxCount()
            xyxyBox = aggDet.box.xyxy()
            digits.append(DigitAtPoint(digit, boxCenter(xyxyBox), xyxyBox))

        return digits, aggregatedDetections


def _index_accessor(indexed_items):
    return indexed_items[0]


def count(iterable):
    return sum(1 for _ in iterable)
