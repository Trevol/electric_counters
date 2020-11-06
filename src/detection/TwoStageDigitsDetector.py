from typing import List, Union, Tuple, Iterable
import numpy as np

from trvo_utils.box_utils import expandBox
from trvo_utils.imutils import imgByBox
from trvo_utils.iter_utils import firstOrDefault

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.ObjectDetectionResult import ObjectDetectionResult
from detection.TwoStageDigitsDetectionResult import TwoStageDigitsDetectionResult, DigitDetection


class TwoStageDigitsDetector:
    counterClass = 0
    screenClass = 1

    def __init__(self, screenDetector, digitsDetector):
        self.screenDetector: DarknetOpencvDetector = screenDetector
        self.digitsDetector: DarknetOpencvDetector = digitsDetector

    def _detectScreen(self, rgbImage) -> Tuple[ObjectDetectionResult, ObjectDetectionResult]:
        detections = self.screenDetector.detect(rgbImage)
        counterDetections = list(filter(lambda d: d.classId == self.counterClass, detections))
        screenDetections = list(filter(lambda d: d.classId == self.screenClass, detections))
        # TODO: если больше 2 обнаружений - выбирать те, которые ближе к центру изображения
        # assert len(counterDetections) <= 1, f"len(counterDetections)={len(counterDetections)}"
        assert len(screenDetections) <= 1, f"len(screenDetections)={len(screenDetections)}"
        # extract counter and screen
        return firstOrDefault(counterDetections), firstOrDefault(screenDetections)

    def detect(self, rgbImage) -> TwoStageDigitsDetectionResult:
        counterDetection, screenDetection = self._detectScreen(rgbImage)
        if counterDetection is None and screenDetection is None:
            return TwoStageDigitsDetectionResult(empty=True)
        if counterDetection is None:
            counterBox, counterScore = None, None
        else:
            counterBox, counterScore = counterDetection.box, counterDetection.classScore

        if screenDetection is None:
            return TwoStageDigitsDetectionResult(
                counterBox=counterBox,
                counterScore=counterScore
            )

        adjustedScreenBox = expandBox(screenDetection.box, .2, relative=True)
        screenImg = imgByBox(rgbImage, adjustedScreenBox)

        digitsDetections = self.digitsDetector.detect(screenImg)

        digitsDetections = [
            DigitDetection(
                digit=d.classId,
                score=d.classScore,
                boxInImage=remapBox(d.box, adjustedScreenBox)
            )
            for d in digitsDetections
        ]

        return TwoStageDigitsDetectionResult(
            counterBox=counterBox,
            counterScore=counterScore,
            screenBox=adjustedScreenBox,
            screenScore=screenDetection.classScore,
            digitDetections=digitsDetections
        )


def remapBox(box: np.ndarray, fromBox):
    # box holds x1y1x2y2-coords in fromBox
    # fromBox holds x1y1x2y2-coords in other Box
    # result is box with coords in other Box
    # example: digit box in screenBox remap to image box
    fromBox_topLeft = fromBox[:2]
    # for each point of box add displacement (top left point of fromBox)
    return box + np.append(fromBox_topLeft, fromBox_topLeft)
