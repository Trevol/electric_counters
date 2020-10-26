from typing import List, Union, Tuple

from trvo_utils.imutils import imgByBox

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.ObjectDetectionResult import ObjectDetectionResult
from detection.TwoStageDigitsDetectionResult import TwoStageDigitsDetectionResult


class TwoStageDigitsDetector:
    counterClass = 0
    screenClass = 1

    def __init__(self, screenDetector, digitsDetector):
        self.screenDetector: DarknetOpencvDetector = screenDetector
        self.digitsDetector = digitsDetector

    def _detectScreen(self, rgbImage) -> Tuple[ObjectDetectionResult, ObjectDetectionResult]:
        detections = self.screenDetector.detect(rgbImage)
        counterDetections = list(filter(lambda d: d.classId == self.counterClass, detections))
        screenDetections = list(filter(lambda d: d.classId == self.screenClass, detections))
        # TODO: если
        assert len(counterDetections) <= 1
        assert len(screenDetections) <= 1
        # extract counter and screen
        return _firstOrDefault(counterDetections), _firstOrDefault(screenDetections)

    def detect(self, rgbImage) -> TwoStageDigitsDetectionResult:
        counterDetection, screenDetection = self._detectScreen(rgbImage)
        if counterDetection is None and screenDetection is None:
            return None
        if screenDetection is None:
            return TwoStageDigitsDetectionResult(
                counterBox=counterDetection.box if counterDetection is not None else None)

        # extract screen image
        imgByBox(rgbImage, screenDetection.box, )
        # detect digits
        # return
        #   counterBox,
        #   screenBox,
        #   screenImage - do we really need this?????
        #   digits:
        #       boxInScreen - coordinates relative to screenBox
        #       boxInImage - with coordinates relative to image
        #       class - 0, 1, 2, 3, ... 9
        pass


def _firstOrDefault(src: Union[List, Tuple], default=None):
    if len(src) == 0:
        return default
    return src[0]
