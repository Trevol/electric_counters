from detection.DarknetOpencvDetector import DarknetOpencvDetector


class TwoStageDigitsDetector:
    def __init__(self, screenDetector, digitsDetector):
        self.screenDetector: DarknetOpencvDetector = screenDetector
        self.digitsDetector = digitsDetector

    def _detectScreen(self, rgbImage):
        self.screenDetector.detect(rgbImage)

    def detect(self, rgbImage):
        screenDetection = self._detectScreen(rgbImage)
