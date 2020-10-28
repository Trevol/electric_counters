from typing import List

import cv2
import numpy as np

from trvo_utils import toInt_array
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import bgr2rgb, imgByBox
from trvo_utils.optFlow_trackers import RectTracker
from trvo_utils.viz_utils import make_bgr_colors

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.TwoStageDigitsDetectionResult import TwoStageDigitsDetectionResult, DigitDetection
from detection.TwoStageDigitsDetector import TwoStageDigitsDetector, remapBox
from with_image_aligning.frame_reader import FrameReader


class PrototypeApp:
    framesPath = "../../images/smooth_frames/1/*.jpg"

    @staticmethod
    def createDetectors():
        cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
        weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.weights'
        yield DarknetOpencvDetector(cfg_file, weights_file, 320)

        # cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
        cfg_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.cfg"
        weights_file = '../counter_digits/best_weights/4/yolov3-tiny-10cls-320.4.weights'
        # weights_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.weights"
        yield DarknetOpencvDetector(cfg_file, weights_file, 320)

    @staticmethod
    def fromBoxInImage_to_boxInScreen(screenBoxInImage, digitBoxInImage):
        tl = screenBoxInImage[:2]
        digitBoxInScreenBox = digitBoxInImage - np.append(tl, tl)
        return digitBoxInScreenBox

    @classmethod
    def draw(cls, img, screenImg, currentDetection: TwoStageDigitsDetectionResult,
             trackedDetection: TwoStageDigitsDetectionResult):
        numOfDigits = 10
        colors = make_bgr_colors(numOfDigits)
        for d in currentDetection.digitDetections:
            x1, y1, x2, y2 = toInt_array(d.boxInImage)
            digitColor = colors[d.digit]
            cv2.rectangle(img, (x1, y1), (x2, y2), digitColor, 1)

            boxInScreenBox = cls.fromBoxInImage_to_boxInScreen(currentDetection.screenBox, d.boxInImage)

            x1, y1, x2, y2 = toInt_array(boxInScreenBox)
            cv2.rectangle(screenImg, (x1, y1), (x2, y2), digitColor, 1)

    @classmethod
    def show(cls, img, currentDetection: TwoStageDigitsDetectionResult, trackedDetection: TwoStageDigitsDetectionResult,
             framePos):
        if currentDetection is not None and currentDetection.screenBox is not None:
            screenImg = imgByBox(img, currentDetection.screenBox)
            cls.draw(img, screenImg, currentDetection, trackedDetection)
        else:
            screenImg = None

        key = imshowWait([img, framePos], screenImg)
        if key == 27:
            return 'esc'

    def frames(self):
        frames = enumerate(FrameReader(self.framesPath, 1).read())
        frames = ((pos, bgr, bgr2rgb(bgr), gray) for pos, (bgr, gray) in frames)
        return frames

    rectTracker = RectTracker()

    def trackDigitDetections(self, prevFrameGray, nextFrameGray,
                             prevDetection: List[DigitDetection]) -> List[DigitDetection]:
        prevBoxes = []
        # boxes, status = self.rectTracker.track(prevFrameGray, nextFrameGray, prevBoxes)
        # recreate detections with new boxes
        return prevDetection

    def run(self):
        screenDetector, digitsDetector = self.createDetectors()
        detector = TwoStageDigitsDetector(screenDetector, digitsDetector)
        prevDetection = None
        prevFrameGray = None
        for framePos, frameBgr, frameRgb, frameGray in self.frames():
            currentDetection = detector.detect(frameRgb)
            trackedDetection = self.trackDigitDetections(prevFrameGray, frameGray, prevDetection)
            if self.show(frameBgr, currentDetection, trackedDetection, framePos) == 'esc':
                break
            prevDetection = currentDetection
            prevFrameGray = frameGray


PrototypeApp().run()
