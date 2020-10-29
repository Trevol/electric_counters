from typing import List

import cv2
import numpy as np

from trvo_utils import toInt_array
from trvo_utils.box_utils import pointInBox
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import bgr2rgb, imgByBox
from trvo_utils.optFlow_trackers import RectTracker
from trvo_utils.viz_utils import make_bgr_colors

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.TwoStageDigitsDetectionResult import TwoStageDigitsDetectionResult, DigitDetection
from detection.TwoStageDigitsDetector import TwoStageDigitsDetector, remapBox
from with_image_aligning.frame_reader import FrameReader


class Draw:
    numOfDigits = 10
    colors = make_bgr_colors(numOfDigits)

    @classmethod
    def digitDetections(cls, img, currentDetections: List[DigitDetection],
                        trackedDetections: List[DigitDetection]):
        allDetections = currentDetections + trackedDetections
        for d in allDetections:
            x1, y1, x2, y2 = toInt_array(d.boxInImage)
            digitColor = cls.colors[d.digit]
            cv2.rectangle(img, (x1, y1), (x2, y2), digitColor, 1)


class Show:
    @staticmethod
    def digitDetections(frame, framePos,
                        currentDetections: List[DigitDetection],
                        trackedDetections: List[DigitDetection]):
        Draw.digitDetections(frame, currentDetections, trackedDetections)
        key = imshowWait([frame, framePos])
        if key == 27:
            return 'esc'


class PrototypeApp:
    framesPath = "../../images/smooth_frames/1/*.jpg"
    rectTracker = RectTracker()

    @staticmethod
    def createDetector():
        cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
        weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.weights'
        screenDetector = DarknetOpencvDetector(cfg_file, weights_file, 320)

        # cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
        cfg_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.cfg"
        weights_file = '../counter_digits/best_weights/4/yolov3-tiny-10cls-320.4.weights'
        # weights_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.weights"
        digitsDetector = DarknetOpencvDetector(cfg_file, weights_file, 320)

        return TwoStageDigitsDetector(screenDetector, digitsDetector)

    @staticmethod
    def fromBoxInImage_to_boxInScreen(screenBoxInImage, digitBoxInImage):
        tl = screenBoxInImage[:2]
        digitBoxInScreenBox = digitBoxInImage - np.append(tl, tl)
        return digitBoxInScreenBox

    def frames(self):
        frames = enumerate(FrameReader(self.framesPath, 1).read())
        frames = ((pos, bgr, bgr2rgb(bgr), gray) for pos, (bgr, gray) in frames)
        return frames

    def trackDigitDetections(self, prevFrameGray, nextFrameGray,
                             prevDetections: List[DigitDetection]) -> List[DigitDetection]:
        prevBoxes = [d.boxInImage for d in prevDetections]
        boxes, status = self.rectTracker.track(prevFrameGray, nextFrameGray, prevBoxes)
        nextDetections = []
        for d, box, boxStatus in zip(prevDetections, boxes, status):
            # TODO: check box status
            nextDetections.append(DigitDetection(d.digit, d.score, box))
        return nextDetections

    def run(self):
        detector = self.createDetector()

        def mouseCallback(event, x, y, flags, userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            pt = x, y
            digitsOnPoint = [d.digit for d in prevDetections if pointInBox(d.boxInImage, pt)]
            print(digitsOnPoint)

        cv2.namedWindow("0")
        cv2.setMouseCallback("0", mouseCallback)

        prevDetections = []
        prevFrameGray = None
        for framePos, frameBgr, frameRgb, frameGray in self.frames():
            currentDetections = detector.detect(frameRgb).digitDetections
            trackedDetections = []
            if len(prevDetections) != 0:
                trackedDetections = self.trackDigitDetections(prevFrameGray, frameGray, prevDetections)

            prevDetections = trackedDetections + currentDetections
            prevFrameGray = frameGray

            if Show.digitDetections(frameBgr, framePos, currentDetections, trackedDetections) == 'esc':
                break


PrototypeApp().run()
