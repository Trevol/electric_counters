from builtins import sum
from itertools import groupby
from typing import List

import cv2
import numpy as np

from trvo_utils import toInt_array
from trvo_utils.box_utils import pointInBox, boxCenter, boxSizeWH
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import bgr2rgb
from trvo_utils.optFlow_trackers import RectTracker
from trvo_utils.viz_utils import make_bgr_colors

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.TwoStageDigitsDetectionResult import DigitDetection
from detection.TwoStageDigitsDetector import TwoStageDigitsDetector
from with_image_aligning.clustering_digits_extractor import ClusteringDigitsExtractor, DigitAtPoint
from with_image_aligning.digit_renderer import DigitRenderer
from with_image_aligning.frame_reader import FrameReader


class Draw:
    numOfDigits = 10
    colors = make_bgr_colors(numOfDigits)

    @staticmethod
    def rectangle(img, box, color, thickness=1):
        x1, y1, x2, y2 = toInt_array(box)
        return cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def rectangleCenter(img, box, color, radius=1, thickness=-1):
        center = tuple(toInt_array(boxCenter(box)))
        return cv2.circle(img, center, radius, color, thickness)

    @classmethod
    def digitDetections(cls, img,
                        detections: List[DigitDetection],
                        showAsCenters):
        for d in detections:
            digitColor = cls.colors[d.digit]
            if showAsCenters:
                cls.rectangleCenter(img, d.boxInImage, digitColor)
            else:
                cls.rectangle(img, d.boxInImage, digitColor)
        return img

    @staticmethod
    def clustersCenters(img, centers):
        for center in centers:
            center = tuple(toInt_array(center))
            cv2.circle(img, center, 1, (0, 255, 0), -1)
        return img

    digitRenderer = DigitRenderer(15)

    @classmethod
    def digitsAtPoints(cls, img, digitsAtPoints: List[DigitAtPoint]):
        if len(digitsAtPoints) == 0:
            return None
        for digitAtPoint in digitsAtPoints:
            cls.digitRenderer.render(img, digitAtPoint.digit, digitAtPoint.point)
        return img


class Show:
    @staticmethod
    def digitDetections(frame, framePos,
                        detections: List[DigitDetection],
                        digitsAtPoints: List[DigitAtPoint],
                        showAsCenters):
        detectionsImg = Draw.digitDetections(frame.copy(), detections, showAsCenters)
        digitsImg = Draw.digitsAtPoints(frame.copy(), digitsAtPoints)
        key = imshowWait([detectionsImg, framePos], frame, digitsImg)
        if key == 27:
            return 'esc'

    @staticmethod
    def clustersCenters(frame, framePos, centers):
        vis = Draw.clustersCenters(frame.copy(), centers)
        key = imshowWait([vis, framePos], frame)
        if key == 27:
            return 'esc'


def groupBy_count_desc(items):
    def count_(items):
        return sum(1 for _ in items)

    def countGroup(group_items):
        return group_items[0], count_(group_items[1])

    def countSelector(group_count):
        return group_count[1]

    grpBy = groupby(sorted(items))
    return sorted(map(countGroup, grpBy), key=countSelector, reverse=True)


class PrototypeApp:
    rectTracker = RectTracker()

    @staticmethod
    def saveDetections(detections, file):
        from pickle import dump, HIGHEST_PROTOCOL
        with open(file, "wb") as f:
            dump(detections, f, HIGHEST_PROTOCOL)

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

    @staticmethod
    def frames(framesPath):
        frames = enumerate(FrameReader(framesPath, 1).read())
        frames = ((pos, bgr, bgr2rgb(bgr), gray) for pos, (bgr, gray) in frames)
        return frames

    def trackDigitDetections(self, prevFrameGray, nextFrameGray,
                             prevDetections: List[DigitDetection]) -> List[DigitDetection]:
        prevBoxes = [d.boxInImage for d in prevDetections]
        boxes, status = self.rectTracker.track(prevFrameGray, nextFrameGray, prevBoxes)

        nextDetections = []
        for prevDetection, box, boxStatus in zip(prevDetections, boxes, status):
            if not boxStatus or self.isAbnormalTrack(prevDetection.boxInImage, box):
                continue
            nextDetections.append(DigitDetection(prevDetection.digit, prevDetection.score, box))
        return nextDetections

    @staticmethod
    def isAbnormalTrack(prevBox, nextBox):
        x1 = nextBox[0]
        y1 = nextBox[1]
        x2 = nextBox[2]
        y2 = nextBox[3]
        if x1 >= x2 or y1 >= y2:
            return True

        # flow = nextBox - prevBox

        # measure sides of prev and next boxes
        prevBoxWH = boxSizeWH(prevBox)
        nextBoxWH = boxSizeWH(nextBox)
        wRatio = prevBoxWH[0] / nextBoxWH[0]
        hRatio = prevBoxWH[1] / nextBoxWH[1]
        delta = .25
        isNormalRatio = 1 + delta > wRatio > 1 - delta and 1 + delta > hRatio > 1 - delta
        if not isNormalRatio:
            return True

        return False

    def run(self):
        def mouseCallback(event, x, y, flags, userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            pt = x, y
            digitsOnPoint = [d.digit for d in prevDetections if pointInBox(d.boxInImage, pt)]
            stats = groupBy_count_desc(digitsOnPoint)
            print(stats)

        framesPath = "../../images/smooth_frames/{}/*.jpg"

        detector = self.createDetector()
        digitExtractor = ClusteringDigitsExtractor()

        cv2.namedWindow("0")
        cv2.setMouseCallback("0", mouseCallback)

        prevDetections = []
        prevFrameGray = None

        framePathId = 3
        for framePos, frameBgr, frameRgb, frameGray in self.frames(framesPath.format(framePathId)):
            print("framePos", framePos)
            currentDetections = detector.detect(frameRgb).digitDetections
            trackedDetections = []
            if len(prevDetections) != 0:
                trackedDetections = self.trackDigitDetections(prevFrameGray, frameGray, prevDetections)

            prevDetections = trackedDetections + currentDetections
            prevFrameGray = frameGray

            digitsAtPoints = digitExtractor.extract(prevDetections, -1)

            # centers = digitExtractor.extractCenters_(prevDetections)
            # print(framePos, len(centers))
            # if Show.clustersCenters(frameBgr, framePos, centers) == 'esc':
            #     break

            if Show.digitDetections(frameBgr, framePos, prevDetections, digitsAtPoints,
                                    showAsCenters=False) == 'esc':
                break

        self.saveDetections(prevDetections, f"digit_detections_{framePathId}.pcl")


PrototypeApp().run()
