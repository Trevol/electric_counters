from builtins import sum
from itertools import groupby
from typing import List

import cv2
import numpy as np

from trvo_utils import toInt_array
from trvo_utils.box_utils import pointInBox, boxCenter
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import bgr2rgb, imgByBox
from trvo_utils.optFlow_trackers import RectTracker
from trvo_utils.timer import timeit
from trvo_utils.viz_utils import make_bgr_colors

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.TwoStageDigitsDetectionResult import TwoStageDigitsDetectionResult, DigitDetection
from detection.TwoStageDigitsDetector import TwoStageDigitsDetector, remapBox
from with_image_aligning.clustering_digits_extractor import ClusteringDigitsExtractor
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


class Show:
    @staticmethod
    def digitDetections(frame, framePos,
                        detections: List[DigitDetection],
                        showAsCenters):
        vis = Draw.digitDetections(frame.copy(), detections, showAsCenters)
        key = imshowWait([vis, framePos], frame)
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

    def frames(self):
        framesPath = "../../images/smooth_frames/2/*.jpg"
        frames = enumerate(FrameReader(framesPath, 1).read())
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
        digitExtractor = ClusteringDigitsExtractor()

        def mouseCallback(event, x, y, flags, userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            pt = x, y
            digitsOnPoint = [d.digit for d in prevDetections if pointInBox(d.boxInImage, pt)]
            stats = groupBy_count_desc(digitsOnPoint)
            print(stats)

        cv2.namedWindow("0")
        cv2.setMouseCallback("0", mouseCallback)

        prevDetections = []
        prevFrameGray = None
        for framePos, frameBgr, frameRgb, frameGray in self.frames():
            currentDetections = detector.detect(frameRgb).digitDetections
            # TODO: count number of frames from first observation (detection)
            trackedDetections = []
            if len(prevDetections) != 0:
                trackedDetections = self.trackDigitDetections(prevFrameGray, frameGray, prevDetections)

            prevDetections = trackedDetections + currentDetections
            prevFrameGray = frameGray

            # centers = digitExtractor.extractCenters_(prevDetections)
            # print(framePos, len(centers))
            # if Show.clustersCenters(frameBgr, framePos, centers) == 'esc':
            #     break

            if Show.digitDetections(frameBgr, framePos, prevDetections,
                                    showAsCenters=False) == 'esc':
                break



PrototypeApp().run()
