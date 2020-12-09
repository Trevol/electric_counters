import time
from typing import List

import cv2
import numpy as np

from trvo_utils import toInt_array
from trvo_utils.box_utils import boxCenter
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import bgr2rgb
from trvo_utils.timer import timeit
from trvo_utils.viz_utils import make_bgr_colors

from core.rect import Rect
from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.TwoStageDigitsDetectionResult import DigitDetection
from detection.TwoStageDigitsDetector import TwoStageDigitsDetector
from with_image_aligning.boxed_object_tracker import BoxedObjectTracker
from with_image_aligning.digits_extractors.aggregating_box_grouping_digit_extractor import \
    AggregatingBoxGroupingDigitExtractor
from with_image_aligning.digits_extractors.box_grouping_digit_extractor import BoxGroupingDigitExtractor
from with_image_aligning.digits_extractors.digit_at_point import DigitAtPoint
from with_image_aligning.digit_renderer import DigitRenderer
from with_image_aligning.frame_reader import FrameReader
from with_image_aligning.types.aggregated_detections import AggregatedDetections


class Draw:
    numOfDigits = 10
    colors = make_bgr_colors(numOfDigits)

    @staticmethod
    def rectangle(img, box, color, thickness=1):
        x1, y1, x2, y2 = toInt_array(box)
        return cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def rectangleCenter(img, xyxyBox, color, radius=1, thickness=-1):
        center = tuple(toInt_array(boxCenter(xyxyBox)))
        return cv2.circle(img, center, radius, color, thickness)

    @classmethod
    def digitDetections(cls, img,
                        detections: List[DigitDetection],
                        showAsCenters):
        for d in detections:
            digitColor = cls.colors[d.digit]
            if showAsCenters:
                cls.rectangleCenter(img, d.xyxyBoxInImage, digitColor)
            else:
                cls.rectangle(img, d.xyxyBoxInImage, digitColor)
        return img

    @staticmethod
    def clustersCenters(img, centers):
        for center in centers:
            center = tuple(toInt_array(center))
            cv2.circle(img, center, 1, (0, 255, 0), -1)
        return img

    digitRenderer = DigitRenderer(15, fontThickness=2)

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
                        detections: List[AggregatedDetections],
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

    @staticmethod
    def nmsBoxes_detections(
            img,
            imgPos,
            boxes: List[np.ndarray],
            detections: List[DigitDetection]):
        green = 0, 255, 0
        imgWithResultingBoxes = img.copy()
        for box in boxes:
            Draw.rectangle(imgWithResultingBoxes, box, green)

        Draw.digitDetections(img, detections, showAsCenters=False)

        key = imshowWait((imgWithResultingBoxes, imgPos), (img, imgPos))
        if key == 27:
            return 'esc'


class PrototypeApp:
    @staticmethod
    def saveDetections(detections, file):
        from pickle import dump, HIGHEST_PROTOCOL
        with open(file, "wb") as f:
            dump(detections, f, HIGHEST_PROTOCOL)

    @staticmethod
    def createDetector():
        cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
        weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/1/yolov3-tiny-2cls-320.weights'
        screenDetector = DarknetOpencvDetector(cfg_file, weights_file, 320)

        # cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
        cfg_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.cfg"
        weights_file = '../counter_digits/best_weights/4/yolov3-tiny-10cls-320.4.weights'
        # weights_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.weights"
        digitsDetector = DarknetOpencvDetector(cfg_file, weights_file, 320)

        return TwoStageDigitsDetector(screenDetector, digitsDetector)

    @staticmethod
    def frames(framesPath):
        frames = enumerate(FrameReader(framesPath, 1).read())
        frames = ((pos, bgr, bgr2rgb(bgr), gray) for pos, (bgr, gray) in frames)
        return frames

    @staticmethod
    def mouseCallback_example(event, x, y, flags, userdata):
        # if event != cv2.EVENT_LBUTTONDOWN:
        #     return
        # pt = x, y
        # digitsOnPoint = [d.digit for d in prevDetections if pointInBox(d.xyxyBoxInImage, pt)]
        # stats = groupBy_count_desc(digitsOnPoint)
        # boxes = [b for b in nmsBoxes if pointInBox(b, pt)]
        # print(boxes, stats)
        pass

    def run(self):
        framesPath = "../../images/smooth_frames/{}/*.jpg"

        detector = self.createDetector()

        # digitExtractor = ClusteringDigitsExtractor()
        digitExtractor = AggregatingBoxGroupingDigitExtractor()

        digitDetectionTracker = BoxedObjectTracker(
            xyxyBoxAccessor=lambda aggDet: aggDet.box.xyxy(),
            nextObjectMaker=lambda prevAggDet, xyxyBox: AggregatedDetections(Rect.fromXyxy(xyxyBox), prevAggDet.score,
                                                                             prevAggDet.digit_counts)
        )

        prevDetections: List[AggregatedDetections] = []
        prevFrameGray = None

        framePathId = 1
        for framePos, frameBgr, frameRgb, frameGray in self.frames(framesPath.format(framePathId)):
            if framePos % 20 == 0:
                print("framePos", framePos)

            currentDetections = detector.detect(frameRgb).digitDetections

            if len(prevDetections) != 0:
                prevDetections = digitDetectionTracker.track(prevFrameGray, frameGray, prevDetections)

            digitsAtPoints, prevDetections = digitExtractor.extract(currentDetections, prevDetections, -1)
            prevFrameGray = frameGray

            # if Show.digitDetections(frameBgr, framePos, currentDetections, digitsAtPoints,
            #                         showAsCenters=False) == 'esc':
            #     break

            if self.show_agg_detections(frameBgr, prevDetections) == 'esc':
                break

        print("framePos", framePos)

    def show_agg_detections(self, bgrFrame, detections: List[AggregatedDetections]):
        for d in detections:
            Draw.rectangle(bgrFrame, d.box.xyxy(), (0, 255, 0))
        if imshowWait(bgrFrame) == 27:
            return 'esc'


PrototypeApp().run()
