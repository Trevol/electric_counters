from dataclasses import dataclass
from glob import glob
import os
from typing import Tuple

import cv2
from trvo_utils.cv2gui_utils import imshowWait_WFK
from trvo_utils.imutils import bgr2rgb

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from consts import BGRColors
from utils_local.vis_utils import drawDetections


class DetectionLogDir:
    @dataclass
    class ParsedPath:
        stamp: str
        num: int
        path: str
        key: Tuple = None

        def __post_init__(self):
            self.key = (self.stamp, self.num)

    def __init__(self, logDir):
        self.logDir = logDir

    def inputFrames(self):
        parsedNames = []
        for path in glob(os.path.join(self.logDir, "*_input.jpg")):
            stamp, id, _ = os.path.basename(path).split("_")
            parsedNames.append(self.ParsedPath(stamp, int(id), path))
        parsedNames.sort(key=lambda p: p.key)
        return (p.path for p in parsedNames)

    def __sample(self):
        """
        2020-09-15-10-30-27-849_35_detectionResult.txt
        2020-09-15-10-30-27-849_35_digits.jpg
        2020-09-15-10-30-27-849_35_input.jpg
        2020-09-15-10-30-27-849_35_inputDrawing.jpg
        2020-09-15-10-30-27-849_35_screen.jpg
        2020-09-15-10-30-27-849_35_screenDrawing.jpg
        """
        pass


def createDetectors():
    cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
    weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.weights'
    yield DarknetOpencvDetector(cfg_file, weights_file, 320)

    # cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
    cfg_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.cfg"
    weights_file = '../counter_digits/best_weights/4/yolov3-tiny-10cls-320.4.weights'
    # weights_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.weights"
    yield DarknetOpencvDetector(cfg_file, weights_file, 320)


def main():
    counter_screen_colors = {
        0: BGRColors.green,
        1: BGRColors.red
    }
    screenDetector, digitsDetector = createDetectors()
    logsDir = "/hdd/Datasets/counters/data/detections_log/1"
    for inputPath in DetectionLogDir(logsDir).inputFrames():
        img = cv2.imread(inputPath)

        detections = screenDetector.detect(bgr2rgb(img))[0]
        drawDetections(img, detections, counter_screen_colors, withScores=True)

        if imshowWait_WFK(img, waitForKeys=[27, ord(' ')]) == 27:
            break


main()
