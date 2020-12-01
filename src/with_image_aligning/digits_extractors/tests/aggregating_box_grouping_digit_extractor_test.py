from typing import List

import cv2
import numpy as np
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.timer import timeit

from with_image_aligning.digits_extractors.aggregating_box_grouping_digit_extractor import \
    AggregatingBoxGroupingDigitExtractor
from with_image_aligning.digits_extractors.tests.test_utils import showDigits


def loadDetections(id):
    from pickle import load
    with open(f"digit_detections_frames_{id}.pcl", "rb") as f:
        numOfObservations = 386
        return load(f), numOfObservations


def main():
    detectionsPerFrame, numOfObservations = loadDetections(1)
    digitExtractor = AggregatingBoxGroupingDigitExtractor()
    aggregatedDetections = []
    for frameNum, detections in enumerate(detectionsPerFrame):
        print(frameNum)
        digitsAtPoints, aggregatedDetections = digitExtractor.extract(detections, aggregatedDetections,
                                                                      numOfObservations)
        showDigits(digitsAtPoints)


main()
