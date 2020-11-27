from typing import List

import cv2
import numpy as np
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.timer import timeit

from with_image_aligning.digits_extractors.box_grouping_digit_extractor import BoxGroupingDigitExtractor
from with_image_aligning.digits_extractors.digit_at_point import DigitAtPoint
from with_image_aligning.digits_extractors.tests.test_utils import loadDetections, showDigits


def main():
    detections, numOfObservations = loadDetections(4)
    digitExtractor = BoxGroupingDigitExtractor()
    digitsAtPoints = digitExtractor.extract(detections, numOfObservations)
    print(len(detections))
    for _ in range(5):
        with timeit():
            digitsAtPoints = digitExtractor.extract(detections, numOfObservations)
    # showDigits(digitsAtPoints)


main()
