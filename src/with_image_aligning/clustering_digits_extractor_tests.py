from typing import List

import cv2
import numpy as np
from trvo_utils.cv2gui_utils import imshowWait

from with_image_aligning.clustering_digits_extractor import ClusteringDigitsExtractor, DigitAtPoint


def loadDetections():
    from pickle import load
    with open("digit_detections_1.pcl", "rb") as f:
        numOfObservations = 386
        return load(f), numOfObservations


def showDigits(digitsAtPoints: List[DigitAtPoint]):
    if len(digitsAtPoints) == 0:
        return
    maxX = int(max(digitsAtPoints, key=lambda d: d.point[0]).point[0])
    maxY = int(max(digitsAtPoints, key=lambda d: d.point[1]).point[1])
    img = np.full([maxY + 100, maxX + 100], 127, np.uint8)
    for digitAtPoint in digitsAtPoints:
        cv2.putText(img, str(digitAtPoint.digit), tuple(np.int32(digitAtPoint.point)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    imshowWait(img)


def main():
    detections, numOfObservations = loadDetections()
    extractor = ClusteringDigitsExtractor()
    digitsAtPoints = extractor.extract(detections, numOfObservations)

    showDigits(digitsAtPoints)


main()
