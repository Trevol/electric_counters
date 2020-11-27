from typing import List

import cv2
import numpy as np
from trvo_utils.cv2gui_utils import imshowWait

from with_image_aligning.digits_extractors.digit_at_point import DigitAtPoint


def loadDetections(id):
    from pickle import load
    with open(f"digit_detections_{id}.pcl", "rb") as f:
        numOfObservations = 386
        return load(f), numOfObservations


def getFontScale(text, fontFace, desiredHeight, thickness):
    fontScale = 20
    (w, h), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
    return fontScale * desiredHeight / h


def showDigits(digitsAtPoints: List[DigitAtPoint]):
    if len(digitsAtPoints) == 0:
        return
    maxX = int(max(digitsAtPoints, key=lambda d: d.point[0]).point[0])
    maxY = int(max(digitsAtPoints, key=lambda d: d.point[1]).point[1])
    img = np.full([maxY + 100, maxX + 100], 127, np.uint8)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontThickness = 1
    fontScale = getFontScale("1", fontFace, 30, fontThickness)

    for digitAtPoint in digitsAtPoints:
        digitStr = str(digitAtPoint.digit)
        textPt = tuple(np.int32(digitAtPoint.point))
        cv2.putText(img, digitStr, textPt, fontFace, fontScale, 255, fontThickness)
    imshowWait(img)
