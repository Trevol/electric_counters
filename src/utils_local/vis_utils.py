from typing import List

import cv2
import numpy as np
from trvo_utils import toInt_array
from trvo_utils.imutils import fit_image_to_shape, scaleBox

from consts import BGRColors
from detection.ObjectDetectionResult import ObjectDetectionResult


def __makeLabel(klass, score, showClass, showScore):
    assert showClass or showScore
    score = int(score * 100)  # score to int: .868 -> 87 - for space economy
    if showClass and showScore:
        return f"{klass} {score}"
    if showClass:
        return str(klass)
    return str(score)


def drawDetections(img, detections: List[ObjectDetectionResult], color, withClasses=False, withScores=False,
                   fontScale=.8):
    def _color(klass):
        if isinstance(color, (list, dict)):
            return color[klass]
        return color

    showLabel = withClasses or withScores
    for det in detections:
        x1, y1, x2, y2 = toInt_array(det.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), _color(det.classId), 1)
        if showLabel:
            lbl = __makeLabel(det.classId, det.classScore, withClasses, withScores)
            cv2.putText(img, lbl, (x1 + 2, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, fontScale, _color(det.classId))
    return img


def drawDigitsDetections(img, detections: List[ObjectDetectionResult], color=BGRColors.green):
    imgWithClasses = np.zeros_like(img)
    imgWithScores = np.zeros_like(img)

    drawDetections(img, detections, color)
    drawDetections(imgWithClasses, detections, color, withClasses=True)
    drawDetections(imgWithScores, detections, color, withScores=True, fontScale=.4)

    result = np.vstack([img, imgWithClasses, imgWithScores])
    return result


def fitImageDetectionsToShape(img, detections: List[ObjectDetectionResult], dstShape):
    img, scale = fit_image_to_shape(img, dstShape)
    return img, scaleDetections(detections, scale), scale


def scaleDetections(detections: List[ObjectDetectionResult], scale):
    if scale == 1:
        return detections
    detections = [
        ObjectDetectionResult(d.classId, d.classScore, scaleBox(d.box, scale))
        for d
        in detections
    ]
    return detections
