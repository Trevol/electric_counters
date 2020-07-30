import cv2
from trvo_utils import toInt_array
from trvo_utils.imutils import fit_image_to_shape, scaleBox

from consts import BGRColors


def __makeLabel(klass, score, showClass, showScore):
    assert showClass or showScore
    score = int(score * 100)  # score to int: .868 -> 87 - for space economy
    klass = int(klass)
    if showClass and showScore:
        return f"{klass} {score}"
    if showClass:
        return str(klass)
    return str(score)


def drawDetections(img, detections, color, withClasses=False, withScores=False):
    showLabel = withClasses or withScores
    for *xyxy, score, klass in detections:
        x1, y1, x2, y2 = toInt_array(xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if showLabel:
            lbl = __makeLabel(klass, score, withClasses, withScores)
            cv2.putText(img, lbl, (x1 + 2, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, .8, color)
    return img


def fitImageDetectionsToShape(img, detections, dstShape):
    img, scale = fit_image_to_shape(img, dstShape)
    return img, scaleDetections(detections, scale), scale


def scaleDetections(detections, scale):
    if scale == 1:
        return detections
    detections = [[*scaleBox(box, scale), score, klass] for *box, score, klass in detections]
    return detections
