from glob import glob
import os

import cv2
from trvo_utils.annotation import PascalVocXmlParser
from trvo_utils.imutils import imSize

from utils.utils import load_classes


def convert():
    labels = load_classes("counter_digits/data/digits.names")
    vocDirs = [
        "/hdd/Datasets/counters/Musson_counters/train/digits",
        "/hdd/Datasets/counters/Musson_counters/val/digits/"
    ]
    yoloFileAnnotations = []

    for vocDir in vocDirs:
        for annFile in sorted(glob(os.path.join(vocDir, "*.xml"))):
            p = PascalVocXmlParser(annFile)
            imgExt = os.path.splitext(p.filename())[1]
            fileName = os.path.splitext(annFile)[0]
            yoloFile = fileName + '.txt'

            imHeight, imWidth = imSize(cv2.imread(fileName + imgExt))
            assert p.size() == (imWidth, imHeight)

            yoloAnnotations = []
            for (x1, y1, x2, y2), label in zip(p.boxes(), p.labels()):
                classId = labels.index(label)
                cx = (x1 + x2) / 2 / imWidth
                cy = (y1 + y2) / 2 / imHeight
                w = (x2 - x1) / imWidth
                h = (y2 - y1) / imHeight
                yoloAnnotations.append((classId, cx, cy, w, h))
            yoloFileAnnotations.append((yoloFile, yoloAnnotations))

    for yoloFile, yoloAnnotations in yoloFileAnnotations:
        with open(yoloFile, "wt") as f:
            for classId, cx, cy, w, h in yoloAnnotations:
                f.write(f"{classId} {cx} {cy} {w} {h}\n")


if __name__ == '__main__':
    convert()
