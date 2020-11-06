import cv2
import numpy as np
from trvo_utils.box_utils import pointInBox
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.timer import timeit


def main():
    img = np.zeros([400, 400], np.uint8)
    pt1 = 123, 123
    pt2 = 23, 23
    cv2.rectangle(img, pt1, pt2, 255, 1)
    imshowWait(img)


main()
