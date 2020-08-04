import cv2
import numpy as np
from trvo_utils.imutils import imshowWait, bgr2rgb, zeros
from trvo_utils.path_utils import list_files

from DarknetDetector import DarknetDetector
from consts import IMAGES_EXTENSIONS, FHD_SHAPE, BGRColors
from counter_digits.dataset_utils.extract_dataset import imgByBox
from utils_local.vis_utils import fitImageDetectionsToShape, drawDetections, drawDigitsDetections


def createDetectors():
    cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
    weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.weights'
    screenDetector = DarknetDetector(cfg_file, weights_file, 320)
    yield screenDetector

    cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
    cfg_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.cfg"
    weights_file = '../counter_digits/best_weights/3/yolov3-tiny-10cls-320.weights'
    weights_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.weights"
    digitsDetector = DarknetDetector(cfg_file, weights_file, 320)
    yield digitsDetector


def enumerate_images(dirs):
    return list_files(dirs, IMAGES_EXTENSIONS)


def extractObjectImage(desiredClass, img, detections, noImage=None, extraSpace=0):
    desiredDetection = next(filter(lambda d: d[5] == desiredClass, detections), None)
    if desiredDetection is None:
        return noImage
    box = desiredDetection[:4]
    objectImg = imgByBox(img, box, extraSpace).copy()
    return objectImg


def main():
    bgrImage = cv2.imread("/home/trevol/IdeaProjects/HelloKotlin/screenBgrImage.png")
    _, digitsDetector = createDetectors()
    digitsDetector.detect()

if __name__ == '__main__':
    main()
