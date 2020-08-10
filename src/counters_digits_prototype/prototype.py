import cv2
import numpy as np
from trvo_utils.imutils import imshowWait, bgr2rgb, zeros, fit_image_to_shape
from trvo_utils.path_utils import list_files

from DarknetOpencvDetector import DarknetOpencvDetector
from DarknetPytorchDetector import DarknetPytorchDetector
from consts import IMAGES_EXTENSIONS, FHD_SHAPE, BGRColors
from counter_digits.dataset_utils.extract_dataset import imgByBox
from utils.datasets import letterbox
from utils_local.vis_utils import fitImageDetectionsToShape, drawDetections, drawDigitsDetections


def createDetectors():
    cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
    weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.weights'
    yield DarknetPytorchDetector(cfg_file, weights_file, 320)
    yield DarknetOpencvDetector(cfg_file, weights_file, 320)

    # cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
    cfg_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.cfg"
    # weights_file = '../counter_digits/best_weights/3/yolov3-tiny-10cls-320.weights'
    weights_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.weights"
    yield DarknetPytorchDetector(cfg_file, weights_file, 320)
    yield DarknetOpencvDetector(cfg_file, weights_file, 320)


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
    noScreenImage = zeros((50, 150))
    screenClass = 1
    dirs = [
        "/hdd/Datasets/counters/8_from_phone",
        # "/hdd/Datasets/counters/Musson_counters_2",
        # "/hdd/Datasets/ElectroCounters/ElectroCounters_4/ElectroCounters/2020-08-01-19-26-39-482"
        # "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets"
    ]
    pytorchScreenDetector, opencvScreenDetector, pytorchDigitsDetector, opencvDigitsDetector = createDetectors()
    for img_file in enumerate_images(dirs):
        imgBgr = cv2.imread(img_file)

        detections = pytorchScreenDetector.detect(bgr2rgb(imgBgr))[0]
        screenImg = extractObjectImage(screenClass, imgBgr, detections, extraSpace=5)

        if screenImg is None:
            screenImg = noScreenImage
        else:
            digitDetections = pytorchDigitsDetector.detect(bgr2rgb(screenImg))[0]
            screenImg = drawDigitsDetections(screenImg, digitDetections, BGRColors.green)

        imgBgr, detections, _ = fitImageDetectionsToShape(imgBgr, detections, FHD_SHAPE)
        drawDetections(imgBgr, detections, BGRColors.green, withScores=True)

        if imshowWait(imgBgr, screenImg) == 27:
            break


if __name__ == '__main__':
    main()
