import cv2
import numpy as np
from trvo_utils.imutils import imshowWait, bgr2rgb, zeros, imgByBox, IMAGES_EXTENSIONS, enumerate_images
from trvo_utils.path_utils import list_files

from DarknetOpencvDetector import DarknetOpencvDetector
from DarknetPytorchDetector import DarknetPytorchDetector
from consts import FHD_SHAPE, BGRColors
from counters_dataset_paths import paths
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


def extractObjectImage(desiredClass, img, detections, noImage=None, extraSpace=0):
    desiredDetection = next(filter(lambda d: d[5] == desiredClass, detections), None)
    if desiredDetection is None:
        return noImage
    box = desiredDetection[:4]
    objectImg = imgByBox(img, box, extraSpace).copy()
    return objectImg


def doDetection(pytorchScreenDetector, pytorchDigitsDetector, imgRgb, imgBgr):
    detections = pytorchScreenDetector.detect(imgRgb)[0]
    screenImg = extractObjectImage(screenClass, imgBgr, detections, extraSpace=5)

    if screenImg is None:
        digitsImg = noScreenImage
    else:
        digitDetections = pytorchDigitsDetector.detect(bgr2rgb(screenImg))[0]
        digitsImg = drawDigitsDetections(screenImg, digitDetections, BGRColors.green)

    screenDetectionImg, detections, _ = fitImageDetectionsToShape(imgBgr, detections, FHD_SHAPE)
    counter_screen_colors = {
        0: BGRColors.green,
        1: BGRColors.red
    }
    drawDetections(screenDetectionImg, detections, counter_screen_colors, withScores=True)
    return screenDetectionImg, digitsImg


noScreenImage = zeros((50, 150))
screenClass = 1


def main():
    hideFromTitle = "/hdd/Datasets/counters/data"
    paths = [
        # "/hdd/Datasets/counters/data/Musson_counters",
        # "/hdd/Datasets/counters/data/Musson_counters_2",
        "/hdd/Datasets/counters/data/Musson_counters_3"
    ]
    pytorchScreenDetector, opencvScreenDetector, pytorchDigitsDetector, opencvDigitsDetector = createDetectors()
    for img_file in enumerate_images(paths[0:]):
        imgBgr = cv2.imread(img_file)
        imgRgb = bgr2rgb(imgBgr)
        pytorchDetectionsImgs = doDetection(pytorchScreenDetector, pytorchDigitsDetector, imgRgb, imgBgr.copy())
        opencvDetectionsImgs = doDetection(opencvScreenDetector, opencvDigitsDetector, imgRgb, imgBgr)
        path = img_file.replace(hideFromTitle, "")
        key = imshowWait([pytorchDetectionsImgs[0], f'trch {path}'], [pytorchDetectionsImgs[1], 'trch'],
                         [opencvDetectionsImgs[0], f'opcv {path}'], [opencvDetectionsImgs[1], 'opcv'])
        if key == 27:
            break
        elif key == ord('a'):
            print('Need to annotate', img_file)


if __name__ == '__main__':
    main()
