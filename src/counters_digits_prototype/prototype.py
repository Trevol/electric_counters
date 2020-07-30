import cv2
from numpy.lib.histograms import _hist_bin_fd
from trvo_utils.imutils import imshowWait, fit_image_to_shape, bgr2rgb
from trvo_utils.path_utils import list_files

from DarknetDetector import DarknetDetector
from consts import IMAGES_EXTENSIONS, FHD_SHAPE, BGRColors
from utils_local.vis_utils import fitImageDetectionsToShape, drawDetections


def createDetectors():
    cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
    weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.weights'
    screenDetector = DarknetDetector(cfg_file, weights_file, 320)
    yield screenDetector

    cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
    weights_file = '../counter_digits/best_weights/3/best_320.weights'
    digitsDetector = DarknetDetector(cfg_file, weights_file, 320)
    yield digitsDetector


def enumerate_images(dirs):
    return list_files(dirs, IMAGES_EXTENSIONS)


def main():
    dirs = [
        "/hdd/Datasets/counters/8_from_phone"
    ]
    screenDetector, digitsDetector = createDetectors()
    for img_file in enumerate_images(dirs):
        img = cv2.imread(img_file)
        det = screenDetector.detect(bgr2rgb(img))[0]
        img, det, _ = fitImageDetectionsToShape(img, det, FHD_SHAPE)
        drawDetections(img, det, BGRColors.green, withScores=True)
        if imshowWait(img) == 27:
            break


if __name__ == '__main__':
    main()
