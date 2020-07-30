from glob import glob

import cv2
import numpy as np
from trvo_utils import toInt_array
from trvo_utils.imutils import imreadRGB, imshowWait, rgb2bgr

from DarknetDetector import DarknetDetector
from consts import BGRColors
from utils_local.vis_utils import drawDetections


def drawDetections_MultiStage(img, detections, color=BGRColors.green):
    imgWithClasses = np.zeros_like(img)
    imgWithScores = np.zeros_like(img)

    drawDetections(img, detections, color)
    drawDetections(imgWithClasses, detections, color, withClasses=True)
    drawDetections(imgWithScores, detections, color, withScores=True)

    result = np.vstack([img, imgWithClasses, imgWithScores])
    return result


def test_detect():
    print("")

    image_files = [
        # '/hdd/Datasets/counters/Musson_counters/train/digits/*.jpg',
        # '/hdd/Datasets/counters/Musson_counters/val/digits/*.jpg',
        # '/hdd/Datasets/counters/1_from_phone/train/digits/*.jpg',
        # '/hdd/Datasets/counters/5_from_phone/digits/*.jpg',
        # '/hdd/Datasets/counters/6_from_phone/digits/*.jpg',
        '/hdd/Datasets/counters/8_from_phone/digits/*.jpg',
    ]

    s = 320
    detector = DarknetDetector(
        cfg_path=f'data/yolov3-tiny-10cls-{s}.cfg',
        weights_path=f'best_weights/3/best_{s}.weights',
        input_size=(s, s),
        device='cpu',
        conf_thres=.3,
        iou_thres=.4
    )

    for im_files in image_files:
        for image_file in sorted(glob(im_files)):
            img = imreadRGB(image_file)
            # img = img[700:850, 760:1250]
            pred = detector.detect(img)

            withDetections = drawDetections_MultiStage(img, pred[0])

            k = imshowWait([rgb2bgr(withDetections), image_file])
            if k == 27:
                return


def test_convert_pt_to_weights():
    from ultralytics_yolo.models import convert

    cfg_file = 'data/yolov3-tiny-10cls-320.cfg'
    weights_file = 'best_weights/3/best_320.pt'

    convert(cfg_file, weights_file)
