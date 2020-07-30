from glob import glob

import cv2
from trvo_utils import toInt_array
from trvo_utils.imutils import imreadRGB, imshowWait, rgb2bgr, bgr2rgb, fit_image_to_shape

from DarknetDetector import DarknetDetector
from consts import BGRColors, FHD_SHAPE
from utils_local.vis_utils import drawDetections, fitImageDetectionsToShape


def test_detect():
    print("")

    image_files = [
        # '/hdd/Datasets/counters/4_from_phone/*.jpg',
        # '/home/trevol/hdd/Datasets/counters/7_from_app/*.jpg',
        # '/home/trevol/hdd/Datasets/counters/0_from_internet/all/*.jp*',
        # '/hdd/Datasets/counters/for_yolo/images/0_from_internet/train/*.jp*',
        # '/hdd/Datasets/counters/for_yolo/images/0_from_internet/val/*.jp*',
        '/hdd/Datasets/counters/Musson_counters/train/*.jpg',
        '/hdd/Datasets/counters/Musson_counters/val/*.jpg'
    ]

    s = 320
    detector = DarknetDetector(
        cfg_path=f'data/yolov3-tiny-2cls-{s}.cfg',
        weights_path=f'best_weights/yolov3-tiny-2cls/{s}/yolov3-tiny-2cls-{s}.weights',
        input_size=(s, s),
        device='cpu',
        conf_thres=.3,
        iou_thres=.4
    )

    for im_files in image_files:
        for image_file in sorted(glob(im_files)):
            img = cv2.imread(image_file)
            detections = detector.detect(bgr2rgb(img))[0]

            img, detections, _ = fitImageDetectionsToShape(img, detections, FHD_SHAPE)
            drawDetections(img, detections, BGRColors.green)

            k = imshowWait([img, image_file])
            if k == 27:
                return


def test_convert_pt_to_weights():
    from ultralytics_yolo.models import convert

    cfg_file = 'data/counters/yolov3-tiny-2cls-320.cfg'
    weights_file = 'best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.pt'

    convert(cfg_file, weights_file)
