from glob import glob

import cv2
from trvo_utils import toInt_array
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import imreadRGB, rgb2bgr, bgr2rgb, fit_image_to_shape

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.DarknetPytorchDetector import DarknetPytorchDetector
from consts import BGRColors, FHD_SHAPE
from utils_local.vis_utils import drawDetections, fitImageDetectionsToShape


def test_detect():
    print("")

    image_files = [
        # '/hdd/Datasets/counters/data/4_from_phone/*.jpg',
        # '/home/trevol/hdd/Datasets/counters/data/7_from_app/*.jpg',
        # '/home/trevol/hdd/Datasets/counters/data/0_from_internet/all/*.jp*',
        # '/hdd/Datasets/counters/data/for_yolo/images/0_from_internet/train/*.jp*',
        # '/hdd/Datasets/counters/data/for_yolo/images/0_from_internet/val/*.jp*',
        # '/hdd/Datasets/counters/data/Musson_counters/*.jpg',
        # "/home/trevol/Repos/experiments_with_lightweight_detectors/electric_counters/images/smooth_frames/1/*.jpg",
        # "/home/trevol/Repos/experiments_with_lightweight_detectors/electric_counters/images/smooth_frames/2/*.jpg"
        # "/home/trevol/Repos/experiments_with_lightweight_detectors/electric_counters/images/smooth_frames/3/*.jpg",
        "/home/trevol/Repos/experiments_with_lightweight_detectors/electric_counters/images/smooth_frames/4/*.jpg"
    ]

    s = 320
    cfg_path = f'data/yolov3-tiny-2cls-{s}.cfg'
    # weights_path = f'best_weights/yolov3-tiny-2cls/{s}/1/yolov3-tiny-2cls-320.weights'
    weights_path = f'best_weights/yolov3-tiny-2cls/{s}/2/best___.weights'
    # detector = DarknetPytorchDetector(
    #     cfg_path=cfg_path,
    #     weights_path=weights_path,
    #     input_size=(s, s),
    #     device='cpu',
    #     conf_thres=.3,
    #     iou_thres=.4
    # )
    detector = DarknetOpencvDetector(
        cfg_path=cfg_path,
        weights_path=weights_path,
        input_size=s,
        conf_thres=.3,
        iou_thres=.4
    )

    for im_files in image_files:
        for image_file in sorted(glob(im_files)):
            img = cv2.imread(image_file)
            detections = detector.detect(bgr2rgb(img))

            img, detections, _ = fitImageDetectionsToShape(img, detections, FHD_SHAPE)
            drawDetections(img, detections, BGRColors.green, withScores=True)

            k = imshowWait([img, image_file])
            if k == 27:
                return


def test_convert_pt_to_weights():
    from ultralytics_yolo.models import convert

    cfg_file = 'data/yolov3-tiny-2cls-320.cfg'
    weights_file = 'best_weights/yolov3-tiny-2cls/320/2/best___.pt'

    convert(cfg_file, weights_file)
