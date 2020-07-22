from glob import glob

import cv2
import numpy as np
from trvo_utils import toInt_array
from trvo_utils.imutils import imreadRGB, imshowWait, rgb2bgr

from DarknetDetector import DarknetDetector


def drawDetections(img, detections):
    green = (0, 200, 0)
    imgWithLabels = np.zeros_like(img)
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = toInt_array(xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), green, 1)

        cv2.rectangle(imgWithLabels, (x1, y1), (x2, y2), green, 1)
        cv2.putText(imgWithLabels, str(int(cls)), (x1+2, y2-3), cv2.FONT_HERSHEY_SIMPLEX, .8, green)
        print(int(cls), " ")
    print("")
    result = np.vstack([img, imgWithLabels])
    return result


def test_detect():
    print("")

    image_files = [
        # '/hdd/Datasets/counters/Musson_counters/train/digits/*.jpg',
        # '/hdd/Datasets/counters/Musson_counters/val/digits/*.jpg',
        '/hdd/Datasets/counters/1_from_phone/train/digits/*.jpg'
    ]

    s = 320
    detector = DarknetDetector(
        cfg_path=f'data/yolov3-tiny-10cls-{s}.cfg',
        weights_path=f'best_weights/best_{s}.weights',
        input_size=(s, s),
        device='cpu',
        conf_thres=.3,
        iou_thres=.4
    )

    for im_files in image_files:
        for image_file in sorted(glob(im_files)):
            img = imreadRGB(image_file)
            pred = detector.detect(img)

            withDetections = drawDetections(img, pred[0])

            k = imshowWait([rgb2bgr(withDetections), image_file])
            if k == 27:
                return


def test_convert_pt_to_weights():
    from ultralytics_yolo.models import convert

    cfg_file = 'data/yolov3-tiny-10cls-320.cfg'
    weights_file = 'best_weights/best_320.pt'

    convert(cfg_file, weights_file)
