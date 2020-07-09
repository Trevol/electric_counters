from glob import glob
from time import sleep

import cv2
import torch
import tqdm
from trvo_utils import toInt
from trvo_utils.imutils import imreadRGB, imshowWait, rgb2bgr
import numpy as np
from trvo_utils.timer import timeit

from models import load_darknet_weights
from nms_tests import nms_opencv
from ultralytics_yolo.models import Darknet
from utils.datasets import letterbox
from utils.utils import non_max_suppression, scale_coords, load_classes


def preprocess(rgb, imSize):
    img = letterbox(rgb, new_shape=imSize)[0]
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    return img.unsqueeze(0)


def test_detect():
    print("")

    # cfg_file = 'ultralytics_yolo/cfg/yolov3-spp.cfg'
    # weights_path = 'weights/yolov3-spp.pt'
    cfg_file = 'data/yolov3-tiny-2cls.cfg'
    weights_path = 'weights/gpu_server/2/best.pt'

    # image_file = 'ultralytics_yolo/data/samples/zidane.jpg'
    # image_files = '/hdd/Datasets/counters/4_from_phone/*.jpg'
    # image_files = '/home/trevol/hdd/Datasets/counters/0_from_internet/all/*.jp*'
    # image_files = '/home/trevol/hdd/Datasets/counters/7_from_app/*.jpg'
    # image_files = '/hdd/Datasets/counters/for_yolo/images/0_from_internet/train/*.jp*'
    # image_files = '/hdd/Datasets/counters/for_yolo/images/0_from_internet/val/*.jp*'
    image_files = '/hdd/Datasets/counters/Musson_counters/train/*.jpg'
    # image_files = '/hdd/Datasets/counters/Musson_counters/val/*.jpg'

    imgsz = (416, 416)
    device = 'cpu'
    conf_thres = .3
    iou_thres = .4

    model = Darknet(cfg_file, imgsz)
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    # load_darknet_weights(model, weights_path)
    model.to(device).eval()

    def nms(predictions, conf_thres, iou_thres):
        return non_max_suppression(predictions, conf_thres, iou_thres,
                                   multi_label=False, classes=None, agnostic=True)

    for image_file in sorted(glob(image_files)):
        img = imreadRGB(image_file)
        input = preprocess(img, imgsz).to(device)

        with torch.no_grad():
            with timeit():
                pred = model(input)[0]
            with timeit():
                pred = model(input)[0]

        pred = nms(pred, conf_thres, iou_thres)

        anyDetections = len(pred) > 0 and pred[0] is not None
        if anyDetections:
            for det in pred:
                det[:, :4] = scale_coords(input.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = list(toInt(*xyxy))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)

        displayImg = rgb2bgr(cv2.resize(img, None, None, .4, .4))
        k = imshowWait([displayImg, image_file])
        if k == 27:
            break


def test_convert_pt_to_weights():
    from ultralytics_yolo.models import convert

    cfg = 'data/yolov3-tiny-2cls.cfg'
    weights_file = 'weights/gpu_server/2/best.pt'

    convert(cfg, weights_file)
