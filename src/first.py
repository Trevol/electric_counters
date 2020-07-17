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


def load_weights(model, weights_path, device):
    if weights_path.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    elif weights_path.endswith('.weights'):  # darknet format
        load_darknet_weights(model, weights_path)
    else:
        raise Exception("Unexpected weights extension " + weights_path)


def nms(predictions, conf_thres, iou_thres):
    return non_max_suppression(predictions, conf_thres, iou_thres,
                               multi_label=False, classes=None, agnostic=True)


def test_detect():
    print("")

    s = 320
    cfg_file = f'data/yolov3-tiny-2cls-{s}.cfg'
    weights_path = f'best_weights/yolov3-tiny-2cls/{s}/yolov3-tiny-2cls-{s}.weights'
    imgsz = (s, s)

    image_files = [
        # '/hdd/Datasets/counters/4_from_phone/*.jpg',
        # '/home/trevol/hdd/Datasets/counters/7_from_app/*.jpg',
        # '/home/trevol/hdd/Datasets/counters/0_from_internet/all/*.jp*',
        # '/hdd/Datasets/counters/for_yolo/images/0_from_internet/train/*.jp*',
        # '/hdd/Datasets/counters/for_yolo/images/0_from_internet/val/*.jp*',
        '/hdd/Datasets/counters/Musson_counters/train/*.jpg',
        '/hdd/Datasets/counters/Musson_counters/val/*.jpg'
    ]

    device = 'cpu'
    conf_thres = .3
    iou_thres = .4

    model = Darknet(cfg_file, imgsz)
    load_weights(model, weights_path, device)

    model.to(device).eval()

    for im_files in image_files:
        for image_file in sorted(glob(im_files)):
            img = imreadRGB(image_file)
            input = preprocess(img, imgsz).to(device)

            with torch.no_grad():
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
                return


def test_convert_pt_to_weights():
    from ultralytics_yolo.models import convert

    cfg_file = 'data/yolov3-tiny-2cls-320.cfg'
    weights_file = 'best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.pt'

    convert(cfg_file, weights_file)
