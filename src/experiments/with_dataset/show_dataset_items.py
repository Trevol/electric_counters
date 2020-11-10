import cv2
import torch
import numpy as np
from trvo_utils import toInt
from trvo_utils.cv2gui_utils import imshowWait, waitKeys

import os

from trvo_utils.imutils import imHW

from experiments.with_dataset.load_images_and_labels_2 import LoadImagesAndLabels2
from utils.utils import xywh2xyxy, plot_images


def image2label(img):
    ext = os.path.splitext(img)[-1]
    return img.replace(ext, '.txt')


def main():
    train_path = "train_split.txt"
    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.20,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.0005,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138 * 1.3,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678 * 1.3,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36 * 1.3,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 2,  # image rotation (+/- deg)
           'translate': 0.05 * 2,  # image translation (+/- fraction)
           'scale': 0.05 * 7,  # image scale (+/- gain) 7
           'shear': 0.641 * 4,  # image shear (+/- deg)
           'lr_flip': False,
           'ud_flip': False
           }
    dataset = LoadImagesAndLabels2(train_path,
                                   img_size=320,
                                   batch_size=16,
                                   augment=True,
                                   hyp=hyp,  # augmentation hyperparameters
                                   rect=True,  # rectangular training
                                   cache_images=False,
                                   single_cls=False,
                                   image2label=image2label)

    for i in range(len(dataset)):
        imgTensor, labels, fName, shapes, src_img = dataset[i]
        # RGB to BGR, from 3xHxW to HxWx3
        imgBgr = imgTensor.numpy()[::-1].transpose(1, 2, 0).copy()
        imgH, imgW = imHW(imgBgr)
        for _, label, cx, cy, w, h in labels:
            cx, cy, w, h = cx * imgW, cy * imgH, w * imgW, h * imgH
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = toInt(x1, y1, x2, y2)
            cv2.rectangle(imgBgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
        displayImg = np.hstack([imgBgr, src_img])
        if imshowWait(displayImg) == 27:
            return
    waitKeys(27)


main()
