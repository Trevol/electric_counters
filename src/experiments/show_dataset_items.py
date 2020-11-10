import torch
import numpy as np
from trvo_utils.cv2gui_utils import imshowWait, waitKeys

from utils.datasets import LoadImagesAndLabels
import os


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
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.05 * 0,  # image scale (+/- gain)
           'shear': 0.641 * 0,  # image shear (+/- deg)
           'lr_flip': False,
           'ud_flip': False
           }
    dataset = LoadImagesAndLabels(train_path,
                                  img_size=320,
                                  batch_size=16,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=True,  # rectangular training
                                  cache_images=False,
                                  single_cls=True,
                                  image2label=image2label)

    for i in range(len(dataset)):
        item = dataset[i]
        imgTensor, labels, fName, shapes = dataset[i]
        # RGB to BGR, from 3xHxW to HxWx3
        imgBgr = imgTensor.numpy()[::-1].transpose(1, 2, 0)

        if imshowWait(imgBgr) == 27:
            return
    waitKeys(27)


main()
