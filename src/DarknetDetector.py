import numpy as np
import torch
from trvo_utils.timer import timeit

from models import Darknet, load_darknet_weights
from utils.datasets import letterbox
from utils.utils import non_max_suppression, scale_coords


class DarknetDetector:
    def __init__(self, cfg_path, weights_path, input_size, device='cpu', conf_thres=.3, iou_thres=.4):
        self.input_size = input_size
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = Darknet(cfg_path, input_size)
        self.load_weights(self.model, weights_path, device)
        self.model.to(device).eval()

    @staticmethod
    def preprocess(rgb, imSize):
        img = letterbox(rgb, new_shape=imSize)[0]
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0
        return img.unsqueeze(0)

    @staticmethod
    def load_weights(model, weights_path, device):
        if weights_path.endswith('.pt'):  # if PyTorch format
            model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
        elif weights_path.endswith('.weights'):  # darknet format
            load_darknet_weights(model, weights_path)
        else:
            raise Exception("Unexpected weights extension " + weights_path)

    @staticmethod
    def nms(predictions, conf_thres, iou_thres):
        return non_max_suppression(predictions, conf_thres, iou_thres, multi_label=False, classes=None, agnostic=True)

    def detect(self, rgbImage):
        input = self.preprocess(rgbImage, self.input_size).to(self.device)
        with torch.no_grad():
            with timeit():
                pred = self.model(input)[0]

        pred = self.nms(pred, self.conf_thres, self.iou_thres)

        anyDetections = len(pred) > 0 and pred[0] is not None
        if anyDetections:
            for det in pred:
                det[:, :4] = scale_coords(input.shape[2:], det[:, :4], rgbImage.shape).round()
        else:
            pred = [[]]
        return pred