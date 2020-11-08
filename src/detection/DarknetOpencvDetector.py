from typing import List

import cv2
import numpy as np
from trvo_utils.imutils import imHW

from detection.ObjectDetectionResult import ObjectDetectionResult
from utils.datasets import letterbox


class DarknetOpencvDetector:
    @staticmethod
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def __init__(self, cfg_path, weights_path, input_size, conf_thres=.5, iou_thres=.4):
        self.input_size = [input_size] * 2 if isinstance(input_size, int) else input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.modelOutputNames = self.getOutputsNames(self.model)

    @staticmethod
    def preprocess(rgb, imSize):
        img = letterbox(rgb, new_shape=imSize)[0]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0)
        return blob

    @staticmethod
    def postprocess(originalShape, outs, confThreshold, nmsThreshold):
        frameHeight, frameWidth = originalShape

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        rawDetections = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    rawDetections.append((detection[:4], classId, confidence))

                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        detections = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            detection = left, top, left + width, top + height, confidences[i], classIds[i]
            detections.append(detection)
        return detections

    def detect(self, rgbImage) -> List[ObjectDetectionResult]:
        input = self.preprocess(rgbImage, self.input_size)
        self.model.setInput(input)
        outs = self.model.forward(self.modelOutputNames)
        detections = self.postprocess(imHW(rgbImage), outs, self.conf_thres, self.iou_thres)
        result = ObjectDetectionResult.fromDetections(detections)
        return result
