import cv2
import torch
import numpy as np


def nms_opencv(predictions, conf_thres, iou_thres):
    assert isinstance(predictions, torch.Tensor)
    output = [None] * predictions.size(0)

    for imId, pred in enumerate(predictions):
        pred = pred[pred[:, 4] > conf_thres]
        if not pred.shape[0]:
            continue
        boxes = [[float(x), float(y), float(w), float(h)] for x, y, w, h in pred[:, :4]]

        scores, classes = pred[:, 5:].max(1)
        scores = scores.numpy().tolist()
        indexes = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        if not len(indexes):
            continue
        indexes = indexes[:, 0]
        output[imId] = torch.tensor([boxes[i] + [scores[i]] + [classes[i]] for i in indexes], dtype=torch.float32)

    return output


def test_cv2_dnn_NMSBoxes():
    print('')

    predictions = torch.zeros([2, 2, 85], dtype=torch.float32)

    predictions[0, 0, :4] = torch.tensor([1, 1, 3, 3])  # box
    predictions[0, 0, 4] = .8  # obj confidence
    predictions[0, 0, 65] = .3  # class score

    predictions[0, 1, :4] = torch.tensor([5, 5, 3, 3])  # box
    predictions[0, 1, 4] = .8  # obj confidence
    predictions[0, 1, 66] = .3  # class score

    o = nms_opencv(predictions, .5, .5)
    o
