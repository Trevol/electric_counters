import cv2
import numpy as np
from trvo_utils.imutils import imreadRGB, rgb2bgr, imshowWait
from trvo_utils.timer import timeit

from utils.datasets import letterbox


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(img, classId, conf, left, top, right, bottom, drawClassId=False):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), (255, 178, 50), 1)

    if drawClassId:
        label = f'{classId}'
        cv2.putText(img, label, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


def drawDetections(img, detections):
    img2 = img.copy()
    for d in detections:
        classId, confidence, x1, y1, x2, y2 = d
        drawPred(img, classId, confidence, x1, y1, x2, y2)
        drawPred(img2, classId, confidence, x1, y1, x2, y2, drawClassId=True)
    return img, img2


def postprocess(originalShape, outs, confThreshold, nmsThreshold):
    frameHeight, frameWidth = originalShape

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
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
        detection = classIds[i], confidences[i], left, top, left + width, top + height
        detections.append(detection)
    return detections


def main():
    confThreshold = 0.3  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    s = 320  # Width/Height of network's input image
    modelConfiguration = f'data/yolov3-tiny-10cls-{s}.cfg'

    modelWeights = f'best_weights/3/yolov3-tiny-10cls-320.weights'

    image_file = '/hdd/Datasets/counters/8_from_phone/digits/P_20200610_120509.jpg'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    frame = imreadRGB(image_file)

    h, w = frame.shape[:2]
    hPadding, vPadding = 0, 0
    newFrame = np.zeros((h + hPadding * 2, w + vPadding * 2, 3), np.uint8)
    newFrame[hPadding:hPadding + h, vPadding:vPadding + w] = frame
    frame = newFrame

    # blob = cv2.dnn.blobFromImage(frame, 1 / 255, (s, s), [0, 0, 0], swapRB=False, crop=False)
    blob2 = preprocess(frame, (s, s))

    # Sets the input to the network
    net.setInput(blob2)

    outputs_names = getOutputsNames(net)
    outs = net.forward(outputs_names)
    detections = postprocess(frame.shape[:2], outs, confThreshold, nmsThreshold)

    visualizations = drawDetections(rgb2bgr(frame), detections)
    imshowWait(*visualizations)


def preprocess_old(rgb, imSize):
    img = letterbox(rgb, new_shape=imSize)[0]
    # imshowWait((img, "DEBUG"))
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)
    img = np.divide(img, 255.0, dtype=np.float32)
    return np.expand_dims(img, axis=0)


def preprocess(rgb, imSize):
    img = letterbox(rgb, new_shape=imSize)[0]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0)
    # img = img.transpose(2, 0, 1)  # to 3x416x416
    # img = np.ascontiguousarray(img)
    # img = np.divide(img, 255.0, dtype=np.float32)
    # return np.expand_dims(img, axis=0)
    return blob


if __name__ == '__main__':
    main()
