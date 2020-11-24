import cv2
from trvo_utils.timer import timeit


class Rect:
    def __init__(self, xywh):
        self.xywh = xywh
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.area = self.w * self.h

    def intersection(self, rect):
        x1 = max(self.x, rect.x)
        y1 = max(self.y, rect.y)
        w = min(self.x + self.w, rect.x + rect.w) - x1
        h = min(self.y + self.h, rect.y + rect.h) - y1
        if w <= 0 or h <= 0:
            return Rect([0, 0, 0, 0])
        return Rect([x1, y1, w, h])

    def IOU(self, rect):
        Aintersect = self.intersection(rect).area
        return Aintersect / (self.area + rect.area - Aintersect)

    def overlap(self, rect):
        return self.IOU(rect)


def getMaxScoreIndex(scores, threshold):
    score_index = []
    for index, score in enumerate(scores):
        if score > threshold:
            score_index.append((score, index))
    return sorted(score_index, key=lambda si: si[0], reverse=True)


def NMSBoxes(bboxes, scores, score_threshold, nms_threshold):
    assert len(bboxes) == len(scores) and score_threshold >= 0 and nms_threshold >= 0
    score_index_vec = getMaxScoreIndex(scores, score_threshold)
    bboxes = [Rect(b) for b in bboxes]
    indexes = []
    for _, index in score_index_vec:
        keep = True
        for keptIndex in indexes:
            if not keep:
                break
            overlap = bboxes[index].overlap(bboxes[keptIndex])
            keep = overlap <= nms_threshold
        if keep:
            indexes.append(index)
    return indexes


def NMSBoxes2(bboxes, scores, score_threshold, nms_threshold):
    assert len(bboxes) == len(scores) and score_threshold >= 0 and nms_threshold >= 0
    # TODO: return corresponding kept indexes for ALL bboxes (for future clustering/indexing)
    score_index_vec = getMaxScoreIndex(scores, score_threshold)
    bboxes = [Rect(b) for b in bboxes]
    indexes = []
    raise NotImplementedError()
    for _, index in score_index_vec:
        keep = True
        for keptIndex in indexes:
            if not keep:
                break
            overlap = bboxes[index].overlap(bboxes[keptIndex])
            keep = overlap <= nms_threshold
        if keep:
            indexes.append(index)
    return indexes


if __name__ == '__main__':
    def run():
        import cv2

        boxes = [
            (1, 1, 10, 10),
            (9, 9, 10, 10)
        ]
        boxes = boxes * 1000
        scores = [.8 for _ in boxes]

        for _ in range(5):
            with timeit():
                indices = NMSBoxes(boxes, scores, .7, .5)
        print("--------------------------")
        for _ in range(5):
            with timeit():
                indices = cv2.dnn.NMSBoxes(boxes, scores, .7, .5)


    run()
