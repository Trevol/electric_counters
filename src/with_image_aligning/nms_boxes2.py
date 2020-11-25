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
    return sorted(score_index, reverse=True)


def getSortedScoreIndex(scores):
    score_index = ((score, index) for index, score in enumerate(scores))
    return sorted(score_index, reverse=True)


def groupBoxes_noScore(boxes, overlap_threshold):
    boxes = [Rect(b) for b in boxes]
    indices = []
    keptIndices = []

    def shouldKeep(box, boxes, keptIndices, overlap_threshold):
        keep = True
        maxOverlap = -1.0
        maxOverlapIndex = -1
        for keptIndex in keptIndices:
            overlap = box.overlap(boxes[keptIndex])
            # если случился (not keep), то вернуть мы должны keep=False
            # поэтому далее keep=False сохраняем
            # но maxOverlapIndex вычисляем для всех элементов keptIndex
            if keep:
                keep = overlap <= overlap_threshold
            if overlap > maxOverlap:
                maxOverlap = overlap
                maxOverlapIndex = keptIndex

        return keep, maxOverlapIndex

    for index, box in enumerate(boxes):
        keep, maxOverlapIndex = shouldKeep(box, boxes, keptIndices, overlap_threshold)
        if keep:
            keptIndices.append(index)
            indices.append(index)
        else:  # not keep
            indices.append(maxOverlapIndex)
    return indices, keptIndices


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


def groupBoxes(boxes, scores, overlap_threshold):
    assert len(boxes) == len(scores) and overlap_threshold >= 0
    # TODO: return corresponding kept indexes for ALL boxes (for future clustering/indexing)
    score_indices = getSortedScoreIndex(scores)
    boxes = [Rect(b) for b in boxes]
    index_groupIndex = []
    keptIndices = []

    def shouldKeep(index, boxes, keptIndices, overlap_threshold):
        box = boxes[index]
        keep = True
        maxOverlap = -1.0
        maxOverlapIndex = -1
        for keptIndex in keptIndices:
            overlap = box.overlap(boxes[keptIndex])
            # если случился (not keep), то вернуть мы должны keep=False
            # поэтому далее keep=False сохраняем
            # но maxOverlapIndex вычисляем для всех элементов keptIndex
            if keep:
                keep = overlap <= overlap_threshold
            if overlap > maxOverlap:
                maxOverlap = overlap
                maxOverlapIndex = keptIndex

        return keep, maxOverlapIndex

    for score, index in score_indices:
        keep, maxOverlapIndex = shouldKeep(index, boxes, keptIndices, overlap_threshold)
        if keep:
            keptIndices.append(index)
            index_groupIndex.append((index, index))
        else:  # not keep
            index_groupIndex.append((index, maxOverlapIndex))
    # restore original boxes order (sort by index) and select groupIndex
    groupIndices = [item[1] for item in sorted(index_groupIndex)]
    return groupIndices, keptIndices


if __name__ == '__main__':
    def runTest():
        boxes = [
            (1, 0, 15, 15),
            (1, 1, 15, 15),
            (0, 0, 15, 15),
            (0, 1, 15, 15),

            (20, 20, 10, 10)
        ]
        scores = [
            .8, .9, .99, .6, .8
        ]

        indices, keptIndices = groupBoxes_noScore(boxes, .7)
        assert (indices == [0, 0, 0, 0, 4])
        assert (keptIndices == [0, 4])

        indices, keptIndices = groupBoxes(boxes, scores, .7)
        assert (indices == [2, 2, 2, 2, 4])
        assert (keptIndices == [2, 4])


    def runBenchmark():
        boxes = [
            (1, 0, 15, 15),
            (1, 1, 15, 15),
            (0, 0, 15, 15),
            (0, 1, 15, 15),

            (20, 20, 10, 10)
        ]
        scores = [
            .8, .9, .99, .6, .8
        ]

        k = 1000
        boxes = boxes * k
        scores = scores * k

        n = 5
        for _ in range(n):
            with timeit():
                indices, keptIndices = groupBoxes_noScore(boxes, .7)

        print("-------------------------")

        for _ in range(n):
            with timeit():
                indices, keptIndices = groupBoxes(boxes, scores, .7)


    runTest()
    runBenchmark()
