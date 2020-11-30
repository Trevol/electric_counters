from typing import List

from core.rect import Rect


def groupBoxes(boxes: List[Rect], scores, overlap_threshold):
    assert (scores is None or len(boxes) == len(scores)) and overlap_threshold >= 0

    if scores is None:
        score_indices = ((None, i) for i in range(len(boxes)))
    else:
        score_indices = getSortedScoreIndex(scores)
    index_groupIndex = []
    keptIndices = []

    for _, index in score_indices:
        keep, maxOverlapIndex = __groupBoxes_shouldKeep(index, boxes, keptIndices, overlap_threshold)
        if keep:
            keptIndices.append(index)
            index_groupIndex.append((index, index))
        else:  # not keep
            index_groupIndex.append((index, maxOverlapIndex))
    # restore original boxes order (sort by index) and select groupIndex
    groupIndices = [item[1] for item in sorted(index_groupIndex)]
    return groupIndices, keptIndices


def __groupBoxes_shouldKeep(index, boxes: List[Rect], keptIndices, overlap_threshold):
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


def _score_accessor(score_index):
    return score_index[0]


def getMaxScoreIndex(scores, threshold):
    score_index = []
    for index, score in enumerate(scores):
        if score > threshold:
            score_index.append((score, index))
    return sorted(score_index, key=_score_accessor, reverse=True)


def getSortedScoreIndex(scores):
    score_index = ((score, index) for index, score in enumerate(scores))
    return sorted(score_index, key=_score_accessor, reverse=True)


def ___NMSBoxes(bboxes, scores, score_threshold, nms_threshold):
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
