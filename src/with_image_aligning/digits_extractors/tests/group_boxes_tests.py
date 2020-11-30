import numpy as np
from trvo_utils.timer import timeit

from core.rect import Rect
from with_image_aligning.digits_extractors.group_boxes import groupBoxes


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
    boxes = [Rect(b) for b in boxes]
    indices, keptIndices = groupBoxes(boxes, None, .7)
    assert (indices == [0, 0, 0, 0, 4])
    assert (keptIndices == [0, 4])

    indices, keptIndices = groupBoxes(boxes, scores, .7)
    assert (indices == [2, 2, 2, 2, 4])
    assert (keptIndices == [2, 4])


def runBenchmark():
    tt = np.float32
    # tt = np.int32
    # tt = lambda d: np.float32(d).tolist()
    tt = lambda d: d
    boxes = [
        tt((1, 0, 15, 15)),
        tt((1, 1, 15, 15)),
        tt((0, 0, 15, 15)),
        tt((0, 1, 15, 15)),

        tt((20, 20, 10, 10))
    ]
    scores = [
        .8, .9, .99, .6, .8
    ]

    k = 1000
    boxes = [Rect(b) for b in boxes * k]
    scores = scores * k

    n = 5
    for _ in range(n):
        with timeit():
            indices, keptIndices = groupBoxes(boxes, scores, .7)

    print("-------------------------")

    for _ in range(n):
        with timeit():
            indices, keptIndices = groupBoxes(boxes, None, .7)


runTest()
runBenchmark()
