import numpy as np
from trvo_utils.box_utils import pointInBox
from trvo_utils.timer import timeit


def pointInBox2(box, point):
    return np.all(box[:2] <= point) and np.all(box[2:] >= point)


def pointInBox3(box, point):
    box = tuple(box)
    point = tuple(point)
    return box[:2] <= point <= box[2:]


def pointInBox4(box, pt):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    x = pt[0]
    y = pt[1]
    return x1 <= x <= x2 and y1 <= y <= y2


def pointInBox5(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def main():
    # point in box
    box = np.float32([1, 2, 6, 7])
    ptIn = 3, 4
    ptOut1 = 0, -1
    ptOut2 = 10, 20

    def measurePt(fn, pt, msg):
        with timeit(msg):
            for _ in range(300):
                fn(box, pt)

    def measure(fn, msg):
        print(f"------{msg}")
        pts = [
            (ptIn, "ptIn"),
            (ptOut1, "ptOut1"),
            (ptOut2, "ptOut2")
        ]
        for pt, msg in pts:
            measurePt(fn, pt, msg)

    measure(pointInBox, "pointInBox")
    measure(pointInBox2, "pointInBox2")
    measure(pointInBox3, "pointInBox3")
    measure(pointInBox4, "pointInBox4")
    measure(pointInBox5, "pointInBox5")


main()
