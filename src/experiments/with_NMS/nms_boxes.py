import cv2
import numpy as np
from trvo_utils.timer import timeit


def main():
    n = 2000
    print(n)
    boxes = [
        *[
             [10, 10, 20, 20],
             [11, 11, 19, 19]
         ] * n,

        *[[50, 50, 20, 20]] * n
    ]
    scores = np.random.rand(len(boxes)).clip(.71, 1.1)

    with timeit():
        indices = cv2.dnn.NMSBoxes(boxes, scores, .7, .7)

    for _ in range(7):
        with timeit():
            indices = cv2.dnn.NMSBoxes(boxes, scores, .7, .7)



main()
