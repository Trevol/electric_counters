from itertools import repeat, groupby
from random import shuffle
import numpy as np
from trvo_utils.iter_utils import groupBy
from trvo_utils.timer import timeit

from detection.TwoStageDigitsDetectionResult import DigitDetection


def main():
    n = 100000
    keys = [1, 2, 3, 4, 5]
    data = []
    for key in keys:
        data += [(key, 1, 2.2, f"{key}") for key in repeat(key, n)]
    shuffle(data)

    with timeit():
        for key, group in groupBy(data):
            list(group)
    with timeit():
        for key, group in groupBy(data, key=lambda o: o[0]):
            list(group)

    for key, group in groupBy(data,
                              key=lambda o: o[0],
                              select=lambda g: g[1:]):
        print(key, list(group)[:3])

    for key, agg in groupBy(data,
                              key=lambda o: o[0],
                              select=lambda g: g[1:],
                              aggregate=lambda g: sum(1 for _ in g)):
        print(key, agg)


main()
