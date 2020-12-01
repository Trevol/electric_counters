from typing import Callable, Any, List

import numpy as np
from trvo_utils.box_utils import boxSizeWH
from trvo_utils.optFlow_trackers import RectTracker


class BoxedObjectTracker:
    rectTracker = RectTracker()

    def __init__(self, xyxyBoxAccessor: Callable[[Any], np.ndarray], nextObjectMaker: Callable[[Any, np.ndarray], Any]):
        self.xyxyBoxAccessor = xyxyBoxAccessor
        self.nextObjectMaker = nextObjectMaker

    def track(self, prevFrameGray, nextFrameGray,
              prevObjects: List) -> List:
        prevBoxes = [self.xyxyBoxAccessor(o) for o in prevObjects]
        boxes, status = self.rectTracker.track(prevFrameGray, nextFrameGray, prevBoxes)

        nextObjects = []
        for prevObject, box, boxStatus in zip(prevObjects, boxes, status):
            if not boxStatus or self.isAbnormalTrack(self.xyxyBoxAccessor(prevObject ), box):
                continue
            nextObjects.append(self.nextObjectMaker(prevObject, box))
        return nextObjects

    @staticmethod
    def isAbnormalTrack(prevBox, nextBox):
        x1 = nextBox[0]
        y1 = nextBox[1]
        x2 = nextBox[2]
        y2 = nextBox[3]
        if x1 >= x2 or y1 >= y2:
            return True

        # flow = nextBox - prevBox

        # measure sides of prev and next boxes
        prevBoxWH = boxSizeWH(prevBox)
        nextBoxWH = boxSizeWH(nextBox)
        wRatio = prevBoxWH[0] / nextBoxWH[0]
        hRatio = prevBoxWH[1] / nextBoxWH[1]
        delta = .25
        isNormalRatio = 1 + delta > wRatio > 1 - delta and 1 + delta > hRatio > 1 - delta
        if not isNormalRatio:
            return True

        return False