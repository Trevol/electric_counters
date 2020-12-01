from trvo_utils.box_utils import xyxy2xywh


class Rect:
    @classmethod
    def fromXyxy(cls, xyxy):
        return cls(xyxy2xywh(xyxy))

    def __init__(self, xywh):
        self.xywh = xywh
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.area = self.w * self.h

    def xyxy(self):
        return [self.x, self.y, self.x + self.w, self.y + self.h]

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
