from glob import glob

import cv2
from trvo_utils.imutils import bgr2gray


class FrameReader:
    def __init__(self, pathname, scale):
        self.pathname = pathname
        self.scale = scale

    @classmethod
    def smoothFrames_1(cls, frameScaleFactor):
        path = "images/smooth_frames/1/*.jpg"
        return cls(path, frameScaleFactor)

    @staticmethod
    def __resize(img, scale):
        if scale == 1:
            return img
        return cv2.resize(img, None, None, scale, scale)

    @classmethod
    def _readResizeGrayscale(cls, fName, scale):
        frame = cv2.imread(fName)
        frame = cls.__resize(frame, scale)
        return frame, bgr2gray(frame)

    def read(self):
        fileNames = sorted(glob(self.pathname))
        frames = (self._readResizeGrayscale(fName, self.scale) for fName in fileNames)
        return frames

    def readAtPositions(self, *positions):
        fileNames = sorted(glob(self.pathname))
        for pos in positions:
            yield self._readResizeGrayscale(fileNames[pos], self.scale)