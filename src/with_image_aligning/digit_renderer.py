import cv2
import numpy as np


class DigitRenderer:
    referenceChar = "1"

    @staticmethod
    def getFontScale(text, desiredHeight, fontFace, thickness):
        startingFontScale = 40
        # calculate size of text for starting scale
        (startingFontWidth, startingFontHeight), _ = cv2.getTextSize(text, fontFace, startingFontScale, thickness)
        # and recalculate scale for required height
        # startingFontScale / desiredScale = startingFonHeight / desiredHeight
        desiredScale = desiredHeight * startingFontScale / startingFontHeight
        return desiredScale

    green = 0, 255, 0

    def __init__(self, fontHeight,
                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                 fontThickness=1,
                 defaultColor=green):
        self.defaultColor = defaultColor or self.green
        self.fontFace = fontFace
        self.fontThickness = fontThickness
        self.fontScale = self.getFontScale(self.referenceChar, fontHeight, fontFace, fontThickness)
        self.fontWH = np.float32(cv2.getTextSize(self.referenceChar, fontFace, self.fontScale, fontThickness)[0])
        # text renders in left_bottom corner of char
        # so displacement from center to this corner is calculated:
        # ord = [centerX, centerY] + [-boxWidth/2, boxHeight/2]
        # displacement = [-boxWidth/2, boxHeight/2] = [-boxWidth/2, boxHeight/2] = boxWH * [-.5, .5]
        self.displacementToOrd = self.fontWH * [-.5, .5]

    def textOrd(self, center: np.ndarray):
        ord = tuple(np.int32(center + self.displacementToOrd))
        return ord

    def render(self, img, digit, point, color=None):
        assert 0 <= digit <= 9
        cv2.putText(img, str(digit), self.textOrd(point), self.fontFace, self.fontScale,
                    color or self.defaultColor,
                    self.fontThickness)