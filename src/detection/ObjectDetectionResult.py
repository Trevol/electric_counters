from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectDetectionResult:
    classId: int
    classScore: float
    box: np.ndarray  # xyxy

    @classmethod
    def fromDetection(cls, detection):
        *xyxy, score, klass = detection
        return ObjectDetectionResult(int(klass), score, np.asarray(xyxy, np.float32))

    @classmethod
    def fromDetections(cls, detections):
        for d in detections:
            *xyxy, score, klass = d
        return [cls.fromDetection(d) for d in detections]
