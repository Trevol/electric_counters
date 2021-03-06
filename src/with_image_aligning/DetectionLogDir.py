import os
from dataclasses import dataclass
from glob import glob
from typing import Tuple


class DetectionLogDir:
    @dataclass
    class ParsedPath:
        stamp: str
        num: int
        path: str
        key: Tuple = None

        def __post_init__(self):
            self.key = (self.stamp, self.num)

    def __init__(self, logDir):
        self.logDir = logDir

    def inputFrames(self):
        parsedNames = []
        for path in glob(os.path.join(self.logDir, "*_input.jpg")):
            stamp, id, _ = os.path.basename(path).split("_")
            parsedNames.append(self.ParsedPath(stamp, int(id), path))
        parsedNames.sort(key=lambda p: p.key)
        return (p.path for p in parsedNames)

    def __sample(self):
        """
        2020-09-15-10-30-27-849_35_detectionResult.txt
        2020-09-15-10-30-27-849_35_digits.jpg
        2020-09-15-10-30-27-849_35_input.jpg
        2020-09-15-10-30-27-849_35_inputDrawing.jpg
        2020-09-15-10-30-27-849_35_screen.jpg
        2020-09-15-10-30-27-849_35_screenDrawing.jpg
        """
        pass