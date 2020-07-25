import cv2
import pathlib
import os


def check(dirConfigs):
    for directory, img2annotationFn in dirConfigs:
        pass


def splitPath(path):
    """
    :param path:
    :return: parentDir, baseName, nameWithoutExt, ext
    """
    parentDir, baseName = os.path.split(path)
    nameWithoutExt, ext = os.path.splitext(baseName)
    return parentDir, baseName, nameWithoutExt, ext


def inplaceAnnotation(imgPath):
    parentDir, baseName, nameWithoutExt, ext = splitPath(imgPath)
    return os.path.join(parentDir, nameWithoutExt + ".xml")


def annotation_at_digits_annotations(imgPath):
    parentDir, baseName, nameWithoutExt, ext = splitPath(imgPath)
    return os.path.join(parentDir, "digits_annotations", nameWithoutExt + ".xml")


if __name__ == '__main__':
    dirConfigs = [
        ("/hdd/Datasets/counters/0_from_internet/train", inplaceAnnotation),
        ("/hdd/Datasets/counters/0_from_internet/val", inplaceAnnotation),
        ("/hdd/Datasets/counters/1_from_phone/train", inplaceAnnotation),
        ("/hdd/Datasets/counters/1_from_phone/val", inplaceAnnotation)
    ]


    def check_main():
        pass


    check_main()
