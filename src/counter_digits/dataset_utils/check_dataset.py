import cv2
from pathlib import Path
import os

from trvo_utils import toInt_array, toInt
from trvo_utils.annotation import PascalVocXmlParser
from trvo_utils.imutils import imshowWait, imSize, fit_image_to_shape

from counter_digits.dataset_utils.extract_dataset import list_files


def display(imgFile, annotationFile):
    img = cv2.imread(imgFile)
    p = PascalVocXmlParser(annotationFile)
    color = 255, 0, 0
    img, scale = fit_image_to_shape(img, (950, 1850))
    for x1, y1, x2, y2 in p.boxes():
        if scale < 1:
            x1, y1, x2, y2 = toInt(x1 * scale, y1 * scale, x2 * scale, y2 * scale)
        else:
            x1, y1, x2, y2 = toInt(x1, y1, x2, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
    return imshowWait([img, imgFile])


def check(dirConfigs):
    imagesExtensions = ['jpg', 'jpeg', 'png']
    for directory, img2annotationFn in dirConfigs:
        for imgFile in list_files([directory], imagesExtensions):
            annotationFile = img2annotationFn(imgFile)
            if not os.path.isfile(annotationFile):
                continue
            key = display(imgFile, annotationFile)
            if key == 27:
                return


def splitPath(path):
    """
    :param path:
    :return: parentDir, baseName, nameWithoutExt, ext
    """
    parentDir, baseName = os.path.split(path)
    nameWithoutExt, ext = os.path.splitext(baseName)
    return parentDir, baseName, nameWithoutExt, ext


if __name__ == '__main__':
    def check_main():
        def inplaceAnnotation(imgPath):
            parentDir, baseName, nameWithoutExt, ext = splitPath(imgPath)
            return os.path.join(parentDir, nameWithoutExt + ".xml")

        def annotation_at_digits_annotations(imgPath):
            parentDir, baseName, nameWithoutExt, ext = splitPath(imgPath)
            return os.path.join(parentDir, "digits_annotations", nameWithoutExt + ".xml")

        dirConfigs = [
            # ("/hdd/Datasets/counters/0_from_internet/train", inplaceAnnotation),
            # ("/hdd/Datasets/counters/0_from_internet/val", inplaceAnnotation),

            # ("/hdd/Datasets/counters/1_from_phone/train", inplaceAnnotation),
            # ("/hdd/Datasets/counters/1_from_phone/val", inplaceAnnotation),
            # ("/hdd/Datasets/counters/1_from_phone/train", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/1_from_phone/val", annotation_at_digits_annotations),

            # ("/hdd/Datasets/counters/2_from_phone/train", inplaceAnnotation),
            # ("/hdd/Datasets/counters/2_from_phone/val", inplaceAnnotation),
            # ("/hdd/Datasets/counters/2_from_phone/train", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/2_from_phone/val", annotation_at_digits_annotations),

            # ("/hdd/Datasets/counters/3_from_phone", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/4_from_phone", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/5_from_phone", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/6_from_phone", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/7_from_app", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/8_from_phone", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/Musson_counters/train", annotation_at_digits_annotations),
            # ("/hdd/Datasets/counters/Musson_counters/val", annotation_at_digits_annotations),

            ("/hdd/Datasets/counters/8_from_phone/digits", inplaceAnnotation),
        ]
        check(dirConfigs)


    check_main()
