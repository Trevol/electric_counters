import os
from glob import glob

import cv2
from trvo_utils import toInt_array
from trvo_utils.annotation import PascalVocXmlParser
from trvo_utils.imutils import imshowWait

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree


def list_files(dirs, extensions):
    for d in dirs:
        files = []
        for e in extensions:
            files.extend(glob(os.path.join(d, '*.' + e)))
        for f in sorted(set(files)):
            yield f


def digitsAnnotationFile(imgFile):
    annotations_dir = 'digits_annotations'
    parent, file_ = os.path.split(imgFile)
    nameWithoutExt = os.path.splitext(file_)[0]
    ann_file = os.path.join(parent, annotations_dir, nameWithoutExt + '.xml')
    if os.path.isfile(ann_file):
        return ann_file
    return None


def imgByBox(srcImg, box, extraSpace=0):
    x1, y1, x2, y2 = toInt_array(box)
    return srcImg[y1 - extraSpace:y2 + extraSpace, x1 - extraSpace:x2 + extraSpace]


def shiftBoxes(boxes, x, y):
    return [(x1 + x, y1 + y, x2 + x, y2 + y) for x1, y1, x2, y2 in boxes]


def extract_screenImg_digitsAnnotations(img_file, ann_file):
    assert ann_file is not None
    p = PascalVocXmlParser(ann_file)

    screenBox = None
    digitBoxes = []
    digitLabels = []
    for b, l in zip(p.boxes(), p.labels()):
        if l == 'screen':
            screenBox = b
        else:
            digitBoxes.append(b)
            digitLabels.append(l)
    if screenBox is None:
        raise Exception(f'screenBox is None. {img_file} {ann_file}')

    extraSpace = 5
    screenImg = imgByBox(cv2.imread(img_file), screenBox, extraSpace=extraSpace)
    x1, y1, *_ = screenBox
    digitBoxes = shiftBoxes(digitBoxes, -(x1 - extraSpace), -(y1 - extraSpace))
    return screenImg, digitBoxes, digitLabels


def drawBoxes(img, boxes, color):
    for b in boxes:
        x1, y1, x2, y2 = toInt_array(b)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    return img


def show(screenImg, digitBoxes, digitLabels):
    drawing = drawBoxes(screenImg.copy(), digitBoxes, (0, 255, 0))
    if imshowWait(drawing) == 27:
        raise Exception('End of show')


def SubElement(parent, tag, text="", attrib={}):
    subEl = ET.SubElement(parent, tag, attrib)
    subEl.text = str(text or "")
    return subEl


def writeAnnotation(annFile, imgFile, imgShape, boxes, labels):
    assert len(boxes)
    assert len(labels)

    root = ET.Element("annotation")
    SubElement(root, 'filename', os.path.basename(imgFile))
    SubElement(root, 'path', imgFile)
    sizeEl = SubElement(root, 'size')
    h, w, d = imgShape
    SubElement(sizeEl, 'width', w)
    SubElement(sizeEl, 'height', h)
    SubElement(sizeEl, 'depth', d)

    for (x1, y1, x2, y2), l in zip(boxes, labels):
        objEl = SubElement(root, 'object')
        SubElement(objEl, 'name', l)
        bndboxEl = SubElement(objEl, 'bndbox')
        SubElement(bndboxEl, 'xmin', x1)
        SubElement(bndboxEl, 'ymin', y1)
        SubElement(bndboxEl, 'xmax', x2)
        SubElement(bndboxEl, 'ymax', y2)

    tree = ElementTree(element=root)
    tree.write(annFile)


def extract_dataset(imagesDirs):
    imagesExtensions = ['jpg', 'jpeg', 'png']
    digitsFolder = 'digits'

    for d in imagesDirs:
        os.makedirs(os.path.join(d, digitsFolder), exist_ok=True)

    results = []
    for img_file in list_files(imagesDirs, imagesExtensions):
        ann_file = digitsAnnotationFile(img_file)
        if ann_file is None:
            continue
        screenImg, digitBoxes, digitLabels = extract_screenImg_digitsAnnotations(img_file, ann_file)
        results.append((img_file, screenImg, digitBoxes, digitLabels))

    for img_file, screenImg, digitBoxes, digitLabels in results:
        parentDir, imgBaseName = os.path.split(img_file)
        nameWithoutExt = os.path.splitext(imgBaseName)[0]
        screenImgFile = os.path.join(parentDir, digitsFolder, imgBaseName)
        annFile = os.path.join(parentDir, digitsFolder, nameWithoutExt + '.xml')

        cv2.imwrite(screenImgFile, screenImg, [cv2.IMWRITE_JPEG_QUALITY, 100])
        writeAnnotation(annFile, screenImgFile, screenImg.shape, digitBoxes, digitLabels)


def main():
    # extract rect with screen, adjust digits coords
    # save screen img and adjusted annotations to ./digits
    imagesDirs = [
        "/hdd/Datasets/counters/1_from_phone/train",
    ]
    extract_dataset(imagesDirs)


if __name__ == '__main__':
    main()
