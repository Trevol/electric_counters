import cv2
from trvo_utils.imutils import imshowWait, bgr2rgb

from DarknetOpencvDetector import DarknetOpencvDetector
from utils.datasets import letterbox


def main():
    origImgFile = '/home/trevol/Repos/Android/ncnn-android-mobilenetssd/app/src/main/assets/20200724_095620.jpg'
    srcImgFile = '/home/trevol/Repos/Android/ncnn-android-mobilenetssd/app/src/main/assets/20200724_095620_letterbox_320.jpg'
    origImg = cv2.imread(origImgFile)
    img, wh_ratio, (dw, dh) = letterbox(origImg, (320, 320))
    print(img.shape, wh_ratio, (dw, dh))
    cv2.imwrite(srcImgFile, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def main():
    # 2, 0.789033    0.247275 0.350422 0.528203 0.415583"
    # 1, 0.602444    0.119947 0.218336 1.055318 1.061704"

    # 1, 0.851439,   0.33955, 0.34615, 0.28018, 0.06406
    # 0, 0.602478,   0.51418, 0.57602, 0.90614, 0.84337

    srcImgFile = '/home/trevol/Repos/Android/ncnn-android-mobilenetssd/app/src/main/assets/20200724_095620_letterbox_320.jpg'

    cfg = "/home/trevol/Repos/Android/android-electro-counters/app/src/main/assets/yolov3-tiny-2cls-320.cfg"
    weights = "/home/trevol/Repos/Android/android-electro-counters/app/src/main/assets/yolov3-tiny-2cls-320.weights"

    img = cv2.imread(srcImgFile)
    imgRgb = bgr2rgb(img)
    detector = DarknetOpencvDetector(cfg, weights, 320)
    detections = detector.detect(imgRgb)[0]
    # for d in detections:
    #     print(d)


if __name__ == '__main__':
    main()
