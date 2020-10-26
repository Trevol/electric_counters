from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import bgr2rgb

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.TwoStageDigitsDetector import TwoStageDigitsDetector
from with_image_aligning.frame_reader import FrameReader


def createDetectors():
    cfg_file = '../counters/data/yolov3-tiny-2cls-320.cfg'
    weights_file = '../counters/best_weights/yolov3-tiny-2cls/320/yolov3-tiny-2cls-320.weights'
    yield DarknetOpencvDetector(cfg_file, weights_file, 320)

    # cfg_file = '../counter_digits/data/yolov3-tiny-10cls-320.cfg'
    cfg_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.cfg"
    weights_file = '../counter_digits/best_weights/4/yolov3-tiny-10cls-320.4.weights'
    # weights_file = "/home/trevol/Repos/Android/camera-samples/CameraXBasic/app/src/main/assets/yolov3-tiny-10cls-320.weights"
    yield DarknetOpencvDetector(cfg_file, weights_file, 320)


def main():
    framesPath = "../../images/smooth_frames/1/*.jpg"
    frames = FrameReader(framesPath, 1).read()

    screenDetector, digitsDetector = createDetectors()
    detector = TwoStageDigitsDetector(screenDetector, digitsDetector)
    for imgBgr, imgGray in frames:
        imgRgb = bgr2rgb(imgBgr)
        detection = detector.detect(imgRgb)
        key = imshowWait(imgBgr)
        if key == 27:
            break


main()
