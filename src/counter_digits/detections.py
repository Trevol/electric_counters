from glob import glob

from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import imreadRGB, rgb2bgr

from detection.DarknetOpencvDetector import DarknetOpencvDetector
from detection.DarknetPytorchDetector import DarknetPytorchDetector
from utils_local.vis_utils import drawDigitsDetections


def test_detect():
    print("")

    image_files = [
        # '/hdd/Datasets/counters/data/Musson_counters/train/digits/*.jpg',
        # '/hdd/Datasets/counters/data/Musson_counters/val/digits/*.jpg',
        # '/hdd/Datasets/counters/data/1_from_phone/train/digits/*.jpg',
        '/hdd/Datasets/counters/data/5_from_phone/digits/*.jpg',
        # '/hdd/Datasets/counters/data/6_from_phone/digits/*.jpg',
        # '/hdd/Datasets/counters/data/8_from_phone/digits/*.jpg',
        # "/home/trevol/IdeaProjects/HelloKotlin/screenBgrImage.png"
    ]

    s = 320
    cfg_path = f'data/yolov3-tiny-10cls-{s}.cfg'
    weights_path = f'best_weights/5/yolov3-tiny-10cls-320.5.weights'
    # detector = DarknetPytorchDetector(
    #     cfg_path=cfg_path,
    #     weights_path=weights_path,
    #     input_size=(s, s),
    #     device='cpu',
    #     conf_thres=.3,
    #     iou_thres=.4
    # )
    detector = DarknetOpencvDetector(
        cfg_path=cfg_path,
        weights_path=weights_path,
        input_size=s,
        conf_thres=.5,
        iou_thres=.4
    )

    for im_files in image_files:
        for image_file in sorted(glob(im_files)):
            img = imreadRGB(image_file)
            detections = detector.detect(img)

            withDetections = drawDigitsDetections(img, detections)

            k = imshowWait([rgb2bgr(withDetections), image_file])
            if k == 27:
                return


def test_convert_pt_to_weights():
    from ultralytics_yolo.models import convert

    cfg_file = 'data/yolov3-tiny-10cls-320.cfg'
    weights_file = 'best_weights/7_no_local_counters/best_7.1.pt'

    convert(cfg_file, weights_file)
