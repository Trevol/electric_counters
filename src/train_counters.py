from models import Darknet, load_darknet_weights


def train():
    cfg = 'data/yolov3-tiny-2cls.cfg'
    weights = '../weights/yolov3-tiny.weights'

    # cfg = '/home/trevol/Repos/experiments_with_lightweight_detectors/electric_counters/ultralytics_yolo/cfg/yolov3.cfg'
    # weights = '../weights/yolov3.weights'

    device = 'cuda'
    model = Darknet(cfg).to(device)
    load_darknet_weights(model, weights)


train()
