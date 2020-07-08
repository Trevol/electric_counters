cd ..
source venv/bin/activate
cd src
export PYTHONPATH=/home/trevol/Repos/experiments_with_lightweight_detectors/electric_counters
export PYTHONPATH=$PYTHONPATH:/home/trevol/Repos/experiments_with_lightweight_detectors/electric_counters/ultralytics_yolo
python train.py --cfg data/yolov3-tiny-2cls.cfg --data data/counters.data --weights weights/yolov3-tiny.pt --device 0 --epochs 1000 --batch-size 16