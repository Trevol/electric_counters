source train_common.sh

python ../../train.py --cfg ../data/yolov3-tiny-2cls-416.cfg \
       --data ../data/counters.data --weights ../../../weights/yolov3-tiny.pt --device 0 --epochs 10 --batch-size 8 \
       --img-size 320 640 416 --input-size 416 --rect --multi-scale