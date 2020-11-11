source train_common.sh

python ../../train.py --cfg ../data/yolov3-tiny-2cls-320.cfg \
       --data ../data/counters.data --weights ../../../weights/yolov3-tiny.pt --device 0 --epochs 1000 --batch-size 16 \
       --img-size 288 608 320 --input-size 320 --rect --multi-scale