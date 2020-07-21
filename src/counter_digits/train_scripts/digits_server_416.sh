source train_common.sh

python ../../train.py --cfg ../data/yolov3-tiny-10cls-416.cfg \
       --data ../data/digits.data --weights ../../../weights/yolov3-tiny.pt --device 0 --epochs 1000 --batch-size 16 \
       --img-size 320 640 416 --input-size 416