source train_common.sh

python ../../train.py --cfg ../data/yolov3-tiny-10cls-320.cfg \
       --data ../data/digits.data --weights ../../../weights/yolov3-tiny.pt --device 0 --epochs 10 --batch-size 8 \
       --img-size 288 608 320 --input-size 320