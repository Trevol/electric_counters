cd ../../..
export PYTHONPATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/ultralytics_yolo

source venv/bin/activate
cd src/counters/train_scripts