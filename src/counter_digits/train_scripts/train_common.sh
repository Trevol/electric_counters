cd ../../..
export PYTHONPATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/ultralytics_yolo

source venv/bin/activate
cd src/counter_digits/train_scripts