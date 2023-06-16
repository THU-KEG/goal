#!/bin/bash
CONFIG_PATH=${1:-config_release}
OUT_DIR=${2:-output}

export CUDA_VISIBLE_DEVICES="7"
export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/goal_caption.json'

# horovodrun -np 2 python src/tasks/run_video_caption.py \
python src/tasks/run_video_caption.py \
      --config $CONFIG_PATH \
      --output_dir $OUT_DIR/$(date '+%Y%m%d%H%M%S')


