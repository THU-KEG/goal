#!/bin/bash
TXT_DB=${1:-/data/val.caption_coco_format.json}
IMG_DB=${2:-/data/clips}
CONFIG_PATH=${3:-config_release/goal_caption.json}

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

STEP='4750'


# horovodrun -np 8 python src/tasks/run_video_caption.py \
python src/tasks/run_video_caption.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 6 \
      --output_dir output/finetune/goal_caption/20230118105337 \
      --config $CONFIG_PATH