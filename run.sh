#!/bin/bash

# Raw Data
DATA_BASE=/data/qiji/DATA/VideoCaption
GOAL_DATA=$DATA_BASE/goal
MSVD_DATA=$DATA_BASE/MSVD
MSRVTT_DATA=$DATA_BASE/MSRVTT-v2
UCOOK2_DATA=$DATA_BASE/YouCook2
ANET_DATA=$DATA_BASE/ActivityNet

video_dir=/data/qiji/DATA/soccernet/videos/
anns_dir=/data/qiji/DATA/soccernet/annotations/
# Processed data
clips_dir=/data/qiji/DATA/soccernet/clips/
commentary_dir=/data/qiji/DATA/soccernet/commentaries/
caption_dir=/data/qiji/DATA/soccernet/captions/

# The BLINK interface that link given entity to wikipedia items
BLINK_API=http://127.0.0.1:9271/blink_entity


# Process data
## Segment clips & generate captions
python data_process/segment.py \
  --video_dir $GOAL_DATA/videos \
  --anns_dir $GOAL_DATA/annotations \
  --clips_dir $GOAL_DATA/clips \
  --commentary_dir $GOAL_DATA/commentaries \
  --caption_dir $GOAL_DATA/captions


## Generate KGs
python data_process/build_kg/gen_kg.py \
  --blink_interface $BLINK_API \
  --commentary_dir $GOAL_DATA/commentaries \
  --kgs_dir $GOAL_DATA/kgs

python data_process/build_kg/stat_kg.py \
  --commentary_dir $GOAL_DATA/commentaries \
  --kgs_dir $GOAL_DATA/kgs


# Analyze data
## Do statistics
python data_analysis/stat_datasets.py \
  --video_dir $GOAL_DATA/videos \
  --anns_dir $GOAL_DATA/annotations \
  --clips_dir $GOAL_DATA/clips \
  --commentary_dir $GOAL_DATA/commentaries \
  --caption_dir $GOAL_DATA/captions \
  --msvd_dir $MSVD_DATA \
  --msrvtt_dir $MSRVTT_DATA \
  --youcook2_dir $UCOOK2_DATA \
  --activitynet_dir $ANET_DATA


# Train KGVC Model with ALPRO
bash models/ALPRO/run_scripts/ft_goal_caption.sh models/ALPRO/config_release models/ALPRO/output