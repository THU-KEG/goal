cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/msvd_qa.json'

horovodrun -np 8 python src/tasks/run_video_qa.py \
      --config $CONFIG_PATH \
      --output_dir /export/home/workspace/experiments/alpro/finetune/msvd_qa/$(date '+%Y%m%d%H%M%S')
