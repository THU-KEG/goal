"""
Modified from UNITER code
"""
import os
import sys
import json
import argparse

from easydict import EasyDict as edict


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object, enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class SharedConfigs(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self, desc="shared config for pretraining and finetuning"):
        parser = argparse.ArgumentParser(description=desc)
        # debug parameters
        parser.add_argument(
            "--debug", type=int, choices=[0, 1], default=0,
            help="debug mode, output extra info & break all loops."
                 "0: disable, 1 enable")
        parser.add_argument(
            "--data_ratio", type=float, default=1.0,
            help="portion of train/val exampels to use,"
                 "e.g., overfit a small set of data")
        
        # Model Architecture
        parser.add_argument(
            "--model_type", type=str, default="pretrain",
            help="type of e2e model to use. Support only 'pretrain' for now. ")
        parser.add_argument(
            "--text_enc_cfg", type=str, default="",
            help="path to the text encoder mdoel config (e.g., the json file of bert config).")
        parser.add_argument(
            "--text_enc_tokenizer_dir", type=str, help="path to tokenizer dir of text encoder model.")
        parser.add_argument(
            "--visual_enc_cfg", type=str, default="",
            help="path to the visual encoder config file."
        )
        parser.add_argument(
            "--max_enc_txt_len", type=int, default=20, help="max text #tokens for the text encoder.")
        parser.add_argument(
            "--max_dec_txt_len", type=int, default=300, help="max text #tokens for the text decoder.")

        # Image Configurations
        parser.add_argument(
            "--img_pixel_mean", type=float, default=None,
            nargs=3, help="image pixel mean")
        parser.add_argument(
            "--img_pixel_std", type=float, default=None,
            nargs=3, help="image pixel std")
        parser.add_argument(
            "--img_input_format", type=str, default="BGR",
            choices=["BGR", "RGB"], help="image input format is BGR for detectron2")
        
        # Data organization
        parser.add_argument(
            "--max_n_example_per_group", type=int, default=5,
            help="max #examples (e.g., captions) paired with each image/video in an input group."
                 "1: each image is paired with a single sent., equivalent to sample by sent.;"
                 "X (X>1): each image can be paired with a maximum of X sent.; X>1 can be used "
                 "to reduce image processing time, including basic transform (resize, etc) and CNN encoding"
        )

        # Training parameters
        parser.add_argument(
            "--train_batch_size", default=128, type=int,
            help="Single-GPU batch size for training for Horovod.")
        parser.add_argument(
            "--val_batch_size", default=128, type=int,
            help="Single-GPU batch size for validation for Horovod.")
        parser.add_argument(
            "--gradient_accumulation_steps", type=int, default=1,
            help="#updates steps to accumulate before performing a backward/update pass."
                 "Used to simulate larger batch size training. The simulated batch size "
                 "is train_batch_size * gradient_accumulation_steps for a single GPU.")
        parser.add_argument("--num_train_epochs", default=10, type=int,
                            help="Total #training epochs.")
        parser.add_argument(
            "--num_valid", default=20, type=int,
            help="Run validation X times during training and checkpoint.")
        parser.add_argument(
            "--min_valid_steps", default=100, type=int,
            help="minimum #steps between two validation runs")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="initial learning rate.")
        parser.add_argument("--optim", default="adamw",
                            choices=["adam", "adamax", "adamw"],
                            help="optimizer")
        parser.add_argument("--betas", default=[0.9, 0.98],
                            nargs=2, help="beta for adam optimizer")
        parser.add_argument("--decay", default="linear",
                            choices=["linear", "invsqrt"],
                            help="learning rate decay method")
        parser.add_argument("--dropout", default=0.1, type=float,
                            help="tune dropout regularization")
        parser.add_argument("--weight_decay", default=1e-3, type=float,
                            help="weight decay (L2) regularization")
        parser.add_argument("--grad_norm", default=2.0, type=float,
                            help="gradient clipping (-1 for no clipping)")
        parser.add_argument(
            "--warmup_ratio", default=0.1, type=float,
            help="to perform linear learning rate warmup for. (invsqrt decay)")
        parser.add_argument("--transformer_lr_mul", default=1.0, type=float,
                            help="lr_mul for transformer")
        parser.add_argument("--step_decay_epochs", type=int,
                            nargs="+", help="multi_step decay epochs")
        
        # Outputs
        parser.add_argument(
            "--log_interval", default=10, type=int,
            help="record every a few steps on tensorboard.")
        
        parser.add_argument(
            "--save_steps_ratio", default=0.01, type=float,
            help="save every 0.01*global steps to resume after preemption,"
                 "not used for checkpointing.")
        parser.add_argument(
                    "--output_dir", type=str,
                    help="dir to store model checkpoints & training meta.")

        # Inference
        parser.add_argument("--e2e_weights_path", type=str,
                            help="path to e2e model weights")
        # inference only, please include substring `inference'
        # in the option to avoid been overwrite by loaded options,
        # see start_inference() in run_vqa_w_hvd.py
        parser.add_argument("--inference_model_step", default=-1, type=str,
                            help="pretrained model checkpoint step")
        parser.add_argument(
            "--do_inference", default=0, type=int, choices=[0, 1],
            help="perform inference run. 0: disable, 1 enable")
        parser.add_argument(
            "--inference_split", default="val",
            help="For val, the data should have ground-truth associated it."
                 "For test*, the data comes with no ground-truth.")
        parser.add_argument("--inference_txt_db", type=str,
                            help="path to txt_db file for inference")
        parser.add_argument("--inference_img_db", type=str,
                            help="path to img_db file for inference")
        parser.add_argument("--inference_batch_size", type=int, default=64,
                            help="single-GPU batch size for inference")
        
        # Device parameters
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument(
            "--fp16", type=int, choices=[0, 1], default=0,
            help="Use 16-bit float precision instead of 32-bit."
                 "0: disable, 1 enable")
        parser.add_argument("--n_workers", type=int, default=4,
                            help="#workers for data loading")
        parser.add_argument("--pin_mem", type=int, choices=[0, 1], default=1,
                            help="pin memory. 0: disable, 1 enable")

        # can use config files, will only overwrite unset parameters
        parser.add_argument("--config", help="JSON config files")
        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        args = parse_with_config(parsed_args)

        # convert to all [0, 1] options to bool, including these task specific ones
        zero_one_options = [
            "fp16", "pin_mem", "use_itm", "use_mlm", "use_itc", "debug", #"freeze_cnn",
            "do_inference",
        ]
        for option in zero_one_options:
            if hasattr(args, option):
                setattr(args, option, bool(getattr(args, option)))

        # basic checks
        # This is handled at TrainingRestorer
        # if exists(args.output_dir) and os.listdir(args.output_dir):
        #     raise ValueError(f"Output directory ({args.output_dir}) "
        #                      f"already exists and is not empty.")
        if args.step_decay_epochs and args.decay != "multi_step":
            Warning(
                f"--step_decay_epochs epochs set to {args.step_decay_epochs}"
                f"but will not be effective, as --decay set to be {args.decay}")

        assert args.gradient_accumulation_steps >= 1, \
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps} "

        assert 1 >= args.data_ratio > 0, \
            f"--data_ratio should be [1.0, 0), but get {args.data_ratio}"

        return args

    def get_pretraining_args(self):
        # pre-training args
        self.parser.add_argument(
            "--use_itm", type=int, choices=[0, 1], default=0,
            help="enable itm loss. 0: disable, 1 enable")
        self.parser.add_argument(
            "--use_mlm", type=int, choices=[0, 1], default=0,
            help="enable mlm loss. 0: disable, 1 enable")
        self.parser.add_argument(
            "--use_itc", type=int, choices=[0, 1], default=0,
            help="enable itc loss. 0: disable, 1 enable")
        
        # sparse pretraining-specific settings
        self.parser.add_argument(
            "--crop_img_size", type=int, default=256,
            help="crop size during pre-training.")
        self.parser.add_argument(
            "--resize_size", type=int, default=288,
            help="resize frames to square, ignoring aspect ratio.")

        args = self.parse_args()
        return args

    
    def get_video_caption_args(self):
        self.parser.add_argument("--task", type=str, choices=["goal_caption"], help="Video captioning.")
        self.parser.add_argument("--video_dec_model", type=str, choices=['transformer', 'gpt2'], help='the decoder model.')
        
        args = self.parse_args()
        
        return args


shared_configs = SharedConfigs()
