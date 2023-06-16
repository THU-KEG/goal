import os
import torch
import random
import numpy as np
import copy
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import AlproBaseDataset
from src.datasets.randaugment import TemporalConsistentRandomAugment
from src.modeling.transformer.video_transformer import VideoTransformer
from src.evaluation.utils_caption_evaluate import evaluate_on_coco_caption


class AlproVideoCaptionDataset(AlproBaseDataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    enc_tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """
    open_ended_qa_names = ["goal_caption"]

    def __init__(self, task_type, datalist, enc_tokenizer, dec_tokenizer, img_lmdb_dir, dec_output_fn,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20,
                 ensemble_n_clips=1, is_train=False, random_sample_clips=True, 
                 video_fmt='.mp4', img_db_type='lmdb', sid2vid=None):
        super(AlproVideoCaptionDataset, self).__init__(
            datalist, enc_tokenizer, img_lmdb_dir, img_db_type=img_db_type,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.is_train = is_train
        self.task_type = task_type
        self.random_sample_clips = random_sample_clips
        self.dec_tokenizer = dec_tokenizer
        self.sid2vid = sid2vid
        self.video_fmt = video_fmt
        self.dec_output_fn = dec_output_fn

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)


    def __getitem__(self, index):
        # skip error videos:
        num_retries = 5
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            if self.ensemble_n_clips > 1:
                raise NotImplementedError('Do not support multiple clips for now.')
            else:
                # video_path = os.path.join(self.img_db_dir, vid_id + self.video_fmt) 
                video_path = vid_id
                vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            if self.randaug:
                vid_frm_array = self.randaug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # examples = [self._get_single_example(e) for e in examples]
            return dict(
                vid=vid_frm_array,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        
    # def _get_single_example(self, data):
    #     example = dict(
    #         q_str=data["question"],
    #         question_id=data["question_id"],
    #         label=data["answer"]
    #     )
    #     if self.task_type in self.open_ended_qa_names:
    #         if self.return_label:
    #             example["label"] = self.ans2label[example["label"]]
    #     if not self.return_label:
    #         example["label"] = None
    #     return example
    def convert_datalist_to_coco(self, datalist):
        """
        Args:
            datalist: list(tpule(dict)), where each dict contains 'caption', 'sample_id' field.
        """
        ann_coco = []
        imgs_coco = []
        for vid, examples in datalist:
            for X in examples:
                ann_coco.append({'image_id': vid, 'caption':X['caption'], 'id':X['sample_id']})
                imgs_coco.append({'id':vid, 'file_name':vid})
        data_coco = {'annotations': ann_coco, 'type': 'caption', 'images': imgs_coco, 'info': 'dummy', 'licenses':'qiji'}
        return data_coco


    def evaluate_caption(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "sample_ids": [int],
                    "pred_cap_ids": a batch of lists, each refers to the predicted sequence of indices.
                    "pred_lens": a batch of int, each refers to a length of pred seq.
                    "gold_cap_ids: a batch of lists, each refers to the golden sequence of indices.
                }
        Returns:
            TGIF-QA score
        """
        gold_coco = self.convert_datalist_to_coco(self.datalist)

        pred_datalist = []
        for X in results:
            # pred_captions = VideoTransformer.process_output(X['pred_cap_ids'], X['pred_lens'], self.tokenizer)
            pred_captions = self.dec_output_fn(X['pred_cap_ids'], X['pred_lens'], self.tokenizer)
            pred_datalist.extend([(
                self.sid2vid[X['sample_ids'][i]],
                [{
                'caption':cap,
                'sample_id': X['sample_ids'][i],
                }]) for i,cap in enumerate(pred_captions)])
        pred_coco = self.convert_datalist_to_coco(pred_datalist)

        eval_result = evaluate_on_coco_caption(pred_coco, gold_coco)

        return {
            'Bleu_1': eval_result['Bleu_1'],
            'Bleu_2': eval_result['Bleu_2'],
            'Bleu_3': eval_result['Bleu_3'],
            'Bleu_4': eval_result['Bleu_4'],
            'METEOR': eval_result['METEOR'],
            'ROUGE_L': eval_result['Bleu_1'],
            'CIDEr': eval_result['CIDEr'],
            # 'SPICE': eval_result['SPICE'],
        }

    def generate_caption(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "sample_ids": [int],
                    "pred_cap_ids": a batch of lists, each refers to the predicted sequence of indices.
                    "pred_lens": a batch of int, each refers to a length of pred seq.
                }
        Returns:
        """
        pred_datalist = []
        for X in results:
            # pred_captions = VideoTransformer.process_output(X['pred_cap_ids'], X['pred_lens'], self.tokenizer)
            pred_captions = self.dec_output_fn(X['pred_cap_ids'], X['pred_lens'], self.tokenizer)
            pred_datalist.extend([(
                self.sid2vid[X['sample_ids'][i]],
                [{
                'caption':cap,
                'sample_id': X['sample_ids'][i],
                }]) for i,cap in enumerate(pred_captions)])
        pred_coco = self.convert_datalist_to_coco(pred_datalist)

        return pred_coco


class VideoCaptionCollator(object):
    def __init__(self, enc_tokenizer, dec_tokenizer, dec_input_fn, max_enc_length=20, max_dec_length=300, task_type="goal_caption"):
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.max_enc_length = max_enc_length
        self.max_dec_length = max_dec_length
        self.task_type = task_type
        self.dec_input_fn = dec_input_fn

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        
        # prompt inputs
        text_str_list = [str(d["prompt"]) for d in text_examples]  # (B, )
        batch_enc = self.enc_tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_enc_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        prompt_input_ids = batch_enc.input_ids  # (B, L)
        prompt_input_mask = batch_enc.attention_mask  # (B, L)
        
        # target caption
        text_str_list = [str(d["caption"]) for d in text_examples]  # (B, )
        # target_input_ids, target_input_mask, gold_output_ids = VideoTransformer.prepare_input(self.dec_tokenizer, self.max_dec_length, text_str_list)
        # target_input_ids, target_input_mask, gold_output_ids = self.dec_input_fn(self.dec_tokenizer, self.max_dec_length, text_str_list)
        target_input_ids, target_input_mask = self.dec_input_fn(self.dec_tokenizer, self.max_dec_length, text_str_list)
        
        sample_ids = [d["sample_id"] for d in text_examples]

        return dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            prompt_input_ids=prompt_input_ids,
            prompt_input_mask=prompt_input_mask,
            target_input_ids=target_input_ids,
            target_input_mask=target_input_mask,
            # gold_output_ids=gold_output_ids,
            n_examples_list=n_examples_list,  # used to create image feature copies.
            sample_ids=sample_ids  # used to create image feature copies.
        )
