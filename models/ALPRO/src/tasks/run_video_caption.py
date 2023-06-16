import math
import os
import random
import time
from collections import defaultdict
import numpy as np
from os.path import join

import horovod.torch as hvd
import torch
from apex import amp
from easydict import EasyDict as edict
from src.configs.config import shared_configs
from src.datasets.data_utils import ImageNorm, mk_input_group
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
# from src.datasets.dataset_video_qa import (AlproVideoQADataset,
#                                            VideoQACollator)
# from src.modeling.alpro_models import AlproForSequenceClassification

from src.modeling.alpro_models import AlproForVideoTextGeneration
from src.datasets.dataset_video_caption import AlproVideoCaptionDataset, VideoCaptionCollator

from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from src.utils.basic_utils import (get_rounded_percentage, load_json,
                                   load_jsonl, save_json)
from src.utils.distributed import all_gather_list
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.utils.load_save import (ModelSaver,
                                 load_state_dict_with_pos_embed_resizing,
                                 save_training_meta)
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from src.utils.misc import NoOp, set_random_seed, zero_none_grad
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertConfig, BertTokenizerFast
from transformers import AutoTokenizer


def mk_caption_dataloader(task_type, anno_path, lmdb_dir, cfg, enc_tokenizer, dec_tokenizer,max_dec_txt_len,
                          dec_input_fn, dec_output_fn, is_train=True):
    """
    Returns:
        list(dict), each dict is
            goal_caption: {
                "caption": "New style kick offs which.",
                "video_id": "video6513",
                }
    """
    raw_datalist = load_json(anno_path)['annotations']
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    if cfg.data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]
        LOGGER.info(f"Use {100 * cfg.data_ratio}% of the loaded data: {len(raw_datalist)}")

    datalist = []
    sid2vid = {}
    for raw_d in raw_datalist:
        d = dict(
            caption=raw_d["caption"],
            vid_id=raw_d["image_id"],
            sample_id=raw_d["id"],
            # prompt="What is the commentary on the current video clip ?"
            prompt=raw_d["context"]
        )
        datalist.append(d)
        sid2vid[raw_d["id"]] = raw_d["image_id"]

    LOGGER.info(f"datalist {len(datalist)}")

    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vid_id"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")


    frm_sampling_strategy = cfg.frm_sampling_strategy
    if not is_train:
        # frm_sampling_strategy = "middle"
        frm_sampling_strategy = "uniform"
    
    if 'goal' in cfg.task:
        video_fmt = '.avi'
    else:
        video_fmt = '.mp4'

    dataset = AlproVideoCaptionDataset(
        task_type=cfg.task,
        datalist=group_datalist,
        enc_tokenizer=enc_tokenizer,
        dec_tokenizer=dec_tokenizer,
        img_lmdb_dir=lmdb_dir,
        dec_output_fn=dec_output_fn,
        max_img_size=cfg.crop_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        ensemble_n_clips=cfg.train_n_clips if is_train else cfg.inference_n_clips,
        is_train=is_train,
        img_db_type='rawvideo',
        video_fmt=video_fmt,
        sid2vid=sid2vid
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    collator = VideoCaptionCollator(enc_tokenizer=enc_tokenizer, dec_tokenizer=dec_tokenizer,
                                   max_enc_length=cfg.max_enc_txt_len, max_dec_length=max_dec_txt_len,
                                   dec_input_fn=dec_input_fn, task_type=cfg.task)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=collator.collate_batch)
    return dataloader


def setup_dataloaders(cfg, enc_tokenizer, dec_tokenizer, max_dec_txt_len, dec_input_fn, dec_output_fn):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loader = mk_caption_dataloader(
        task_type=cfg.task,
        anno_path=cfg.train_datasets[0].txt[cfg.task],
        lmdb_dir=cfg.train_datasets[0].img, cfg=cfg,
        enc_tokenizer=enc_tokenizer, dec_tokenizer=dec_tokenizer, max_dec_txt_len=max_dec_txt_len, is_train=True,
        dec_input_fn=dec_input_fn, dec_output_fn=dec_output_fn
    )
    val_loader = mk_caption_dataloader(
        task_type=cfg.task,
        anno_path=cfg.val_datasets[0].txt[cfg.task],
        lmdb_dir=cfg.val_datasets[0].img, cfg=cfg,
        enc_tokenizer=enc_tokenizer, dec_tokenizer=dec_tokenizer, max_dec_txt_len=max_dec_txt_len, is_train=False,
        dec_input_fn=dec_input_fn, dec_output_fn=dec_output_fn
    )
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loader = PrefetchLoader(val_loader, img_norm)
    return train_loader, val_loader


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    # add_attr_list = [
    #     "num_labels", "classifier", "cls_hidden_scale",
    #     "loss_type",
    # ]
    # for k in add_attr_list:
    #     setattr(model_cfg, k, cfg[k])

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")

    video_enc_cfg = load_json(cfg.visual_model_cfg)
    video_enc_cfg['num_frm'] = cfg.num_frm
    video_enc_cfg['img_size'] = cfg.crop_img_size
    
    video_dec_cfg = load_json(cfg.video_dec_cfg)[cfg.video_dec_model]

    model = AlproForVideoTextGeneration(
        cfg,
        model_cfg, 
        input_format=cfg.img_input_format,
        video_enc_cfg=video_enc_cfg,
        dec_cfg=video_dec_cfg,
        )
    
    total_parameters = sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad])
    LOGGER.info("Total parameters: %d" % total_parameters)

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        num_patches = (cfg.crop_img_size // video_enc_cfg['patch_size']) ** 2
        # NOTE strict is False if loaded from ALBEF ckpt
        load_state_dict_with_pos_embed_resizing(model, 
                                                cfg.e2e_weights_path, 
                                                num_patches=num_patches, 
                                                num_frames=cfg.num_frm, 
                                                strict=False,
                                                remove_text_encoder_prefix=True
                                                )
        # LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        # load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    else:
        LOGGER.info(f"Loading visual weights from {cfg.visual_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            visual_weights_path=cfg.visual_weights_path,
            bert_weights_path=cfg.bert_weights_path
        )

    # if cfg.freeze_cnn:
    #     model.freeze_cnn_backbone()
    model.to(device)

    LOGGER.info("Setup model done!")
    return model




@torch.no_grad()
def validate(model, val_loader, cfg, train_global_step, eval_score=True):
    """use eval_score=False when doing inference on test sets where captions are not available"""
    model.eval()

    loss = 0.
    n_ex = 0
    cap_results = []
    st = time.time()
    debug_step = 5
    pbar = tqdm(total=len(val_loader))
    for val_step, batch in enumerate(val_loader):
        # forward pass
        sample_ids = batch["sample_ids"]
        bsz = len(batch["visual_inputs"])
        mini_batch = dict()
        for k, v in batch.items():
            if k != "visual_inputs":
                mini_batch[k] = v

        n_ex += len(sample_ids)
        pool_method = cfg.score_agg_func
        # could be 1, where only a single clip is used
        num_clips = cfg.train_n_clips
        num_frm = cfg.num_frm
        # (B, T=num_clips*num_frm, C, H, W) --> (B, num_clips, num_frm, C, H, W)
        new_visual_shape = (bsz, num_clips, num_frm) + batch["visual_inputs"].shape[2:]
        visual_inputs = batch["visual_inputs"].view(*new_visual_shape)
        logits = []
        bsz_lens = [] # (num_frm, B)
        losses = []
        for clip_idx in range(num_clips):
            # (B, num_frm, C, H, W)
            mini_batch["visual_inputs"] = visual_inputs[:, clip_idx]
            mini_batch["n_examples_list"] = batch["n_examples_list"]
            outputs = model.forward_inference(mini_batch)
            logits.append(outputs["logits"].cpu())
            bsz_lens.append(outputs['pred_lens'])
            _loss = outputs["loss"].sum().item() if isinstance(
                outputs["loss"], torch.Tensor) else 0
            losses.append(_loss)
        loss += (sum(losses) / num_clips)
        bsz_lens = torch.cat(bsz_lens, dim=0)

        logits = torch.stack(logits)  # (num_frm, B, L, V)
        if pool_method == "mean":
            logits = logits.mean(0)  # (B, L, V)
        elif pool_method == "max":
            logits = logits.max(0)[0]  # (B, L, V)
        elif pool_method == "lse":
            logits = logits.permute(1, 0, 2).contiguous()  # (B, num_frm, 5), pooling will be done in CE
            logits = torch.logsumexp(logits, dim=1)  # torch.exp alone might be too large and unstable
        else:
            raise ValueError(f"Invalid value for pool_method, "
                             f"got {pool_method}, expect one of [`mean`, `max`, `lse`]")

        # Note: assume values that are beyond the seq lens are zero.
        # pred_lens = torch.where((logits==0.0).all(2), torch.zeros((logits.shape[:2])), torch.ones((logits.shape[:2])))
        # pred_lens = pred_lens.sum(1).int().tolist()
        pred_lens = bsz_lens


        if cfg.task in ["goal_caption"]:
            # cross entropy
            # pred_labels = logits.max(dim=-1)[1].data.tolist()
            pred_cap_ids = torch.argmax(logits, axis=-1).data # (B, L)
        else:
            # mse
            # preds = (logits + 0.5).long().clamp(min=1, max=10)
            # pred_labels = preds.data.squeeze().tolist()
            raise KeyError("Not supported dataset yet.")
        # for sid, pred_label in zip(sample_ids, pred_captions):
        cap_results.append(dict(
            sample_ids=sample_ids,
            pred_cap_ids=pred_cap_ids,
            pred_lens=pred_lens
        ))
        cap_coco = val_loader.dataset.generate_caption(cap_results)
        # )
        pbar.update(1)
        if cfg.debug and val_step >= debug_step:
            break

    if cfg.debug:
        LOGGER.info((sample_ids, cap_coco))
    n_ex_per_rank = all_gather_list(n_ex)
    loss = sum(all_gather_list(loss))
    n_ex = sum(all_gather_list(n_ex))
    # average loss for each example
    val_log = {f'valid/loss': float(loss / n_ex)}
    if eval_score:
        LOGGER.info(f"Caption Task [{cfg.task}], "
                    f"{len(cap_results)} caption_results,"
                    f"3 examples here: {cap_results[:3]}")

        cap_scores = val_loader.dataset.evaluate_caption(cap_results)
        # print(f"{hvd.rank()}: {vqa_scores}")

        # Gather scores
        scores_per_rank = all_gather_list(cap_scores)
        gathered_scores = {}

        # FIXME: Gather scores become complicated due to np.mean and dict format.
        for scores_k, _ in cap_scores.items():
            gathered_v = 0
            for rank_id, n in enumerate(n_ex_per_rank):
                curr_acc = scores_per_rank[rank_id][scores_k]
                gathered_v += curr_acc
            gathered_v = gathered_v * 1. / len(n_ex_per_rank)
            if cfg.task not in ["goal_caption"]:
                gathered_scores[scores_k] = get_rounded_percentage(gathered_v)
            else:
                gathered_scores[scores_k] = round(gathered_v, 2)

    else:
        LOGGER.info("eval_score = False, no scores are calculated.")
        gathered_scores = 0

    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds."
                f"{gathered_scores}")

    model.train()
    # return cap_results, gathered_scores
    return cap_coco, gathered_scores


def start_training(cfg):
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    cfg.n_gpu = n_gpu
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    model = setup_model(cfg, device=device)
    model.train()
    optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level='O2',
        keep_batchnorm_fp32=True)

    # prepare data
    enc_tokenizer = BertTokenizerFast.from_pretrained(cfg.enc_tokenizer_dir)
    # if cfg.video_dec_model == 'transformer':
    #     video_dec_cfg = load_json(cfg.video_dec_cfg)['transformer']
    #     dec_tokenizer = AutoTokenizer.from_pretrained(video_dec_cfg['tokenizer_dir'])
    # elif cfg.video_dec_model == 'gpt2':
    #     video_dec_cfg = load_json(cfg.video_dec_cfg)['gpt2']
    #     dec_tokenizer = GPT2Tokenizer(video_dec_cfg['vocab_file'], video_dec_cfg['merges_file'])
    # train_loader, val_loader = setup_dataloaders(cfg, enc_tokenizer, dec_tokenizer, video_dec_cfg['max_dec_length'])
    train_loader, val_loader = setup_dataloaders(cfg, enc_tokenizer, model.decoder.tokenizer, model.decoder.max_dec_length, 
                                    model.decoder.prepare_input, model.decoder.process_output)

    # compute the number of steps and update cfg
    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))
    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
        save_training_meta(cfg)
        LOGGER.info("Saving training done...")
        TB_LOGGER.create(join(cfg.output_dir, 'log'))
        pbar = tqdm(total=cfg.num_train_steps)
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 3
    running_loss = RunningMeter('train_loss')
    for step, batch in enumerate(InfiniteIterator(train_loader)):
    # for step, batch in enumerate(train_loader):
        # forward pass
        bsz = len(batch["visual_inputs"])
        mini_batch = dict()
        for k, v in batch.items():
            if k != "visual_inputs":
                mini_batch[k] = v

        pool_method = cfg.score_agg_func
        # could be 1, where only a single clip is used
        num_clips = cfg.train_n_clips
        num_frm = cfg.num_frm
        # (B, T=num_clips*num_frm, C, H, W) --> (B, num_clips, num_frm, C, H, W)
        new_visual_shape = (bsz, num_clips, num_frm) + batch["visual_inputs"].shape[2:]
        visual_inputs = batch["visual_inputs"].view(*new_visual_shape)
        logits = []
        for clip_idx in range(num_clips):
            # (B, num_frm, C, H, W)
            mini_batch["visual_inputs"] = visual_inputs[:, clip_idx]
            mini_batch["n_examples_list"] = batch["n_examples_list"]
            # outputs = forward_step(model, mini_batch, cfg)
            outputs = model(mini_batch)
            logits.append(outputs)
            # the losses are cross entropy and mse, no need to * num_labels

            loss = outputs['loss']

        loss = loss.mean()

        running_loss(loss.item())
        # backward pass
        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            n_epoch = int(1. * total_train_batch_size * global_step
                          / total_n_examples)
            # learning rate scheduling cnn
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs,
                multi_step_epoch=n_epoch)

            # Hardcoded param group length
            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                    param_group['lr'] = lr_this_step

            if step % cfg.log_interval == 0:
                TB_LOGGER.add_scalar(
                    "train/lr", lr_this_step, global_step)

            TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer),
                    cfg.grad_norm)
                TB_LOGGER.add_scalar(
                    "train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
            restorer.step()
            pbar.update(1)

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(
                    model, val_loader, cfg, global_step)
                model_saver.save(step=global_step, model=model)
        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        cap_results, cap_scores = validate(
            model, val_loader, cfg, global_step)
        model_saver.save(step=global_step, model=model)


def start_inference(cfg):
    set_random_seed(cfg.seed)
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True

    inference_res_dir = join(
        cfg.output_dir,
        f"results_{os.path.splitext(os.path.basename(cfg.inference_txt_db))[0]}/"
        f"step_{cfg.inference_model_step}_{cfg.inference_n_clips}_{cfg.score_agg_func}"
    )

    if hvd.rank() == 0:
        os.makedirs(inference_res_dir, exist_ok=True)
        save_json(cfg, join(inference_res_dir, "raw_args.json"),
                  save_pretty=True)

    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    # overwrite cfg with stored_cfg,
    # but skip keys containing the keyword 'inference'
    stored_cfg_path = join(cfg.output_dir, "log/args.json")
    stored_cfg = edict(load_json(stored_cfg_path))
    for k, v in cfg.items():
        if k in stored_cfg and "inference" not in k:
            setattr(cfg, k, stored_cfg[k])

    # setup models
    cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    e2e_weights_path = join(
        cfg.output_dir, f"ckpt/model_step_{cfg.inference_model_step}.pt")
    cfg.e2e_weights_path = e2e_weights_path
    model = setup_model(cfg, device=device)
    model.eval()

    # FIXME separate scaling for each loss
    model = amp.initialize(
        model, enabled=cfg.fp16, opt_level='O2')

    # prepare data
    # video_dec_cfg = load_json(cfg.video_dec_cfg)
    enc_tokenizer = BertTokenizerFast.from_pretrained(cfg.enc_tokenizer_dir)
    # dec_tokenizer = AutoTokenizer.from_pretrained(video_dec_cfg['tokenizer_dir'])
    dec_tokenizer = model.decoder.tokenizer

    global_step = 0
    # prepare data
    cfg.data_ratio = 1.
    val_loader = mk_caption_dataloader(
        task_type=cfg.task,
        anno_path=cfg.inference_txt_db,
        lmdb_dir=cfg.inference_img_db,
        cfg=cfg,
        enc_tokenizer=enc_tokenizer,
        dec_tokenizer=dec_tokenizer,
        # max_dec_txt_len=video_dec_cfg['max_dec_length'],
        max_dec_txt_len=model.decoder.max_dec_length,
        is_train=False,
        dec_input_fn=model.decoder.prepare_input, dec_output_fn=model.decoder.process_output
    )
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    val_loader = PrefetchLoader(val_loader, img_norm)

    LOGGER.info(cfg)
    LOGGER.info("Starting inference...")
    LOGGER.info(f"***** Running inference with {n_gpu} GPUs *****")
    LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

    LOGGER.info(f'Step {global_step}: start validation')
    cap_results, cap_scores = validate(
        model, val_loader, cfg, global_step,
        eval_score=True)  # cfg.inference_split == "val"
    
    if hvd.rank() == 0:
        save_json(cfg, join(inference_res_dir, "merged_args.json"),
                  save_pretty=True)
        save_json(cap_scores, join(inference_res_dir, "scores.json"),
                  save_pretty=True)

    # ###### Saving with Horovod ####################
    # dummy sync
    _ = None
    all_gather_list(_)
    if n_gpu > 1:
        # with retrial, as azure blob fails occasionally.
        max_save_load_trial = 10
        save_trial = 0
        while save_trial < max_save_load_trial:
            try:
                LOGGER.info(f"Save results trial NO. {save_trial}")
                save_json(
                    cap_results,
                    join(inference_res_dir, f"results_rank{hvd.rank()}.json"))
                break
            except Exception as e:
                save_trial += 1
    # dummy sync
    _ = None
    all_gather_list(_)
    # join results
    if n_gpu > 1 and hvd.rank() == 0:
        cap_results = []
        for rk in range(n_gpu):
            cap_results.extend(load_json(
                join(inference_res_dir, f"results_rank{rk}.json")))
        LOGGER.info(f'results joined')

    if hvd.rank() == 0:
        save_json(
            cap_results,
            join(inference_res_dir, f"results_all.json"))
        LOGGER.info(f'all results written')


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    input_cfg = shared_configs.get_video_caption_args()
    if input_cfg.do_inference:
        # assert hvd.size() == 1, \
        #     "Please use single GPU for evaluation! " \
        #     "Multi-GPU might miss some examples."
        start_inference(input_cfg)
    else:
        start_training(input_cfg)
