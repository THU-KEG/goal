import os
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
import datetime
import re
import tqdm
import collections
import stanza
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from nltk.tokenize import sent_tokenize, word_tokenize
from multiprocessing import cpu_count, Process, Lock, Manager, Pool
import random
import argparse
from utils import *



# The class for satistics
class Stat():
    def __init__(self,):
        self.clip_lens = [] # all clips
        self.sent_lens = [] # [(sent_len1, sent_len2, ...), ], each tuple refer to a clip with multiple sents
        self.sent_durs = [] # [(sent_dur1, sent_ddur2, ...), ], each tuple refer to a clip with multiple sents
        self.verb_lens = [] # # [(verb_n1, verb_n2, ...), ], each tuple refer to a clip with multiple sents
        self.adj_lens = [] # # [(verb_n1, verb_n2, ...), ], each tuple refer to a clip with multiple sents
        self.adv_lens = [] # # [(verb_n1, verb_n2, ...), ], each tuple refer to a clip with multiple sents
        self.noun_lens = [] # # [(verb_n1, verb_n2, ...), ], each tuple refer to a clip with multiple sents
        self.sent_tree_depths = [] # # [(depth1, depth2, ...), ], each tuple refer to a clip with multiple sents
        self.vocab = collections.Counter()
    
    def calculate(self,):
        self.dur_per_clip = np.mean(self.clip_lens)
        self.sents_per_clip = np.mean([len(slens) for slens in self.sent_lens])
        self.words_per_clip = np.mean([sum([l for l in slens]) for slens in self.sent_lens])
        self.words_per_sent = np.mean([l for slens in self.sent_lens for l in slens])
        self.words_cover_clip = sum(self.sent_durs) / max(1e-10, sum(self.clip_lens))
        self.words_per_sec = sum([l for slens in self.sent_lens for l in slens]) / max(1e-10, sum(self.clip_lens))
        self.verbs_per_sec = np.mean([l for vlens in self.verb_lens for l in vlens])
        self.adjs_per_sec = np.mean([l for vlens in self.adj_lens for l in vlens])
        self.advs_per_sec = np.mean([l for vlens in self.adv_lens for l in vlens])
        self.nouns_per_sec = np.mean([l for vlens in self.noun_lens for l in vlens])
        self.syntactic_complexity = np.mean([l for tree_depths in self.sent_tree_depths for l in tree_depths])
        self.total_clips = sum([1 for cl in self.clip_lens])
        self.total_sents = sum([1 for sl in self.sent_lens for s in sl])
        self.vocab_size = len(self.vocab)
        # compute ratio of long sentences
        LENS = list(range(5, 20, 5))
        all_lens = np.array([l for slens in self.sent_lens for l in slens])
        self.sent_lens_ratios = {}
        for l in LENS:
            self.sent_lens_ratios[l] = (all_lens > l).astype(int).sum() / len(all_lens)
        # compute var, std
        self.sent_lens_var = np.var(all_lens)
        self.sent_lens_std = np.std(all_lens)


        
    def update(self, sta):
        self.clip_lens.extend(sta.clip_lens)
        self.sent_lens.extend(sta.sent_lens)
        self.sent_durs.extend(sta.sent_durs)
        self.verb_lens.extend(sta.verb_lens)
        self.adj_lens.extend(sta.adj_lens)
        self.advb_lens.extend(sta.adv_lens)
        self.noun_lens.extend(sta.noun_lens)
        self.vocab.update(sta.vocab)
        
    def log_all(self, save_file=None, plot_name=None):
        if save_file is not None:
            print("Saving logs to file ...")
        print("dur_per_clip: %s" % self.dur_per_clip, file=save_file)
        print("sents_per_clip: %s" % self.sents_per_clip, file=save_file)
        print("words_per_clip: %s" % self.words_per_clip, file=save_file)        
        print("words_per_sent: %s" % self.words_per_sent, file=save_file)        
        print("words_per_sec: %s" % self.words_per_sec, file=save_file)        
        print("verbs_per_sec: %s" % self.verbs_per_sec, file=save_file)  
        print("adjs_per_sec: %s" % self.adjs_per_sec, file=save_file)  
        print("advs_per_sec: %s" % self.advs_per_sec, file=save_file)  
        print("nouns_per_sec: %s" % self.nouns_per_sec, file=save_file)  
        print("syntactic_complexity: %s" % self.syntactic_complexity, file=save_file)        
        print("vocab_size: %s" % self.vocab_size, file=save_file)
        print("words coverage of each clip: %.2f" % self.words_cover_clip, file=save_file)
        print("Total # of clips: %s" % self.total_clips, file=save_file)
        print("Total # of sentences: %s" % self.total_sents, file=save_file)
        print("Ratio of sentence lengths: %s " % json.dumps(self.sent_lens_ratios), file=save_file)
        print("Variance of sentence lengths: %s " % self.sent_lens_var, file=save_file)
        print("Standard-variance of sentence lengths: %s " % self.sent_lens_std, file=save_file)
        print("Standard-variance of sentence lengths: %s " % json.dumps(self.sent_lens_ratios), file=save_file)
        if save_file is not None:
            save_file.close()
        if plot_name is not None:
            datasets = [self.clip_lens, flat_list_of_list(self.sent_lens), stat.sent_durs]
            bins = [50, 50, 50]
            titles = ["lengths of clips (seconds)", "lengths of sentences (words)", "durations of sentences (seconds)"]
            display_analysis_hists(datasets, bins, titles, cols=3, save_name=plot_name)
        
            

            
            
def stat_multi_clips(list_of_clipSentsSentdur, local_rank=0, n_sample_sents=None, n_sample_syntax=None, seed=0):
        """
          Args:
            list_of_clip_with_sents: [(f_clip, [sent1, sent2, ...]), ...]
        """
        stanford_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',
                                      use_gpu=False, download_method=None, logging_level='ERROR')

        if local_rank == 0:
            data = tqdm.tqdm(list_of_clipSentsSentdur)
        else:
            pass
            data = list_of_clipSentsSentdur

        random.seed(seed)
        to_parse_sentences = []
        to_stat_sentences = []
        stat = Stat()
        for i, (f_clip, sents, sdur) in enumerate(data):
            # Statistics clip
            if os.path.exists(f_clip):
                fcap = cv2.VideoCapture(f_clip)
                FPS = fcap.get(cv2.CAP_PROP_FPS)
                FCOUNT = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
                fcap.release()
                if FCOUNT == 0:
                    raise BaseException
                stat.clip_lens.append(FCOUNT / FPS)
            # Statistics s/data/qiji/DATA/VideoCaption/s_words = [s.split() for s in sents]
            # s_words = [s.split() for s in sents]
            # s_words = [word_tokenize(s) for s in sents]
            to_parse_sentences.append(sents)
            # to_stat_sentences.extend([(i, s_words[ii], sdur) for ii in range(len(s_words))])
            to_stat_sentences.extend([(i, sents[ii], sdur) for ii in range(len(sents))])
        stat.vocab.update([w for sent in to_stat_sentences for w in sent[1].split()])
        
        if n_sample_sents is not None:
            random.seed(seed)
            to_stat_sentences = random.sample(to_stat_sentences, n_sample_sents)

        # Sentence analysis
        # stat.vocab.update([w for sent in to_stat_sentences for w in sent[1]])
        sent_lens = collections.defaultdict(list)
        sent_durs = collections.defaultdict(list)
        for sent in to_stat_sentences:
            sent_lens[sent[0]].append(len(sent[1].split()))
            # sent_durs[sent[0]].append(sent[2])
            sent_durs[sent[0]] = sent[2]
        stat.sent_lens = list(sent_lens.values())
        stat.sent_durs = list(sent_durs.values())
            
        # Syntatic analysis
        if n_sample_syntax is not None:
            to_parse_sentences = random.sample(to_parse_sentences, n_sample_syntax)
            to_parse_sentences = {i:sents for i,sents in enumerate(to_parse_sentences)}
        else:
            to_parse_sentences = collections.defaultdict(list)
            for clip_id, sent, sdur in to_stat_sentences:
                to_parse_sentences[clip_id].append(sent)
        
        if local_rank == 0:
            data = tqdm.tqdm(to_parse_sentences.items())
        else:
            data = to_parse_sentences.items()
        for clip_id, sents in data:
            depths, n_verbs, n_nouns, n_adjs, n_advs = [], [], [], [], []
            for cur_sent in sents:
                parse_out = stanford_parser(cur_sent)
                # POS
                poss = collections.Counter()
                poss.update([w.upos for w in parse_out.iter_words()])
                n_verbs.append(poss['VERB'])
                n_adjs.append(poss['ADJ'])
                n_advs.append(poss['ADV'])
                n_nouns.append(poss['NOUN'])
                # Constituency
                if len(parse_out.sentences) > 0:
                    depths.append(parse_out.sentences[0].constituency.depth())
                else:
                    depths.append(0)
            stat.verb_lens.append(n_verbs)
            stat.adj_lens.append(n_adjs)
            stat.adv_lens.append(n_advs)
            stat.noun_lens.append(n_nouns)
            stat.sent_tree_depths.append(depths)
        return stat
    
    
# Statistics GOAL
def stat_goal(commentary_dir, clips_dir, n_samples=None, workers=None, n_sample_sents=10000, n_sample_syntax=100, seed=71):

    # stanford_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',
    #                                   use_gpu=False, download_method=None, logging_level='ERROR')
    stat = Stat()
    tot_durs = 0
    clips_sentences = []
    files_comm = os.listdir(commentary_dir)
    for f_comm in tqdm.tqdm(files_comm):
        if not f_comm.endswith(".json"):
            continue

        with open(os.path.join(commentary_dir, f_comm), "r") as f:
            comm = json.load(f)
        game_id = f_comm.replace(".json", "")
        for X in comm:
            # clips
            clip_path = os.path.join(clips_dir, game_id, X["clip"])
            # sents
            sents = sent_tokenize(str(X["commentary"]))
            clips_sentences.append((clip_path, sents, X['duration']))

            # basic statistics
            fcap = cv2.VideoCapture(clip_path)
            FPS = fcap.get(cv2.CAP_PROP_FPS)
            FCOUNT = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
            fcap.release()
            if FCOUNT == 0:
                raise BaseException
            tot_durs += FCOUNT / FPS
    
    if n_samples is not None:
        random.seed(seed)
        clips_sentences = random.sample(clips_sentences, n_samples)
    
    if workers is None:
        print("Total # of clips: %s" % len(clips_sentences))
        print("Total hours: %s" % (tot_durs / 3600))
        print("Total sentences: %s" % sum([len(slens[1]) for slens in clips_sentences]))
        stat = stat_multi_clips(clips_sentences, local_rank=0,  n_sample_sents=n_sample_sents, n_sample_syntax=n_sample_syntax, seed=seed)
        stat.calculate()
    elif workers > 0: # multi-processes
        all_stats = []
        pool = Pool(processes=workers)
        chunks = list(range(0, len(clips_sentences), len(clips_sentences)//workers))[:-1] + [len(clips_sentences)]
        for i in range(1, len(chunks)):
            lines = clips_sentences[chunks[i-1]: chunks[i]]
            all_stats.append(
                pool.apply_async(stat_multi_clips, args=(lines, ), kwds={"local_rank":i-1})
            )
        pool.close(); pool.join()
        stat = Stat()
        for sta in all_stats:
            stat.update(sta.get())
        stat.calculate()
    return stat


def stat_coco_format(files_caption, n_samples=None, workers=None, n_sample_sents=10000, n_sample_syntax=100, seed=71):
    """
    """
    stanford_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',
                                  use_gpu=False, download_method=None, logging_level='ERROR')
    ann_coco = []
    for f_cap in files_caption:
        with open(f_cap) as f:
            ann_coco.extend(json.load(f)['annotations'])
        
    clip_anns = collections.defaultdict(list)
    for X in ann_coco:
        clip_anns[X['image_id']].append(X['caption'])
        
    clip_anns = [(x, clip_anns[x], 0) for x in clip_anns]
    if n_samples is not None:
        random.seed(seed)
        clip_anns = random.sample(clip_anns, n_samples)
    
    if workers is None:
        stat = stat_multi_clips(clip_anns, local_rank=0, n_sample_sents=n_sample_sents, n_sample_syntax=n_sample_syntax, seed=seed)
        stat.calculate()
        print("Total # of clips: %s" % len(clip_anns))
        print("Total sentences: %s" % sum([len(slens) for slens in clip_anns]))
    elif workers > 0: # multi-processes
        all_stats = []
        pool = Pool(processes=workers)
        chunks = list(range(0, len(clip_anns), len(clip_anns)//workers))[:-1] + [len(clip_anns)]
        for i in range(1, len(chunks)):
            lines = clip_anns[chunks[i-1]: chunks[i]]
            all_stats.append(
                pool.apply_async(stat_multi_clips, args=(lines, ), kwds={"local_rank":i-1})
            )
        pool.close(); pool.join()
        stat = Stat()
        for sta in all_stats:
            stat.update(sta.get())
        stat.calculate()
    
    return stat


# Statistics YouCook2
def stat_youcook2(files_dir, n_samples=None, workers=None, n_sample_sents=10000, n_sample_syntax=100, seed=71):
    
    all_captions = [] # train, val, test
    with open(os.path.join(files_dir, 'training.caption.tsv')) as f:
        all_captions.extend(f.readlines())
    with open(os.path.join(files_dir, 'validation.caption.tsv')) as f:
        all_captions.extend(f.readlines())
    with open(os.path.join(files_dir, 'testing.caption.tsv')) as f:
        all_captions.extend(f.readlines())
        
    clip_anns = collections.defaultdict(list)
    for line in all_captions:
        v_path, caps = line.strip().split('\t')
        caption = json.loads(caps)[0]['caption']
        clip_anns[v_path].append(caption)
    
    new_clip_anns = []
    for v_path in clip_anns:
        if len(clip_anns[v_path]) > 1:
            print("There are more than 1 captions in YouCook2.")
        new_clip_anns.append([v_path, clip_anns[v_path], 0])
    
    clip_anns = new_clip_anns
    if n_samples is not None:
        random.seed(seed)
        clip_anns = random.sample(clip_anns, n_samples)
    
    if workers is None:
        stat = stat_multi_clips(clip_anns, local_rank=0, n_sample_sents=n_sample_sents, n_sample_syntax=n_sample_syntax, seed=seed)
        stat.calculate()
        print("Total # of clips: %s" % len(clip_anns))
        print("Total sentences: %s" % sum([len(slens) for slens in clip_anns]))
    elif workers > 0: # multi-processes
        all_stats = []
        pool = Pool(processes=workers)
        chunks = list(range(0, len(clip_anns), len(clip_anns)//workers))[:-1] + [len(clip_anns)]
        for i in range(1, len(chunks)):
            lines = clip_anns[chunks[i-1]: chunks[i]]
            all_stats.append(
                pool.apply_async(stat_multi_clips, args=(lines, ), kwds={"local_rank":i-1})
            )
        pool.close(); pool.join()
        stat = Stat()
        for sta in all_stats:
            stat.update(sta.get())
        stat.calculate()
    
    return stat


# Stat ActivityNet
def sta_activitynet(data_dir, n_samples=None, workers=None, n_sample_sents=10000, n_sample_syntax=100, seed=71):
    
    clip_anns = []
    with open(os.path.join(data_dir, 'train.json')) as f:
        data = json.load(f)
        for vid in data:
            clip_anns.append([vid, data[vid]['sentences'], data[vid]['duration']])
    with open(os.path.join(data_dir, 'val_1.json')) as f:
        data = json.load(f)
        for vid in data:
            clip_anns.append([vid, data[vid]['sentences'], data[vid]['duration']])
    with open(os.path.join(data_dir, 'val_2.json')) as f:
        data = json.load(f)
        for vid in data:
            clip_anns.append([vid, data[vid]['sentences'], data[vid]['duration']])

    
    if n_samples is not None:
        random.seed(seed)
        clip_anns = random.sample(clip_anns, n_samples)
    
    if workers is None:
        stat = stat_multi_clips(clip_anns, local_rank=0, n_sample_sents=n_sample_sents, n_sample_syntax=n_sample_syntax, seed=seed)
        stat.calculate()
        print("Total # of clips: %s" % len(clip_anns))
        print("Total sentences: %s" % sum([len(slens) for slens in clip_anns]))
    elif workers > 0: # multi-processes
        all_stats = []
        pool = Pool(processes=workers)
        chunks = list(range(0, len(clip_anns), len(clip_anns)//workers))[:-1] + [len(clip_anns)]
        for i in range(1, len(chunks)):
            lines = clip_anns[chunks[i-1]: chunks[i]]
            all_stats.append(
                pool.apply_async(stat_multi_clips, args=(lines, ), kwds={"local_rank":i-1})
            )
        pool.close(); pool.join()
        stat = Stat()
        for sta in all_stats:
            stat.update(sta.get())
        stat.calculate()
    
    return stat
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # GOAL
    parser.add_argument('video_dir', type=str, default="/data/qiji/DATA/soccernet/videos/")
    parser.add_argument('anns_dir', type=str, default="/data/qiji/DATA/soccernet/annotations/")
    parser.add_argument('clips_dir', type=str, default="/data/qiji/DATA/soccernet/clips/")
    parser.add_argument('commentary_dir', type=str, default="/data/qiji/DATA/soccernet/commentaries/")
    parser.add_argument('caption_dir', type=str, default="/data/qiji/DATA/soccernet/captions/")
    # Others
    parser.add_argument('msvd_dir', type=str, default="/data/qiji/DATA/VideoCaption/SwinBERT/MSVD")
    parser.add_argument('msrvtt_dir', type=str, default="/data/qiji/DATA/VideoCaption/SwinBERT/MSRVTT-v2")
    parser.add_argument('youcook2_dir', type=str, default="/data/qiji/DATA/VideoCaption/SwinBERT/YouCook2")
    parser.add_argument('activitynet_dir', type=str, default="/data/qiji/DATA/VideoCaption/ActivityNet")
    # Specify
    parser.add_argument('--dataset', type=str, choices=['goal', 'msvd', 'msrvtt', 'activitynet', 'youcook2'], default='goal')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed = args.seed

    if args.dataset == 'goal':
        # Statistics GOAL
        # stat = stat_goal(commentary_dir, n_samples=None, workers=None, n_sample_sents=10000, n_sample_syntax=100, seed=seed)
        stat = stat_goal(args.commentary_dir, args.clips_dir, n_samples=None, workers=None, n_sample_sents=None, n_sample_syntax=100, seed=seed)
        with open(f"logs/stat_goal_seed{seed}.txt", 'w') as f:
            stat.log_all(save_file=f, plot_name="logs/goal.png")
    elif args.dataset == 'msvd':
        # Statistics MSVD
        files = map(lambda x:os.path.join(args.msvd_dir, x), ["train.caption_coco_format.json", "val.caption_coco_format.json", "test.caption_coco_format.json"])
        stat_msvd_train = stat_coco_format(files, n_samples=None, n_sample_sents=10000, n_sample_syntax=100, seed=71)
        # stat_msvd_train = stat_coco_format(files, n_samples=None, n_sample_sents=10000, n_sample_syntax=None, seed=71)
        with open("logs/stat_msvd_seed{}.txt".format(seed), 'w') as f:
            stat_msvd_train.log_all(save_file=f)
    elif args.dataset == 'msrvtt':
        # Statistics MSRVTT
        files = map(lambda x:os.path.join(args.msrvtt_dir, x), ["train.caption_coco_format.json", "val.caption_coco_format.json", "test.caption_coco_format.json"])
        stat_msrvtt = stat_coco_format(files, n_samples=None, n_sample_sents=10000, n_sample_syntax=100, seed=71)
        # stat_msrvtt = stat_coco_format(files, n_samples=None, n_sample_sents=10000, n_sample_syntax=None, seed=71)
        with open("logs/stat_msrvtt_seed{}.txt".format(seed), 'w') as f:
            stat_msrvtt.log_all(save_file=f)
    elif args.dataset == 'youcook2':
        # Statistics YouCook2
        stat_youcook2 = stat_youcook2(args.youcook2_dir, n_samples=None, n_sample_sents=10000, n_sample_syntax=100, seed=71)
        # stat_youcook2 = stat_youcook2(youcook2_dir, n_samples=None, n_sample_sents=10000, n_sample_syntax=None, seed=71)
        with open("logs/stat_youcook2_seed{}.txt".format(seed), 'w') as f:
            stat_youcook2.log_all(save_file=f)
    elif args.dataset == 'activitynet':
        # Statistics ActivityNet   
        sat_activitynet = sta_activitynet(args.activitynet_dir, n_samples=None, n_sample_sents=10000, n_sample_syntax=100, seed=seed)
        # sat_activitynet = sta_activitynet(activitynet_dir, n_samples=None, n_sample_sents=10000, n_sample_syntax=None, seed=seed)
        with open("logs/stat_activitynet_seed{}.txt".format(seed), 'w') as f:
            sat_activitynet.log_all(save_file=f)
    else:
        raise ValueError("Not supported dataset.")