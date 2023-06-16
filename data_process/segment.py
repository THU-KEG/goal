import os
import cv2
import argparse
import numpy as np
from PIL import Image
import collections
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re
import tqdm
import operator
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count, Process, Lock, Manager, Pool




def extract_clips(params, max_off=float('Inf'), local_rank=0):
    """ Extract specify clips from the given video according to the offsets and durations using cv2.
      Args:
        params: [(f_video, offset, duration, save_name), ...]
    """
    if local_rank == 0:
        data = tqdm.tqdm(params)
    else:
        data = params
        
    for (f_v, off, dur, s_name) in data:
        # check legal offset
        if off > max_off:
            raise IndexError("The offset is out of the frame numbers.")
        fcap = cv2.VideoCapture(f_v)
        FPS = fcap.get(cv2.CAP_PROP_FPS)
        FCOUNT = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
        W, H = int(fcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(fcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        frame = fcap.set(cv2.CAP_PROP_POS_FRAMES, off*FPS)
        # save_path = os.path.join(save_dir, "{}.avi".format(i+start_idx))

        writer = cv2.VideoWriter(s_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), FPS, (W, H))

        suc, frame = fcap.read()
        ii = 0
        while suc and ii<dur*FPS:
            writer.write(frame)
            suc, frame = fcap.read()
            ii += 1
        writer.release()
        fcap.release()



def segment(f_ann, clips_dir, commentary_dir, video_dir):

    # Segment
    os.makedirs(commentary_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)
    
    df_sheets = pd.read_excel(f_ann, sheet_name=None, engine="openpyxl")
    waiting_extracts = collections.defaultdict(dict)
    for sh_name in tqdm.tqdm(df_sheets):
        if sh_name == "progress":
            continue
        
        df = df_sheets[sh_name]
        game_id = name_map[sh_name]
        
        # read corresponding video
        # fcap = cv2.VideoCapture(os.path.join(video_dir, game_id+"_720p.mkv"))
        # FPS = fcap.get(cv2.CAP_PROP_FPS)
        # FCOUNT = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
        # W, H = int(fcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(fcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # pbar = tqdm.tqdm(total=df.shape[0])
        result = []
        i = 0
        while i < df.shape[0]:
        # for i in range(df.shape[0]):
            ofs, dur, comm, ent, clss = df.iloc[i][["offset", "duration", "proof_read_text", "entity", "classification"]]
            if pd.isna(comm) or comm=='': # legal check
                i += 1
                continue

            # Merge strategy
            if dur < 2 or len(str(comm).split()) < 3:
                # if i != 0 and ofs - df.iloc[i-1]["duration"] - df.iloc[i-1]["offset"] < 4: # merge to last
                if i != 0 and ofs - df.iloc[i-1]["duration"] - df.iloc[i-1]["offset"] < 5 and \
                    not (pd.isna(df.iloc[i-1]["proof_read_text"]) or df.iloc[i-1]["proof_read_text"]==''): # merge to last, V1
                    # change last
                    last_X = result.pop()
                    dur = ofs - last_X["offset"] + dur
                    assert dur > 0
                    ofs = last_X["offset"]
                    comm = "{} {}".format(last_X["commentary"], comm)
                    ent = "{} {}".format(last_X["entity"], ent)
                    clss = "{}/{}".format(last_X["class"], clss)
                    last_clip_path = os.path.join(clips_dir, game_id, last_X["clip"])
                    # os.remove(last_clip_path)
                    waiting_extracts[game_id].pop(last_X["clip"])
                # elif i != df.shape[0]-1 and ofs + dur - df.iloc[i-1]["offset"] < 4: # merge to next
                elif i != df.shape[0]-1 and ofs + dur - df.iloc[i+1]["offset"] < 5 and \
                    not (pd.isna(df.iloc[i+1]["proof_read_text"]) or df.iloc[i+1]["proof_read_text"]==''): # merge to next, V1
                    # skip to next
                    i += 1
                    next_ofs, next_dur, next_comm, next_ent, next_clss = \
                        df.iloc[i][["offset", "duration", "proof_read_text", "entity", "classification"]]
                    # dur = dur + next_dur # TO-DO: consider the interval between two commentaries
                    dur = next_ofs - ofs + next_dur
                    assert dur > 0
                    comm = "{} {}".format(comm, next_comm)
                    ent = "{} {}".format(ent, next_ent)
                    clss = "{}/{}".format(clss, next_clss)
                    # pbar.update(1)
                else:
                    i += 1
                    continue


            # Segment strategy
            extra = 2
            # if i == 0:
            #     s_frame = max(ofs*FPS, (ofs-extra)*FPS)
            # else:
            #     s_frame = max(X["offset"]*FPS, (ofs-extra)*FPS)
            # s_frame = max(0, (ofs-extra)*FPS)
            # n_frame = (dur+extra)*FPS
            # print(ofs,dur, s_frame, n_frame)

            max_ofs = 45 * 60
            if ofs > max_ofs:
                i += 1
                continue
                
            waiting_extracts[game_id][f'{i}.avi'] = [
                os.path.join(video_dir, game_id+"_720p.mkv"),
                max(0, ofs-extra), dur+extra,
                os.path.join(clips_dir, game_id, f'{i}.avi'),
            ]
            os.makedirs(os.path.join(clips_dir, game_id), exist_ok=True)

            if type(clss) == datetime.datetime:
                clss = "{},{}".format(clss.month, clss.day)
            X = {
                "offset": ofs,
                "duration": dur,
                "commentary": comm,
                "entity": ent,
                "class": clss,
                "clip": f"{i}.avi"
            }
            result.append(X)
            i += 1
            # pbar.update(1)
        # save
        with open(os.path.join(commentary_dir, game_id+".json"), "w") as f:
            try:
                json.dump(result, f)
            except:
                print(sh_name)

    # Extract clips with multi process
    waiting_extracts = [ext for gid, g_exts in waiting_extracts.items() for k, ext in g_exts.items()]
    print("Total clips waiting to extract: %s" % len(waiting_extracts))
    max_ofs = 45 * 60
    workers = 20 # further dstribute to 4 workers
    pool = Pool(processes=workers)
    chunks = list(range(0, len(waiting_extracts), len(waiting_extracts)//workers))[:-1] + [len(waiting_extracts)]
    for i in range(1, len(chunks)):
        lines = waiting_extracts[chunks[i-1]: chunks[i]]
        pool.apply_async(extract_clips, args=(lines, ), kwds={"max_off":max_ofs, "local_rank":i-1})
    pool.close(); pool.join()



def generate_captions(caption_dir, commentary_dir, clips_dir):

    os.makedirs(caption_dir, exist_ok=True)

    caption_result = []
    vid2path = []
    tot_samples = 0
    for f_comm in tqdm.tqdm(os.listdir(commentary_dir)):
        game_id = f_comm.replace(".json", "")

        if not f_comm.endswith(".json"): continue
        # Load commentaries
        with open(os.path.join(commentary_dir, f_comm), "r") as f:
            comm = json.load(f)
        
        for X in comm:
            v_path = os.path.join(clips_dir, game_id, X['clip'])
            new_X = {
                'image_id': v_path,
                'caption': X['commentary'],
                'id': tot_samples,
            }
            caption_result.append(new_X)
            vid2path.append({'id': v_path, 'file_name': v_path})
            tot_samples += 1
    # Split
    train_set, test_set = train_test_split(caption_result, train_size=0.8, test_size=0.2, shuffle=True, random_state=71)
    val_set, test_set = train_test_split(test_set, train_size=0.5, test_size=0.5, shuffle=False)
    print("# of examples train: {}, val: {}, test: {}".format(len(train_set), len(val_set), len(test_set)))
            
    # Save as COCO format
    with open(os.path.join(caption_dir, 'train.caption_coco_format.json'), 'w') as f:
        json.dump({'annotations': train_set, 'images': vid2path, 'type': 'caption', 'info': 'dummy'}, f)
    with open(os.path.join(caption_dir, 'val.caption_coco_format.json'), 'w') as f:
        json.dump({'annotations': val_set, 'images': vid2path, 'type': 'caption', 'info': 'dummy'}, f)
    with open(os.path.join(caption_dir, 'test.caption_coco_format.json'), 'w') as f:
        json.dump({'annotations': test_set, 'images': vid2path, 'type': 'caption', 'info': 'dummy'}, f)


def generate_captions_knowledge(caption_dir, commentary_dir, clips_dir, anns_dir):
    # Statistics entity mentions over commentaries
    os.makedirs(caption_dir, exist_ok=True)
    # Get background
    bg_knowledge = {}
    with open(os.path.join(anns_dir, 'Whoscored.json')) as f:
        for line in f:
            bgdata = json.loads(line)
            for ftime in ('Full time:', 'Finale'):
                if ftime in bgdata['score_info']:
                    match = re.match(r'^(\d+).*?(\d+)$', bgdata['score_info'][ftime])
                    score = match.groups()
                    break
            team_h = bgdata['team_home']['name']
            team_a = bgdata['team_away']['name']
            for d in ('Date:', 'Data:'):
                if d in bgdata['score_info']:
                    match = re.match(r'.*?,\s+(.*)', bgdata['score_info'][d])
                    if match: date = match.group(1)
                    break
            try:
                date = datetime.datetime.strptime(date, "%d-%b-%y").strftime("%Y-%m-%d")
            except:
                date = ''
            k1 = date
            k2 = "{}_{}_-_{}_{}".format(team_h, score[0], score[1], team_a)
            reg_gid = ".*?({}).*?({}).*".format(k1, k2)
            bg_knowledge[reg_gid] = bgdata
    print('Total games in Whoscored data:', len(bg_knowledge))

    caption_result = []
    vid2path = []
    tot_samples = 0
    for f_comm in tqdm.tqdm(os.listdir(commentary_dir)):
        game_id = f_comm.replace(".json", "")

        # Get whoscored context
        bg_context = ""
        for bg_k in bg_knowledge:
            if re.match(bg_k, game_id):
                try:
                    bg_knowledge[bg_k]['team_home'].pop('news'); bg_knowledge[bg_k]['team_away'].pop('news')
                except:
                    print(bg_knowledge[bg_k]['team_home'])
                bg_context += str(bg_knowledge[bg_k]['team_home'])
                bg_context += str(bg_knowledge[bg_k]['team_away'])
                bg_context += str(bg_knowledge[bg_k]['history_list'])

        # Get KG context
        # kg_knowledge = pd.read_excel(os.path.join(kgs_dir, game_id, '1-hop_kgs.xlsx'), sheet_name=None, engine="openpyxl")

        if not f_comm.endswith(".json"): continue
        # Load commentaries
        with open(os.path.join(commentary_dir, f_comm), "r") as f:
            comm = json.load(f)
        
        for X in comm:
            v_path = os.path.join(clips_dir, game_id, X['clip'])

            # Get entity mention info
            ent_mentions = []
            for tag in ('team', 'player'):
                pattern = re.compile(r".*?\[{}\]\s?(.*?)\s?\[{}\].*?".format(tag,tag))
                match = pattern.match(str(X["entity"]))
                if match: ent_mentions.extend(pattern.findall(str(X["entity"])))
                
            new_X = {
                'image_id': v_path,
                'caption': X['commentary'],
                'id': tot_samples,
                'context': bg_context,
                'entities': ent_mentions
            }
            caption_result.append(new_X)
            vid2path.append({'id': v_path, 'file_name': v_path})
            tot_samples += 1
    # Split
    train_set, test_set = train_test_split(caption_result, train_size=0.8, test_size=0.2, shuffle=True, random_state=71)
    val_set, test_set = train_test_split(test_set, train_size=0.5, test_size=0.5, shuffle=False)
    print("# of examples train: {}, val: {}, test: {}".format(len(train_set), len(val_set), len(test_set)))
            
    # Save as COCO format
    path = os.path.join(caption_dir, 'knowledge')
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'train.caption_coco_format.json'), 'w') as f:
        json.dump({'annotations': train_set, 'images': vid2path, 'type': 'caption', 'info': 'dummy'}, f)
    with open(os.path.join(path, 'val.caption_coco_format.json'), 'w') as f:
        json.dump({'annotations': val_set, 'images': vid2path, 'type': 'caption', 'info': 'dummy'}, f)
    with open(os.path.join(path, 'test.caption_coco_format.json'), 'w') as f:
        json.dump({'annotations': test_set, 'images': vid2path, 'type': 'caption', 'info': 'dummy'}, f)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=str, default="/data/qiji/DATA/soccernet/videos/")
    parser.add_argument('anns_dir', type=str, default="/data/qiji/DATA/soccernet/annotations/")
    parser.add_argument('clips_dir', type=str, default="/data/qiji/DATA/soccernet/clips/")
    parser.add_argument('commentary_dir', type=str, default="/data/qiji/DATA/soccernet/commentaries/")
    parser.add_argument('caption_dir', type=str, default="/data/qiji/DATA/soccernet/captions/")
    args = parser.parse_args()


    # View commentary annotation
    f_ann = os.path.join(args.anns_dir, "AnnotationResults-20220903.xlsx")
    df_map = pd.read_excel(f_ann, sheet_name="progress", engine="openpyxl")
    df_map = pd.DataFrame(df_map)
    df_map.columns=["sheet_name", "status", "video_name"]
    name_map = {s_name:v_name for i, (s_name, v_name) in enumerate(zip(df_map["sheet_name"], df_map["video_name"]))}

    df = pd.read_excel(f_ann, sheet_name="new_19", engine="openpyxl")
    df.shape
    df.info() # videw indecies, datatype and mememory
    # df.describe() # view summary of digital columns
    # df.dtypes # view type of each field
    # df.axes # view names of row and columns
    # df.columns # view column names
    df.head()

    # Do segmentation for all videos
    segment(f_ann, args.clips_dir, args.commentary_dir, args.video_dir)
    # Generate captions of COCO format
    generate_captions(args.caption_dir, args.commentary_dir, args.clips_dir)
    # Generate captions of COCO format incorporating knowledge
    generate_captions_knowledge(args.caption_dir, args.commentary_dir, args.clips_dir, args.anns_dir)
