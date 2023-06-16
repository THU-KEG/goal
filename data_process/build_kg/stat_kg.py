import os, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import json
import tqdm
import argparse


# Plot distribution of relation types
def plot_distribution_bars(bar_values, bar_labels, x_label=None, y_label=None, title=None, color=None, save_name=None):
    """
      Args:
    """
    
    fig, ax = plt.subplots(figsize=(24, 8), dpi=200)
    
    rects = ax.bar(bar_labels, bar_values, label=bar_labels, color=color)
    # ax.bar_label(rects, padding=3)
        
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=15)
    ax.set_xticklabels(bar_labels, rotation=30, ha='right', fontsize=15)
    # ax.legend()
    
    plt.yticks(fontsize=15)
    fig.tight_layout()
    plt.show()

    if save_name:
      fig.savefig(save_name)



# The class for satistics
class Stat():
    def __init__(self,):
        self.player_ments_videos = [] # the number of mentions of player type per video
        self.team_ments_videos = []
        self.ments_sents = [] # the number of mentions of player/team type per sentence
        self.rels_types_videos = [] # the number of relations of any type per video
        self.rels_types_sents = [] # the number of relations of any type per sentence
        self.tris_videos = [] # the number of triples per video
        self.tris_sents = [] # the number of triples per sentence
        self.rels_types = collections.Counter() # the number of relation types
        self.total_tris = 0



def stat_kgs(commentary_dir, kgs_dir, plot=False):
    # Statistics entity mentions over commentaries
    stat = Stat()
    for f_comm in tqdm.tqdm(os.listdir(commentary_dir)):
        if not f_comm.endswith(".json"):
            continue
        
        # Load commentaries
        with open(os.path.join(commentary_dir, f_comm), "r") as f:
            comm = json.load(f)
        game_id = f_comm.replace(".json", "")
        
        # Load linked info to wikipedia for all mentions
        with open(os.path.join(kgs_dir, game_id, "linked_entities.json")) as f:
            wikipedia_entities = json.load(f)
            mention_to_eid = {men_name: wikipedia_entities[men_name][-1] for men_name in wikipedia_entities}
            eid_to_mention = {wikipedia_entities[men_name][-1]: men_name for men_name in wikipedia_entities}
            
        # Load kgs of wikidata for all entities
        eid_to_kgs = pd.read_excel(os.path.join(kgs_dir, game_id, "1-hop_kgs.xlsx"), sheet_name=None, engine="openpyxl")
        distinct_rels_video = set()
        for eid in eid_to_kgs:
            cur_kg = eid_to_kgs[eid]
            stat.rels_types.update(cur_kg["relation_label"].values.tolist())
            distinct_rels_video.update(cur_kg["relation_label"].values.tolist())
        stat.rels_types_videos.append(len(distinct_rels_video))
            
        n_teams_video, n_players_video, n_rels_video, n_tris_video = 0, 0, 0, 0
        for X in comm:
            # statistic sentence info
            men_teams = re.findall(r".*?\[team\]\s?(.*?)\s?\[team\].*?", str(X["entity"]))
            men_players = re.findall(r".*?\[player\]\s?(.*?)\s?\[player\].*?", str(X["entity"]))
            
            n_teams_video += len(men_teams)
            n_players_video += len(men_players)
            
            n_men_sent, n_tri_sent = 0, 0
            distinct_rels_sent = set()
            for men_name in men_teams + men_players:
                if men_name == "":
                    continue
                n_men_sent += 1
                if men_name in mention_to_eid:
                    if mention_to_eid[men_name] in eid_to_kgs:
                        cur_kg = eid_to_kgs[mention_to_eid[men_name]]
                        distinct_rels_sent.update(cur_kg["relation_label"].values.tolist())
                        n_tri_sent += len(cur_kg)
                        
                        
            stat.ments_sents.append(n_men_sent)
            stat.rels_types_sents.append(len(distinct_rels_sent))
            stat.tris_sents.append(n_tri_sent)
            
            # n_rels_video += n_rel_sent
            n_tris_video += n_tri_sent
        
        stat.player_ments_videos.append(n_players_video)
        stat.team_ments_videos.append(n_teams_video)
        stat.tris_videos.append(n_tris_video)

    print("Average number of player-mentions per video: %.2f" % np.mean(stat.player_ments_videos))
    print("Average number of team-mentions per video: %.2f" % np.mean(stat.team_ments_videos))
    print("Average number of relation types per video: %.2f" % np.mean(stat.rels_types_videos))
    print("Average number of triples per video: %.2f" % np.mean(stat.tris_videos))
    print("Total number of relation types: %.2f" % len(stat.rels_types))
    print("Total number of triples: %.2f" % sum(stat.tris_videos))

    if plot:
        relations = sorted(stat.rels_types.items(), key=lambda x: x[1], reverse=True)
        # relations = relations[1:51] # trunk
        relations = relations[1:51] # trunk
        rel_labels, rel_count = [r[0] for r in relations], [r[1] for r in relations]
        plot_distribution_bars(rel_count, rel_labels, color="#5DADE2", save_name="imgs/dist_relations.pdf")




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('commentary_dir', type=str, default="/data/qiji/DATA/soccernet/commentaries/")
    parser.add_argument('kgs_dir', type=str, default="/data/qiji/DATA/soccernet/kgs/")
    args = parser.parse_args()

    stat_kgs(args.commentary_dir, args.kgs_dir, plot=True)