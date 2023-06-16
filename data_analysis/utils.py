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

def flat_list_of_list(ll):
    return [l2 for l1 in ll for l2 in l1]


def display_analysis_hists(datasets, binses=None, titles=None, cols=4, save_name=None):
    
    titles = titles if titles is not None else [""]*len(datasets)
    rows = max(1, round(len(datasets) // cols))
#     plt.figure(figsize=(20,20*rows//cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20,20*rows//cols), tight_layout=True)
    if rows <= 1:
        axes = np.array([axes])
    i = 0
    # for data, title in zip(datasets, titles):
    for idx, data in enumerate(datasets):
        bins = binses[idx] if binses is not None else 10
        title = titles[idx] if titles is not None else None
        
        axes[i//cols, i%cols].set_title(title, fontsize=9)
        axes[i//cols, i%cols].xaxis.set_minor_locator(MultipleLocator(1))
        # freq, bins, patches = axes[i//cols, i%cols].hist(data, bins=max(data), rwidth=0.5)
        freq, bins, patches = axes[i//cols, i%cols].hist(data, bins=bins, rwidth=0.5)
        
        # x coordinate for labels
        bin_centers = np.diff(bins)*0.5 + bins[:-1]

        # n = 0
        # for fr, x, patch in zip(freq, bin_centers, patches):
        #     height = int(freq[n])
        #     axes[i//cols, i%cols].annotate("{}".format(height),
        #                xy = (x, height),             # top left corner of the histogram bar
        #                xytext = (0,0.2),             # offsetting label position above its bar
        #                textcoords = "offset points", # Offset (in points) from the *xy* value
        #                ha = 'center', va = 'bottom'
        #                )
        #     n += 1
    
        # annotate mean/min/max values
        _mean = round(np.mean(data),3)
        _min = round(np.min(data),3)
        _max = round(np.max(data),3)
        text = "mean: {}\n min: {}\n max: {}".format(_mean, _min, _max)
        # axis_bbox = axes[i//cols, i%cols].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        axes[i//cols, i%cols].annotate(text,
                    xy=(bin_centers[-1]*0.75, np.max(freq)*0.8),
                    # xy=(bin_centers[-1]-50, np.max(freq)/2),
                    # xy=(x-2, 4000),
                    # xytext=(0, 30),  # 3 points vertical offset
                    # textcoords="offset points",
                    bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
                    ha='left',
                    va='bottom',
                    color='red',
                    fontsize=10,
                    transform=axes[i//cols, i%cols].transAxes)
        # ticks = [patch.xy[0]+patch.get_width()/2 for patch in patches]
        # axes[i//cols, i%cols].set_xticks(ticks)
        i += 1
    if save_name:
        fig.savefig(save_name, dpi=200)
    plt.show()