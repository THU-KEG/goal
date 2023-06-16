# GOAL

Despite the recent emergence of video captioning models, how to generate vivid, fine-grained video descriptions based on the background knowledge (i.e., long and informative commentary about the domain-specific scenes with appropriate reasoning) is still far from being solved, which however has great applications such as automatic sports narrative. Based on soccer game videos and synchronized commentary data, we present GOAL, a benchmark of over $8.9$k soccer video clips, $22$k sentences, and $42$k knowledge triples for proposing a challenging new task setting as Knowledge-grounded Video Captioning (KGVC). We experimentally test existing state-of-the-art (SOTA) methods on this resource to demonstrate the future directions for improvement in this challenging task. We hope that our data resource can serve researchers and developers interested in knowledge-grounded cross-modal applications.


![task](https://cloud.tsinghua.edu.cn/f/78def9986f2b40378bc9/?dl=1)

## News !! 

* Please refer to [the paper](https://arxiv.org/abs/2303.14655) for more details.


## Data Access

The data source including raw annotations and generated captions and KGs can be download:

| Data             | Description                                           | Download Link |
| ---------------- | ----------------------------------------------------- | ------------- |
| Annotation       | The file of annotations for all game videos.          |     [Raw link](https://cloud.tsinghua.edu.cn/f/297fa6cb191a4301b01c/?dl=1)          |
| Commentaries     | The files of commentaries for all game videos.        |     [Raw link](https://cloud.tsinghua.edu.cn/f/7d5ba8b55a724b15a1e9/?dl=1)          |
| Captions    | The files captions with segmented clips for all game videos. |      [Raw link](https://cloud.tsinghua.edu.cn/f/451ccb1fd1f34ac48a2d/?dl=1)         |
| KGs    | The files stored knowledge graphs for all game videos.          |     [Raw link](https://cloud.tsinghua.edu.cn/f/300171c48bf14f17a5cc/?dl=1)          |


## Process the raw data

We provide the source code for preparing the training data.

Please download the raw video from [SoccerNet](https://www.soccer-net.org/data) according to the annotation file, and run the `run.sh` scripts according to the command lines. The code includes:

- clips generation
- captions generation
- knowledge graph generation
- data analysis


### Train the KGVC model

We provide the source code to build a KGVC model based ALPRO and a Transformer decoder. Please see the code in `models` dir for details. The code includes:

- Train a KGVC model based on pretrained ALPRO
- Specify multiple datasets for training and inference



## Reference

```bibtex
@article{qi2023goal,
  title={GOAL: A challenging knowledge-grounded video captioning benchmark for real-time soccer commentary generation},
  author={Qi, Ji and Yu, Jifan and Tu, Teng and Gao, Kunyu and Xu, Yifan and Guan, Xinyu and Wang, Xiaozhi and Dong, Yuxiao and Xu, Bin and Hou, Lei and others},
  journal={arXiv preprint arXiv:2303.14655},
  year={2023}
}
 ```
