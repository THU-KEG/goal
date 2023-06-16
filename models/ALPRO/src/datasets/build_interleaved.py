""" Build image-text interleaved dataset from image-text pairs.
"""
import os
import json
import random
import collections
import tqdm



def load_coco_anns(anns):
    """ Load annotations with coco format.
      Args:
        anns: a dict including fileds of `anns`, `img_dirs`.
    """
    datasets_coco_format = []
    for f_name in anns['anns']:
        with open(f_name) as f:
            datasets_coco_format.append(json.load(f))
    return datasets_coco_format


def load_vg_anns(anns):
    """ Load Visual Genome annotations and convert it to coco format.
      Args:
        anns: a dict including fileds of `anns`, `img_dirs`.
    """
    datasets_coco_format = []
    for f_name, img_dir in zip(anns['anns'], anns['img_dirs']):
        with open(f_name) as f:
            regions_cap_vg = json.load(f)
        
        rt_caps = []
        rt_imgid2path = []

        tot_samples = 0
        for multi_regs in tqdm.tqdm(regions_cap_vg, desc='reading VG annotations'):
            img_id = multi_regs['id']
            img_path = os.path.join(img_dir, '%s.jpg' % img_id)
            for reg in multi_regs['regions']:
                rt_caps.append({
                    'image_id': img_id,
                    'caption': reg['phrase'],
                    'id': tot_samples
                })
                rt_imgid2path.append({
                    'id': img_id,
                    'file_name': img_path
                })

        vg_coco_format = {
            'annotations': rt_caps,
            'images': rt_imgid2path,
            'info':'Visual Genome',
            'type':'caption'
            }
        datasets_coco_format.append(vg_coco_format)

    return datasets_coco_format



def load_cc3m_anns(anns):
    """
    """
    return []

def load_cc12m_anns(anns):
    """
    """
    return []

def load_sbu_anns(anns):
    """
    """
    return []

def load_laion_anns(anns):
    """
    """
    return []




def build_interleaved_ann(datasets, dataname, intra_size=5, out_f='interleaved.json'):
    """ Build image-text interleaved dataset with the following data structure:
        [
            [
                {
                'image': image local path,
                'caption': a sentence,
                'url': online image url,
                'dataset': dataset name,
                },
            ] * intra_size
            ...
        ]
      Args:
        datasets: a list of datasets of coco format.
        dataname: the name of datasets.
    """
    if os.path.exists(out_f):
        return
    
    # Load all images into a set
    mix_anns = []
    for data in datasets:
        vid2path = {str(img['id']):img['file_name'] for img in data['images']}
        for X in data['annotations']:
            new_X = {
                'image': vid2path[str(X['image_id'])],
                'caption': X['caption'],
                'dataset': dataname
            }
            mix_anns.append(new_X)

    rt_anns= []
    tot = len(mix_anns)
    # freqs = [0] * tot
    remains = [i for i in range(tot) for ii in range(intra_size)]
    random.shuffle(remains)
    for X in tqdm.tqdm(mix_anns, desc='building interleaved for %s' % dataname):
        # sample `intra_size` samples
        # probs = freqs / freqs.sum()
        # intra_idxs = np.random.choice(a=range(tot), size=intra_size-1, p=probs, replace=False) # random sample its context
        # freqs[intra_idxs] += 1
        intra_idxs = random.sample(remains, k=intra_size-1)
        rt_anns.append([mix_anns[idx] for idx in intra_idxs] + [X])
        for idx in intra_idxs:
            remains.pop(idx)

    print('Total interleaved cliques: %s\n Total image-caption pairs: %s\n' % (tot, tot*intra_size))

    with open(out_f, 'w') as f:
        json.dump(rt_anns, f)
    


if __name__ == '__main__':

    random.seed(71)

    out_dir = '/data/qiji/DATA/alpro/fla/'

    data_config = {
        'coco_anns':{
            'anns': ['/data/qiji/DATA/coco/annotations/captions_train2014.json', '/data/qiji/DATA/coco/annotations/captions_val2014.json'],
            'img_dirs': ['/data/qiji/DATA/coco/train2014', '/data/qiji/DATA/coco/val2014']
        },
        'vg_anns':{
            'anns': ['/data/qiji/DATA/vg/region_descriptions.json'],
            'img_dirs': ['/data/qiji/DATA/vg/VG_100K']
        },
        'cc3m_anns': {},
        'cc12m_anns': {},
        'sbu_anns': {},
        'laion_anns': {}
    }

    # datasets = {}
    for name, anns in data_config.items():
        # load to unified coco format
        datasets_coco_format = eval('load_'+name)(anns)
        # datasets[name] = datasets_coco_format # a list of multiple datasets each with coco format

        build_interleaved_ann(datasets_coco_format, name, intra_size=5, out_f=os.path.join(out_dir, name+'_interleaved.json'))



