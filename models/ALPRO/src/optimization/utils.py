import torch
from torch.optim import Adam, Adamax, SGD
from src.optimization.adamw import AdamW


def setup_e2e_optimizer(model, opts):
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(model.parameters(), lr=opts.learning_rate, betas=opts.betas)

    return optimizer



def setup_e2e_grouped_optimizer(model, opts):
    ''' Assign different learning rates to different groups.
    '''
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    
    visual_backbones, textual_backbones, decoder_backbones, all_new = [], [], [], []
    for n, p in model.named_parameters():
        if n.start_with('visual_encoder') and p.requires_grad:
            visual_backbones.append(p)
        elif n.start_with('text_encoder') and p.requires_grad:
            textual_backbones.append(p)
        elif n.start_with('decoder.decoder.transformer') and 'crossattention' not in n and p.requires_grad:
            decoder_backbones.append(p)
        else:
            all_new.append(p)

    param_dicts = [
        {
            'params': visual_backbones,
            'lr': opts.lr_visual_backbone,
        },
        {
            'params': textual_backbones,
            'lr': opts.textual_backbone,
        },
        {
            'params': decoder_backbones,
            'lr': opts.lr_decoder_backbone
        },
        {
            'params': all_new,
            'lr': opts.learning_rate
        }
    ]

    optimizer = OptimCls(param_dicts, lr=opts.learning_rate, betas=opts.betas)

    return optimizer

