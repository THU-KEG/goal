"""
optimizer learning rate scheduling helpers
"""
import torch
from math import ceil
from collections import Counter
import math


def noam_schedule(step, warmup_step=4000):
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def multi_step_schedule(n_epoch, milestones, gamma=0.5):
    milestones = list(sorted(milestones))
    for i, m in enumerate(milestones):
        if n_epoch < m:
            return gamma**i
    return gamma**(len(milestones)+1)


def get_lr_sched(global_step, decay, learning_rate,
                 num_train_steps, warmup_ratio=0.1,
                 decay_epochs=[], multi_step_epoch=-1):
    warmup_steps = int(warmup_ratio*num_train_steps)
    if decay == 'linear':
        lr_this_step = learning_rate * warmup_linear(
            global_step, warmup_steps, num_train_steps)
    elif decay == 'invsqrt':
        lr_this_step = learning_rate * noam_schedule(
            global_step, warmup_steps)
    elif decay == 'constant':
        lr_this_step = learning_rate
    elif decay == "multi_step":
        assert multi_step_epoch >= 0
        lr_this_step = learning_rate * multi_step_schedule(
            multi_step_epoch, decay_epochs)
    if lr_this_step <= 0:
        # save guard for possible miscalculation of train steps
        lr_this_step = 1e-8
    return lr_this_step


class MultiStepWarmupLR:
    def __init__(self, decay_rate=0.1, lr_milestones=[20000, 40000], warm_up_steps=5000, min_decay_rate=0.01) -> None:
        self.deacy_rate = decay_rate
        self.lr_milestones = lr_milestones
        self.warm_up_steps = warm_up_steps
        self.min_decay_rate = min_decay_rate

    def __call__(self, steps):
        if steps < self.warm_up_steps:
            rate = (steps+1)/self.warm_up_steps
        else:
            rate = self.deacy_rate ** len([m for m in self.lr_milestones if m <= steps])
        # make sure lr is not too small
        if rate <= self.min_decay_rate:
            return self.min_decay_rate
        else:
            return rate
        
class CosineWarmupLR:
    def __init__(self, max_T=100, warm_up_steps=5, min_decay_rate=0.01) -> None:
        self.max_T = max_T
        self.warm_up_steps = warm_up_steps
        self.min_decay_rate = min_decay_rate

    def __call__(self, steps):
        if steps < self.warm_up_steps:
            rate = (steps+1)/self.warm_up_steps
        else:
            rate = 0.5 * (math.cos((steps - self.warm_up_steps) / (self.max_T - self.warm_up_steps) * math.pi) + 1)
        # make sure lr is not too small
        if rate <= self.min_decay_rate:
            return self.min_decay_rate
        else:
            return rate

class LinearStepWarmupLR:
    def __init__(self, warmup_step, tot_step) -> None:
        self.warmup_step = warmup_step
        self.tot_step = tot_step
    
    def __call__(self, steps):
        if steps < self.warmup_step:
            return steps / self.warmup_step
        return max(0, (self.tot_step-steps)/(self.tot_step-self.warmup_step))


def setup_lr_scheduler(optimizer,
                       lr_schedule,
                       num_train_steps,
                       steplr_step=1000,
                       warmup_ratio=0.1,
                       msteplr_milestones=[10000, 20000],
                       msteplr_decay=0.1):
    ''' Get the learning rate scheduler to match the optimizer with grouped parameters.
    '''
    if lr_schedule == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, steplr_step)
    elif lr_schedule == 'LinearStepWarmupLR':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=LinearStepWarmupLR(
                warmup_step=num_train_steps*warmup_ratio,
                tot_step=num_train_steps,
            )
        )
    elif lr_schedule == 'MultiStepWarmupLR':        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=MultiStepWarmupLR(
                decay_rate=msteplr_decay,
                lr_milestones=msteplr_milestones,
                warm_up_steps=num_train_steps*warmup_ratio
            )
        )
    elif lr_schedule == 'CosineWarmupLR':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=CosineWarmupLR(
                max_T=num_train_steps,
                warm_up_steps=num_train_steps*warmup_ratio
            )
        )

    return lr_scheduler