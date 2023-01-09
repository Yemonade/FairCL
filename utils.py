import json
import math
import random

import numpy as np
import torch


def set_seed(seed):
    """ set the random seed """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunks(arr, m):
    """ split the arr into N chunks. """

    return [arr[i:i + n] for i in range(0, len(arr), n)]


def get_curriculum_stages(y, s, idx_path, N=2):
    """ split the index into N shares in origin order.
    Arguments:
        idx_path(str): the index storage path
        N(int): the number of stages

    Return:

    """
    with open(idx_path) as f:
        idx = json.load(f)
    idx = np.array(idx)
    y_sorted = y[idx]
    s_sorted = s[idx]

    classes = np.unique(y)
    s_feat = np.unique(s)

    # N = 2 时，就是每个 subgroup 按顺序取一半，放在一起
    # step1: get the subgroup: {(g0, y0): [...]}
    all_grp = {}
    # step2: calculate each group size in one stage
    grp_size = {}
    # y
    for g in s_feat:
        for label in classes:
            # order messy?
            sub_grp = idx[(s_sorted == g) & (y_sorted == label)]
            all_grp[(g, label)] = sub_grp
            grp_size[(g, label)] = len(sub_grp) / N

    res = []
    for i in range(N):
        tmp = []
        print('=' * 20)
        for g, label in all_grp.keys():
            size = int(grp_size[(g, label)])
            sub_grp = all_grp[(g, label)]
            j = i * size
            print('g:%d, y:%d ==> size:%d' % (g, label, size))
            tmp.append(sub_grp[j: j + size])
        res.append(np.concatenate(tmp))

    return res


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, root_path='checkpoint/', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.root_path = root_path
        self.trace_func = trace_func

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        file_path = self.root_path + model.scope_name + str(model.batch_id) + '.pt'
        if model.debias:
            save_dict = {
                'model_clf_model_state_dict': model.clf_model.state_dict(),
                'model_clf_opt_state_dict': model.classifier_opt.state_dict(),
                'model_clf_lr_scheduler_state_dict': model.clf_lr_scheduler.state_dict(),
                'model_adv_model_state_dict': model.adv_model.state_dict(),
                'model_adversary_opt_state_dict': model.adversary_opt.state_dict(),
                'model_adv_lr_scheduler': model.adv_lr_scheduler.state_dict()
            }
        else:
            save_dict = {
                'model_clf_model_state_dict': model.clf_model.state_dict(),
                'model_clf_opt_state_dict': model.classifier_opt.state_dict(),
                'model_clf_lr_scheduler_state_dict': model.clf_lr_scheduler.state_dict()
            }
        torch.save(save_dict, file_path)
        self.val_loss_min = val_loss
