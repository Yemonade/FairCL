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


def get_curriculum_stages(idx_path, N=3):
    """ split the index into N shares in origin order.
    Arguments:
        idx_path(str): the index storage path
        N(int): the number of stages

    Return:

    """
    with open(idx_path) as f:
        idx = json.load(f)

    n = int(math.ceil(len(idx) / float(N)))
    return [idx[i:i + n] for i in range(0, len(idx), n)]