import numpy as np


def loss_dp(x, s, pred):
    """ Calculate the Demographic parity. """
    N = x.shape[0]

    idx_grp_0 = [i for i in range(N) if s[i] == 0]
    idx_grp_1 = [i for i in range(N) if s[i] == 1]

    pred_grp_0 = np.sum(pred[idx_grp_0])
    pred_grp_1 = np.sum(pred[idx_grp_1])
    return pred_grp_1 / len(idx_grp_1) - pred_grp_0 / len(idx_grp_0)
