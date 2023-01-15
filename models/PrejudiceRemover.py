import scipy.special
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Prejudice Index implementation
def pr_loss(output_a, output_b, eta):
    N_a = output_a.shape[0]
    N_b = output_b.shape[0]
    Dxisi = torch.Tensor((N_b, N_a))  # aa sample, #cc sample
    # Pr[y|s]
    y_pred_a = torch.sum(output_a)
    y_pred_b = torch.sum(output_b)
    P_ys = torch.stack((y_pred_b, y_pred_a), dim=0) / Dxisi
    # print("P_ys: ", P_ys)
    P = torch.cat((output_a, output_b), dim=0)
    P_y = torch.sum(P) / (output_a.shape[0] + output_b.shape[0])
    # print("P_y", P_y)
    P_s1y1 = torch.log(P_ys[1] + 1e-8) - torch.log(P_y + 1e-8)
    P_s1y0 = torch.log(1 - P_ys[1] + 1e-8) - torch.log(1 - P_y + 1e-8)
    P_s0y1 = torch.log(P_ys[0] + 1e-8) - torch.log(P_y + 1e-8)
    P_s0y0 = torch.log(1 - P_ys[0] + 1e-8) - torch.log(1 - P_y + 1e-8)

    P_s1y1 = torch.flatten(P_s1y1)
    P_s1y0 = torch.flatten(P_s1y0)
    P_s0y1 = torch.flatten(P_s0y1)
    P_s0y0 = torch.flatten(P_s0y0)

    # PI
    PI_s1y1 = output_a * P_s1y1
    PI_s1y0 = (1 - output_a) * P_s1y0
    PI_s0y1 = output_b * P_s0y1
    PI_s0y0 = (1 - output_b) * P_s0y0
    # print("PI_s0y0: ", PI_s1y1, PI_s1y0, PI_s0y1, PI_s0y0)
    PI = torch.sum(PI_s1y1) + torch.sum(PI_s1y0) + torch.sum(PI_s0y1) + torch.sum(PI_s0y0)
    PI = eta * PI
    return PI


class classifier_model(nn.Module):
    def __init__(self, feature, Hneuron1, output, dropout):
        super(classifier_model, self).__init__()
        self.feature = feature
        self.hN1 = Hneuron1
        self.output = output
        self.dropout = dropout
        self.FC1 = nn.Linear(self.feature, self.hN1)
        self.FC2 = nn.Linear(self.hN1, self.output)
        self.sigmoid = torch.sigmoid
        self.relu = F.relu
        self.Dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        # print("FC1.weight: ", torch.isnan(self.FC1.weight.data).any())
        x = self.Dropout(self.relu(self.FC1(x)))
        x_logits = self.FC2(x)
        x_pred = self.sigmoid(x_logits)
        return x_pred, x_logits



