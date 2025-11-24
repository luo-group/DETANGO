import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, ndcg_score
from scipy.stats import spearmanr
from math import pi, sqrt


def get_alphabet():
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    map_a2i = {j:i for i,j in enumerate(alphabet)}
    map_i2a = {i:j for i,j in enumerate(alphabet)}
    return alphabet, map_a2i, map_i2a


def evaluate_metrics_spearman(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)

    sr, _ = spearmanr(label, pred)

    return sr


class DetangoLoss(nn.Module):

    def __init__(self):
        super(DetangoLoss, self).__init__()

    def forward(self, esm1v, esm1v_fit, minus_ddg, minus_ddg_pred, function_plausibility, mask):

        ''' with mask (e.g., abundance) '''
        loss_p_fit = F.mse_loss(esm1v, esm1v_fit, reduction='none')
        loss_p_fit = (loss_p_fit * mask).sum() / mask.sum()
        loss_s_pred = F.mse_loss(minus_ddg, minus_ddg_pred, reduction='none')
        loss_s_pred = (loss_s_pred * mask).sum() / mask.sum()

        loss_q_reg = torch.norm(function_plausibility, p=2, dim=1).mean()

        loss = loss_p_fit + loss_s_pred + loss_q_reg 

        loss_dict = {'loss': loss.item(),
                    'loss_p_fit': loss_p_fit.item(),
                    'loss_s_pred': loss_s_pred.item(),
                    'loss_q_reg': loss_q_reg.item(),
                    }

        return loss, loss_dict


def transform_df_to_matrix(df, sequence_wt, column, shift=0, fillna=0, col_mut='mutant', flip=False):

    alphabet, map_a2i, map_i2a = get_alphabet()

    matrix = np.zeros((len(sequence_wt), 20))
    for mut, value in df[[col_mut, column]].values:
        wt, pos, mt = mut[0], int(mut[1:-1])+shift, mut[-1]
        assert sequence_wt[pos] == wt
        if np.isnan(value):
            matrix[pos, map_a2i[mt]] = fillna
        else:
            matrix[pos, map_a2i[mt]] = value

    if flip:
        matrix = 0 - matrix

    return matrix