import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from Bio import SeqIO
from tqdm import tqdm
import pickle

from sklearn.preprocessing import PowerTransformer
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from detango.utils import get_alphabet


class DetangoDataset(Dataset):
    def __init__(self, protein, sequence_wt, stability_col):

        alphabet, map_a2i, map_i2a = get_alphabet()

        self.alphabet = alphabet
        self.protein = protein
        self.sequence_wt = sequence_wt
        self.seq_len = len(self.sequence_wt)
        self.stability_col = stability_col

        df = pd.read_csv(f'data/{protein}/collection.csv', index_col=0)
        self.df = df

        matrix_esm1v = torch.zeros((len(self.sequence_wt), len(alphabet)))
        matrix_stability = torch.zeros((len(self.sequence_wt), len(alphabet)))
        matrix_mask = torch.ones((len(self.sequence_wt), len(alphabet)))
        for mut, esm1v, stability in self.df[['mutant', 'esm1v', self.stability_col]].values:
            wt, pos, mut = mut[0], int(mut[1:-1]), mut[-1]
            assert self.sequence_wt[pos] == wt
            matrix_esm1v[pos, map_a2i[mut]] = esm1v

            if pd.isna(stability):
                matrix_mask[pos, map_a2i[mut]] = 0
            else:
                matrix_stability[pos, map_a2i[mut]] = stability
        
        # mask the wt amino acids if abundance data is used
        if self.stability_col in ['abundance']:
            for pos, aa in enumerate(self.sequence_wt):
                matrix_mask[pos, map_a2i[aa]] = 0

        self.matrix_esm1v = matrix_esm1v
        self.matrix_stability = matrix_stability
        self.matrix_mask = matrix_mask

        self.representations = torch.load(f'data/{protein}/{protein}.pt')['representations'][33]
        assert self.seq_len == self.representations.shape[0]

        self.positions = torch.arange(self.seq_len).reshape(-1, 1)
        self.wt_AAs = torch.tensor([map_a2i[aa] for aa in self.sequence_wt])

    def __getitem__(self, idx):
        return [self.representations[idx], self.positions[idx], self.wt_AAs[idx], self.matrix_esm1v[idx], self.matrix_stability[idx], self.matrix_mask[idx]]

    def __len__(self):
        return self.seq_len