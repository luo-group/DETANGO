import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1, activation=nn.GELU):
    layers = []
    in_dim = input_dim
    for i in range(num_layers):
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_dim)
        ])
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class DetangoModel(nn.Module):
    def __init__(self,
                 args,):
        super(DetangoModel, self).__init__()

        emb_dim = 1280
        vocab_size = 20

        lm_hidden_dim = args.lm_hidden_dim
        ss_hidden_dim = args.ss_hidden_dim
        proj_hidden_dim = args.proj_hidden_dim
        dropout = args.dropout
        lm_layers = args.lm_layers
        ss_layers = args.ss_layers
        proj_layers = args.proj_layers

        self.LM_q = build_mlp(emb_dim, lm_hidden_dim, vocab_size, num_layers=lm_layers, dropout=dropout)

        self.LM_s = build_mlp(emb_dim, lm_hidden_dim, vocab_size, num_layers=lm_layers, dropout=dropout)

        self.LM_ss = build_mlp(1, ss_hidden_dim, 1, num_layers=ss_layers, dropout=dropout)

        self.project_s = build_mlp(emb_dim, proj_hidden_dim, emb_dim, num_layers=proj_layers, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.LM_q+self.LM_s+self.LM_ss+self.project_s:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, repr, pos, minus_ddg):

        '''projection'''
        pos = pos.reshape(-1)
        
        emb_s = self.project_s(repr)
        emb_q = repr - emb_s
        
        q = self.LM_q(emb_q.detach())
        s = self.LM_s(emb_s)

        ss = self.LM_ss(minus_ddg.unsqueeze(-1)).squeeze(-1)

        return s, q, ss