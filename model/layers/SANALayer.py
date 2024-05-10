
import torch
from torch import nn
from utils import tools
import torch.nn.functional as F


class SANA(nn.Module):

    def __init__(self, h_dim, d_in, att_layers, dropout, temperature=1.0):
        super(SANA, self).__init__()
        """"""
        self.temperature = temperature ** 2
        self.lins = []
        for i in range(att_layers):
            if i == 0:
                self.lins.append(torch.nn.Linear(2 * d_in, h_dim))
            else:
                self.lins.append(torch.nn.Linear(h_dim, h_dim))

        self.trans_lin = torch.nn.ModuleList(self.lins)
        self.att_lin = torch.nn.Linear(h_dim + d_in, 1)
        # print(layers)
        self.dropout = nn.Dropout(dropout)
        # self.bn = nn.BatchNorm1d(d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.bn = BatchNorm(d_in)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.att_lin]:
            m.reset_parameters()
        for m in self.trans_lin:
            m.reset_parameters()

    def forward(self, h_l, s_l):
        # h_l,s_l = self.bn(h_l),self.bn(s_l)
        # h_sum = h_l + s_l
        # h1,h2 = h_l / h_sum, s_l/h_sum
        h1, h2 = h_l, s_l
        z = torch.cat([h1, h2], dim=-1)
        # h1,h2 = self.trans_lin(h_l),self.trans_lin(s_l)
        for m in self.trans_lin:
            z = F.relu(self.dropout(m(z)))
        al1 = self.att_lin(torch.cat([z, h1], dim=-1))
        al2 = self.att_lin(torch.cat([z, h2], dim=-1))
        att = F.softmax(torch.cat([al1 / self.temperature + 1e-8, al2 / self.temperature + 1e-8], dim=1), dim=1)
        # print(att)
        return self.layer_norm((att[:, 0] * h1.T + att[:, 1] * h2.T).T)



