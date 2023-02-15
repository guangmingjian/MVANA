
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class VLRLayers(nn.Module):

    def __init__(self, d_kv, device, gcn_h_dim, att_drop_ratio=0.5):
        super().__init__()
        self.d_kv = d_kv
        self.k_fc = nn.Linear(gcn_h_dim, d_kv)
        self.v_fc = nn.Linear(gcn_h_dim, d_kv)
        self.device = device
        self.dropout = nn.Dropout(att_drop_ratio)
        self.layer_norm = nn.LayerNorm(d_kv, eps=1e-6)

    def reset_parameters(self):
        self.k_fc.reset_parameters()
        self.v_fc.reset_parameters()

    def forward(self, gnn_out, batch):
        sz_c = gnn_out.size(0)
        sz_b = len(th.unique(batch))
        batch_data = th.empty([sz_b, sz_c, self.d_kv, self.d_kv]).to(self.device)
        for cha in range(sz_c):
            # sz_b x n x dh
            batch_cha, mask = to_dense_batch(gnn_out[cha], batch)
            K = self.k_fc(batch_cha)
            att = self.dropout(F.softmax(self.v_fc(batch_cha), dim=-2))
            new_em = th.matmul(K.transpose(1, 2), att)
            batch_data[:, cha, :, :] = new_em
        return self.layer_norm(batch_data)
