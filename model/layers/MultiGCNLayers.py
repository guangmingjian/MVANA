

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from model.layers.DataTransform import DataTransform, feature_mask
import numpy as np

gnn_dict = {
    "GCN": gnn.GCNConv,
    "GraphSAGE": gnn.SAGEConv,
    "GAT": gnn.GATConv
}


class MultiGCNLayers(nn.Module):

    def __init__(self, d_in, d_h, d_out, sz_c, sz_l, mask_rate, drop_rate, device, g_name="GCN",
                 g_norm=False, alpha=None, is_em=False, MFeatype='SFM'):
        super().__init__()
        self.MFeatype = MFeatype
        self.mask_rate = mask_rate
        self.is_em = is_em
        self.alpha = alpha
        self.sz_c = sz_c
        d_in = d_in
        d_h = d_h
        self.d_out = d_out
        self.sz_l = sz_l
        self.drop_rate = drop_rate
        self.device = device
        if self.MFeatype == 'MFM':
            rates = [rate / 10 for rate in range(int(mask_rate * 10 + 1))]
            self.all_mask_rate = np.random.choice(rates,self.sz_c,replace=True)
        g_name = g_name
        self.gcn_layer = nn.ModuleList(self.channal_block(d_in, d_h, d_out, g_name, g_norm))
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)

        # self.all_smu = nn.ModuleList([SMULayer() for _ in range(self.sz_c)])
        # self.Dropout = nn.Dropout(self.drop_rate)

    def reset_parameters(self):
        for layer in self.gcn_layer:
            for f in layer:
                if type(f) in list(gnn_dict.values()):
                    f.reset_parameters()
                if type(f) == gnn.BatchNorm:
                    f.reset_parameters()

    def channal_block(self, d_in, d_h, d_out, g_name, g_norm):
        layer = []
        for c in range(self.sz_c):
            t_layer = []
            for l in range(self.sz_l):
                if l == 0:
                    t_layer.append(self.get_gnn(d_in, d_h, g_name))
                elif l == (self.sz_l - 1):
                    t_layer.append(self.get_gnn(d_h, d_out, g_name))
                else:
                    t_layer.append(self.get_gnn(d_h, d_h, g_name))
                t_layer.append(nn.Dropout(self.drop_rate))
                if g_norm:
                    t_layer.append(gnn.GraphSizeNorm())
                    if l == (self.sz_l - 1):
                        t_layer.append(gnn.BatchNorm(d_out))
                    else:
                        t_layer.append(gnn.BatchNorm(d_h))
                t_layer.append(nn.ReLU())
            layer.append(nn.ModuleList(t_layer))
            # layer.append(nn.Sequential(*t_layer))
        return layer
        # pass

    def get_gnn(self, d_in, d_out, g_name):
        """"""
        if g_name == 'GAT':
            return gnn.GATConv(d_in, d_out, 1)  # 1 head
        else:
            return gnn_dict[g_name](d_in, d_out)

    def fea_mask(self, x,i):
        if self.MFeatype == 'SFM':
            h = feature_mask(x, self.mask_rate, self.device)
        elif self.MFeatype == 'TFM':
            if self.training:
                # ra = random.sample(self.rates,1)[0]
                # self.DT = DataTransform(mask_rate, self.device)
                h = feature_mask(x, self.mask_rate, self.device)
            else:
                h = x
        elif self.MFeatype == "MFM":
            if self.training:
                # ra = random.sample(self.rates,1)[0]
                # self.DT = DataTransform(mask_rate, self.device)
                h = feature_mask(x, self.all_mask_rate[i], self.device)
            else:
                h = x
        else:
            raise NotImplementedError
        return h

    def forward(self, x, edge, batch, all_smu=None):
        # channal = torch.empty([self.sz_c, x.size(0), self.d_out]).to(self.device)
        smu = torch.empty([self.sz_c, x.size(0), self.d_out]).to(self.device)

        for i, layer in enumerate(self.gcn_layer):
            h = self.fea_mask(x,i)
            # h = self.DT(x)
            all_hs = []
            for f in layer:
                if type(f) in list(gnn_dict.values()):
                    h = f(h, edge)
                    all_hs.append(h)
                elif type(f) == gnn.GraphSizeNorm:
                    h = f(h, batch)
                else:
                    h = f(h)
            if all_smu is not None:
                if self.is_em:
                    h_smu = x
                else:
                    h_smu = all_hs[0]
                    all_hs = all_hs[1:]
                for ch in all_hs:
                    h_smu = all_smu[i](h_smu, ch)
                h_smu = self.alpha * h_smu + float(1.0 - self.alpha) * h
                smu[i] = h_smu
            else:
                smu[i] = h
        # channal =
        batchs = torch.ones([self.sz_c, batch.size(0)]).to(self.device) * batch
        return self.layer_norm(smu), batchs


# print(isinstance(model,MultiGCNLayers))
if __name__ == '__main__':
    model = MultiGCNLayers(10, 32, 2, 3, 3, 0.2, "cuda:0")
    print(model)
