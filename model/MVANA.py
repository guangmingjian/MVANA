
import torch.nn.functional as F
from torch import nn
import torch
from model.layers.LeNet import LeNet
from model.layers.MultiGCNLayers import MultiGCNLayers
from model.layers.SANALayer import SANA
from model.layers.VLRLayers import VLRLayers
from utils import tools
from model.layers.SimilarMeasure import SimilarMeasure


class MVANA(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels, gcn_nums, dropout, g_name,
                 sz_c, graph_norm, gcn_h_dim, alpha, device, beta,
                 d_kv, att_drop_ratio, gcn_dropout=0.0, mask_rate=0.0, cnn_out_ker=8, cnn_dropout=0.0,
                 hidden_kernel=8, is_cbn=False, is_em=False, MFeatype='SFM'):
        super(MVANA, self).__init__()
        """"""
        # *******************************def params******************************
        self.device = device
        # if g_name != "GAT":
        gcn_h_dim = gcn_h_dim // sz_c
        self.alpha = alpha
        self.beta = beta
        self.sz_c = sz_c
        # self.gcn_out = net_params["gcn_out"]

        # *******************************def models******************************
        self.is_em = is_em
        if is_em:
            self.fea_embed = nn.Sequential(nn.Linear(in_channels, gcn_h_dim))
            # self.fea_embed = nn.Sequential(nn.Linear(in_channels, gcn_h_dim), nn.Linear(gcn_h_dim, gcn_h_dim))
            in_channels = gcn_h_dim
        self.mgl = MultiGCNLayers(in_channels, gcn_h_dim, gcn_h_dim, sz_c, gcn_nums, mask_rate=mask_rate,
                                  drop_rate=gcn_dropout, device=self.device, g_name=g_name, g_norm=graph_norm,
                                  alpha=self.alpha,
                                  MFeatype=MFeatype)
        self.loss1 = SimilarMeasure("mean")
        if self.alpha == 0:
            self.sana = None
        else:
            self.smus = nn.ModuleList(
                [SANA(gcn_h_dim, gcn_h_dim, 2, att_drop_ratio, temperature=1.0) for _ in range(sz_c)])
        self.vlr = VLRLayers(d_kv, self.device, gcn_h_dim, att_drop_ratio)
        kernal = 5
        if d_kv == 32:
            padding = 0
        else:
            padding = 1
        self.cnn_net = self.cnn_net = LeNet(d_kv, self.sz_c, kernal, hidden_kernel, cnn_out_ker, padding, cnn_dropout,
                                            is_cbn)
        cnn_size = self.cnn_net.cal_conv_size()
        cnn_out_dim = cnn_out_ker * cnn_size * cnn_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(cnn_out_dim, gcn_h_dim)
        self.fc2 = nn.Linear(gcn_h_dim, out_channels)

    def reset_parameters(self):
        if self.is_em:
            self.fea_embed.apply(tools.weight_reset)
        if self.smus is not None:
            for m in self.smus:
                m.reset_parameters()
        all_res = [self.cnn_net, self.vlr, self.mgl, self.fc1, self.fc2]
        for res in all_res:
            if res != None:
                res.reset_parameters()

    def forward(self, data, edge_index, batch):
        # *******************************feature embedding******************************
        if self.is_em:
            data = self.fea_embed(data)
        # *******************************multi-channel encoder******************************
        z, batches = self.mgl(data, edge_index, batch, self.smus)
        self.div_loss = self.loss1.correlation_loss(z)
        # *******************************CNN Decoder******************************
        batch_data = self.vlr(z, batch)  # VLR
        z = self.cnn_net(batch_data)  # CPB
        z = F.relu(self.dropout(self.fc1(z)))
        z = self.fc2(z)
        return z

    def loss(self, y_pre, y_true):
        l1 = F.cross_entropy(y_pre, y_true)
        return l1 + self.beta * self.div_loss
        # return self.alpha * self.div_loss + F.nll_loss(y_pre, y_true)
