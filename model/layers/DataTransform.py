

from torch import nn
import torch.nn.functional as F
import torch


def feature_mask(x,mask_rate,device):
    drop_rates = torch.ones(x.size(0)) * mask_rate
    masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(device)
    x = masks * x
    return x

class DataTransform(nn.Module):
    """"""
    # trans_type FeatureMask DropNode DropEdge

    def __init__(self, mask_rate, device):
        super(DataTransform, self).__init__()
        """"""
        self.device = device
        self.mask_rate = mask_rate

    def feature_mask(self,x):
        drop_rates = torch.ones(x.size(0)) * self.mask_rate
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(self.device)
        x = masks * x
        return x

    def forward(self, x):
        """"""
        return self.feature_mask(x)
