
import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(self, d_kv, size_c, kernel, hidden_kernel,out_kernel,padding=0,cnn_dropout=0.0,is_bn=False):
        super(LeNet, self).__init__()
        self.d_kv = d_kv
        self.padding = padding
        self.kernel = kernel
        self.conv1 = nn.Conv2d(size_c, hidden_kernel, kernel, padding=padding)
        self.conv2 = nn.Conv2d(hidden_kernel, out_kernel, kernel, padding=padding)
        if is_bn:
            self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_kernel),nn.BatchNorm2d(out_kernel)])
        self.dropout = nn.Dropout(cnn_dropout)
        self.is_bn = is_bn
        self.reset_parameters()


    def reset_parameters(self):
        all_res = [self.conv1, self.conv2]
        for res in all_res:
            res.reset_parameters()
        if self.is_bn:
            for bn in self.bn:
                bn.reset_parameters()

    def cal_conv_size(self, ):
        conv_l = self.d_kv
        for _ in range(2):
            conv_l = (conv_l - self.kernel + 2 * self.padding) // 1 + 1
            # print(conv_l)
            conv_l = (conv_l - 2 + 2 * 0) // 2 + 1
            # print(conv_l)
        return conv_l

    def forward(self, x):
        for i,conv in enumerate([self.conv1,self.conv2]):
            x = conv(x)
            if self.is_bn:
                x = self.bn[i](x)
            x = self.dropout(F.relu(x))
            x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)