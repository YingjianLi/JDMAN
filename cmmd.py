#!/usr/bin/env python
# encoding: utf-8


import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
min_var_est = 1e-8


class CMMD_loss(nn.Module):
    def __init__(self):
        super(CMMD_loss, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0]) # number of all samples, including source and target
        total = torch.cat([source, target], dim=0)  # connect them, from m*d, n*d to (m+n)*d
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)

    def forward(self, source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size_s = int(source.size()[0])
        s_label = s_label.cpu()
        s_label = s_label.view(batch_size_s,1) # batchsize = 64, 32 source and 32 target, 31 is the class number?
        s_label = torch.zeros(batch_size_s, 7).scatter_(1, s_label.data, 1)
        s_label = Variable(s_label).cuda()
        batch_size_t = int(target.size()[0])
        t_label = t_label.cpu()
        t_label = t_label.view(batch_size_t, 1)
        t_label = torch.zeros(batch_size_t, 7).scatter_(1, t_label.data, 1)
        t_label = Variable(t_label).cuda()

        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = 0
        XX = kernels[:batch_size_s, :batch_size_s] # kernel 是个对称阵
        YY = kernels[batch_size_s:, batch_size_s:]
        XY = kernels[:batch_size_s, batch_size_s:]
        # print(str(kernels))
        # print(s_label.size(),t_label.size())
        # print(XX.size(),YY.size(),XY.size())
        loss += torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX) \
                + torch.mean(torch.mm(t_label, torch.transpose(t_label, 0, 1)) * YY) \
                - torch.mean(2 * torch.mm(s_label, torch.transpose(t_label, 0, 1)) * XY)  # 非方阵
        return loss



