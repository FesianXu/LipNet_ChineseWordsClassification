# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from codes.utils import resnet
from codes.utils.utils import import_class

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class cnnfeature(nn.Module):
    def __init__(self, cnnDropout, backbone, cnnType):
        super().__init__()
        self.MetaFeatureModel = import_class(backbone)
        self.feature = self.MetaFeatureModel(cnnDropout=cnnDropout)['model']
        self.feature.fc = Identity()
        self.feature.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.cnnType = cnnType
        
    def forward(self, inputv):
        # inputv shape (N, C, T, H, W)
        cnn = inputv
        N, C, T, H, W = cnn.size()
        if self.cnnType == '2d':
            cnn = cnn.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
        cnn = self.feature(cnn)
        if self.cnnType == '3d':
            N, T, _ = cnn.size()
        cnn = cnn.view(N, T, -1)
        return cnn
    
    def get_outshape(self):
        return self.MetaFeatureModel()['output_size']
    
if __name__ == '__main__':
    model = cnnfeature().cuda()
  
    print(model)
    inputv = torch.tensor(np.random.normal(size=(4, 3, 16, 112,112))).cuda().float()
    output = model(inputv)
    print(output.shape)