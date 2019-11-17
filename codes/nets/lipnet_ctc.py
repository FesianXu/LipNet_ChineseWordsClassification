# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from codes.nets.cnnFeatures import cnnfeature

class LipNet(torch.nn.Module):
    def __init__(self,
                 backbone,
                 word_class=313,
                 char_class=431,
                 blank_char_class=432,
                 channel=3,
                 attention=False,
                 cnnDropout=0.4,
                 gruDropout=0.5,
                 fcDropout=0.5,
                 cnnType='2d'):
        super(LipNet, self).__init__()
        # Cnn
        # conv3d kernel
        self.feature3d = cnnfeature(cnnDropout=cnnDropout, backbone=backbone, cnnType=cnnType)
        # Rnn
        self.gru1 = nn.GRU(self.feature3d.get_outshape(), 256, bidirectional=True, batch_first=True,dropout=gruDropout, num_layers=2) 
        
        # Fc
        self.fc_char = nn.Sequential(
            nn.Dropout(fcDropout),
            nn.Linear(512, blank_char_class))
        
        self.fc_word = nn.Sequential(
            nn.Dropout(fcDropout),
            nn.Linear(512, word_class)
        )

        self.reg_length = nn.Sequential(
            nn.Dropout(fcDropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        self.gru1.flatten_parameters()
        N, C, T, H, W = x.size()
        cnn = self.feature3d(x)
        rnn, _ = self.gru1(cnn)

        logit_word_ = self.fc_word(rnn)
        logit_word = torch.mean(logit_word_, dim=1)
        
        logit_char = self.fc_char(rnn)
        logit_char = logit_char.permute(1,0,2)

        reg_length = self.reg_length(rnn) * 4
        reg_length = torch.mean(reg_length, dim=1).squeeze(-1)
        return logit_char, logit_word, logit_word_.log_softmax(-1), reg_length

    
if __name__ == '__main__':
    model = LipNet().cuda()
    inputv = torch.rand(size=(4,3,24,60,120)).cuda()
    inputv2 = torch.zeros(size=(2, 3, 24, 60, 120)).cuda()
    inputv = torch.cat((inputv, inputv2), dim=0)
    out = model(inputv)
    # print(out[0].shape)
    