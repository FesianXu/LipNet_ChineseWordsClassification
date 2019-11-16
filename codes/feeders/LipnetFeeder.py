# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from configs.configs import GlobalConfig
import random
import imgaug as ia
import imgaug.augmenters as iaa

gc = GlobalConfig()

class LipnetDataset(Dataset):
    def __init__(self, tmode, length=None, data_augment=False):
        # train and eval are dividing the whole training set while the whole_training 
        # set are using the whole training set to train the model
        # this only happened when you already make sure your model can work and try
        # to tune the performance
        assert tmode in ('train', 'eval', 'test', 'whole_train')

        self._tmode = tmode
        self._length = length
        if tmode in ('train', 'eval', 'whole_train'):
            self._datapath = gc.train_img_datapath
            if tmode == 'train':
                self._dataindex = gc.train_index
            elif tmode == 'eval':
                self._dataindex = gc.eval_index
            else:
                self._dataindex = gc.whole_train_index
        else:
            self._datapath = gc.test_img_datapath
            self._dataindex = gc.test_index
        self.len_dataset = len(self._dataindex)
        self.data_augment = data_augment
        self.trans = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((gc.pic_height, gc.pic_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0, 0, 0], [1, 1, 1]) 
                            ])

        if data_augment and tmode in ('train', 'whole_train'):
            self.data_augment_transfer = iaa.Sequential([
                #  iaa.Fliplr(0.5), # horizontal flips
                #  iaa.ContrastNormalization((0.75, 1.5)),
                #  iaa.Affine(
                #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #     rotate=(-10, 10),
                #     shear=(-2, 2)
                # ),
                 iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                
            ], random_order=True)
            # data augmentation
        else:
            self.data_augment_transfer = None

    def __len__(self):
        if not self._length:
            return self.len_dataset
        return self._length
    
    def __getitem__(self, index):
        if self._tmode in ('train', 'eval', 'whole_train'):
            hashname, label = self._dataindex[index]
        else:
            hashname = self._dataindex[index]

        foldername = self._datapath+hashname+'/'
        imglen = len(os.listdir(foldername))
        files = [os.path.join(foldername, ('{}' + '.jpg').format(i)) for i in range(1, imglen+1)]
        files = list(filter(lambda path: os.path.exists(path), files))
        frames = [cv2.imread(file) for file in files ] 
        frames_ = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frames]
        length = len(frames_)
        vlm = torch.zeros((gc.nchannels, gc.seq_max_lens, gc.pic_height, gc.pic_width))
        
        if length <= gc.seq_max_lens:
            for i in range(length):
                vlm[:, i] = self.trans(frames_[i])
        else:
            # downsample
            for i in range(gc.seq_max_lens):
                vlm[:, i] = self.trans(frames_[i])
        
        if self.data_augment_transfer:
            vlm = vlm.permute(1,2,3,0).data.cpu().numpy()
            vlm = self.data_augment_transfer(images=vlm)
            vlm = torch.tensor(vlm).permute(3, 0, 1,2)

        if self._tmode in ('train', 'eval', 'whole_train'):
            return {'volume': vlm, 
                    'label': torch.LongTensor([int(label)]), 
                    'length': length,
                    'hashname': hashname}
        else:
            return {'volume':vlm,
                    'hashname':hashname,
                    'length': length}

if __name__ == '__main__':
    dataset = LipnetDataset(tmode='train', data_augment=True)
    # print(dataset[0])
    a = dataset[100]

    for each in range(len(dataset)):
        print(each)
        dataset[each]