
import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict

class Naive3DCNN(nn.Module):
    def __init__(self, cnnDropout=0.5):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(3, 32, kernel_size=(3, 5,5), stride=(1, 2,2), padding=(1,2,2))),
            ('norm', nn.BatchNorm3d(32)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
            ]))
        
        self.features2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(32, 64, kernel_size=(3, 5,5), stride=(1, 1, 1), padding=(1,2,2))),
            ('norm', nn.BatchNorm3d(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
        ]))

        self.features3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1))),
            ('norm', nn.BatchNorm3d(96)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
        ]))

    def forward(self, inputv):
        cnn = self.features(inputv)
        cnn = self.features2(cnn)
        cnn = self.features3(cnn)   
        cnn = cnn.permute(0, 2, 1, 3, 4).contiguous()
        batch, seq, channel, height, width = cnn.size()
        cnn = cnn.view(batch, seq, -1)
        return cnn

    @staticmethod
    def get_outshape():
        return 96*3*7


class ST_splitted_CNN(nn.Module):
    def __init__(self, cnnDropout=0.5):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('s_conv', nn.Conv3d(3, 32, kernel_size=(3, 3,3), stride=(1, 1,1), padding=(1,1,1))),
            ('norm', nn.BatchNorm3d(32)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
            ]))
        
        self.features2 = nn.Sequential(OrderedDict([
            ('s_conv', nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))),
            ('norm', nn.BatchNorm3d(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(p=cnnDropout)),
            ('t_conv', nn.Conv3d(64, 64, kernel_size=(3, 3,3), stride=(1,1, 1), padding=(1,1,1))),
            ('norm', nn.BatchNorm3d(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
        ]))

        self.features3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1))),
            ('norm', nn.BatchNorm3d(96)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
        ]))

        self.features4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(96, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1))),
            ('norm', nn.BatchNorm3d(128)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
        ]))

    def forward(self, inputv):
        cnn = self.features(inputv)
        cnn = self.features2(cnn)
        cnn = self.features3(cnn)  
        cnn = self.features4(cnn) 
        cnn = cnn.permute(0, 2, 1, 3, 4).contiguous()
        batch, seq, channel, height, width = cnn.size()
        cnn = cnn.view(batch, seq, -1)
        return cnn

    @staticmethod
    def get_outshape():
        return 2688

class ShringkedNaiveCNN(nn.Module):
    def __init__(self, cnnDropout=0.5):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(3, 32, kernel_size=(3, 5,5), stride=(1, 2,2), padding=(1,2,2))),
            ('norm', nn.BatchNorm3d(32)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
            ]))
        
        self.features2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(32, 64, kernel_size=(3, 5,5), stride=(1, 1, 1), padding=(1,2,2))),
            ('norm', nn.BatchNorm3d(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
        ]))

        self.features3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1))),
            ('norm', nn.BatchNorm3d(96)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('dropout', nn.Dropout(p=cnnDropout))
        ]))

    def forward(self, inputv):
        cnn = self.features(inputv)
        cnn = self.features2(cnn)
        cnn = self.features3(cnn)   
        cnn = cnn.permute(0, 2, 1, 3, 4).contiguous()
        batch, seq, channel, height, width = cnn.size()
        cnn = cnn.view(batch, seq, -1)
        return cnn

    @staticmethod
    def get_outshape():
        return 96*3*7



def naive_3dcnn(**kwargs):
    model = Naive3DCNN(**kwargs)
    return {
        'model': model, 
        'output_size': Naive3DCNN.get_outshape()
    }

def st_splitted_cnn(**kwargs):
    model = ST_splitted_CNN(**kwargs)
    return {
        'model': model, 
        'output_size': ST_splitted_CNN.get_outshape()
    }

def shrinked_naive_cnn(**kwargs):
    Model = ShringkedNaiveCNN
    model = Model(**kwargs)
    return {
        'model': model, 
        'output_size': Model.get_outshape()
    }


if __name__ == '__main__':
    model = ShringkedNaiveCNN(cnnDropout=0.3)
    inputv = torch.rand(size=(8, 3, 24, 60, 120))
    out = model(inputv)
    print(out.shape)