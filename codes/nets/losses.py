import torch.nn as nn
import torch

class LipSeqLoss(nn.Module):
    def __init__(self, device, iscuda=True):
        super(LipSeqLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction='none')
        self._iscuda = iscuda
        self._device = device

    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i, ], target.squeeze(1)).unsqueeze(1))
        loss = torch.cat(loss, 1)
        
        mask = torch.zeros(loss.size(0), loss.size(1)).float()
        if self._iscuda:
            mask = mask.to(self._device)
        
        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            mask[i, L-1] = 1.0
        loss = (loss * mask).sum() / mask.sum()
        return loss