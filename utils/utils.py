#---->read yaml
import yaml
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
    
class Score_pooling(nn.Module):
    def __init__(self, input_dim=64, output_dim=1, pooling_mode='max', net='minet'):
        super(Score_pooling, self).__init__()
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode
        self.fc = nn.Linear(input_dim, output_dim)
        self.net = net
    
    def choice_pooling(self, x):
        if self.pooling_mode == 'max':
            return torch.max(x, dim=0, keepdim=True)[0]
        if self.pooling_mode == 'lse':
            return torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))
        if self.pooling_mode == 'ave':
            return torch.mean(x, dim=0, keepdim=True)

    def forward(self, x):
        if self.net == 'MInet':
            x = self.choice_pooling(x)
            emb = x

        x = self.fc(x)
        output = torch.sigmoid(x)

        if self.net == 'minet':
            output = self.choice_pooling(output)
            emb = None

        # if self.net == 'minet':
        #     x = self.fc(x)
        #     x = torch.sigmoid(x)
        #     output = self.choice_pooling(x)
        # else: #MI-net
        #     x = self.choice_pooling(x)
        #     x = self.fc(x)
        #     output = torch.sigmoid(x)
        return output, emb