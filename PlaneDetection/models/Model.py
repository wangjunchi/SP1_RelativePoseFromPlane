"""
Model framework used from "https://github.com/JunjH/Revisiting_Single_Depth_Estimation". Only the last module is
different to estimate planes.
"""
import torch

import torch.nn as nn

from PlaneDetection.models import modules
from PlaneDetection.models import senet

class Model(nn.Module):
    def __init__(self, num_features, block_channel, pretrained='imagenet'):

        super(Model, self).__init__()

        self.E = modules.E_senet(senet.senet154(pretrained=None))
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out
