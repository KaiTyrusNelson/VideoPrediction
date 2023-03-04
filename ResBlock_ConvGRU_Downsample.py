import torch
import torch.nn as nn
import convgru
import Resblock

class ResBlock_ConvGRU_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock_ConvGRU_Downsample, self).__init__()
        
        self.out_channels = out_channels

        # ResBlock
        res = Resblock.ResBlock(in_channels, out_channels)
        
        # ConvGRU TO MAKE
        
        self.conv_gru = convgru.ConvGRU(out_channels, 3)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        
        # Downsampling
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, hidden_state = None):
        # ResBlock
        x = res(x)
        # ConvGRU
        
        h = self.conv_gru(x, hidden_state)
        x = self.relu3(h)
        x = self.bn3(x)
        
        # FIX THIS!!!! THIS NEEDS TO BE PASSED A HIDDEN STATE
        residual = x
        # TAKES THE RESIDUAL FOR CONCATENTATION
        
        # Downsampling
        x = self.downsample(x)
        
        return x, h, residual